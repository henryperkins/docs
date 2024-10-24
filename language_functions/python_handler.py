"""
python_handler.py

This module defines the PythonHandler class, which is responsible for extracting code structures, inserting docstrings, and validating Python code. It utilizes the radon library for complexity metrics and libcst for code transformations.
"""

import logging
import os
import tempfile
import subprocess
import ast
from typing import Dict, Any, Optional, List, Union

# Import metrics calculation
from metrics import calculate_code_metrics  # Assuming this calculates all necessary metrics

# External dependencies with proper error handling
try:
    from radon.complexity import cc_visit
    from radon.metrics import h_visit, mi_visit
except ImportError:
    logging.error("radon is not installed. Please install it using 'pip install radon'.")
    raise

try:
    import libcst as cst
    from libcst import FunctionDef, ClassDef, SimpleStatementLine, Expr, SimpleString
except ImportError:
    logging.error("libcst is not installed. Please install it using 'pip install libcst'.")
    raise

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class PythonHandler(BaseHandler):
    """Handler for Python language analysis and metrics calculation."""

    def __init__(self, function_schema: Dict[str, Any]):
        """Initialize the Python handler."""
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extracts the structure of the Python code."""
        try:
            tree = ast.parse(code)
            code_structure = {
                "classes": [],
                "functions": [],
                "variables": [],
                "constants": [],
                "decorators": [],
                "context_managers": [],
                "comprehensions": [],
            }

            if metrics is None:
                metrics = calculate_code_metrics(code)

            halstead_metrics = metrics.get("halstead", {})
            function_complexity = metrics.get("function_complexity", {})

            class CodeVisitor(ast.NodeVisitor):
                def __init__(self, file_path: str):
                    self.scope_stack = []
                    self.file_path = file_path
                    self.comments = self._extract_comments(code, tree)
                    self.current_class = None  # Keep track of current class

                def _extract_comments(self, code: str, tree: ast.AST) -> Dict[int, List[str]]:
                    comments = {}
                    for lineno, line in enumerate(code.splitlines(), start=1):
                        if line.strip().startswith("#"):
                            comments.setdefault(lineno, []).append(line.strip().lstrip("#").strip())
                    return comments

                def _get_method_type(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
                    if isinstance(node, ast.AsyncFunctionDef):
                        return "async"
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name) and decorator.id in ["classmethod", "staticmethod"]:
                            return decorator.id
                        elif isinstance(decorator, ast.Attribute) and decorator.attr in ["classmethod", "staticmethod"]:
                            return decorator.attr  # Handle @decorators.classmethod
                    return "instance"

                def visit_FunctionDef(self, node: ast.FunctionDef):
                    self._visit_function(node)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                    self._visit_function(node, is_async=True)

                def _visit_function(self, node, is_async=False):
                    self.scope_stack.append(node)
                    full_name = ".".join([scope.name for scope in self.scope_stack if hasattr(scope, "name")])
                    complexity = function_complexity.get(full_name, 0)
                    func_halstead = halstead_metrics.get(full_name, {})
                    decorators = [ast.unparse(d) for d in node.decorator_list] if hasattr(ast, "unparse") else []
                    docstring = ast.get_docstring(node) or ""
                    args_info = self._process_function_args(node)
                    function_info = {
                        "name": node.name,
                        "docstring": docstring,
                        "args": args_info,  # Use args_info here
                        "async": is_async,
                        "complexity": complexity,
                        "halstead": func_halstead,
                        "decorators": decorators,
                        "lineno": node.lineno,
                        "end_lineno": getattr(node, 'end_lineno', None),  # Include end_lineno
                        "type": self._get_method_type(node) if self.current_class else "function",
                    }
                    if not self.current_class:  # Add to functions only if not in a class
                        code_structure["functions"].append(function_info)
                    self.generic_visit(node)
                    self.scope_stack.pop()


                def _process_function_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict[str, Any]]:
                    """Process function arguments and their annotations."""
                    args_info = []
                    for arg in node.args.args:
                        if arg.arg != "self":  # Skip self parameter
                            arg_info = {
                                "name": arg.arg,
                                "annotation": ast.unparse(arg.annotation) if arg.annotation and hasattr(ast, "unparse") else None,
                                "has_default": False  # You can add logic for default values later
                            }
                            args_info.append(arg_info)
                    return args_info


                def visit_ClassDef(self, node: ast.ClassDef):
                    """Visit a class definition node."""
                    prev_class = self.current_class
                    self.current_class = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
                        "methods": [],
                        "bases": [ast.unparse(base) for base in node.bases] if hasattr(ast, "unparse") else [],
                        "decorators": [ast.unparse(d) for d in node.decorator_list] if hasattr(ast, "unparse") else [],
                        "lineno": node.lineno,
                        "end_lineno": getattr(node, 'end_lineno', None),
                        "complexity": function_complexity.get(node.name, 0),
                        "halstead": halstead_metrics.get(node.name, {}),
                    }
                    code_structure["classes"].append(self.current_class)

                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self.visit_FunctionDef(item) if isinstance(item, ast.FunctionDef) else self.visit_AsyncFunctionDef(item)
                            if code_structure["functions"]:  # Check for functions to pop
                                method_info = code_structure["functions"].pop()
                                # Check if method_info is for the current class. This is crucial!
                                if method_info.get("lineno", 0) >= node.lineno and method_info.get("end_lineno", 0) <= getattr(node, "end_lineno", float('inf')):
                                    self.current_class["methods"].append(method_info)


                    self.generic_visit(node)
                    self.current_class = prev_class


                def visit_Assign(self, node: ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            is_constant = var_name.isupper()
                            var_type = self._infer_type(node.value)  # Corrected type inference
                            var_info = {
                                "name": var_name,
                                "type": var_type,
                                "description": self._extract_description(target.lineno),
                                "file": os.path.basename(self.file_path),
                                "line": target.lineno,
                                "value": ast.unparse(node.value) if hasattr(ast, "unparse") else str(node.value),  # Get value
                                "comments": self.comments.get(target.lineno, []),
                            }
                            if is_constant:
                                code_structure["constants"].append(var_info)
                            else:
                                code_structure["variables"].append(var_info)
                    self.generic_visit(node)


                def _infer_type(self, value: ast.AST) -> str:
                    if isinstance(value, ast.Constant):
                        return type(value.value).__name__
                    # Add more type inference logic as needed
                    return "Unknown"


                def _extract_description(self, lineno: int) -> str:
                    return " ".join(self.comments.get(lineno - 1, []) + self.comments.get(lineno, [])) or "No description provided."


                def visit_With(self, node: ast.With):
                    for item in node.items:
                        if isinstance(item.context_expr, ast.Call):
                            context_manager = ast.unparse(item.context_expr) if hasattr(ast, "unparse") else ""
                            if context_manager:
                                code_structure["context_managers"].append(context_manager)
                    self.generic_visit(node)

                def visit_AsyncWith(self, node: ast.AsyncWith):
                    for item in node.items:
                        if isinstance(item.context_expr, ast.Call):
                            context_manager = f"async {ast.unparse(item.context_expr)}" if hasattr(ast, "unparse") else ""
                            if context_manager:
                                code_structure["context_managers"].append(context_manager)
                    self.generic_visit(node)

                def visit_ListComp(self, node: ast.ListComp):
                    code_structure["comprehensions"].append("ListComprehension")
                    self.generic_visit(node)

                def visit_DictComp(self, node: ast.DictComp):
                    code_structure["comprehensions"].append("DictComprehension")
                    self.generic_visit(node)

                def visit_SetComp(self, node: ast.SetComp):
                    code_structure["comprehensions"].append("SetComprehension")
                    self.generic_visit(node)

                def visit_GeneratorExp(self, node: ast.GeneratorExp):
                    code_structure["comprehensions"].append("GeneratorExpression")
                    self.generic_visit(node)


            visitor = CodeVisitor(file_path)
            visitor.visit(tree)

            # Add metrics to the structure
            code_structure["metrics"] = metrics  # Include all calculated metrics

            logger.debug(f"Extracted structure for '{file_path}': {code_structure}")
            return code_structure

        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e.text.strip()} at line {e.lineno}, offset {e.offset}")
            return {}
        except Exception as e:
            logger.error(f"Error extracting Python structure: {e}", exc_info=True)
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts docstrings into the Python code based on the provided documentation.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The source code with inserted documentation.
        """
        try:
            tree = ast.parse(code)
            for func in documentation.get("functions", []):
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == func["name"]:
                        node.body.insert(0, ast.Expr(value=ast.Str(s=func["docstring"])))
            return ast.unparse(tree)
        except Exception as e:
            logger.error(f"Error inserting docstrings: {e}")
            raise

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates the Python code using a linter like pylint.

        Args:
            code (str): The Python code to validate.
            file_path (Optional[str]): The path to the file being validated.

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
                temp_file.write(code.encode("utf-8"))
                temp_file_path = temp_file.name

            result = subprocess.run(["pylint", temp_file_path], capture_output=True, text=True)
            os.unlink(temp_file_path)

            if result.returncode == 0:
                logger.info("Code validation passed.")
                return True
            else:
                logger.error(f"Code validation failed: {result.stdout}\n{result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error validating code: {e}")
            return False

    def _format_google_docstring(self, docstring: str) -> str:
        """
        Formats a docstring in Google style.

        Args:
            docstring (str): The original docstring.

        Returns:
            str: The formatted docstring.
        """
        # Placeholder for actual Google style formatting
        return docstring

# Example usage for testing purposes
if __name__ == "__main__":
    sample_code = """
def add(a, b):
    return a + b

class Calculator:
    def subtract(self, a, b):
        if a > b:
            return a - b
        else:
            return b - a
    """

    handler = PythonHandler(function_schema={})
    structure = handler.extract_structure(sample_code, "sample.py")
    print(structure)