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
# from metrics import calculate_all_metrics
# from ..metrics import calculate_all_metrics
from metrics import calculate_all_metrics
# External dependencies
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
    """Handler for Python language."""

    def __init__(self, function_schema: Dict[str, Any]):
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str) -> Dict[str, Any]:
        """
        Extracts the structure of the Python code, analyzing functions, classes, and assignments.

        Args:
            code (str): The source code to analyze.
            file_path (str): The file path for code reference.

        Returns:
            Dict[str, Any]: A detailed structure of the code components.
        """
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

            # Calculate all metrics
            metrics = calculate_all_metrics(code)
        
            # Add metrics to code structure
            code_structure.update({
                "halstead": metrics["halstead"],
                "complexity": metrics["complexity"],
                "maintainability_index": metrics["maintainability_index"]
            })
        
            # Store function complexity for use in visitor
            function_complexity = metrics["function_complexity"]

            class CodeVisitor(ast.NodeVisitor):
                """AST visitor for traversing Python code structures and extracting functional and class definitions."""

                def __init__(self, file_path: str):
                    """Initializes the CodeVisitor for traversing AST nodes."""
                    self.scope_stack = []
                    self.file_path = file_path
                    self.comments = self._extract_comments(code, tree)

                def _extract_comments(self, code: str, tree: ast.AST) -> Dict[int, List[str]]:
                    comments = {}
                    for lineno, line in enumerate(code.splitlines(), start=1):
                        stripped = line.strip()
                        if stripped.startswith("#"):
                            comment = stripped.lstrip("#").strip()
                            comments.setdefault(lineno, []).append(comment)
                    return comments

                def _get_method_type(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
                    """
                    Determines the type of method based on decorators and context.

                    Args:
                        node: The AST node for the method.

                    Returns:
                        str: The method type (instance, class, static, or async).
                    """
                    if isinstance(node, ast.AsyncFunctionDef):
                        return "async"

                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name):
                            if decorator.id == "classmethod":
                                return "class"
                            elif decorator.id == "staticmethod":
                                return "static"
                        elif isinstance(decorator, ast.Attribute):
                            # Handle cases like @decorators.classmethod
                            if decorator.attr in ["classmethod", "staticmethod"]:
                                return decorator.attr
                    return "instance"

                def visit_FunctionDef(self, node: ast.FunctionDef):
                    self._visit_function(node)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                    self._visit_function(node, is_async=True)

                def _visit_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool = False) -> None:
                    self.scope_stack.append(node)
                    full_name = ".".join([scope.name for scope in self.scope_stack if hasattr(scope, "name")])
                    complexity = function_complexity.get(full_name, 0)
                    decorators = [ast.unparse(d) for d in node.decorator_list] if hasattr(ast, "unparse") else []
                    docstring = ast.get_docstring(node) or ""
                    function_info = {
                        "name": node.name,
                        "docstring": docstring,
                        "args": [arg.arg for arg in node.args.args if arg.arg != "self"],
                        "async": is_async,
                        "complexity": complexity,
                        "decorators": decorators,
                    }
                    if not any(isinstance(parent, ast.ClassDef) for parent in self.scope_stack[:-1]):
                        code_structure["functions"].append(function_info)
                    self.generic_visit(node)
                    self.scope_stack.pop()

                def visit_ClassDef(self, node: ast.ClassDef):
                    self.scope_stack.append(node)
                    class_docstring = ast.get_docstring(node) or ""
                    class_info = {
                        "name": node.name,
                        "docstring": class_docstring,
                        "methods": [],
                        "decorators": [ast.unparse(d) for d in node.decorator_list] if hasattr(ast, "unparse") else [],
                    }
                    for body_item in node.body:
                        if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self.scope_stack.append(body_item)
                            full_method_name = ".".join(
                                [scope.name for scope in self.scope_stack if hasattr(scope, "name")]
                            )
                            complexity = function_complexity.get(full_method_name, 0)
                            decorators = (
                                [ast.unparse(d) for d in body_item.decorator_list] if hasattr(ast, "unparse") else []
                            )
                            method_docstring = ast.get_docstring(body_item) or ""
                            method_info = {
                                "name": body_item.name,
                                "docstring": method_docstring,
                                "args": [arg.arg for arg in body_item.args.args if arg.arg != "self"],
                                "async": isinstance(body_item, ast.AsyncFunctionDef),
                                "complexity": complexity,
                                "decorators": decorators,
                                "type": self._get_method_type(body_item),
                            }
                            class_info["methods"].append(method_info)
                            self.scope_stack.pop()
                    code_structure["classes"].append(class_info)
                    self.generic_visit(node)
                    self.scope_stack.pop()

                def visit_Assign(self, node: ast.Assign):
                    for target in node.targets:
                        self._process_target(target, node.value)
                    self.generic_visit(node)

                def _process_target(self, target: ast.AST, value: ast.AST) -> None:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        is_constant = var_name.isupper()
                        var_type = self._infer_type(value)
                        description = self._extract_description(target.lineno)
                        example = self._extract_example(target.lineno)
                        references = self._extract_references(target.lineno)
                        var_info = {
                            "name": var_name,
                            "type": var_type,
                            "description": description,
                            "file": os.path.basename(self.file_path),
                            "line": target.lineno,
                            "link": f"https://github.com/user/repo/blob/main/{self.file_path}#L{target.lineno}",
                            "example": example,
                            "references": references,
                        }
                        if is_constant:
                            code_structure["constants"].append(var_info)
                        else:
                            code_structure["variables"].append(var_info)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for elt in target.elts:
                            self._process_target(elt, value)
                    elif isinstance(target, ast.Attribute):
                        var_name = target.attr
                        is_constant = var_name.isupper()
                        var_type = self._infer_type(value)
                        description = self._extract_description(target.lineno)
                        example = self._extract_example(target.lineno)
                        references = self._extract_references(target.lineno)
                        var_info = {
                            "name": var_name,
                            "type": var_type,
                            "description": description,
                            "file": os.path.basename(self.file_path),
                            "line": target.lineno,
                            "link": f"https://github.com/user/repo/blob/main/{self.file_path}#L{target.lineno}",
                            "example": example,
                            "references": references,
                        }
                        if is_constant:
                            code_structure["constants"].append(var_info)
                        else:
                            code_structure["variables"].append(var_info)

                def _infer_type(self, value: ast.AST) -> str:
                    if isinstance(value, ast.Constant):
                        return type(value.value).__name__
                    elif isinstance(value, ast.List):
                        return "List"
                    elif isinstance(value, ast.Tuple):
                        return "Tuple"
                    elif isinstance(value, ast.Dict):
                        return "Dict"
                    elif isinstance(value, ast.Set):
                        return "Set"
                    elif isinstance(value, ast.Call):
                        return "Call"
                    elif isinstance(value, ast.BinOp):
                        return "BinOp"
                    elif isinstance(value, ast.UnaryOp):
                        return "UnaryOp"
                    elif isinstance(value, ast.Lambda):
                        return "Lambda"
                    elif isinstance(value, ast.Name):
                        return "Name"
                    else:
                        return "Unknown"

                def _extract_description(self, lineno: int) -> str:
                    comments = self.comments.get(lineno - 1, []) + self.comments.get(lineno, [])
                    if comments:
                        return " ".join(comments)
                    return "No description provided."

                def _extract_example(self, lineno: int) -> str:
                    comments = self.comments.get(lineno + 1, [])
                    if comments:
                        return " ".join(comments)
                    return "No example provided."

                def _extract_references(self, lineno: int) -> str:
                    comments = self.comments.get(lineno + 2, [])
                    if comments:
                        return " ".join(comments)
                    return "N/A"

                def visit_With(self, node: ast.With):
                    for item in node.items:
                        if isinstance(item.context_expr, ast.Call):
                            context_manager = ast.unparse(item.context_expr) if hasattr(ast, "unparse") else ""
                            code_structure.setdefault("context_managers", []).append(context_manager)
                    self.generic_visit(node)

                def visit_AsyncWith(self, node: ast.AsyncWith):
                    for item in node.items:
                        if isinstance(item.context_expr, ast.Call):
                            context_manager = ast.unparse(item.context_expr) if hasattr(ast, "unparse") else ""
                            code_structure.setdefault("context_managers", []).append(context_manager)
                    self.generic_visit(node)

                def visit_ListComp(self, node: ast.ListComp):
                    code_structure.setdefault("comprehensions", []).append("ListComprehension")
                    self.generic_visit(node)

                def visit_DictComp(self, node: ast.DictComp):
                    code_structure.setdefault("comprehensions", []).append("DictComprehension")
                    self.generic_visit(node)

                def visit_SetComp(self, node: ast.SetComp):
                    code_structure.setdefault("comprehensions", []).append("SetComprehension")
                    self.generic_visit(node)

                def visit_GeneratorExp(self, node: ast.GeneratorExp):
                    code_structure.setdefault("comprehensions", []).append("GeneratorExpression")
                    self.generic_visit(node)

            visitor = CodeVisitor(file_path)
            visitor.visit(tree)
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
        logger.debug("Starting docstring insertion for Python code (Google Style).")
        try:
            docstrings_mapping = {}
            for func_doc in documentation.get("functions", []):
                name = func_doc.get("name")
                if name:
                    docstrings_mapping[name] = self._format_google_docstring(func_doc)
            for class_doc in documentation.get("classes", []):
                class_name = class_doc.get("name")
                if class_name:
                    docstrings_mapping[class_name] = self._format_google_docstring(class_doc)
                    for method_doc in class_doc.get("methods", []):
                        method_name = method_doc.get("name")
                        if method_name:
                            full_method_name = f"{class_name}.{method_name}"
                            docstrings_mapping[full_method_name] = self._format_google_docstring(method_doc)

            class DocstringInserter(cst.CSTTransformer):
                def __init__(self, docstrings_mapping: Dict[str, str]):
                    self.docstrings_mapping = docstrings_mapping
                    self.scope_stack = []

                def visit_FunctionDef(self, node: FunctionDef):
                    self.scope_stack.append(node.name.value)

                def leave_FunctionDef(self, original_node: FunctionDef, updated_node: FunctionDef) -> FunctionDef:
                    full_name = ".".join(self.scope_stack)
                    docstring = self.docstrings_mapping.get(full_name)
                    if docstring and not original_node.get_docstring():
                        new_doc = SimpleStatementLine([Expr(SimpleString(f'"""{docstring}"""'))])
                        new_body = [new_doc] + list(updated_node.body.body)
                        updated_node = updated_node.with_changes(body=updated_node.body.with_changes(body=new_body))
                        logger.debug(f"Inserted docstring for function: {full_name}")
                    self.scope_stack.pop()
                    return updated_node

                def visit_ClassDef(self, node: ClassDef):
                    self.scope_stack.append(node.name.value)

                def leave_ClassDef(self, original_node: ClassDef, updated_node: ClassDef) -> ClassDef:
                    full_name = ".".join(self.scope_stack)
                    docstring = self.docstrings_mapping.get(full_name)
                    if docstring and not original_node.get_docstring():
                        new_doc = SimpleStatementLine([Expr(SimpleString(f'"""{docstring}"""'))])
                        new_body = [new_doc] + list(updated_node.body.body)
                        updated_node = updated_node.with_changes(body=updated_node.body.with_changes(body=new_body))
                        logger.debug(f"Inserted docstring for class: {full_name}")
                    self.scope_stack.pop()
                    return updated_node

            tree = cst.parse_module(code)
            inserter = DocstringInserter(docstrings_mapping)
            modified_tree = tree.visit(inserter)
            modified_code = modified_tree.code
            logger.debug("Docstring insertion completed successfully.")
            return modified_code
        except Exception as e:
            logger.error(f"Error inserting docstrings: {e}", exc_info=True)
            return code

    def _format_google_docstring(self, doc: Dict[str, Any]) -> str:
        """
        Formats a docstring in Google style.

        Args:
            doc (Dict[str, Any]): The documentation details.

        Returns:
            str: The formatted docstring.
        """
        docstring = f'{doc.get("docstring", "")}\n\n'

        arguments = doc.get("arguments", [])
        if arguments:
            docstring += "Args:\n"
            for arg in arguments:
                arg_name = arg.get("name", "unknown")
                arg_type = arg.get("type", "Any")
                arg_description = arg.get("description", "")
                default_value = arg.get("default_value")

                docstring += f"    {arg_name} ({arg_type}): {arg_description}"
                if default_value is not None:
                    docstring += f" (Default: {default_value})"
                docstring += "\n"

        return_type = doc.get("return_type")
        return_description = doc.get("return_description", "")
        if return_type:
            docstring += f"\nReturns:\n    {return_type}: {return_description}\n"

        return docstring.strip()

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates the modified Python code for syntax correctness.

        Args:
            code (str): The modified source code.
            file_path (Optional[str]): Path to the Python source file (optional).

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        logger.debug("Starting Python code validation.")
        try:
            ast.parse(code)
            logger.debug("Syntax validation passed.")
            if file_path:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
                    tmp.write(code)
                    temp_file = tmp.name
                try:
                    result = subprocess.run(
                        ["flake8", temp_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    if result.returncode != 0:
                        logger.error(f"Flake8 validation failed for {file_path}:\n{result.stdout}\n{result.stderr}")
                        return False
                    else:
                        logger.debug("Flake8 validation passed.")
                except FileNotFoundError:
                    logger.error(
                        "flake8 is not installed or not found in PATH. Please install it using 'pip install flake8'."
                    )
                    return False
                except subprocess.SubprocessError as e:
                    logger.error(f"Subprocess error during flake8 execution: {e}")
                    return False
                finally:
                    try:
                        os.remove(temp_file)
                    except OSError as e:
                        logger.error(f"Error deleting temporary file {temp_file}: {e}")
            else:
                logger.warning("File path not provided for flake8 validation. Skipping flake8.")
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error during validation: {e.text.strip()} at line {e.lineno}, offset {e.offset}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during code validation: {e}", exc_info=True)
            return False
