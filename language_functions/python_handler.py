import logging
import os
import sys
import tempfile
import subprocess
import ast
from typing import Dict, Any, Optional

# External dependencies
try:
    from radon.complexity import cc_visit
    from radon.metrics import h_visit, mi_visit
except ImportError as e:
    logging.error("radon is not installed. Please install it using 'pip install radon'.")
    raise

try:
    import libcst as cst
    from libcst import FunctionDef, ClassDef, SimpleStatementLine, Expr, SimpleString
except ImportError as e:
    logging.error("libcst is not installed. Please install it using 'pip install libcst'.")
    raise

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class PythonHandler(BaseHandler):
    """Handler for Python language."""

    def __init__(self, function_schema: Dict[str, Any]):
        """
        Initializes the PythonHandler with the provided function schema.

        Args:
            function_schema (Dict[str, Any]): The schema used for function operations.
        """
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str) -> Dict[str, Any]:
        """Extracts the structure of the Python code, analyzing functions, classes, and assignments.

        Args:
            code (str): The source code to analyze.
            file_path (Optional[str]): The file path for code reference.

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
                "halstead": {},
                "maintainability_index": None,
                "decorators": [],
                "context_managers": [],
                "comprehensions": [],
            }
            complexity_scores = cc_visit(code)
            function_complexity = {score.fullname: score.complexity for score in complexity_scores}
            halstead_metrics = h_visit(code)
            if halstead_metrics:
                metrics = halstead_metrics[0]
                code_structure["halstead"] = {
                    "volume": metrics.volume,
                    "difficulty": metrics.difficulty,
                    "effort": metrics.effort,
                }
            mi_score = mi_visit(code, True)
            code_structure["maintainability_index"] = mi_score

            class CodeVisitor(ast.NodeVisitor):
                """AST visitor for traversing Python code structures and extracting functional and class definitions."""

                def __init__(self):
                    """Initializes the CodeVisitor for traversing AST nodes."""
                    self.scope_stack = []

                def visit_FunctionDef(self, node: ast.FunctionDef):
                    """Visits a function definition node."""
                    self._visit_function(node)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                    """Visits an async function definition node."""
                    self._visit_function(node, is_async=True)

                def _visit_function(self, node: ast.FunctionDef, is_async: bool = False) -> None:
                    """Handles both sync and async functions."""
                    self.scope_stack.append(node)
                    full_name = ".".join([scope.name for scope in self.scope_stack if hasattr(scope, 'name')])
                    complexity = function_complexity.get(full_name, 0)
                    decorators = [ast.unparse(d) for d in node.decorator_list] if hasattr(ast, 'unparse') else []
                    function_info = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
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
                    """Visits a class definition node."""
                    self.scope_stack.append(node)
                    class_info = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
                        "methods": [],
                        "decorators": [ast.unparse(d) for d in node.decorator_list] if hasattr(ast, 'unparse') else []
                    }
                    for body_item in node.body:
                        if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self.scope_stack.append(body_item)
                            full_method_name = ".".join([scope.name for scope in self.scope_stack if hasattr(scope, 'name')])
                            complexity = function_complexity.get(full_method_name, 0)
                            decorators = [ast.unparse(d) for d in body_item.decorator_list] if hasattr(ast, 'unparse') else []
                            method_info = {
                                "name": body_item.name,
                                "docstring": ast.get_docstring(body_item) or "",
                                "args": [arg.arg for arg in body_item.args.args if arg.arg != "self"],
                                "async": isinstance(body_item, ast.AsyncFunctionDef),
                                "complexity": complexity,
                                "decorators": decorators,
                            }
                            class_info["methods"].append(method_info)
                            self.scope_stack.pop()
                    code_structure["classes"].append(class_info)
                    self.generic_visit(node)
                    self.scope_stack.pop()

                def visit_Assign(self, node: ast.Assign):
                    """Processes assignment nodes for extracting code comprehension details."""
                    for target in node.targets:
                        self._process_target(target)
                    self.generic_visit(node)

                def _process_target(self, target: ast.AST) -> None:
                    """Recursively processes assignment targets to extract variable names.

                    Args:
                        target (ast.AST): The assignment target node.
                    """
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        if var_name.isupper():
                            code_structure["constants"].append(var_name)
                        else:
                            code_structure["variables"].append(var_name)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for elt in target.elts:
                            self._process_target(elt)
                    elif isinstance(target, ast.Attribute):
                        var_name = target.attr
                        if var_name.isupper():
                            code_structure["constants"].append(var_name)
                        else:
                            code_structure["variables"].append(var_name)

                def visit_With(self, node: ast.With):
                    """Processes 'with' statements."""
                    for item in node.items:
                        if isinstance(item.context_expr, ast.Call):
                            context_manager = ast.unparse(item.context_expr) if hasattr(ast, 'unparse') else ""
                            code_structure.setdefault("context_managers", []).append(context_manager)
                    self.generic_visit(node)

                def visit_AsyncWith(self, node: ast.AsyncWith):
                    """Processes 'async with' statements."""
                    for item in node.items:
                        if isinstance(item.context_expr, ast.Call):
                            context_manager = ast.unparse(item.context_expr) if hasattr(ast, 'unparse') else ""
                            code_structure.setdefault("context_managers", []).append(context_manager)
                    self.generic_visit(node)

                def visit_ListComp(self, node: ast.ListComp):
                    """Tracks list comprehensions."""
                    code_structure.setdefault("comprehensions", []).append("ListComprehension")
                    self.generic_visit(node)

                def visit_DictComp(self, node: ast.DictComp):
                    """Tracks dictionary comprehensions."""
                    code_structure.setdefault("comprehensions", []).append("DictComprehension")
                    self.generic_visit(node)

                def visit_SetComp(self, node: ast.SetComp):
                    """Tracks set comprehensions."""
                    code_structure.setdefault("comprehensions", []).append("SetComprehension")
                    self.generic_visit(node)

                def visit_GeneratorExp(self, node: ast.GeneratorExp):
                    """Tracks generator expressions."""
                    code_structure.setdefault("comprehensions", []).append("GeneratorExpression")
                    self.generic_visit(node)

            visitor = CodeVisitor()
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
        Inserts docstrings into Python code based on the provided documentation.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The source code with inserted documentation.
        """
        logger.debug("Starting docstring insertion for Python code.")
        try:
            docstrings_mapping = {}
            for func_doc in documentation.get("functions", []):
                name = func_doc.get("name")
                if name:
                    docstrings_mapping[name] = func_doc.get("docstring", "")
            for class_doc in documentation.get("classes", []):
                class_name = class_doc.get("name")
                if class_name:
                    docstrings_mapping[class_name] = class_doc.get("docstring", "")
                    for method_doc in class_doc.get("methods", []):
                        method_name = method_doc.get("name")
                        if method_name:
                            full_method_name = f"{class_name}.{method_name}"
                            docstrings_mapping[full_method_name] = method_doc.get("docstring", "")

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
                    logger.error("flake8 is not installed or not found in PATH. Please install it using 'pip install flake8'.")
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