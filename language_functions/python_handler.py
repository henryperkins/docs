import logging
import os
import sys
import tempfile
import subprocess
import ast
from typing import Dict, Any, Optional, List

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

                def __init__(self, file_path: str):
                    """Initializes the CodeVisitor for traversing AST nodes."""
                    self.scope_stack = []
                    self.file_path = file_path
                    # To track parent nodes for comment extraction
                    self.comments = self._extract_comments(code, tree)

                def _extract_comments(self, code: str, tree: ast.AST) -> Dict[int, List[str]]:
                    """Extracts comments from the source code and maps them to line numbers.

                    Args:
                        code (str): The source code.
                        tree (ast.AST): The parsed AST.

                    Returns:
                        Dict[int, List[str]]: Mapping from line numbers to list of comments.
                    """
                    comments = {}
                    for lineno, line in enumerate(code.splitlines(), start=1):
                        stripped = line.strip()
                        if stripped.startswith("#"):
                            comment = stripped.lstrip("#").strip()
                            comments.setdefault(lineno, []).append(comment)
                    return comments

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
                    """Visits a class definition node."""
                    self.scope_stack.append(node)
                    class_docstring = ast.get_docstring(node) or ""
                    class_info = {
                        "name": node.name,
                        "docstring": class_docstring,
                        "methods": [],
                        "decorators": [ast.unparse(d) for d in node.decorator_list] if hasattr(ast, 'unparse') else []
                    }
                    for body_item in node.body:
                        if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self.scope_stack.append(body_item)
                            full_method_name = ".".join([scope.name for scope in self.scope_stack if hasattr(scope, 'name')])
                            complexity = function_complexity.get(full_method_name, 0)
                            decorators = [ast.unparse(d) for d in body_item.decorator_list] if hasattr(ast, 'unparse') else []
                            method_docstring = ast.get_docstring(body_item) or ""
                            method_info = {
                                "name": body_item.name,
                                "docstring": method_docstring,
                                "args": [arg.arg for arg in body_item.args.args if arg.arg != "self"],
                                "async": isinstance(body_item, ast.AsyncFunctionDef),
                                "complexity": complexity,
                                "decorators": decorators,
                                "type": "async" if isinstance(body_item, ast.AsyncFunctionDef) else "instance"
                            }
                            class_info["methods"].append(method_info)
                            self.scope_stack.pop()
                    code_structure["classes"].append(class_info)
                    self.generic_visit(node)
                    self.scope_stack.pop()

                def visit_Assign(self, node: ast.Assign):
                    """Processes assignment nodes for extracting variable information."""
                    for target in node.targets:
                        self._process_target(target, node.value)
                    self.generic_visit(node)

                def _process_target(self, target: ast.AST, value: ast.AST) -> None:
                    """Recursively processes assignment targets to extract variable information.

                    Args:
                        target (ast.AST): The assignment target node.
                        value (ast.AST): The value assigned to the target.
                    """
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
                            "references": references
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
                            "references": references
                        }
                        if is_constant:
                            code_structure["constants"].append(var_info)
                        else:
                            code_structure["variables"].append(var_info)
                    # Handle other target types if necessary

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

                # Helper methods for detailed information
                def _infer_type(self, value: ast.AST) -> str:
                    """Infers the type of a variable based on its assigned value.

                    Args:
                        value (ast.AST): The value assigned to the variable.

                    Returns:
                        str: The inferred type as a string.
                    """
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
                    """Extracts the description of a variable from inline comments.

                    Args:
                        lineno (int): The line number of the variable assignment.

                    Returns:
                        str: The extracted description.
                    """
                    comments = self.comments.get(lineno - 1, []) + self.comments.get(lineno, [])
                    if comments:
                        return " ".join(comments)
                    return "No description provided."

                def _extract_example(self, lineno: int) -> str:
                    """Extracts an example usage of the variable from comments.

                    Args:
                        lineno (int): The line number of the variable assignment.

                    Returns:
                        str: The example usage.
                    """
                    comments = self.comments.get(lineno + 1, [])
                    if comments:
                        return " ".join(comments)
                    return "No example provided."

                def _extract_references(self, lineno: int) -> str:
                    """Extracts references related to the variable from comments.

                    Args:
                        lineno (int): The line number of the variable assignment.

                    Returns:
                        str: The references.
                    """
                    comments = self.comments.get(lineno + 2, [])
                    if comments:
                        return " ".join(comments)
                    return "N/A"

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
