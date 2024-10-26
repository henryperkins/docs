# python_handler.py

import ast
import logging
import subprocess
import tempfile
from typing import Dict, Any, Optional, List
from jsonschema import validate, ValidationError
from radon.complexity import cc_visit, cc_rank
from radon.metrics import mi_visit, h_visit

from .base_handler import BaseHandler
from metrics import (
    calculate_code_metrics,
    DEFAULT_EMPTY_METRICS
)

logger = logging.getLogger(__name__)

class PythonHandler(BaseHandler):
    """Handler for Python language analysis and metrics calculation."""

    def __init__(self, function_schema: Dict[str, Any]):
        """Initialize the Python handler."""
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extracts the structure of the Python code and calculates complexity."""

        try:
            if metrics is None:
                metrics_result = calculate_code_metrics(code, file_path, language="python")
                if metrics_result.success:
                    metrics = metrics_result.metrics
                else:
                    logger.warning(f"Metrics calculation failed for {file_path}: {metrics_result.error}")
                    metrics = DEFAULT_EMPTY_METRICS

            tree = ast.parse(code)
            code_structure = {
                "docstring_format": "Google",
                "summary": "",
                "changes_made": [],
                "functions": [],
                "classes": [],
                "variables": [],
                "constants": [],
                "imports": [],
                "metrics": metrics,
            }

            class CodeVisitor(ast.NodeVisitor):
                def __init__(self, file_path: str):
                    self.scope_stack = []
                    self.file_path = file_path
                    self.current_class = None

                def _calculate_complexity(self, node):
                    try:
                        complexity_blocks = cc_visit(ast.unparse(node))
                        total_complexity = sum(block.complexity for block in complexity_blocks)
                        complexity_rank = cc_rank(total_complexity)
                        return total_complexity, complexity_rank
                    except Exception as e:
                        logger.error(f"Error calculating complexity: {e}")
                        return 0, "A"

                def visit_Module(self, node):
                    code_structure["summary"] = ast.get_docstring(node) or ""
                    for n in node.body:
                        if isinstance(n, (ast.Import, ast.ImportFrom)):
                            module_name = n.module if isinstance(n, ast.ImportFrom) else None
                            for alias in n.names:
                                imported_name = alias.name
                                full_import_path = f"{module_name}.{imported_name}" if module_name else imported_name
                                code_structure["imports"].append(full_import_path)
                    self.generic_visit(node)

                def visit_FunctionDef(self, node):
                    self._visit_function(node)

                def visit_AsyncFunctionDef(self, node):
                    self._visit_function(node, is_async=True)

                def _visit_function(self, node, is_async=False):
                    self.scope_stack.append(node)
                    function_info = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
                        "args": self._get_args(node),  # Extract arguments with type annotations
                        "async": is_async,
                        "returns": self._get_return_type(node),  # Extract return type
                        "lineno": node.lineno,
                        "end_lineno": node.end_lineno,
                    }

                    complexity, rank = self._calculate_complexity(node)
                    function_info["complexity"] = complexity
                    function_info["complexity_rank"] = rank

                    code_structure["functions"].append(function_info)
                    self.generic_visit(node)
                    self.scope_stack.pop()

                def _get_args(self, node):
                    """Extracts function arguments with type annotations."""
                    args = []
                    for arg in node.args.args:
                        if arg.arg != "self":
                            arg_info = {"name": arg.arg}
                            if arg.annotation:
                                arg_info["type"] = ast.unparse(arg.annotation)
                            args.append(arg_info)
                    return args

                def _get_return_type(self, node):
                    """Extracts the return type annotation."""
                    if node.returns:
                        return ast.unparse(node.returns)
                    return None

                def visit_ClassDef(self, node):
                    self.current_class = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
                        "methods": [],
                        "lineno": node.lineno,
                        "end_lineno": node.end_lineno,
                    }
                    complexity, rank = self._calculate_complexity(node)
                    self.current_class["complexity"] = complexity
                    self.current_class["complexity_rank"] = rank

                    code_structure["classes"].append(self.current_class)
                    self.scope_stack.append(node)
                    self.generic_visit(node)
                    self.scope_stack.pop()
                    self.current_class = None

                def visit_Assign(self, node):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            is_constant = var_name.isupper()
                            try:
                                var_value = ast.literal_eval(node.value)
                                var_type = type(var_value).__name__
                            except ValueError:
                                var_value = None
                                var_type = "Unknown"

                            var_info = {
                                "name": var_name,
                                "type": var_type,
                                "value": var_value,
                                "lineno": node.lineno,
                                "end_lineno": node.end_lineno,
                            }
                            if is_constant:
                                code_structure["constants"].append(var_info)
                            else:
                                code_structure["variables"].append(var_info)

            visitor = CodeVisitor(file_path)
            visitor.visit(tree)

            try:
                validate(instance=code_structure, schema=self.function_schema)
            except ValidationError as e:
                logger.warning(f"Schema validation failed: {e}")

            return code_structure

        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e.text.strip()} at line {e.lineno}, offset {e.offset}")
            return {"error": str(e), "metrics": DEFAULT_EMPTY_METRICS}
        except Exception as e:
            logger.error(f"Error extracting Python structure from {file_path}: {e}", exc_info=True)
            return {"error": str(e), "metrics": DEFAULT_EMPTY_METRICS}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """Inserts docstrings into the Python code."""
        try:
            tree = ast.parse(code)
            docstring_format = documentation.get("docstring_format", "Google")
            transformer = DocstringTransformer(documentation, docstring_format, preserve_existing=False)
            modified_tree = transformer.visit(tree)
            return ast.unparse(modified_tree)
        except Exception as e:
            logger.error(f"Error inserting docstrings: {e}", exc_info=True)
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """Validates the Python code using pylint."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
                temp_file.write(code.encode("utf-8"))
                temp_file_path = temp_file.name

            result = subprocess.run(
                ["pylint", temp_file_path],
                capture_output=True,
                text=True,
                check=False
            )
            os.unlink(temp_file_path)

            if result.returncode == 0:
                logger.info("Code validation passed.")
                return True
            else:
                logger.error(f"Code validation failed: {result.stdout}\n{result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error validating code: {e}", exc_info=True)
            return False

class DocstringTransformer(ast.NodeTransformer):
    """Transformer for inserting docstrings into AST nodes."""

    def __init__(self, documentation: Dict[str, Any], docstring_format: str, preserve_existing=False):
        self.documentation = documentation
        self.docstring_format = docstring_format
        self.preserve_existing = preserve_existing

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Adds or updates docstring to function definitions."""
        for func in self.documentation.get("functions", []):
            if func["name"] == node.name:
                if not self.preserve_existing or not ast.get_docstring(node):
                    docstring = self._format_docstring(func["docstring"], self.docstring_format, func.get("args", []), func.get("returns"))
                    node.docstring = docstring
                break
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Adds or updates docstring to async function definitions."""
        for func in self.documentation.get("functions", []):
            if func["name"] == node.name:
                if not self.preserve_existing or not ast.get_docstring(node):
                    docstring = self._format_docstring(func["docstring"], self.docstring_format, func.get("args", []), func.get("returns"))
                    node.docstring = docstring
                break
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Adds or updates docstring to class definitions."""
        for cls in self.documentation.get("classes", []):
            if cls["name"] == node.name:
                if not self.preserve_existing or not ast.get_docstring(node):
                    docstring = self._format_docstring(cls["docstring"], self.docstring_format)
                    node.docstring = docstring
                break
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Adds or updates docstring to the module."""
        if not self.preserve_existing or not ast.get_docstring(node):
            docstring = self._format_docstring(self.documentation.get("summary", ""), self.docstring_format)
            node.docstring = docstring
        return node

    def _format_docstring(self, docstring: str, format: str = "Google", args: List[Dict] = None, returns: Optional[str] = None) -> str:
        """Formats the docstring according to the specified format."""
        if format == "Google":
            formatted_docstring = docstring.strip() + "\n\n"
            if args:
                formatted_docstring += "Args:\n"
                for arg in args:
                    formatted_docstring += f"    {arg['name']} ({arg.get('type', 'Any')}): {arg.get('description', '')}\n"
            if returns:
                formatted_docstring += f"\nReturns:\n    {returns}\n"
            return formatted_docstring
        elif format == "NumPy":
            # ... (Implement NumPy formatting)
            pass
        # ... (Handle other formats)
        return docstring
