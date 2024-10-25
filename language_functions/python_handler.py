"""
python_handler.py

This module defines the PythonHandler class, which is responsible for extracting code structures, 
inserting docstrings, and validating Python code.
"""

import logging
import os
import tempfile
import subprocess
import ast
import asyncio
import aiofiles
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
from .base_handler import BaseHandler


# Import the enhanced metrics calculation
from metrics import (
    calculate_code_metrics,
    MetricsResult,
    HalsteadCalculationError,
    ComplexityCalculationError
)

logger = logging.getLogger(__name__)

# Default empty metrics structure
DEFAULT_EMPTY_METRICS = {
    "maintainability_index": 0.0,
    "cyclomatic": 0.0,
    "halstead": {
        "h1": 0, "h2": 0, "N1": 0, "N2": 0,
        "vocabulary": 0, "length": 0,
        "calculated_length": 0.0, "volume": 0.0,
        "difficulty": 0.0, "effort": 0.0,
        "time": 0.0, "bugs": 0.0
    },
    "function_complexity": {},
    "raw": None,
    "quality": None
}

class PythonHandler(BaseHandler):
    """Handler for Python language analysis and metrics calculation."""

    def __init__(self, function_schema: Dict[str, Any]):
        """Initialize the Python handler."""
        self.function_schema = function_schema

    def extract_structure(
        self, 
        code: str, 
        file_path: str, 
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extracts the structure of the Python code with enhanced metrics handling.

        Args:
            code (str): The source code to analyze.
            file_path (str): Path to the source file.
            metrics (Optional[Dict[str, Any]]): Pre-calculated metrics, if available.

        Returns:
            Dict[str, Any]: The extracted code structure with metrics.
        """
        try:
            # Calculate metrics if not provided
            if metrics is None:
                try:
                    metrics_result = calculate_code_metrics(code, file_path)
                    if metrics_result.success:
                        metrics = metrics_result.metrics
                    else:
                        logger.warning(f"Metrics calculation failed for {file_path}: {metrics_result.error}")
                        metrics = DEFAULT_EMPTY_METRICS
                except (HalsteadCalculationError, ComplexityCalculationError) as e:
                    logger.error(f"Error calculating metrics for {file_path}: {e}")
                    metrics = DEFAULT_EMPTY_METRICS

            tree = ast.parse(code)
            code_structure = {
                "classes": [],
                "functions": [],
                "variables": [],
                "constants": [],
                "decorators": [],
                "context_managers": [],
                "comprehensions": [],
                "metrics": metrics,  # Include metrics in structure
            }

            class CodeVisitor(ast.NodeVisitor):
                def __init__(self, file_path: str):
                    self.scope_stack = []
                    self.file_path = file_path
                    self.comments = self._extract_comments(code, tree)
                    self.current_class = None

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
                            return decorator.attr
                    return "instance"

                def visit_FunctionDef(self, node: ast.FunctionDef):
                    self._visit_function(node)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                    self._visit_function(node, is_async=True)

                def _visit_function(self, node, is_async=False):
                    self.scope_stack.append(node)
                    full_name = ".".join([scope.name for scope in self.scope_stack if hasattr(scope, "name")])
                    
                    function_metrics = metrics.get("function_complexity", {}).get(full_name, {})
                    complexity = function_metrics.get("complexity", 0)
                    halstead = function_metrics.get("halstead", {})
                    
                    decorators = [ast.unparse(d) for d in node.decorator_list] if hasattr(ast, "unparse") else []
                    docstring = ast.get_docstring(node) or ""
                    args_info = self._process_function_args(node)
                    
                    function_info = {
                        "name": node.name,
                        "docstring": docstring,
                        "args": args_info,
                        "async": is_async,
                        "complexity": complexity,
                        "halstead": halstead,
                        "decorators": decorators,
                        "lineno": node.lineno,
                        "end_lineno": getattr(node, 'end_lineno', None),
                        "type": self._get_method_type(node) if self.current_class else "function",
                    }
                    
                    if not self.current_class:
                        code_structure["functions"].append(function_info)
                    
                    self.generic_visit(node)
                    self.scope_stack.pop()

                def _process_function_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict[str, Any]]:
                    args_info = []
                    for arg in node.args.args:
                        if arg.arg != "self":
                            arg_info = {
                                "name": arg.arg,
                                "annotation": ast.unparse(arg.annotation) if arg.annotation and hasattr(ast, "unparse") else None,
                                "has_default": False
                            }
                            args_info.append(arg_info)
                    return args_info

                def visit_ClassDef(self, node: ast.ClassDef):
                    prev_class = self.current_class
                    self.current_class = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
                        "methods": [],
                        "bases": [ast.unparse(base) for base in node.bases] if hasattr(ast, "unparse") else [],
                        "decorators": [ast.unparse(d) for d in node.decorator_list] if hasattr(ast, "unparse") else [],
                        "lineno": node.lineno,
                        "end_lineno": getattr(node, 'end_lineno', None),
                        "complexity": metrics.get("function_complexity", {}).get(node.name, 0),
                        "halstead": metrics.get("halstead", {}),
                    }
                    code_structure["classes"].append(self.current_class)

                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self.visit_FunctionDef(item) if isinstance(item, ast.FunctionDef) else self.visit_AsyncFunctionDef(item)
                            if code_structure["functions"]:
                                method_info = code_structure["functions"].pop()
                                if method_info.get("lineno", 0) >= node.lineno and method_info.get("end_lineno", 0) <= getattr(node, "end_lineno", float('inf')):
                                    self.current_class["methods"].append(method_info)

                    self.generic_visit(node)
                    self.current_class = prev_class

                def visit_Assign(self, node: ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            is_constant = var_name.isupper()
                            var_type = self._infer_type(node.value)
                            var_info = {
                                "name": var_name,
                                "type": var_type,
                                "description": self._extract_description(target.lineno),
                                "file": os.path.basename(self.file_path),
                                "line": target.lineno,
                                "value": ast.unparse(node.value) if hasattr(ast, "unparse") else str(node.value),
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
                    elif isinstance(value, ast.List):
                        return "list"
                    elif isinstance(value, ast.Dict):
                        return "dict"
                    elif isinstance(value, ast.Set):
                        return "set"
                    elif isinstance(value, ast.Tuple):
                        return "tuple"
                    elif isinstance(value, ast.Call):
                        if isinstance(value.func, ast.Name):
                            return value.func.id
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

            logger.debug(f"Extracted structure for '{file_path}': {code_structure}")
            return code_structure

        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e.text.strip()} at line {e.lineno}, offset {e.offset}")
            return {"error": str(e), "metrics": DEFAULT_EMPTY_METRICS}
        except Exception as e:
            logger.error(f"Error extracting Python structure from {file_path}: {e}", exc_info=True)
            return {"error": str(e), "metrics": DEFAULT_EMPTY_METRICS}

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
            transformer = DocstringTransformer(documentation)
            modified_tree = transformer.visit(tree)
            return ast.unparse(modified_tree)
        except Exception as e:
            logger.error(f"Error inserting docstrings: {e}", exc_info=True)
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates the Python code using pylint.

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
    
    def __init__(self, documentation: Dict[str, Any]):
        self.documentation = documentation

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Adds docstring to function definitions."""
        node = self.generic_visit(node)
        for func in self.documentation.get("functions", []):
            if func["name"] == node.name:
                docstring = ast.Expr(value=ast.Str(s=func["docstring"]))
                node.body.insert(0, docstring)
                break
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Adds docstring to class definitions."""
        node = self.generic_visit(node)
        for cls in self.documentation.get("classes", []):
            if cls["name"] == node.name:
                docstring = ast.Expr(value=ast.Str(s=cls["docstring"]))
                node.body.insert(0, docstring)
                break
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Adds docstring to async function definitions."""
        node = self.generic_visit(node)
        for func in self.documentation.get("functions", []):
            if func["name"] == node.name:
                docstring = ast.Expr(value=ast.Str(s=func["docstring"]))
                node.body.insert(0, docstring)
                break
        return node

    def _format_google_docstring(self, docstring: str) -> str:
        """
        Formats a docstring in Google style.

        Args:
            docstring (str): The original docstring.

        Returns:
            str: The formatted docstring.
        """
        sections = {
            "Args": [],
            "Returns": [],
            "Raises": [],
            "Examples": [],
            "Notes": []
        }
        
        current_section = None
        main_description = []
        
        for line in docstring.split('\n'):
            line = line.strip()
            
            # Check if this line starts a new section
            if line.endswith(':') and line[:-1] in sections:
                current_section = line[:-1]
                continue
                
            # Add line to appropriate section
            if current_section:
                sections[current_section].append(line)
            else:
                main_description.append(line)
        
        # Format the docstring
        formatted = '\n'.join(line for line in main_description if line)
        
        for section, lines in sections.items():
            if lines:
                formatted += f"\n\n{section}:\n"
                formatted += '\n'.join(f"    {line}" for line in lines if line)
        
        return formatted

def _validate_docstring_format(self, docstring: str) -> bool:
    """
    Validates that a docstring follows the Google format.

    Args:
        docstring (str): The docstring to validate.

    Returns:
        bool: True if the docstring is valid, False otherwise.
    """
    required_sections = ["Args", "Returns"]
    found_sections = set()
    
    lines = docstring.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line.endswith(':'):
            section = line[:-1]
            if section in required_sections:
                found_sections.add(section)
                current_section = section
    
    return all(section in found_sections for section in required_sections)