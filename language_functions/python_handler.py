import ast
import logging
import subprocess
from typing import Dict, Any, Optional
from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class PythonHandler(BaseHandler):
    """Handler for Python language."""

    def __init__(self, function_schema):
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """
        Extracts the code structure from Python source code, conforming to the schema.
        """
        try:
            tree = ast.parse(code)
            code_structure = {
                "modules": [],
                "classes": [],
                "functions": [],
                "variables": [],
                "constants": []
            }

            class CodeVisitor(ast.NodeVisitor):
                def visit_FunctionDef(self, node):
                    function_info = {
                        "name": node.name,
                        "description": "",  # Placeholder for AI to fill
                        "parameters": [],
                        "returns": {
                            "type": self._get_type_annotation(node.returns),
                            "description": ""
                        },
                        "raises": self._extract_exceptions(node),
                        "examples": [],
                        "decorators": [ast.unparse(dec) for dec in node.decorator_list],
                        "async": isinstance(node, ast.AsyncFunctionDef),
                        "static": False,  # Top-level functions are not static
                        "visibility": "public" if not node.name.startswith("_") else "private"
                    }
                    # Extract parameters
                    for arg in node.args.args:
                        param_info = {
                            "name": arg.arg,
                            "type": self._get_type_annotation(arg.annotation),
                            "description": "",
                            "default": None,
                            "optional": False
                        }
                        # Determine if parameter has a default value
                        defaults = node.args.defaults
                        default_values = [self._get_constant_value(d) for d in defaults]
                        if arg in node.args.args[-len(defaults):]:
                            param_info["default"] = default_values[node.args.args.index(arg) - (len(node.args.args) - len(defaults))]
                            param_info["optional"] = True
                        function_info["parameters"].append(param_info)
                    code_structure["functions"].append(function_info)
                    self.generic_visit(node)

                def visit_ClassDef(self, node):
                    class_info = {
                        "name": node.name,
                        "description": "",  # Placeholder for AI to fill
                        "inherits": [self._get_class_name(base) for base in node.bases],
                        "methods": [],
                        "attributes": []
                    }
                    # Visit methods and attributes within the class
                    for body_item in node.body:
                        if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_info = {
                                "name": body_item.name,
                                "description": "",
                                "parameters": [],
                                "returns": {
                                    "type": self._get_type_annotation(body_item.returns),
                                    "description": ""
                                },
                                "raises": self._extract_exceptions(body_item),
                                "examples": [],
                                "decorators": [ast.unparse(dec) for dec in body_item.decorator_list],
                                "async": isinstance(body_item, ast.AsyncFunctionDef),
                                "static": any(isinstance(dec, ast.Name) and dec.id == 'staticmethod' for dec in body_item.decorator_list),
                                "visibility": "public" if not body_item.name.startswith("_") else "private"
                            }
                            # Extract parameters, excluding 'self'
                            for arg in body_item.args.args:
                                if arg.arg != 'self':
                                    param_info = {
                                        "name": arg.arg,
                                        "type": self._get_type_annotation(arg.annotation),
                                        "description": "",
                                        "default": None,
                                        "optional": False
                                    }
                                    # Determine if parameter has a default value
                                    defaults = body_item.args.defaults
                                    default_values = [self._get_constant_value(d) for d in defaults]
                                    if arg in body_item.args.args[-len(defaults):]:
                                        param_info["default"] = default_values[body_item.args.args.index(arg) - (len(body_item.args.args) - len(defaults))]
                                        param_info["optional"] = True
                                    method_info["parameters"].append(param_info)
                            class_info["methods"].append(method_info)
                        elif isinstance(body_item, ast.Assign):
                            # Handle attributes
                            for target in body_item.targets:
                                if isinstance(target, ast.Name):
                                    attribute_info = {
                                        "name": target.id,
                                        "type": self._infer_type(body_item.value),
                                        "description": ""
                                    }
                                    class_info["attributes"].append(attribute_info)
                    code_structure["classes"].append(class_info)
                    self.generic_visit(node)

                def visit_Assign(self, node):
                    # Handle global variables and constants
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            value = self._get_constant_value(node.value)
                            if target.id.isupper():
                                constant_info = {
                                    "name": target.id,
                                    "value": value,
                                    "description": ""
                                }
                                code_structure["constants"].append(constant_info)
                            else:
                                variable_info = {
                                    "name": target.id,
                                    "type": self._infer_type(node.value),
                                    "description": ""
                                }
                                code_structure["variables"].append(variable_info)
                    self.generic_visit(node)

                def _get_type_annotation(self, annotation) -> str:
                    if annotation:
                        return ast.unparse(annotation)
                    return ""

                def _get_class_name(self, base) -> str:
                    if isinstance(base, ast.Name):
                        return base.id
                    return ast.unparse(base)

                def _get_constant_value(self, node):
                    if isinstance(node, ast.Constant):
                        return node.value
                    return None

                def _infer_type(self, node) -> str:
                    if isinstance(node, ast.Constant):
                        return type(node.value).__name__
                    elif isinstance(node, (ast.List, ast.Tuple)):
                        return 'list' if isinstance(node, ast.List) else 'tuple'
                    elif isinstance(node, ast.Dict):
                        return 'dict'
                    elif isinstance(node, ast.Call):
                        return ast.unparse(node.func)
                    return ""

                def _extract_exceptions(self, node) -> list:
                    exceptions = []
                    for n in ast.walk(node):
                        if isinstance(n, ast.Raise):
                            if n.exc:
                                exc_type = self._get_exception_name(n.exc)
                                if exc_type:
                                    exceptions.append({
                                        "type": exc_type,
                                        "description": ""
                                    })
                    return exceptions

                def _get_exception_name(self, node) -> Optional[str]:
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            return node.func.id
                        elif isinstance(node.func, ast.Attribute):
                            return ast.unparse(node.func)
                    elif isinstance(node, ast.Name):
                        return node.id
                    elif isinstance(node, ast.Attribute):
                        return ast.unparse(node)
                    return None

            visitor = CodeVisitor()
            visitor.visit(tree)
            return code_structure
        except Exception as e:
            logger.error(f"Error extracting Python structure: {e}")
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts Python docstrings based on the provided structured documentation.
        """
        logger.debug("Starting insert_docstrings")
        try:
            tree = ast.parse(code)
            docstrings_mapping = {}

            # Prepare docstrings mapping for functions
            for func_doc in documentation.get("functions", []):
                name = func_doc.get("name")
                if name:
                    docstrings_mapping[name] = self.format_function_docstring(func_doc)

            # Prepare docstrings mapping for classes and methods
            for class_doc in documentation.get("classes", []):
                class_name = class_doc.get("name")
                if class_name:
                    # Class docstring
                    docstrings_mapping[class_name] = class_doc.get("description", "")
                    # Methods
                    for method_doc in class_doc.get("methods", []):
                        method_name = method_doc.get("name")
                        if method_name:
                            full_method_name = f"{class_name}.{method_name}"
                            docstrings_mapping[full_method_name] = self.format_function_docstring(method_doc)

            # Assign parent attributes to nodes
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    child.parent = node

            class DocstringInserter(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    parent = getattr(node, "parent", None)
                    if isinstance(parent, ast.ClassDef):
                        full_name = f"{parent.name}.{node.name}"
                    else:
                        full_name = node.name

                    doc_content = docstrings_mapping.get(full_name)
                    if doc_content:
                        doc_node = ast.Expr(value=ast.Constant(value=doc_content))
                        node.body.insert(0, doc_node)
                        logger.debug(f"Inserted docstring for function: {full_name}")
                    self.generic_visit(node)
                    return node

                def visit_AsyncFunctionDef(self, node):
                    return self.visit_FunctionDef(node)

                def visit_ClassDef(self, node):
                    doc_content = docstrings_mapping.get(node.name)
                    if doc_content:
                        doc_node = ast.Expr(value=ast.Constant(value=doc_content))
                        node.body.insert(0, doc_node)
                        logger.debug(f"Inserted docstring for class: {node.name}")
                    self.generic_visit(node)
                    return node

            inserter = DocstringInserter()
            modified_tree = inserter.visit(tree)
            ast.fix_missing_locations(modified_tree)
            modified_code = ast.unparse(modified_tree)
            logger.debug("Completed inserting Python docstrings")
            return modified_code
        except Exception as e:
            logger.error(f"Error inserting Python docstrings: {e}")
            return code

    def format_function_docstring(self, func: Dict[str, Any]) -> str:
        """
        Formats the function documentation into a Google-style docstring.
        """
        lines = [f"{func.get('description', '').strip()}"]

        params = func.get('parameters', [])
        if params:
            lines.append("\nArgs:")
            for param in params:
                param_line = f"    {param['name']}"
                if param.get('type'):
                    param_line += f" ({param['type']})"
                param_line += f": {param.get('description', '').strip()}"
                lines.append(param_line)

        returns = func.get('returns', {})
        if returns and returns.get('description'):
            lines.append("\nReturns:")
            ret_line = "    "
            if returns.get('type'):
                ret_line += f"{returns.get('type')}: "
            ret_line += returns.get('description', '').strip()
            lines.append(ret_line)

        raises = func.get('raises', [])
        if raises:
            lines.append("\nRaises:")
            for exc in raises:
                exc_line = f"    {exc.get('type', '')}: {exc.get('description', '').strip()}"
                lines.append(exc_line)

        return '\n'.join(lines)

    def validate_code(self, code: str, file_path: str = None) -> bool:
        """
        Validates the modified Python code for syntax correctness.
        """
        try:
            ast.parse(code)
            logger.debug("Python syntax validation successful.")

            # Optionally, validate using external tools like flake8 if file_path is available
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                result = subprocess.run(["flake8", file_path], capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"flake8 validation failed:\n{result.stdout}")
                    return False

            return True
        except SyntaxError as e:
            logger.error(f"Python syntax validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Python code validation: {e}")
            return False