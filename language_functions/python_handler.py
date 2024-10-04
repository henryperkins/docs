import ast
import logging
import subprocess
from typing import Dict, Any
from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class PythonHandler(BaseHandler):
    """Handler for Python language."""

    def __init__(self, function_schema):
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """Parses Python code to extract classes and functions."""
        try:
            tree = ast.parse(code)
            structure = {'classes': [], 'functions': []}

            # Assign parent attributes to nodes
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    child.parent = node

            # Now traverse the tree and extract structure
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Extract classes
                    class_structure = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node),
                        'methods': []
                    }
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method = {
                                'name': item.name,
                                'docstring': ast.get_docstring(item)
                            }
                            class_structure['methods'].append(method)
                    structure['classes'].append(class_structure)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Extract functions
                    structure['functions'].append({
                        'name': node.name,
                        'docstring': ast.get_docstring(node)
                    })

            return structure
        except Exception as e:
            logger.error(f"Error extracting Python structure: {e}")
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """Inserts Python docstrings based on the provided documentation."""
        logger.debug("Starting insert_docstrings")
        try:
            tree = ast.parse(code)
            docstrings_mapping = {}

            if "functions" in documentation:
                for func_doc in documentation["functions"]:
                    name = func_doc.get("name")
                    doc = func_doc.get("docstring", "")
                    if name and doc:
                        docstrings_mapping[name] = doc

            if "classes" in documentation:
                for class_doc in documentation["classes"]:
                    class_name = class_doc.get("name")
                    class_docstring = class_doc.get("docstring", "")
                    if class_name and class_docstring:
                        docstrings_mapping[class_name] = class_docstring
                    for method_doc in class_doc.get("methods", []):
                        method_name = method_doc.get("name")
                        full_method_name = f"{class_name}.{method_name}"
                        method_docstring = method_doc.get("docstring", "")
                        if method_name and method_docstring:
                            docstrings_mapping[full_method_name] = method_docstring

            # Assign parents to nodes to identify top-level functions
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    child.parent = node

            class FunctionDocInserter(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    # Determine if it's a method or a top-level function
                    parent = getattr(node, "parent", None)
                    if isinstance(parent, ast.ClassDef):
                        full_name = f"{parent.name}.{node.name}"
                    else:
                        full_name = node.name
                    if full_name in docstrings_mapping:
                        doc_content = docstrings_mapping[full_name]
                        node.body.insert(0, ast.Expr(value=ast.Constant(value=doc_content)))
                        logger.debug(f"Inserted docstring for function: {full_name}")
                    return node

                def visit_AsyncFunctionDef(self, node):
                    # Similar to FunctionDef
                    parent = getattr(node, "parent", None)
                    if isinstance(parent, ast.ClassDef):
                        full_name = f"{parent.name}.{node.name}"
                    else:
                        full_name = node.name
                    if full_name in docstrings_mapping:
                        doc_content = docstrings_mapping[full_name]
                        node.body.insert(0, ast.Expr(value=ast.Constant(value=doc_content)))
                        logger.debug(f"Inserted docstring for async function: {full_name}")
                    return node

                def visit_ClassDef(self, node):
                    if node.name in docstrings_mapping:
                        doc_content = docstrings_mapping[node.name]
                        node.body.insert(0, ast.Expr(value=ast.Constant(value=doc_content)))
                        logger.debug(f"Inserted docstring for class: {node.name}")
                    self.generic_visit(node)
                    return node

            inserter = FunctionDocInserter()
            modified_tree = inserter.visit(tree)
            ast.fix_missing_locations(modified_tree)
            modified_code = ast.unparse(modified_tree)
            logger.debug("Completed inserting Python docstrings")
            return modified_code
        except Exception as e:
            logger.error(f"Error inserting Python docstrings: {e}")
            return code

    def validate_code(self, code: str, file_path: str = None) -> bool:
        """Validates the modified Python code for syntax correctness."""
        try:
            ast.parse(code)
            logger.debug("Python syntax validation successful.")
            
            # Optionally, validate using external tools like flake8 if file_path is available
            if file_path:
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
