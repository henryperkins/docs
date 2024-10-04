# language_functions/python_handler.py

import ast
import astor
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def insert_python_docstrings(original_code: str, documentation: Dict[str, Any]) -> str:
    """Inserts docstrings into Python functions and classes based on the provided documentation."""
    logger.debug("Starting insert_python_docstrings")
    try:
        tree = ast.parse(original_code)
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
                methods = class_doc.get("methods", [])
                for method_doc in methods:
                    method_name = method_doc.get("name")
                    full_method_name = f"{class_name}.{method_name}"
                    method_docstring = method_doc.get("docstring", "")
                    if method_name and method_docstring:
                        docstrings_mapping[full_method_name] = method_docstring
        
        # Overview Handling
        overview = documentation.get("overview", "")
        if overview:
            docstrings_mapping["overview"] = overview
        
        def sanitize_docstring(docstring: str) -> str:
            """Cleans or formats a given docstring to ensure it adheres to documentation standards."""
            lines = docstring.strip().splitlines()
            sanitized_lines = [line.rstrip() for line in lines]
            return "\n".join(sanitized_lines)
        
        def set_parent(node, parent=None):
            """Assigns a parent reference to a given node."""
            for child in ast.iter_child_nodes(node):
                setattr(child, "parent", node)
                set_parent(child, node)
        
        set_parent(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent = getattr(node, "parent", None)
                if isinstance(parent, ast.ClassDef):
                    full_name = f"{parent.name}.{node.name}"
                else:
                    full_name = node.name
                if full_name in docstrings_mapping:
                    doc_content = sanitize_docstring(docstrings_mapping[full_name])
                    if ast.get_docstring(node, clean=False) is not None:
                        if (
                            isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Constant)
                            and isinstance(node.body[0].value.value, str)
                        ):
                            node.body.pop(0)
                    docstring_node = ast.Expr(value=ast.Str(s=doc_content))
                    node.body.insert(0, docstring_node)
                    logger.debug(f"Inserted docstring in function/method: {full_name}")
            elif isinstance(node, ast.ClassDef):
                if node.name in docstrings_mapping:
                    doc_content = sanitize_docstring(docstrings_mapping[node.name])
                    if ast.get_docstring(node, clean=False) is not None:
                        if (
                            isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Constant)
                            and isinstance(node.body[0].value.value, str)
                        ):
                            node.body.pop(0)
                    docstring_node = ast.Expr(value=ast.Str(s=doc_content))
                    node.body.insert(0, docstring_node)
                    logger.debug(f"Inserted docstring in class: {node.name}")
        
        # Handle Module-Level Overview
        if "overview" in docstrings_mapping:
            doc_content = sanitize_docstring(docstrings_mapping["overview"])
            if ast.get_docstring(tree, clean=False) is not None:
                if (
                    isinstance(tree.body[0], ast.Expr)
                    and isinstance(tree.body[0].value, ast.Constant)
                    and isinstance(tree.body[0].value.value, str)
                ):
                    tree.body.pop(0)
            docstring_node = ast.Expr(value=ast.Str(s=doc_content))
            tree.body.insert(0, docstring_node)
            logger.debug("Inserted module-level overview docstring.")
        
        ast.fix_missing_locations(tree)
        modified_code = astor.to_source(tree)
        logger.debug("Completed inserting Python docstrings")
        return modified_code
