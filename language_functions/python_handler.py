import logging
import os
from typing import Dict, Any, Optional
from .base_handler import BaseHandler

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
        """
        Parses Python code to extract classes and functions.

        Args:
            code (str): The Python code to be processed.
            file_path (str): Path to the Python source file.

        Returns:
            Dict[str, Any]: A structured representation of the analyzed code.
        """
        try:
            import ast
            from radon.complexity import cc_visit
            from radon.metrics import h_visit, mi_visit

            module_name = os.path.splitext(os.path.basename(file_path))[0] if file_path else 'module'
            tree = ast.parse(code)
            code_structure = {
                'classes': [],
                'functions': [],
                'variables': [],
                'constants': [],
                'halstead': {},
                'maintainability_index': None,
            }

            # Extract complexity scores using radon
            complexity_scores = cc_visit(code)
            function_complexity = {}
            for score in complexity_scores:
                # Use 'classname' attribute instead of 'class_name'
                if score.classname:
                    full_name = f"{score.classname}.{score.name}"
                else:
                    full_name = score.name
                function_complexity[full_name] = score.complexity

            # Extract Halstead metrics and Maintainability Index
            halstead_metrics = h_visit(code)
            if halstead_metrics:
                # 'h_visit' returns a dictionary; extract the first item
                metrics = next(iter(halstead_metrics.values()))
                total_halstead = {
                    'volume': metrics.volume,
                    'difficulty': metrics.difficulty,
                    'effort': metrics.effort
                }
                code_structure['halstead'] = total_halstead

            mi_score = mi_visit(code, True)
            code_structure['maintainability_index'] = mi_score

            class CodeVisitor(ast.NodeVisitor):
                """AST Node Visitor to extract classes, functions, variables, and constants."""

                def __init__(self):
                    self.scope_stack = []

                def visit_FunctionDef(self, node):
                    """Processes a function definition."""
                    self.scope_stack.append(node.name)
                    full_name = '.'.join(self.scope_stack)
                    complexity = function_complexity.get(full_name, 0)
                    function_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node) or '',
                        'args': [arg.arg for arg in node.args.args if arg.arg != 'self'],
                        'async': isinstance(node, ast.AsyncFunctionDef),
                        'complexity': complexity
                    }
                    code_structure['functions'].append(function_info)
                    self.generic_visit(node)
                    self.scope_stack.pop()

                def visit_ClassDef(self, node):
                    """Processes a class definition."""
                    self.scope_stack.append(node.name)
                    class_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node) or '',
                        'methods': []
                    }
                    for body_item in node.body:
                        if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self.scope_stack.append(body_item.name)
                            full_method_name = '.'.join(self.scope_stack)
                            complexity = function_complexity.get(full_method_name, 0)
                            method_info = {
                                'name': body_item.name,
                                'docstring': ast.get_docstring(body_item) or '',
                                'args': [arg.arg for arg in body_item.args.args if arg.arg != 'self'],
                                'async': isinstance(body_item, ast.AsyncFunctionDef),
                                'complexity': complexity
                            }
                            class_info['methods'].append(method_info)
                            self.scope_stack.pop()
                    code_structure['classes'].append(class_info)
                    self.generic_visit(node)
                    self.scope_stack.pop()

                def visit_Assign(self, node):
                    """Processes variable assignments."""
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            if var_name.isupper():
                                code_structure['constants'].append(var_name)
                            else:
                                code_structure['variables'].append(var_name)
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
            import ast

            tree = ast.parse(code)
            docstrings_mapping = {}

            # Map functions and classes to their docstrings
            for func_doc in documentation.get('functions', []):
                name = func_doc.get('name')
                if name:
                    docstrings_mapping[name] = func_doc.get('docstring', '')

            for class_doc in documentation.get('classes', []):
                class_name = class_doc.get('name')
                if class_name:
                    docstrings_mapping[class_name] = class_doc.get('docstring', '')
                    for method_doc in class_doc.get('methods', []):
                        method_name = method_doc.get('name')
                        if method_name:
                            full_method_name = f"{class_name}.{method_name}"
                            docstrings_mapping[full_method_name] = method_doc.get('docstring', '')

            class DocstringInserter(ast.NodeTransformer):
                """AST Node Transformer to insert docstrings."""

                def visit_FunctionDef(self, node):
                    """Inserts docstring into function if available."""
                    full_name = node.name
                    parent = getattr(node, 'parent', None)
                    if isinstance(parent, ast.ClassDef):
                        full_name = f"{parent.name}.{node.name}"
                    docstring = docstrings_mapping.get(full_name)
                    if docstring and not ast.get_docstring(node):
                        new_doc = ast.Expr(value=ast.Constant(value=docstring))
                        node.body.insert(0, new_doc)
                        logger.debug(f"Inserted docstring for function: {full_name}")
                    self.generic_visit(node)
                    return node

                def visit_AsyncFunctionDef(self, node):
                    """Inserts docstring into async function if available."""
                    return self.visit_FunctionDef(node)

                def visit_ClassDef(self, node):
                    """Inserts docstring into class if available."""
                    doc_content = docstrings_mapping.get(node.name)
                    if doc_content and not ast.get_docstring(node):
                        doc_node = ast.Expr(value=ast.Constant(value=doc_content))
                        node.body.insert(0, doc_node)
                        logger.debug(f"Inserted docstring for class: {node.name}")
                    self.generic_visit(node)
                    return node

            # Add parent references to AST nodes
            def add_parent_links(node, parent=None):
                for child in ast.iter_child_nodes(node):
                    setattr(child, 'parent', node)
                    add_parent_links(child, node)

            add_parent_links(tree)

            inserter = DocstringInserter()
            modified_tree = inserter.visit(tree)
            ast.fix_missing_locations(modified_tree)
            modified_code = ast.unparse(modified_tree)
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
            import ast

            # Check for syntax errors
            ast.parse(code)
            logger.debug("Syntax validation passed.")

            if file_path:
                import subprocess

                # Write code to a temporary file for flake8 validation
                temp_file = f"{file_path}.temp.py"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(code)

                # Run flake8 on the temporary file
                result = subprocess.run(['flake8', temp_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Remove the temporary file
                os.remove(temp_file)

                if result.returncode != 0:
                    logger.error(f"Flake8 validation failed for {file_path}:\n{result.stdout}\n{result.stderr}")
                    return False
                else:
                    logger.debug("Flake8 validation passed.")
            else:
                logger.warning("File path not provided for flake8 validation. Skipping flake8.")

            return True

        except SyntaxError as e:
            logger.error(f"Syntax error during validation: {e.text.strip()} at line {e.lineno}, offset {e.offset}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during code validation: {e}", exc_info=True)
            return False
