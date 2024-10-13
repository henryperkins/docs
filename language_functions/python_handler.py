import os
import ast
import logging
import subprocess
from typing import Dict, Any, Optional
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit  # Ensure these are imported
from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class PythonHandler(BaseHandler):
    """The `PythonHandler` class extends `BaseHandler` and is responsible for processing Python code by extracting its structure, validating it, and inserting appropriate docstrings."""
    'Handler for Python language.'

    def __init__(self, function_schema):
        """Initializes instances of the class with required parameters.

        Args:
            self (PythonHandler): The instance of the class.
            function_schema (Any): The schema used for function operations.

        Returns:
            None: This constructor does not return a value."""
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str=None) -> Dict[str, Any]:
        """Analyzes the provided code and returns its structure in a dictionary form.

        Args:
            self (PythonHandler): The instance of the class.
            code (str): The Python code to be processed.
            file_path (str): The path to the file being analyzed.

        Returns:
            Dict[str, Any]: A structured representation of the analyzed code."""
        try:
            tree = ast.parse(code)
            code_structure = {'modules': [], 'classes': [], 'functions': [], 'variables': [], 'constants': []}

            # Calculate complexity scores
            complexity_scores = cc_visit(code)
            function_complexity = {f"{score.classname}.{score.name}": score.complexity for score in complexity_scores if hasattr(score, 'classname')}
            function_complexity.update({score.name: score.complexity for score in complexity_scores if not hasattr(score, 'classname')})

            # Calculate Halstead metrics
            halstead_metrics = h_visit(code)
            # Calculate Maintainability Index
            maintainability_index = mi_visit(code, True)

            class CodeVisitor(ast.NodeVisitor):
                """`CodeVisitor` explores the AST to gather information about functions, classes, and assignments for further analysis."""

                def visit_FunctionDef(self, node):
                    """Processes a function definition within the AST."""
                    self._process_function(node, is_async=False)
                    self.generic_visit(node)

                def visit_AsyncFunctionDef(self, node):
                    """Processes an async function definition within the AST."""
                    self._process_function(node, is_async=True)
                    self.generic_visit(node)

                def visit_ClassDef(self, node):
                    """Processes a class definition within the AST."""
                    class_info = {'name': node.name, 'description': '', 'inherits': [self._get_class_name(base) for base in node.bases], 'methods': [], 'attributes': []}
                    for body_item in node.body:
                        if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_info = {'name': body_item.name, 'description': '', 'parameters': [], 'returns': {'type': self._get_type_annotation(body_item.returns), 'description': ''}, 'raises': self._extract_exceptions(body_item), 'examples': [], 'decorators': [ast.unparse(dec) for dec in body_item.decorator_list], 'async': isinstance(body_item, ast.AsyncFunctionDef), 'static': any((isinstance(dec, ast.Name) and dec.id == 'staticmethod' for dec in body_item.decorator_list)), 'visibility': 'public' if not body_item.name.startswith('_') else 'private', 'complexity': function_complexity.get(f"{node.name}.{body_item.name}", 0)}
                            defaults = body_item.args.defaults
                            default_values = [self._get_constant_value(d) for d in defaults]
                            start = len(body_item.args.args) - len(default_values) if len(default_values) < len(body_item.args.args) else 0
                            for i, arg in enumerate(body_item.args.args):
                                if arg.arg == 'self':
                                    continue
                                param_info = {'name': arg.arg, 'type': self._get_type_annotation(arg.annotation), 'description': '', 'default': None, 'optional': False}
                                if i >= start:
                                    param_info['default'] = default_values[i - start]
                                    param_info['optional'] = True
                                method_info['parameters'].append(param_info)
                            class_info['methods'].append(method_info)
                        elif isinstance(body_item, ast.Assign):
                            for target in body_item.targets:
                                if isinstance(target, ast.Name):
                                    attribute_info = {'name': target.id, 'type': self._infer_type(body_item.value), 'description': ''}
                                    class_info['attributes'].append(attribute_info)
                    code_structure['classes'].append(class_info)
                    self.generic_visit(node)

                def visit_Assign(self, node):
                    """Processes an assignment within the AST."""
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            value = self._get_constant_value(node.value)
                            if target.id.isupper():
                                constant_info = {'name': target.id, 'value': value, 'description': ''}
                                code_structure['constants'].append(constant_info)
                            else:
                                variable_info = {'name': target.id, 'type': self._infer_type(node.value), 'description': ''}
                                code_structure['variables'].append(variable_info)
                    self.generic_visit(node)

                def _process_function(self, node, is_async: bool):
                    """Processes a given function node, determining its asynchronous state."""
                    function_info = {'name': node.name, 'description': '', 'parameters': [], 'returns': {'type': self._get_type_annotation(node.returns), 'description': ''}, 'raises': self._extract_exceptions(node), 'examples': [], 'decorators': [ast.unparse(dec) for dec in node.decorator_list], 'async': is_async, 'static': False, 'visibility': 'public' if not node.name.startswith('_') else 'private', 'complexity': function_complexity.get(node.name, 0)}
                    defaults = node.args.defaults
                    default_values = [self._get_constant_value(d) for d in defaults]
                    start = len(node.args.args) - len(default_values)
                    for i, arg in enumerate(node.args.args):
                        param_info = {'name': arg.arg, 'type': self._get_type_annotation(arg.annotation), 'description': '', 'default': None, 'optional': False}
                        if i >= start:
                            param_info['default'] = default_values[i - start]
                            param_info['optional'] = True
                        function_info['parameters'].append(param_info)
                    logger.debug(f"Extracted parameters for function '{node.name}': {function_info['parameters']}")
                    code_structure['functions'].append(function_info)

                def _get_type_annotation(self, annotation) -> str:
                    """Obtains type annotation from the AST node."""
                    if annotation:
                        return ast.unparse(annotation)
                    return ''

                def _get_class_name(self, base) -> str:
                    """Extracts the class's name from the AST node."""
                    if isinstance(base, ast.Name):
                        return base.id
                    return ast.unparse(base)

                def _get_constant_value(self, node):
                    """Extracts constant values from the AST node."""
                    if isinstance(node, ast.Constant):
                        return node.value
                    return None

                def _infer_type(self, node) -> str:
                    """Infers and returns the type from the AST node."""
                    if isinstance(node, ast.Constant):
                        return type(node.value).__name__
                    elif isinstance(node, (ast.List, ast.Tuple)):
                        return 'list' if isinstance(node, ast.List) else 'tuple'
                    elif isinstance(node, ast.Dict):
                        return 'dict'
                    elif isinstance(node, ast.Call):
                        return ast.unparse(node.func)
                    return ''

                def _extract_exceptions(self, node) -> list:
                    """Pulls exception information from the AST node."""
                    exceptions = []
                    for n in ast.walk(node):
                        if isinstance(n, ast.Raise):
                            if n.exc:
                                exc_type = self._get_exception_name(n.exc)
                                if exc_type:
                                    exceptions.append({'type': exc_type, 'description': ''})
                    return exceptions

                def _get_exception_name(self, node) -> Optional[str]:
                    """Determines the exception name from an AST node."""
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
            code_structure['halstead'] = halstead_metrics
            code_structure['maintainability_index'] = maintainability_index
            return code_structure
        except Exception as e:
            logger.error(f'Error extracting Python structure: {e}')
            return {}
        
    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """Integrates docstrings into the code as defined by the documentation schema.

        Args:
            self (PythonHandler): The instance of the class.
            code (str): The source code to be updated with docstrings.
            documentation (Dict[str, Any]): A map of docstrings assigned to code structures.

        Returns:
            str: Code with integrated docstrings."""
        logger.debug('Starting insert_docstrings')
        try:
            tree = ast.parse(code)
            docstrings_mapping = {}
            for func_doc in documentation.get('functions', []):
                name = func_doc.get('name')
                if name:
                    docstring = self.format_function_docstring(func_doc)
                    docstrings_mapping[name] = docstring
                    logger.debug(f"Function '{name}' docstring: {docstring}")
            for class_doc in documentation.get('classes', []):
                class_name = class_doc.get('name')
                if class_name:
                    class_docstring = class_doc.get('description', '')
                    docstrings_mapping[class_name] = class_docstring
                    logger.debug(f"Class '{class_name}' docstring: {class_docstring}")
                    for method_doc in class_doc.get('methods', []):
                        method_name = method_doc.get('name')
                        if method_name:
                            full_method_name = f'{class_name}.{method_name}'
                            method_docstring = self.format_function_docstring(method_doc)
                            docstrings_mapping[full_method_name] = method_docstring
                            logger.debug(f"Method '{full_method_name}' docstring: {method_docstring}")

            class DocstringInserter(ast.NodeTransformer):
                """`DocstringInserter` modifies AST nodes to add docstrings wherever necessary."""

                def visit_FunctionDef(self, node):
                    """Processes a function definition within the AST."""
                    parent = getattr(node, 'parent', None)
                    if isinstance(parent, ast.ClassDef):
                        full_name = f'{parent.name}.{node.name}'
                    else:
                        full_name = node.name
                    doc_content = docstrings_mapping.get(full_name)
                    if doc_content:
                        doc_node = ast.Expr(value=ast.Constant(value=doc_content))
                        node.body.insert(0, doc_node)
                        logger.debug(f'Inserted docstring for function: {full_name}')
                    self.generic_visit(node)
                    return node

                def visit_AsyncFunctionDef(self, node):
                    """Processes an async function definition within the AST."""
                    return self.visit_FunctionDef(node)

                def visit_ClassDef(self, node):
                    """Processes a class definition within the AST."""
                    doc_content = docstrings_mapping.get(node.name)
                    if doc_content:
                        doc_node = ast.Expr(value=ast.Constant(value=doc_content))
                        node.body.insert(0, doc_node)
                        logger.debug(f'Inserted docstring for class: {node.name}')
                    self.generic_visit(node)
                    return node

            inserter = DocstringInserter()
            modified_tree = inserter.visit(tree)
            ast.fix_missing_locations(modified_tree)
            modified_code = ast.unparse(modified_tree)
            logger.debug('Completed inserting Python docstrings')
            return modified_code
        except Exception as e:
            logger.error(f'Error inserting Python docstrings: {e}')
            return code

    def format_function_docstring(self, func: Dict[str, Any]) -> str:
        """Generates a formatted docstring from a function definition.

        Args:
            self (PythonHandler): The instance of the class.
            func (Dict[str, Any]): The function's details and associated documentation.

        Returns:
            str: A finalized docstring for the provided function."""
        lines = [f'{func.get("description", "").strip()}']
        params = func.get('parameters', [])
        if params:
            lines.append('\nArgs:')
            for param in params:
                param_line = f'    {param["name"]}'
                if param.get('type'):
                    param_line += f' ({param["type"]})'
                param_line += f': {param.get("description", "").strip()}'
                lines.append(param_line)
        returns = func.get('returns', {})
        if returns and returns.get('description'):
            lines.append('\nReturns:')
            ret_line = '    '
            if returns.get('type'):
                ret_line += f'{returns.get("type")}: '
            ret_line += returns.get('description', '').strip()
            lines.append(ret_line)
        raises = func.get('raises', [])
        if raises:
            lines.append('\nRaises:')
            for exc in raises:
                exc_line = f'    {exc.get("type", "")}: {exc.get("description", "").strip()}'
                lines.append(exc_line)
        return '\n'.join(lines)

    def validate_code(self, code: str, file_path: Optional[str]=None) -> bool:
        """Performs validation checks on the provided source code.

        Args:
            self (PythonHandler): The instance of the class.
            code (str): The code content to validate.
            file_path (Optional[str]): A file path for additional validation criteria.

        Returns:
            bool: Validation result indicating code is either valid or not."""
        logger.debug('Starting Python code validation.')
        try:
            ast.parse(code)
            logger.debug('Python syntax validation successful.')
            if file_path:
                temp_file = f'{file_path}.temp'
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(code)
                process = subprocess.run(['flake8', temp_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                os.remove(temp_file)
                if process.returncode != 0:
                    flake8_output = process.stdout + process.stderr
                    logger.error(f'Flake8 validation failed for {file_path}:\n{flake8_output}')
                    return False
                else:
                    logger.debug('Flake8 validation successful.')
            else:
                logger.warning('File path not provided for flake8 validation. Skipping flake8.')
            return True
        except SyntaxError as e:
            logger.error(f'Python syntax error: {e}')
            return False
        except Exception as e:
            logger.error(f'Unexpected error during Python code validation: {e}')
            return False