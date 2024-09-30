# language_functions.py

import ast
import json
import astor
import asyncio
import logging
import subprocess
import tempfile
from typing import Optional, Dict
import file_handlers
from bs4 import BeautifulSoup, Comment
import tinycss2

logger = logging.getLogger(__name__)

# Python Handlers
def extract_python_structure(file_content: str) -> dict:
    """Extracts the structure of Python code."""
    try:
        tree = ast.parse(file_content)
        parent_map = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent_map[child] = node

        functions = []
        classes = []

        def get_node_source(node):
            try:
                return ast.unparse(node)
            except AttributeError:
                return astor.to_source(node).strip()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "description": ast.get_docstring(node) or "",
                    "parameters": [],
                    "returns": {"type": "Any", "description": ""},
                    "decorators": [],
                    "examples": []
                }
                # Extract parameters
                for arg in node.args.args:
                    arg_type = "Any"
                    if arg.annotation:
                        arg_type = get_node_source(arg.annotation)
                    param_info = {
                        "name": arg.arg,
                        "type": arg_type,
                        "description": ""
                    }
                    func_info["parameters"].append(param_info)
                # Extract return type
                if node.returns:
                    func_info["returns"]["type"] = get_node_source(node.returns)
                # Extract decorators
                for decorator in node.decorator_list:
                    func_info["decorators"].append(get_node_source(decorator))
                parent = parent_map.get(node)
                if isinstance(parent, ast.ClassDef):
                    class_name = parent.name
                    class_obj = next(
                        (cls for cls in classes if cls["name"] == class_name), None
                    )
                    if not class_obj:
                        class_obj = {
                            "name": class_name,
                            "description": ast.get_docstring(parent) or "",
                            "bases": [get_node_source(base) for base in parent.bases],
                            "decorators": [],
                            "methods": [],
                            "examples": []
                        }
                        # Extract class decorators
                        for decorator in parent.decorator_list:
                            class_obj["decorators"].append(get_node_source(decorator))
                        classes.append(class_obj)
                    class_obj["methods"].append(func_info)
                else:
                    functions.append(func_info)
            elif isinstance(node, ast.ClassDef):
                class_exists = any(cls["name"] == node.name for cls in classes)
                if not class_exists:
                    class_info = {
                        "name": node.name,
                        "description": ast.get_docstring(node) or "",
                        "bases": [get_node_source(base) for base in node.bases],
                        "decorators": [],
                        "methods": [],
                        "examples": []
                    }
                    # Extract class decorators
                    for decorator in node.decorator_list:
                        class_info["decorators"].append(get_node_source(decorator))
                    classes.append(class_info)
        return {"language": "python", "functions": functions, "classes": classes}
    except Exception as e:
        logger.error(f"Error parsing Python code: {e}")
        return {}

def insert_python_docstrings(file_content: str, documentation: dict) -> str:
    """Inserts docstrings into Python code, including parameters and return descriptions."""
    try:
        tree = ast.parse(file_content)
        parent_map = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent_map[child] = node

        func_doc_map = {func["name"]: func for func in documentation.get("functions", [])}
        class_doc_map = {cls["name"]: cls for cls in documentation.get("classes", [])}
        method_doc_map = {}
        for cls in documentation.get("classes", []):
            for method in cls.get("methods", []):
                method_doc_map[(cls["name"], method["name"])] = method

        class DocstringInserter(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                parent = parent_map.get(node)
                if isinstance(parent, ast.ClassDef):
                    key = (parent.name, node.name)
                    doc_info = method_doc_map.get(key)
                else:
                    doc_info = func_doc_map.get(node.name)
                if doc_info:
                    # Construct docstring
                    description = doc_info.get('description', '')
                    param_docs = ''
                    for param in doc_info.get("parameters", []):
                        param_docs += f":param {param['type']} {param['name']}: {param['description']}\n"
                    return_doc = ''
                    returns = doc_info.get("returns", {})
                    if returns.get("type"):
                        return_doc = f":return {returns['type']}:{returns.get('description', '')}\n"
                    full_docstring = description.strip() + '\n\n' + param_docs + return_doc
                    # Insert docstring if not already present
                    if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                        if hasattr(ast, "Constant"):  # Python 3.8+
                            docstring_node = ast.Expr(value=ast.Constant(value=full_docstring.strip()))
                        else:
                            docstring_node = ast.Expr(value=ast.Str(s=full_docstring.strip()))
                        node.body.insert(0, docstring_node)
                        logger.debug(f"Inserted docstring into function '{node.name}'.")
                return node

            def visit_ClassDef(self, node):
                self.generic_visit(node)
                class_doc = class_doc_map.get(node.name)
                if class_doc:
                    description = class_doc.get("description", "")
                    # Insert docstring if not already present
                    if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                        if hasattr(ast, "Constant"):
                            docstring_node = ast.Expr(value=ast.Constant(value=description.strip()))
                        else:
                            docstring_node = ast.Expr(value=ast.Str(s=description.strip()))
                        node.body.insert(0, docstring_node)
                        logger.debug(f"Inserted docstring into class '{node.name}'.")
                return node

        inserter = DocstringInserter()
        new_tree = inserter.visit(tree)
        ast.fix_missing_locations(new_tree)  # Ensure the AST is properly fixed
        new_code = astor.to_source(new_tree)
        return new_code
    except Exception as e:
        logger.error(f"Error inserting docstrings into Python code: {e}")
        return file_content

def is_valid_python_code(code: str) -> bool:
    """Checks if the given code is valid Python code."""
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in Python code: {e}")
        return False

# JavaScript/TypeScript Handlers
async def extract_js_ts_structure(file_path: str, file_content: str, language: str) -> dict:
    """
    Extracts the structure of JavaScript or TypeScript code using a Node.js script.

    Parameters:
        file_path (str): The path to the file being processed.
        file_content (str): The content of the file (not used in this function but kept for consistency).
        language (str): The programming language ('javascript' or 'typescript').

    Returns:
        dict: A dictionary containing the extracted structure of the code.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            'node', 'extract_structure.js', file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if stderr:
            logger.error(f"Error from Node.js script for '{file_path}': {stderr.decode().strip()}")
            return {}

        try:
            structure = json.loads(stdout.decode())
            logger.debug(f"Extracted structure from '{file_path}'.")
            return structure
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error for '{file_path}': {e}")
            return {}

    except FileNotFoundError:
        logger.error("Node.js or extract_structure.js not found. Ensure Node.js is installed and the script path is correct.")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error while extracting JS/TS structure from '{file_path}': {e}")
        return {}

def insert_js_ts_docstrings(content: str, documentation: dict) -> str:
    """
    Inserts JSDoc comments into JavaScript or TypeScript code.

    Parameters:
        content (str): The original source code.
        documentation (dict): The documentation dictionary containing 'functions' and 'classes'.

    Returns:
        str: The updated source code with inserted docstrings/comments.
    """
    try:
        if not content:
            logger.error("No source code provided.")
            return ""

        # Ensure that 'source_code' is included in the documentation
        documentation['source_code'] = content

        # Prepare the input for the Node.js script
        input_data = json.dumps(documentation)

        # Run the Node.js script to insert docstrings
        process = subprocess.Popen(
            ['node', 'insert_docstrings.js'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(input=input_data)

        if stderr:
            logger.error(f"Error from insert_docstrings.js: {stderr.strip()}")
            return content  # Return the original content if there's an error

        logger.debug("Inserted docstrings into JavaScript/TypeScript code.")
        return stdout.strip()

    except FileNotFoundError:
        logger.error("Node.js or insert_docstrings.js not found. Ensure Node.js is installed and the script path is correct.")
        return content
    except Exception as e:
        logger.error(f"Error inserting JS/TS docstrings: {e}")
        return content

# HTML Handlers
def extract_html_structure(file_content: str) -> dict:
    """Extracts the structure of HTML code."""
    try:
        soup = BeautifulSoup(file_content, "html.parser")
        elements = []

        def traverse(node):
            for child in node.children:
                if isinstance(child, str):
                    continue
                if child.name:
                    element_info = {
                        "tag": child.name,
                        "attributes": dict(child.attrs),
                        "text": child.get_text(strip=True),
                        "description": "",  # Replaced 'docstring' with 'description'
                    }
                    elements.append(element_info)
                    traverse(child)

        traverse(soup)
        logger.debug("Extracted HTML structure.")
        return {"language": "html", "elements": elements}

    except Exception as e:
        logger.error(f"Error parsing HTML code: {e}")
        return {}

def insert_html_comments(file_content: str, documentation: dict) -> str:
    """Inserts comments into HTML code, preventing duplicates."""
    try:
        soup = BeautifulSoup(file_content, "html.parser")
        elements = documentation.get("elements", [])

        element_map = {}
        for element in elements:
            key = (
                element["tag"],
                tuple(sorted(element["attributes"].items())),
                element["text"],
            )
            element_map[key] = element.get("description", "")

        def traverse_and_insert(node):
            for child in node.children:
                if isinstance(child, str):
                    continue
                if child.name:
                    key = (
                        child.name,
                        tuple(sorted(child.attrs.items())),
                        child.get_text(strip=True),
                    )
                    description = element_map.get(key)
                    if description:
                        # Check for existing comment with the same description
                        existing_comment = child.find_previous_sibling(string=lambda text: isinstance(text, Comment) and description in text)
                        if not existing_comment:
                            comment = Comment(f" {description} ")
                            child.insert_before(comment)
                            logger.debug(f"Inserted comment for HTML tag '{child.name}'.")
                    traverse_and_insert(child)

        traverse_and_insert(soup)
        return str(soup)

    except Exception as e:
        logger.error(f"Error inserting comments into HTML code: {e}")
        return file_content

# CSS Handlers
def extract_css_structure(file_content: str) -> dict:
    """Extracts the structure of CSS code."""
    try:
        stylesheet = tinycss2.parse_stylesheet(file_content, skip_whitespace=True)
        rules = []
        for rule in stylesheet:
            if rule.type == "qualified-rule":
                selector = tinycss2.serialize(rule.prelude).strip()
                declarations = tinycss2.serialize(rule.content).strip()
                rules.append({"selector": selector, "declarations": declarations, "description": ""})
            elif rule.type == "at-rule":
                # Handle at-rules (e.g., @media, @import)
                at_keyword = rule.at_keyword
                prelude = tinycss2.serialize(rule.prelude).strip()
                content = tinycss2.serialize(rule.content).strip() if rule.content else None
                rules.append({"selector": f"@{at_keyword} {prelude}", "declarations": content or "", "description": ""})
        logger.debug("Extracted CSS structure.")
        return {"language": "css", "rules": rules}
    except Exception as e:
        logger.error(f"Error parsing CSS code: {e}")
        return {}

def insert_css_docstrings(content: str, documentation: dict) -> str:
    """
    Inserts comments into CSS code based on the provided documentation.

    Parameters:
        content (str): The original CSS code.
        documentation (dict): The documentation data containing 'rules'.

    Returns:
        str: The updated CSS code with inserted comments.
    """
    try:
        lines = content.split('\n')
        updated_lines = []
        for line in lines:
            stripped_line = line.strip()
            # Check if the line is a selector
            if stripped_line.endswith('{'):
                selector = stripped_line[:-1].strip()
                # Find the corresponding rule in the documentation
                for rule in documentation.get('rules', []):
                    if selector == rule['selector']:
                        # Insert the comment above the rule
                        description = rule.get('description', '')
                        if description:  # Only add comment if description exists
                            comment = f"/* {description} */"
                            updated_lines.append(comment)
                            logger.debug(f"Inserted comment for CSS selector '{selector}'.")
                        break
                updated_lines.append(line)
            else:
                updated_lines.append(line)
        return '\n'.join(updated_lines)
    except Exception as e:
        logger.error(f"Error inserting comments into CSS code: {e}")
        return content
