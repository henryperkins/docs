# language_functions.py

import os
import fnmatch
import json
import ast
import asyncio
import subprocess
import logging
import traceback
import tempfile
import astor
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup, Comment
import tinycss2

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set logging level to DEBUG

# Create formatter with module, function, and line number
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(funcName)s:%(lineno)d:%(message)s')

# Create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Change to DEBUG for more verbosity on console
console_handler.setFormatter(formatter)

# Create file handler which logs debug and higher level messages
file_handler = logging.FileHandler('language_functions.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Add handlers to the logger
if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def is_syntax_valid(code: str) -> bool:
    """Check for syntax validity using flake8."""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name
    try:
        result = subprocess.run(["flake8", temp_file_path], capture_output=True, text=True)
        if result.returncode == 0:
            logger.debug(f"flake8: No issues found in '{temp_file_path}'.")
            return True
        else:
            logger.error(f"flake8 found issues in '{temp_file_path}':\n{result.stdout}")
            return False
    except Exception as e:
        logger.error(f"Error running flake8 on '{temp_file_path}': {e}", exc_info=True)
        return False
    finally:
        os.remove(temp_file_path)


# Python-specific functions
def extract_python_structure(code: str) -> Dict[str, Any]:
    """
    Extracts the structure of Python code, including functions (sync and async) and classes with their methods.

    Parameters:
        code (str): The Python source code.

    Returns:
        Dict[str, Any]: A dictionary containing the structure of the code.
    """
    logger.debug("Starting extract_python_structure")
    logger.debug(f"Input code: {code[:100]}...")  # Log first 100 characters of the code for brevity
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        # Initialize the structure dictionary
        structure = {
            "functions": [],
            "classes": []
        }
        logger.debug("Successfully parsed code into AST")

        # Helper function to recursively set parent attributes
        def set_parent(node, parent=None):
            for child in ast.iter_child_nodes(node):
                child.parent = parent
                set_parent(child, parent=node)

        # Set parent attributes for all nodes
        set_parent(tree)

        # Traverse all AST nodes
        for node in ast.walk(tree):
            # Handle top-level functions (excluding methods within classes)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not isinstance(getattr(node, 'parent', None), ast.ClassDef):
                    func_type = "async" if isinstance(node, ast.AsyncFunctionDef) else "function"
                    logger.debug(f"Found top-level {func_type}: {node.name}")
                    structure["functions"].append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node),
                        "async": isinstance(node, ast.AsyncFunctionDef)
                    })
            # Handle classes and their methods
            elif isinstance(node, ast.ClassDef):
                methods = []
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_type = "async" if isinstance(child, ast.AsyncFunctionDef) else "function"
                        methods.append({
                            "name": child.name,
                            "args": [arg.arg for arg in child.args.args],
                            "docstring": ast.get_docstring(child),
                            "async": isinstance(child, ast.AsyncFunctionDef),
                            "type": method_type
                        })
                logger.debug(f"Found class: {node.name} with methods: {methods}")
                structure["classes"].append({
                    "name": node.name,
                    "methods": methods
                })

        logger.debug(f"Extracted structure: {structure}")
        return structure

    except SyntaxError as se:
        logger.error(f"Syntax error in Python code: {se}")
        logger.error(f"Problematic code:\n{code}")
        return {}
    except Exception as e:
        logger.error(f"Error extracting Python structure: {e}", exc_info=True)
        return {}


def insert_python_docstrings(original_code: str, documentation: Dict[str, Any]) -> str:
    """
    Inserts docstrings into Python functions and classes based on provided documentation.

    Parameters:
        original_code (str): The original Python source code.
        documentation (Dict[str, Any]): A dictionary containing documentation details,
                                        primarily the 'summary' and 'changes_made'.

    Returns:
        str: The modified Python source code with inserted docstrings.
    """
    logger.debug("Starting insert_python_docstrings")
    try:
        tree = ast.parse(original_code)
        summary = documentation.get("summary", "").strip()

        if not summary:
            logger.warning("No summary provided in documentation. Skipping docstring insertion.")
            return original_code

        # Truncate summary to prevent overly long docstrings (optional)
        max_length = 300  # Adjust as needed
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if not docstring:
                    docstring_node = ast.Expr(value=ast.Constant(value=summary, kind=None))
                    node.body.insert(0, docstring_node)
                    node_lineno = getattr(node, 'lineno', 'unknown')
                    node_type = 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class'
                    logger.debug(f"Inserted docstring in {node_type}: {node.name} at line {node_lineno}")

        modified_code = astor.to_source(tree)
        logger.debug("Completed inserting Python docstrings")
        return modified_code
    except Exception as e:
        logger.error(f"Error inserting Python docstrings: {e}", exc_info=True)
        return original_code


def is_valid_python_code(code: str) -> bool:
    """
    Validates Python code for syntax correctness.

    Parameters:
        code (str): The Python source code.

    Returns:
        bool: True if the code is valid, False otherwise.
    """
    logger.debug("Starting is_valid_python_code")
    logger.debug(f"Code to validate (first 100 chars): {code[:100]}...")
    try:
        ast.parse(code)
        logger.debug("Python code is valid.")
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in Python code: {e}")
        return False


# JS/TS-specific functions
async def extract_js_ts_structure(file_path: str, code: str, language: str) -> Optional[Dict[str, Any]]:
    logger.debug('Starting extract_js_ts_structure')
    try:
        # Define the path to the JS/TS extraction script
        script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'extract_structure.js')
        if not os.path.isfile(script_path):
            logger.error(f"JS/TS extraction script '{script_path}' not found.")
            return None

        # Use a temporary file to pass code to the script
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.js' if language == 'javascript' else '.ts') as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        # Run the Node.js script
        process = await asyncio.create_subprocess_exec(
            'node', script_path, temp_file_path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()
        os.remove(temp_file_path)  # Clean up the temp file

        if process.returncode != 0:
            logger.error(f"JS/TS extraction script error for '{file_path}': {stderr.decode().strip()}")
            return None

        # Parse the JSON output from the script
        try:
            structure = json.loads(stdout.decode())
            logger.debug(f'Extracted JS/TS structure: {structure}')
            return structure
        except json.JSONDecodeError as e:
            logger.error(f'Invalid JSON output from JS/TS extraction script: {e}')
            logger.error(f'Script output:\n{stdout.decode()}')
            return None

    except Exception as e:
        logger.error(f"Exception in extract_js_ts_structure: {e}", exc_info=True)
        return None

def insert_js_ts_docstrings(original_code: str, documentation: Dict[str, Any]) -> str:
    """
    Inserts docstrings into JavaScript/TypeScript code by running an external Node.js script.

    Parameters:
        original_code (str): The original JS/TS source code.
        documentation (Dict[str, Any]): A dictionary containing documentation details.

    Returns:
        str: The modified JS/TS source code with inserted docstrings.
    """
    logger.debug("Starting insert_js_ts_docstrings")
    logger.debug(f"Original code (first 100 chars): {original_code[:100]}...")
    logger.debug(f"Documentation: {documentation}")
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'insert_docstrings.js')
        if not os.path.isfile(script_path):
            logger.error(f"JS/TS insertion script '{script_path}' not found.")
            return original_code

        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.js') as code_file:
            code_file.write(original_code)
            code_file_path = code_file.name

        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as doc_file:
            json.dump(documentation, doc_file)
            doc_file_path = doc_file.name

        process = subprocess.run(
            ['node', script_path, code_file_path, doc_file_path],
            capture_output=True,
            text=True
        )

        os.remove(code_file_path)
        os.remove(doc_file_path)

        if process.returncode != 0:
            logger.error(f"JS/TS insertion script error: {process.stderr.strip()}")
            return original_code

        modified_code = process.stdout
        logger.debug("Completed inserting JS/TS docstrings")
        return modified_code
    except Exception as e:
        logger.error(f"Error inserting JS/TS docstrings: {e}", exc_info=True)
        return original_code


# HTML-specific functions
def extract_html_structure(code: str) -> Dict[str, Any]:
    """
    Extracts the structure of HTML code, including tags and their attributes.

    Parameters:
        code (str): The HTML source code.

    Returns:
        Dict[str, Any]: A dictionary containing the HTML structure.
    """
    logger.debug("Starting extract_html_structure")
    logger.debug(f"HTML code (first 100 chars): {code[:100]}...")
    try:
        soup = BeautifulSoup(code, 'html.parser')
        structure = {
            "tags": []
        }
        for tag in soup.find_all(True):
            children = []
            for child in tag.children:
                if child.name:
                    children.append({
                        "name": child.name,
                        "attributes": child.attrs
                    })
            structure["tags"].append({
                "name": tag.name,
                "attributes": tag.attrs,
                "children": children
            })
            logger.debug(f"Extracted tag: {tag.name}")

        # If no tags are found, explicitly note it
        if not structure["tags"]:
            logger.info("No HTML tags found in the source code.")

        logger.debug(f"Extracted structure: {structure}")
        return structure
    except Exception as e:
        logger.error(f"Error extracting HTML structure: {e}", exc_info=True)
        return {}


def insert_html_comments(original_code: str, documentation: Dict[str, Any]) -> str:
    """
    Inserts comments into HTML code based on provided documentation.

    Parameters:
        original_code (str): The original HTML source code.
        documentation (Dict[str, Any]): A dictionary containing documentation details.

    Returns:
        str: The modified HTML source code with inserted comments.
    """
    logger.debug("Starting insert_html_comments")
    logger.debug(f"Original HTML code (first 100 chars): {original_code[:100]}...")
    logger.debug(f"Documentation: {documentation}")
    try:
        soup = BeautifulSoup(original_code, 'html.parser')
        existing_comments = soup.find_all(string=lambda text: isinstance(text, Comment))

        summary = documentation.get("summary", "").strip()
        changes = documentation.get("changes_made", [])

        if not summary and not changes:
            logger.warning("No summary or changes provided in documentation. Skipping comment insertion.")
            return original_code

        new_comment_parts = []
        if summary:
            new_comment_parts.append(f" Summary: {summary} ")
        if changes:
            changes_formatted = "; ".join(changes)
            new_comment_parts.append(f" Changes: {changes_formatted} ")

        new_comment_text = " ".join(new_comment_parts)

        duplicate = False
        for comment in existing_comments:
            if new_comment_text.strip() in comment:
                duplicate = True
                break

        if not duplicate:
            comment = Comment(new_comment_text)
            if soup.body:
                soup.body.insert(0, comment)
            else:
                soup.insert(0, comment)
            logger.debug("Inserted new HTML comment.")
        else:
            logger.debug("HTML comment already exists. Skipping insertion.")

        modified_code = str(soup)
        logger.debug("Completed inserting HTML comments")
        return modified_code
    except Exception as e:
        logger.error(f"Error inserting HTML comments: {e}", exc_info=True)
        logger.error(f"Problematic HTML Code:\n{original_code}")
        return original_code


# CSS-specific functions
def extract_css_structure(code: str) -> Dict[str, Any]:
    """
    Extracts the structure of CSS code, including selectors and declarations.

    Parameters:
        code (str): The CSS source code.

    Returns:
        Dict[str, Any]: A dictionary containing the CSS structure.
    """
    logger.debug("Starting extract_css_structure")
    logger.debug(f"CSS code (first 100 chars): {code[:100]}...")
    try:
        rules = tinycss2.parse_stylesheet(code, skip_whitespace=True, skip_comments=True)
        structure = {
            "rules": []
        }
        for rule in rules:
            if rule.type == 'qualified-rule':
                selectors = ''.join([token.serialize() for token in rule.prelude]).strip()
                declarations = []
                for decl in tinycss2.parse_declaration_list(rule.content):
                    if decl.type == 'declaration':
                        declarations.append({
                            "property": decl.name,
                            "value": ''.join([token.serialize() for token in decl.value]).strip()
                        })
                structure["rules"].append({
                    "selectors": selectors,
                    "declarations": declarations
                })
                logger.debug(f"Extracted rule: {selectors}")

        # If no rules are found, explicitly note it
        if not structure["rules"]:
            logger.info("No CSS rules found in the source code.")

        logger.debug(f"Extracted structure: {structure}")
        return structure
    except Exception as e:
        logger.error(f"Error extracting CSS structure: {e}", exc_info=True)
        return {}


def insert_css_docstrings(original_code: str, documentation: Dict[str, Any]) -> str:
    """
    Inserts comments into CSS code based on provided documentation.

    Parameters:
        original_code (str): The original CSS source code.
        documentation (Dict[str, Any]): A dictionary containing documentation details.

    Returns:
        str: The modified CSS source code with inserted comments.
    """
    logger.debug("Starting insert_css_docstrings")
    logger.debug(f"Original CSS code (first 100 chars): {original_code[:100]}...")
    logger.debug(f"Documentation: {documentation}")
    try:
        # CSS doesn't have a standard comment syntax recognized by BeautifulSoup, so we'll manually insert comments
        summary = documentation.get("summary", "").strip()
        changes = documentation.get("changes_made", [])

        if not summary and not changes:
            logger.warning("No summary or changes provided in documentation. Skipping comment insertion.")
            return original_code

        new_comment_parts = []
        if summary:
            new_comment_parts.append(f" Summary: {summary} ")
        if changes:
            changes_formatted = "; ".join(changes)
            new_comment_parts.append(f" Changes: {changes_formatted} ")

        new_comment_text = "/*" + " ".join(new_comment_parts) + "*/\n"

        # Check if the first line is already a comment with the same text
        if original_code.startswith(new_comment_text):
            logger.debug("CSS comment already exists. Skipping insertion.")
            return original_code

        # Insert the new comment at the beginning of the CSS code
        modified_code = new_comment_text + original_code
        logger.debug("Inserted new CSS comment.")
        logger.debug("Completed inserting CSS docstrings")
        return modified_code
    except Exception as e:
        logger.error(f"Error inserting CSS docstrings: {e}", exc_info=True)
        return original_code
