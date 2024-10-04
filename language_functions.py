import os
import sys
import json
import ast
import subprocess
import logging
import astor
import tempfile
from typing import Optional, Dict, Any, List
from bs4 import BeautifulSoup, Comment
import tinycss2
import esprima

# Set minimum Python version requirements
if sys.version_info < (3, 9):
    raise Exception("This script requires Python 3.9 or higher")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set logging level to DEBUG

# Create formatter with module, function, and line number
formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(funcName)s:%(lineno)d:%(message)s")

# Create file handler which logs debug and higher level messages
file_handler = logging.FileHandler("language_functions.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Change to DEBUG for more verbosity on console
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# language_functions.py
script_path_parser = os.path.join(os.path.dirname(__file__), "scripts", "acorn_parser.js")
script_path_inserter = os.path.join(os.path.dirname(__file__), "scripts", "acorn_inserter.js")


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
    except FileNotFoundError:
        logger.error("flake8 is not installed or not found in PATH.")
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
        structure = {"functions": [], "classes": []}
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
                if not isinstance(getattr(node, "parent", None), ast.ClassDef):
                    func_type = "async" if isinstance(node, ast.AsyncFunctionDef) else "function"
                    logger.debug(f"Found top-level {func_type}: {node.name}")
                    structure["functions"].append(
                        {
                            "name": node.name,
                            "args": [arg.arg for arg in node.args.args],
                            "docstring": ast.get_docstring(node),
                            "async": isinstance(node, ast.AsyncFunctionDef),
                        }
                    )
            # Handle classes and their methods
            elif isinstance(node, ast.ClassDef):
                methods = []
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_type = "async" if isinstance(child, ast.AsyncFunctionDef) else "function"
                        methods.append(
                            {
                                "name": child.name,
                                "args": [arg.arg for arg in child.args.args],
                                "docstring": ast.get_docstring(child),
                                "async": isinstance(child, ast.AsyncFunctionDef),
                                "type": method_type,
                            }
                        )
                logger.debug(f"Found class: {node.name} with methods: {methods}")
                structure["classes"].append(
                    {"name": node.name, "methods": methods, "docstring": ast.get_docstring(node)}
                )

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
    """Inserts docstrings into Python functions and classes based on the provided documentation details."""
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
            if isinstance(node, ast.Module) and "overview" in docstrings_mapping:
                doc_content = sanitize_docstring(docstrings_mapping["overview"])
                if ast.get_docstring(node, clean=False) is not None:
                    if (
                        isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    ):
                        node.body.pop(0)
                docstring_node = ast.Expr(value=ast.Str(s=doc_content))
                node.body.insert(0, docstring_node)
                logger.debug("Inserted module-level overview docstring.")
        
        ast.fix_missing_locations(tree)
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
    logger.debug("Validating Python code syntax.")
    try:
        compile(code, '<string>', 'exec')
        logger.debug("Python code is syntactically valid.")
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in Python code: {e}")
        return False
    

# JavaScript/TypeScript-specific functions
async def extract_js_ts_structure(
    file_path: str, code: str, language: str, function_schema: dict = None
) -> Optional[Dict[str, Any]]:
    """
    Extracts the structure of JavaScript/TypeScript code using acorn_parser.js.

    Parameters:
        file_path (str): The path to the JS/TS file.
        code (str): The JS/TS source code.
        language (str): The programming language ('javascript' or 'typescript').
        function_schema (dict): The function schema for validation (optional).

    Returns:
        Optional[Dict[str, Any]]: The extracted structure as a dictionary, or None if extraction fails.
    """
    logger.debug("Starting extract_js_ts_structure")
    try:
        # Prepare data to send to Node.js script
        data_to_send = {"code": code, "language": language, "functionSchema": function_schema}

        # Path to the acorn_parser.js script
        script_path = os.path.join(os.path.dirname(__file__), "scripts", "acorn_parser.js")

        # Check if the script exists
        if not os.path.exists(script_path):
            logger.error(f"acorn_parser.js script not found at {script_path}")
            return None

        # Run the Node.js script as a subprocess
        process = subprocess.run(["node", script_path], input=json.dumps(data_to_send), capture_output=True, text=True)

        if process.returncode == 0:
            # Parse the JSON output from acorn_parser.js
            structure = json.loads(process.stdout)
            logger.debug(f"Successfully extracted JS/TS structure: {structure}")
            return structure
        else:
            logger.error(f"Error running acorn_parser.js: {process.stderr}")
            return None

    except Exception as e:
        logger.error(f"Exception in extract_js_ts_structure: {e}", exc_info=True)
        return None


def insert_js_ts_docstrings(
    original_code: str, documentation: Dict[str, Any], language: str
) -> str:
    """Inserts JSDoc comments into JavaScript or TypeScript code based on the provided documentation."""
    logger.debug("Starting insert_js_ts_docstrings")
    try:
        documentation_json = json.dumps(documentation)
        data_to_send = {
            "code": original_code,
            "documentation": documentation,
            "language": language,
        }
        script_path = os.path.join(
            os.path.dirname(__file__), "scripts", "acorn_inserter.js"
        )
        process = subprocess.run(
            ["node", script_path],
            input=json.dumps(data_to_send),
            capture_output=True,
            text=True,
        )
        if process.returncode == 0:
            modified_code = process.stdout
            logger.debug("Completed inserting JSDoc docstrings")
            return modified_code
        else:
            logger.error(f"Error running acorn_inserter.js: {process.stderr}")
            return original_code
    except Exception as e:
        logger.error(f"Exception in insert_js_ts_docstrings: {e}", exc_info=True)
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
    try:
        soup = BeautifulSoup(code, "lxml")
        structure = {"tags": []}
        for tag in soup.find_all(True):
            structure["tags"].append({"name": tag.name, "attributes": tag.attrs})
            logger.debug(f"Extracted tag: {tag.name}")
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
    try:
        soup = BeautifulSoup(original_code, "lxml")  # Use 'lxml' parser

        # Prepare the comment content
        summary = documentation.get("summary", "").strip()
        changes = documentation.get("changes_made", [])

        if not summary and not changes:
            logger.warning("No summary or changes provided in documentation. Skipping comment insertion.")
            return original_code

        new_comment_parts = []
        if summary:
            new_comment_parts.append(f"Summary: {summary}")
        if changes:
            changes_formatted = "; ".join(changes)
            new_comment_parts.append(f"Changes: {changes_formatted}")

        new_comment_text = "<!-- " + " | ".join(new_comment_parts) + " -->"

        # Insert the comment at the beginning of the body or the document
        comment = Comment(" " + " | ".join(new_comment_parts) + " ")
        if soup.body:
            soup.body.insert(0, comment)
            logger.debug("Inserted comment at the beginning of the body.")
        else:
            soup.insert(0, comment)
            logger.debug("Inserted comment at the beginning of the document.")

        modified_code = soup.prettify()
        logger.debug("Completed inserting HTML comments")
        return modified_code
    except Exception as e:
        logger.error(f"Error inserting HTML comments: {e}", exc_info=True)
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
    try:
        rules = tinycss2.parse_rule_list(code, skip_whitespace=True, skip_comments=True)
        structure = {"rules": []}
        for rule in rules:
            if rule.type == "qualified-rule":
                selectors = "".join([token.serialize() for token in rule.prelude]).strip()
                declarations = []
                for decl in tinycss2.parse_declaration_list(rule.content):
                    if decl.type == "declaration":
                        declarations.append(
                            {
                                "property": decl.lower_name,
                                "value": "".join([token.serialize() for token in decl.value]).strip(),
                            }
                        )
                structure["rules"].append({"selectors": selectors, "declarations": declarations})
                logger.debug(f"Extracted rule: {selectors}")
            else:
                logger.debug(f"Ignored rule of type: {rule.type}")
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
    try:
        # Prepare the comment content
        summary = documentation.get("summary", "").strip()
        changes = documentation.get("changes_made", [])

        if not summary and not changes:
            logger.warning("No summary or changes provided in documentation. Skipping comment insertion.")
            return original_code

        new_comment_parts = []
        if summary:
            new_comment_parts.append(f"Summary: {summary}")
        if changes:
            changes_formatted = "; ".join(changes)
            new_comment_parts.append(f"Changes: {changes_formatted}")

        new_comment_text = "/* " + " | ".join(new_comment_parts) + " */\n"

        # Insert the comment at the beginning of the file
        modified_code = new_comment_text + original_code
        logger.debug("Inserted new CSS comment at the beginning of the file.")
        logger.debug("Completed inserting CSS docstrings")
        return modified_code
    except Exception as e:
        logger.error(f"Error inserting CSS docstrings: {e}", exc_info=True)
        return original_code
