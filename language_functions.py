# language_functions.py

import os  # Added import
import fnmatch
import json
import ast
import asyncio
import subprocess
import logging
import traceback
from typing import Optional

logger = logging.getLogger(__name__)

# Python-specific functions
def extract_python_structure(code: str) -> Dict[str, Any]:
    """
    Extracts the structure of a Python codebase.
    
    Parameters:
        code (str): The Python code as a string.
    
    Returns:
        Dict[str, Any]: A dictionary representing the code structure.
    """
    try:
        tree = ast.parse(code)
        structure = {
            "functions": [],
            "classes": []
        }
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                structure["functions"].append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node)
                })
            elif isinstance(node, ast.ClassDef):
                structure["classes"].append({
                    "name": node.name,
                    "methods": [
                        {
                            "name": method.name,
                            "args": [arg.arg for arg in method.args.args],
                            "docstring": ast.get_docstring(method)
                        }
                        for method in node.body if isinstance(method, ast.FunctionDef)
                    ]
                })
        return structure
    except Exception as e:
        logger.error(f"Error extracting Python structure: {e}", exc_info=True)
        return {}

def insert_python_docstrings(original_code: str, documentation: Dict[str, Any]) -> str:
    """
    Inserts docstrings into Python functions and classes based on the provided documentation.
    
    Parameters:
        original_code (str): The original Python code.
        documentation (Dict[str, Any]): The documentation containing summaries and changes.
    
    Returns:
        str: The Python code with inserted docstrings.
    """
    try:
        tree = ast.parse(original_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                if not docstring:
                    # Insert a docstring
                    summary = documentation.get("summary", "")
                    node.body.insert(0, ast.Expr(value=ast.Str(s=summary)))
            elif isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                if not docstring:
                    # Insert a docstring
                    summary = documentation.get("summary", "")
                    node.body.insert(0, ast.Expr(value=ast.Str(s=summary)))
        return astor.to_source(tree)
    except Exception as e:
        logger.error(f"Error inserting Python docstrings: {e}", exc_info=True)
        return original_code

def is_valid_python_code(code: str) -> bool:
    """
    Validates Python code by attempting to parse it.
    
    Parameters:
        code (str): The Python code to validate.
    
    Returns:
        bool: True if the code is valid, False otherwise.
    """
    try:
        ast.parse(code)
        logger.debug("Python code is valid.")
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in Python code: {e}")
        return False

# JS/TS-specific functions
async def extract_js_ts_structure(file_path: str, code: str, language: str) -> Optional[Dict[str, Any]]:
    """
    Extracts the structure of a JavaScript or TypeScript codebase.
    
    Parameters:
        file_path (str): The path to the JS/TS file.
        code (str): The JS/TS code as a string.
        language (str): 'javascript' or 'typescript'.
    
    Returns:
        Optional[Dict[str, Any]]: A dictionary representing the code structure or None if extraction fails.
    """
    try:
        # Use a Node.js script to parse the JS/TS code structure
        script_path = os.path.join(os.path.dirname(__file__), 'extract_structure.js')
        if not os.path.isfile(script_path):
            logger.error(f"JS/TS extraction script '{script_path}' not found.")
            return None
        
        # Write the code to a temporary file
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.js' if language == 'javascript' else '.ts') as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        
        # Execute the Node.js script
        process = await asyncio.create_subprocess_exec(
            'node', script_path, temp_file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        # Remove the temporary file
        os.remove(temp_file_path)
        
        if process.returncode != 0:
            logger.error(f"JS/TS extraction script error for '{file_path}': {stderr.decode().strip()}")
            return None
        
        structure = json.loads(stdout.decode())
        return structure
    except Exception as e:
        logger.error(f"Unexpected error while extracting JS/TS structure from '{file_path}': {e}", exc_info=True)
        return None

def insert_js_ts_docstrings(original_code: str, documentation: Dict[str, Any]) -> str:
    """
    Inserts JSDoc comments into JS/TS functions based on the provided documentation.
    
    Parameters:
        original_code (str): The original JS/TS code.
        documentation (Dict[str, Any]): The documentation containing summaries and changes.
    
    Returns:
        str: The JS/TS code with inserted JSDoc comments.
    """
    try:
        # Use a Node.js script to insert docstrings
        script_path = os.path.join(os.path.dirname(__file__), 'insert_docstrings.js')
        if not os.path.isfile(script_path):
            logger.error(f"JS/TS insertion script '{script_path}' not found.")
            return original_code
        
        # Write the original code and documentation to temporary files
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.js') as code_file:
            code_file.write(original_code)
            code_file_path = code_file.name
        
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as doc_file:
            json.dump(documentation, doc_file)
            doc_file_path = doc_file.name
        
        # Execute the Node.js script
        process = subprocess.run(
            ['node', script_path, code_file_path, doc_file_path],
            capture_output=True,
            text=True
        )
        
        # Remove temporary files
        os.remove(code_file_path)
        os.remove(doc_file_path)
        
        if process.returncode != 0:
            logger.error(f"JS/TS insertion script error: {process.stderr.strip()}")
            return original_code
        
        return process.stdout
    except Exception as e:
        logger.error(f"Error inserting JS/TS docstrings: {e}", exc_info=True)
        return original_code

# HTML-specific functions
def extract_html_structure(code: str) -> Dict[str, Any]:
    """
    Extracts the structure of an HTML codebase.
    
    Parameters:
        code (str): The HTML code as a string.
    
    Returns:
        Dict[str, Any]: A dictionary representing the code structure.
    """
    try:
        soup = BeautifulSoup(code, 'html.parser')
        structure = {
            "tags": []
        }
        for tag in soup.find_all(True):
            structure["tags"].append({
                "name": tag.name,
                "attributes": tag.attrs,
                "children": [child.name for child in tag.children if child.name]
            })
        return structure
    except Exception as e:
        logger.error(f"Error extracting HTML structure: {e}", exc_info=True)
        return {}

def insert_html_comments(original_code: str, documentation: Dict[str, Any]) -> str:
    """
    Inserts HTML comments based on the provided documentation.
    
    Parameters:
        original_code (str): The original HTML code.
        documentation (Dict[str, Any]): The documentation containing summaries and changes.
    
    Returns:
        str: The HTML code with inserted comments.
    """
    try:
        soup = BeautifulSoup(original_code, 'html.parser')
        comment_text = f" Summary: {documentation.get('summary', '')} Changes: {', '.join(documentation.get('changes', []))} "
        comment = Comment(comment_text)
        if soup.body:
            soup.body.insert(0, comment)
        else:
            soup.insert(0, comment)
        return str(soup)
    except Exception as e:
        logger.error(f"Error inserting HTML comments: {e}", exc_info=True)
        return original_code

# CSS-specific functions
def extract_css_structure(code: str) -> Dict[str, Any]:
    """
    Extracts the structure of a CSS codebase.
    
    Parameters:
        code (str): The CSS code as a string.
    
    Returns:
        Dict[str, Any]: A dictionary representing the code structure.
    """
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
        return structure
    except Exception as e:
        logger.error(f"Error extracting CSS structure: {e}", exc_info=True)
        return {}
        
def insert_css_docstrings(original_code: str, documentation: Dict[str, Any]) -> str:
    """
    Inserts CSS comments based on the provided documentation.
    
    Parameters:
        original_code (str): The original CSS code.
        documentation (Dict[str, Any]): The documentation containing summaries and changes.
    
    Returns:
        str: The CSS code with inserted comments.
    """
    try:
        soup = BeautifulSoup(original_code, 'html.parser')  # Using BeautifulSoup for simplicity
        existing_comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        
        # Construct the new comment
        new_comment_text = f" Summary: {documentation.get('summary', '')} Changes: {', '.join(documentation.get('changes', []))} "
        
        # Check for duplication
        duplicate = False
        for comment in existing_comments:
            if new_comment_text.strip() in comment:
                duplicate = True
                break
        
        if not duplicate:
            comment = Comment(new_comment_text)
            soup.insert(0, comment)
            logger.debug("Inserted new CSS comment.")
        else:
            logger.debug("CSS comment already exists. Skipping insertion.")
        
        return str(soup)
    except Exception as e:
        logger.error(f"Error inserting CSS docstrings: {e}", exc_info=True)
        return original_code
