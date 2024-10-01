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
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Python-specific functions
def extract_python_structure(code: str) -> Dict[str, Any]:
    logger.debug("Starting extract_python_structure")
    logger.debug(f"Input code: {code[:100]}...")  # Log first 100 characters of the code for brevity
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
                logger.debug(f"Extracted function: {node.name}")
            elif isinstance(node, ast.ClassDef):
                methods = [
                    {
                        "name": method.name,
                        "args": [arg.arg for arg in method.args.args],
                        "docstring": ast.get_docstring(method)
                    }
                    for method in node.body if isinstance(method, ast.FunctionDef)
                ]
                structure["classes"].append({
                    "name": node.name,
                    "methods": methods
                })
                logger.debug(f"Extracted class: {node.name} with methods: {methods}")
        logger.debug("Completed extracting Python structure")
        return structure
    except Exception as e:
        logger.error(f"Error extracting Python structure: {e}", exc_info=True)
        return {}

def insert_python_docstrings(original_code: str, documentation: Dict[str, Any]) -> str:
    logger.debug("Starting insert_python_docstrings")
    logger.debug(f"Original code: {original_code[:100]}...")  # Log first 100 characters of the code for brevity
    logger.debug(f"Documentation: {documentation}")
    try:
        tree = ast.parse(original_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                if not docstring:
                    summary = documentation.get("summary", "")
                    node.body.insert(0, ast.Expr(value=ast.Str(s=summary)))
                    logger.debug(f"Inserted docstring in function: {node.name}")
            elif isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                if not docstring:
                    summary = documentation.get("summary", "")
                    node.body.insert(0, ast.Expr(value=ast.Str(s=summary)))
                    logger.debug(f"Inserted docstring in class: {node.name}")
        modified_code = astor.to_source(tree)
        logger.debug("Completed inserting Python docstrings")
        return modified_code
    except Exception as e:
        logger.error(f"Error inserting Python docstrings: {e}", exc_info=True)
        return original_code

def is_valid_python_code(code: str) -> bool:
    logger.debug("Starting is_valid_python_code")
    logger.debug(f"Code to validate: {code[:100]}...")  # Log first 100 characters of the code for brevity
    try:
        ast.parse(code)
        logger.debug("Python code is valid.")
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in Python code: {e}")
        return False

# JS/TS-specific functions
async def extract_js_ts_structure(file_path: str, code: str, language: str) -> Optional[Dict[str, Any]]:
    logger.debug("Starting extract_js_ts_structure")
    logger.debug(f"File path: {file_path}, Language: {language}")
    logger.debug(f"Code: {code[:100]}...")  # Log first 100 characters of the code for brevity
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'extract_structure.js')
        if not os.path.isfile(script_path):
            logger.error(f"JS/TS extraction script '{script_path}' not found.")
            return None
        
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.js' if language == 'javascript' else '.ts') as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        
        process = await asyncio.create_subprocess_exec(
            'node', script_path, temp_file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        os.remove(temp_file_path)
        
        if process.returncode != 0:
            logger.error(f"JS/TS extraction script error for '{file_path}': {stderr.decode().strip()}")
            return None
        
        structure = json.loads(stdout.decode())
        logger.debug(f"Extracted JS/TS structure: {structure}")
        return structure
    except Exception as e:
        logger.error(f"Unexpected error while extracting JS/TS structure from '{file_path}': {e}", exc_info=True)
        return None

def insert_js_ts_docstrings(original_code: str, documentation: Dict[str, Any]) -> str:
    logger.debug("Starting insert_js_ts_docstrings")
    logger.debug(f"Original code: {original_code[:100]}...")  # Log first 100 characters of the code for brevity
    logger.debug(f"Documentation: {documentation}")
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'insert_docstrings.js')
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
    logger.debug("Starting extract_html_structure")
    logger.debug(f"HTML code: {code[:100]}...")  # Log first 100 characters of the code for brevity
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
        logger.debug("Completed extracting HTML structure")
        return structure
    except Exception as e:
        logger.error(f"Error extracting HTML structure: {e}", exc_info=True)
        return {}

def insert_html_comments(original_code: str, documentation: Dict[str, Any]) -> str:
    logger.debug("Starting insert_html_comments")
    logger.debug(f"Original HTML code: {original_code[:100]}...")  # Log first 100 characters of the code for brevity
    logger.debug(f"Documentation: {documentation}")
    try:
        soup = BeautifulSoup(original_code, 'html.parser')
        existing_comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        
        new_comment_text = f" Summary: {documentation.get('summary', '')} Changes: {', '.join(documentation.get('changes', []))} "
        
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
        return original_code

# CSS-specific functions
def extract_css_structure(code: str) -> Dict[str, Any]:
    logger.debug("Starting extract_css_structure")
    logger.debug(f"CSS code: {code[:100]}...")  # Log first 100 characters of the code for brevity
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
        logger.debug("Completed extracting CSS structure")
        return structure
    except Exception as e:
        logger.error(f"Error extracting CSS structure: {e}", exc_info=True)
        return {}

def insert_css_docstrings(original_code: str, documentation: Dict[str, Any]) -> str:
    logger.debug("Starting insert_css_docstrings")
    logger.debug(f"Original CSS code: {original_code[:100]}...")  # Log first 100 characters of the code for brevity
    logger.debug(f"Documentation: {documentation}")
    try:
        soup = BeautifulSoup(original_code, 'html.parser')
        existing_comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        
        new_comment_text = f" Summary: {documentation.get('summary', '')} Changes: {', '.join(documentation.get('changes', []))} "
        
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
        
        modified_code = str(soup)
        logger.debug("Completed inserting CSS docstrings")
        return modified_code
    except Exception as e:
        logger.error(f"Error inserting CSS docstrings: {e}", exc_info=True)
        return original_code
