import os
import sys
import json
import logging
import subprocess
import ast
import astor
import tinycss2
from bs4 import BeautifulSoup, Comment
from .utils import (
    is_valid_extension,
    get_language,
    generate_documentation_prompt,
    fetch_documentation,
    is_binary,
)
import aiofiles
import aiohttp
import asyncio
from typing import Set, List
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)

# Python handlers
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
                    "args": [],
                    "returns": {"type": "Any"},
                    "decorators": [],
                    "docstring": ast.get_docstring(node) or ""
                }
                for arg in node.args.args:
                    arg_type = "Any"
                    if arg.annotation:
                        arg_type = get_node_source(arg.annotation)
                    func_info["args"].append({
                        "name": arg.arg,
                        "type": arg_type
                    })
                if node.returns:
                    func_info["returns"]["type"] = get_node_source(node.returns)
                for decorator in node.decorator_list:
                    func_info["decorators"].append(get_node_source(decorator))
                parent = parent_map.get(node)
                if isinstance(parent, ast.ClassDef):
                    class_name = parent.name
                    class_obj = next(
                        (cls for cls in classes if cls["name"] == class_name), None)
                    if not class_obj:
                        class_obj = {
                            "name": class_name,
                            "bases": [get_node_source(base) for base in parent.bases],
                            "methods": [],
                            "decorators": [],
                            "docstring": ast.get_docstring(parent) or ""
                        }
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
                        "bases": [get_node_source(base) for base in node.bases],
                        "methods": [],
                        "decorators": [],
                        "docstring": ast.get_docstring(node) or ""
                    }
                    for decorator in node.decorator_list:
                        class_info["decorators"].append(get_node_source(decorator))
                    classes.append(class_info)
        return {
            "language": "python",
            "functions": functions,
            "classes": classes
        }
    except Exception as e:
        logger.error(f"Error parsing Python code: {e}")
        return {}

def insert_python_docstrings(file_content: str, docstrings: dict) -> str:
    """Inserts docstrings into Python code."""
    try:
        tree = ast.parse(file_content)
        parent_map = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent_map[child] = node

        func_doc_map = {func['name']: func['docstring']
                        for func in docstrings.get('functions', [])}
        class_doc_map = {cls['name']: cls['docstring']
                         for cls in docstrings.get('classes', [])}
        method_doc_map = {}
        for cls in docstrings.get('classes', []):
            for method in cls.get('methods', []):
                method_doc_map[(cls['name'], method['name'])] = method['docstring']

        class DocstringInserter(ast.NodeTransformer):

            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                parent = parent_map.get(node)
                if isinstance(parent, ast.ClassDef):
                    key = (parent.name, node.name)
                    docstring = method_doc_map.get(key)
                else:
                    docstring = func_doc_map.get(node.name)
                if docstring:
                    docstring_node = ast.Expr(value=ast.Constant(value=docstring) if hasattr(ast, 'Constant') else ast.Str(s=docstring))
                    if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                        node.body.insert(0, docstring_node)
                    else:
                        node.body[0] = docstring_node
                return node

            def visit_ClassDef(self, node):
                self.generic_visit(node)
                docstring = class_doc_map.get(node.name)
                if docstring:
                    docstring_node = ast.Expr(value=ast.Constant(value=docstring) if hasattr(ast, 'Constant') else ast.Str(s=docstring))
                    if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                        node.body.insert(0, docstring_node)
                    else:
                        node.body[0] = docstring_node
                return node

        inserter = DocstringInserter()
        new_tree = inserter.visit(tree)
        new_code = astor.to_source(new_tree)

        try:
            ast.parse(new_code)
        except SyntaxError as e:
            logger.error(f"Syntax error in modified Python code: {e}")
            return file_content

        return new_code
    except Exception as e:
        logger.error(f"Error inserting docstrings into Python code: {e}")
        return file_content

def is_valid_python_code(code: str) -> bool:
    """Checks if the given code is valid Python code."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

# JavaScript/TypeScript handlers
def extract_js_ts_structure(file_content: str, language: str) -> dict:
    """Extracts the structure of JavaScript/TypeScript code using Esprima."""
    try:
        # Write the content to a temporary file
        with open('temp_code.js', 'w', encoding='utf-8') as temp_file:
            temp_file.write(file_content)

        # Run Esprima to parse the code
        result = subprocess.run(
            ["esprima", "-c", "-t", "temp_code.js"],
            capture_output=True,
            text=True,
            check=True
        )

        tree = json.loads(result.stdout)

        functions = []
        classes = []

        def traverse(node, parent_class=None):
            if isinstance(node, dict):
                if node.get("type") == "FunctionDeclaration":
                    func_info = {
                        "name": node["id"]["name"] if node.get("id") else "anonymous",
                        "params": [p.get("name", "param") for p in node.get("params", [])],
                        "loc": node.get("loc"),
                        "docstring": ""
                    }
                    if parent_class:
                        parent_class["methods"].append(func_info)
                    else:
                        functions.append(func_info)
                elif node.get("type") == "ClassDeclaration":
                    class_info = {
                        "name": node["id"]["name"],
                        "methods": [],
                        "loc": node.get("loc"),
                        "docstring": ""
                    }
                    classes.append(class_info)
                    traverse(node.get("body", {}), parent_class=class_info)
                elif node.get("type") == "MethodDefinition":
                    func_info = {
                        "name": node["key"]["name"],
                        "params": [p.get("name", "param") for p in node.get("value", {}).get("params", [])],
                        "loc": node.get("loc"),
                        "docstring": ""
                    }
                    if parent_class:
                        parent_class["methods"].append(func_info)
                else:
                    for key, value in node.items():
                        if isinstance(value, dict):
                            traverse(value, parent_class)
                        elif isinstance(value, list):
                            for item in value:
                                traverse(item, parent_class)
            elif isinstance(node, list):
                for item in node:
                    traverse(item, parent_class)

        traverse(tree)

        # Remove the temporary file
        os.remove('temp_code.js')

        return {
            "language": language,
            "functions": functions,
            "classes": classes,
            "tree": tree,
            "source_code": file_content
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"Esprima parsing failed: {e.stderr}")
        return {}
    except Exception as e:
        logger.error(f"Error during JS/TS structure extraction: {e}")
        return {}

def insert_js_ts_docstrings(docstrings: dict) -> str:
    """Inserts JSDoc comments into JavaScript/TypeScript code."""
    try:
        source_code = docstrings['source_code']
        tree = docstrings['tree']

        inserts = []

        # Function to map locations to positions in the code
        def get_position(loc):
            lines = source_code.splitlines()
            start_line = loc['start']['line'] - 1
            start_column = loc['start']['column']
            pos = sum(len(line) + 1 for line in lines[:start_line]) + start_column
            return pos

        for item_type in ['functions', 'classes']:
            for item in docstrings.get(item_type, []):
                docstring = item['docstring']
                if docstring and item.get('loc'):
                    position = get_position(item['loc'])
                    formatted_comment = format_jsdoc_comment(docstring)
                    inserts.append((position, formatted_comment))
                    if item_type == 'classes':
                        for method in item.get('methods', []):
                            docstring = method['docstring']
                            if docstring and method.get('loc'):
                                position = get_position(method['loc'])
                                formatted_comment = format_jsdoc_comment(docstring)
                                inserts.append((position, formatted_comment))

        # Sort inserts by position descending to avoid position shifts
        inserts.sort(key=lambda x: x[0], reverse=True)
        code = source_code
        for position, comment in inserts:
            code = code[:position] + comment + '\n' + code[position:]

        return code

    except Exception as e:
        logger.error(f"Error inserting docstrings into JS/TS code: {e}")
        return docstrings.get('source_code', '')

def format_jsdoc_comment(docstring: str) -> str:
    """Formats a docstring into a JSDoc comment block."""
    comment_lines = ['/**']
    for line in docstring.strip().split('\n'):
        comment_lines.append(f' * {line}')
    comment_lines.append(' */')
    return '\n'.join(comment_lines)

# HTML handlers
def extract_html_structure(file_content: str) -> dict:
    """Extracts the structure of HTML code."""
    try:
        soup = BeautifulSoup(file_content, 'html.parser')
        elements = []

        def traverse(node):
            for child in node.children:
                if isinstance(child, str):
                    continue
                if child.name:
                    element_info = {
                        'tag': child.name,
                        'attributes': dict(child.attrs),
                        'text': child.get_text(strip=True),
                        'docstring': ''
                    }
                    elements.append(element_info)
                    traverse(child)

        traverse(soup)

        return {
            'language': 'html',
            'elements': elements
        }

    except Exception as e:
        logger.error(f"Error parsing HTML code: {e}")
        return {}

def insert_html_comments(file_content: str, docstrings: dict) -> str:
    """Inserts comments into HTML code."""
    try:
        soup = BeautifulSoup(file_content, 'html.parser')
        elements = docstrings.get('elements', [])

        element_map = {}
        for element in elements:
            key = (element['tag'], tuple(sorted(element['attributes'].items())), element['text'])
            element_map[key] = element['docstring']

        def traverse_and_insert(node):
            for child in node.children:
                if isinstance(child, str):
                    continue
                if child.name:
                    key = (child.name, tuple(sorted(child.attrs.items())), child.get_text(strip=True))
                    docstring = element_map.get(key)
                    if docstring:
                        comment = Comment(f" {docstring} ")
                        child.insert_before(comment)
                    traverse_and_insert(child)

        traverse_and_insert(soup)
        return str(soup)

    except Exception as e:
        logger.error(f"Error inserting comments into HTML code: {e}")
        return file_content

# CSS handlers
def extract_css_structure(file_content: str) -> dict:
    """Extracts the structure of CSS code."""
    try:
        rules = tinycss2.parse_stylesheet(file_content)
        style_rules = []

        for rule in rules:
            if rule.type == 'qualified-rule':
                prelude = tinycss2.serialize(rule.prelude).strip()
                content = tinycss2.serialize(rule.content).strip()
                rule_info = {
                    'selector': prelude,
                    'declarations': content,
                    'docstring': ''
                }
                style_rules.append(rule_info)

        return {
            'language': 'css',
            'rules': style_rules
        }

    except Exception as e:
        logger.error(f"Error parsing CSS code: {e}")
        return {}

def insert_css_comments(file_content: str, docstrings: dict) -> str:
    """Inserts comments into CSS code."""
    try:
        rules = tinycss2.parse_stylesheet(file_content)
        style_rules = docstrings.get('rules', [])
        rule_map = {}
        for rule in style_rules:
            key = rule['selector']
            docstring = rule.get('docstring', '')
            if key in rule_map:
                rule_map[key] += f"\n{docstring}"
            else:
                rule_map[key] = docstring

        modified_content = ''
        inserted_selectors = set()

        for rule in rules:
            if rule.type == 'qualified-rule':
                selector = tinycss2.serialize(rule.prelude).strip()
                if selector not in inserted_selectors:
                    docstring = rule_map.get(selector)
                    if docstring:
                        modified_content += f"/* {docstring} */\n"
                    inserted_selectors.add(selector)
                modified_content += tinycss2.serialize(rule).strip() + '\n'
            else:
                modified_content += tinycss2.serialize(rule).strip() + '\n'

        return modified_content

    except Exception as e:
        logger.error(f"Error inserting comments into CSS code: {e}")
        return file_content

# Process a single file
async def process_file(
    session: aiohttp.ClientSession,
    file_path: str,
    skip_types: Set[str],
    output_file: str,
    semaphore: asyncio.Semaphore,
    output_lock: asyncio.Lock,
    model_name: str
) -> None:
    """Processes a single file to generate and insert documentation."""
    try:
        _, ext = os.path.splitext(file_path)
        if not is_valid_extension(ext, skip_types) or is_binary(file_path):
            logger.debug(f"Skipping file '{file_path}'.")
            return

        language = get_language(ext)
        if language == 'plaintext':
            logger.debug(f"Skipping unsupported language in '{file_path}'.")
            return

        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
        except Exception as e:
            logger.error(f"Failed to read '{file_path}': {e}")
            return

        if language == 'python':
            code_structure = extract_python_structure(content)
        elif language in ['javascript', 'typescript']:
            code_structure = extract_js_ts_structure(content, language)
        elif language == 'html':
            code_structure = extract_html_structure(content)
        elif language == 'css':
            code_structure = extract_css_structure(content)
        else:
            logger.warning(f"Language '{language}' not supported for structured extraction.")
            return

        if not code_structure:
            logger.error(f"Failed to extract structure from '{file_path}'")
            return

        prompt = generate_documentation_prompt(code_structure)

        documentation = await fetch_documentation(session, prompt, semaphore, model_name)
        if not documentation:
            logger.error(f"Failed to generate documentation for '{file_path}'")
            return

        # Extract JSON content from the response
        json_content = extract_json_from_response(documentation)
        if not json_content:
            logger.error(f"Could not extract JSON content from the response for '{file_path}'")
            return

        # Attempt to parse the JSON content
        try:
            updated_code_structure = json.loads(json_content)
            if language in ['javascript', 'typescript']:
                updated_code_structure['tree'] = code_structure['tree']
                updated_code_structure['source_code'] = code_structure['source_code']
        except json.JSONDecodeError as e:
            # Try using json5 as a fallback
            try:
                import json5
                updated_code_structure = json5.loads(json_content)
                if language in ['javascript', 'typescript']:
                    updated_code_structure['tree'] = code_structure['tree']
                    updated_code_structure['source_code'] = code_structure['source_code']
            except (ImportError, json5.JSONDecodeError):
                logger.error(f"Failed to parse documentation JSON for '{file_path}': {e}")
                return

        if not updated_code_structure:
            logger.error(f"Could not update code structure for '{file_path}'. Skipping file.")
            return

        if language == 'python':
            new_content = insert_python_docstrings(content, updated_code_structure)
        elif language in ['javascript', 'typescript']:
            new_content = insert_js_ts_docstrings(updated_code_structure)
        elif language == 'html':
            new_content = insert_html_comments(content, updated_code_structure)
        elif language == 'css':
            new_content = insert_css_comments(content, updated_code_structure)
        else:
            new_content = content

        if language == 'python':
            if not is_valid_python_code(new_content):
                logger.error(f"Modified Python code is invalid. Aborting insertion for '{file_path}'")
                return

        try:
            backup_path = file_path + '.bak'
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.rename(file_path, backup_path)
            logger.info(f"Backup created at '{backup_path}'")

            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(new_content)
            logger.info(f"Inserted comments into '{file_path}'")
        except Exception as e:
            logger.error(f"Error writing to '{file_path}': {e}")

            if os.path.exists(backup_path):
                os.remove(file_path)
                os.rename(backup_path, file_path)
                logger.info(f"Restored original file from backup for '{file_path}'")
            return

        try:
            async with output_lock:
                async with aiofiles.open(output_file, 'a', encoding='utf-8') as f:
                    header = f"# File: {file_path}\n\n"
                    code_block = f"```{language}\n{new_content}\n```\n\n"
                    await f.write(header)
                    await f.write(code_block)
            logger.info(f"Successfully processed and documented '{file_path}'")
        except Exception as e:
            logger.error(f"Error writing documentation for '{file_path}': {e}")

    except Exception as e:
        logger.error(f"Unexpected error processing '{file_path}': {e}")

# Process all files
async def process_all_files(
    file_paths: List[str],
    skip_types: Set[str],
    output_file: str,
    semaphore: asyncio.Semaphore,
    output_lock: asyncio.Lock,
    model_name: str
) -> None:
    """Processes all files asynchronously to generate and insert documentation."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(
                process_file(session, file_path, skip_types, output_file, semaphore, output_lock, model_name)
            )
            for file_path in file_paths
        ]

        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Files"):
            try:
                await f
            except Exception as e:
                logger.error(f"Error processing a file: {e}")
