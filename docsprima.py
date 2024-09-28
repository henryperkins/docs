import os
import sys
import argparse
import asyncio
import aiohttp
import aiofiles
import ast
import logging
import astor
import json
from typing import List, Set, Optional, Dict
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from bs4 import BeautifulSoup, Comment
import tinycss2
import subprocess  # For running esprima

load_dotenv()

logging.basicConfig(
    filename='docs_generation.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not set. Check environment or .env file.")
    sys.exit(1)


SEMAPHORE = None
OUTPUT_LOCK = None
MODEL_NAME = 'gpt-4'

DEFAULT_EXCLUDED_DIRS = {'.git', '__pycache__', 'node_modules'}
DEFAULT_EXCLUDED_FILES = {'.DS_Store'}
DEFAULT_SKIP_TYPES = {'.json', '.md', '.txt', '.csv'}

LANGUAGE_MAPPING = {
    '.py': 'python',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',  # Still supporting TypeScript, but parsing with Esprima
    '.tsx': 'typescript',  # Same as above
    '.html': 'html',
    '.htm': 'html',
    '.css': 'css',
}

def get_language(ext: str) -> str:
    """Determines the programming language based on file extension.

    Args:
        ext: The file extension.

    Returns:
        The programming language (e.g., 'python', 'javascript') or 'plaintext'
        if the extension is not recognized.
    """
    return LANGUAGE_MAPPING.get(ext.lower(), 'plaintext')


def is_binary(file_path: str) -> bool:
    """Checks if a file is binary.

    Args:
        file_path: The path to the file.

    Returns:
        True if the file is binary, False otherwise.
    """
    try:
        with open(file_path, 'rb') as file:
            return b'\0' in file.read(1024)
    except Exception as e:
        logger.error(f"Error checking binary file '{file_path}': {e}")
        return True


def load_config(config_path: str, excluded_dirs: Set[str], excluded_files: Set[str], skip_types: Set[str]) -> None:
    """Loads configuration from a JSON file.

    Args:
        config_path: Path to the configuration file.
        excluded_dirs: Set of excluded directories. Updated by this function.
        excluded_files: Set of excluded files. Updated by this function.
        skip_types: Set of skipped file types. Updated by this function.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
            for key, default_set in [
                ('excluded_dirs', excluded_dirs),
                ('excluded_files', excluded_files),
                ('skip_types', skip_types),
            ]:
                items = config.get(key, [])
                if isinstance(items, list):
                    default_set.update(items)
                else:
                    logger.error(f"'{key}' must be a list in '{config_path}'.")
            logger.info(f"Loaded config from '{config_path}'.")

    except FileNotFoundError:
        logger.warning(f"Config file '{config_path}' not found. Using defaults.")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing config file '{config_path}': {e}")
    except Exception as e:
        logger.error(f"Error loading config file '{config_path}': {e}")


def get_all_file_paths(repo_path: str, excluded_dirs: Set[str], excluded_files: Set[str]) -> List[str]:
    """Gets all file paths in a repository, excluding specified directories and files.

    Args:
        repo_path: Path to the repository.
        excluded_dirs: Set of excluded directories.
        excluded_files: Set of excluded files.

    Returns:
        A list of file paths.
    """
    file_paths = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('.')]
        file_paths.extend([os.path.join(root, file) for file in files if file not in excluded_files and not file.startswith('.')])
    logger.info(f"Collected {len(file_paths)} files from '{repo_path}'.")
    return file_paths


def is_valid_python_code(code: str) -> bool:
    """Checks if the given code is valid Python code.

    Args:
        code: The Python code to check.

    Returns:
        True if the code is valid, False otherwise.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def is_valid_extension(ext: str, skip_types: Set[str]) -> bool:
    """Checks if a file extension is valid (not in the skip list).

    Args:
        ext: The file extension.
        skip_types: Set of file extensions to skip.

    Returns:
        True if the extension is valid, False otherwise.
    """
    return ext.lower() not in skip_types


def extract_python_structure(file_content: str) -> Optional[Dict]:
    """Extracts the structure of Python code.

    Args:
        file_content: The Python code.

    Returns:
        A dictionary representing the code structure, or None if parsing fails.
    """
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
                    func_info["returns"]["type"] = get_node_source(
                        node.returns)

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
                            class_obj["decorators"].append(
                                get_node_source(decorator))
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
                        class_info["decorators"].append(
                            get_node_source(decorator))
                    classes.append(class_info)

        return {
            "language": "python",
            "functions": functions,
            "classes": classes
        }
    except Exception as e:
        logger.error(f"Error parsing Python code: {e}")
        return None
def extract_html_structure(file_content: str) -> Optional[Dict]:
    """Extracts the structure of HTML code.

    Args:
        file_content: The HTML code.

    Returns:
        A dictionary representing the HTML structure, or None if parsing fails.
    """
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
        return None


def extract_css_structure(file_content: str) -> Optional[Dict]:
    """Extracts the structure of CSS code.

    Args:
        file_content: The CSS code.

    Returns:
        A dictionary representing the CSS structure, or None if parsing fails.
    """
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
        return None


def generate_documentation_prompt(
    code_structure: Dict,
    project_info: Optional[str] = None,
    style_guidelines: Optional[str] = None
) -> str:
    """Generates a prompt for the OpenAI API to create documentation.

    Args:
        code_structure: The structured representation of the code.
        project_info: Optional. Additional information about the project.
        style_guidelines: Optional. Specific documentation style guidelines.

    Returns:
        The prompt string.
    """

    language = code_structure.get('language', 'code')
    json_structure = json.dumps(code_structure, indent=2)

    prompt_parts = [
        f"You are an expert {language} developer and technical writer.",
    ]

    if project_info:
        prompt_parts.append(
            f"The code belongs to a project that {project_info}.")

    if style_guidelines:
        prompt_parts.append(
            f"Please follow these documentation style guidelines: {style_guidelines}")

    prompt_parts.append(
        f"""
Given the following {language} code structure in JSON format, generate detailed docstrings or comments for each function, method, class, element, or rule. Include descriptions of all parameters, return types, and any relevant details. Preserve and enhance existing documentation where applicable.

Code Structure:
{json_structure}

Please provide the updated docstrings or comments in the same JSON format, with the 'docstring' fields filled in.
"""
    )

    prompt = '\n'.join(prompt_parts)
    return prompt


async def fetch_documentation(session: aiohttp.ClientSession, prompt: str, retry: int = 3) -> Optional[str]:
    """Fetches generated documentation from the OpenAI API.

    Args:
        session: The aiohttp client session.
        prompt: The prompt to send to the API.
        retry: The number of retry attempts.

    Returns:
        The generated documentation, or None if the request fails.
    """

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.2
    }

    for attempt in range(1, retry + 1):
        try:
            async with SEMAPHORE:
                async with session.post(
                    OPENAI_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=120
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'choices' in data and len(data['choices']) > 0 and 'message' in data['choices'][0]:
                            documentation = data['choices'][0]['message']['content'].strip()
                            logger.info(f"Generated documentation:\n{documentation}")
                            return documentation
                        else:
                            logger.error("Unexpected API response structure.")
                            return None
                    elif response.status in {429, 500, 502, 503, 504}:
                        error_text = await response.text()
                        logger.warning(
                            f"API rate limit or server error (status {response.status}). "
                            f"Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds. "
                            f"Response: {error_text}"
                        )
                        await asyncio.sleep(2 ** attempt)
                    else:
                        error_text = await response.text()
                        logger.error(f"API request failed with status {response.status}: {error_text}")
                        return None
        except asyncio.TimeoutError:
            logger.error(
                f"Request timed out during attempt {attempt}/{retry}. "
                f"Retrying in {2 ** attempt} seconds."
            )
            await asyncio.sleep(2 ** attempt)
        except aiohttp.ClientError as e:
            logger.error(
                f"Client error during API request: {e}. "
                f"Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds."
            )
            await asyncio.sleep(2 ** attempt)

    logger.error("Failed to generate documentation after multiple attempts.")
    return None


def insert_python_docstrings(file_content: str, docstrings: Dict) -> str:
    """Inserts docstrings into Python code.

    Args:
        file_content: The original Python code.
        docstrings: The docstrings to insert.

    Returns:
        The modified Python code with inserted docstrings.
    """

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
                    if docstring:
                        if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                            node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
                        else:
                            node.body[0] = ast.Expr(value=ast.Constant(value=docstring))
                else:
                    docstring = func_doc_map.get(node.name)
                    if docstring:
                        if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                            node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
                        else:
                            node.body[0] = ast.Expr(value=ast.Constant(value=docstring))
                return node

            def visit_ClassDef(self, node):
                self.generic_visit(node)
                docstring = class_doc_map.get(node.name)
                if docstring:
                    if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                        node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
                    else:
                        node.body[0] = ast.Expr(value=ast.Constant(value=docstring))
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
        


def extract_js_ts_structure(file_content: str, language: str) -> Optional[Dict]:
    """Extracts the structure of JavaScript/TypeScript code using Esprima.

    Args:
        file_content: The JS/TS code.
        language: 'javascript' or 'typescript'.

    Returns:
        A dictionary representing the code structure, or None if parsing fails.
    """
    try:
        result = subprocess.run(
            ["esprima", "--comment", file_content],
            capture_output=True,
            text=True,
            check=True
        )
        tree = json.loads(result.stdout)

        functions = []
        classes = []

        def traverse(node, parent_class=None):
            if node["type"] == "FunctionDeclaration":
                func_info = {
                    "name": node["id"]["name"] if node.get("id") else "anonymous",
                    "params": [p.get("name", "param") for p in node["params"]],
                    "loc": node.get("loc"),  # Get location information (lines and columns)
                    "docstring": extract_docstring_from_comments(node)
                }
                if parent_class:
                    parent_class["methods"].append(func_info)
                else:
                    functions.append(func_info)
            elif node["type"] == "ClassDeclaration":
                class_info = {
                    "name": node["id"]["name"],
                    "methods": [],
                    "loc": node.get("loc"),  # Location for classes
                    "docstring": extract_docstring_from_comments(node)
                }
                classes.append(class_info)
                for method_node in node.get("body", {}).get("body", []):
                    traverse(method_node, parent_class=class_info)
            elif node["type"] == "MethodDefinition":  # MethodDefinition handling
                func_info = {
                    "name": node["key"]["name"],  # Extract method name
                    "params": [p.get("name", "param") for p in node.get("value", {}).get("params", [])],  # Extract params
                    "loc": node.get("loc"),  # Location for methods
                    "docstring": extract_docstring_from_comments(node),
                    # ... other method information as needed ...
                }
                if parent_class:  # Methods should always have a parent class
                    parent_class["methods"].append(func_info)
                else:  # This should ideally not happen, but handle it just in case
                    logger.warning("MethodDefinition found outside of a class.")
                    functions.append(func_info)  # Or handle differently


            # ... Add handling for other node types as needed (ArrowFunctionExpression, etc.)

        def extract_docstring_from_comments(node):
            docstring = ""
            if "leadingComments" in node:
                for comment in node["leadingComments"]:
                    if comment["type"] == "Block" and comment["value"].startswith("*"):
                        docstring = comment["value"].lstrip("*").strip()
                        break
            return docstring

        traverse(tree)

        return {
            "language": language,
            "functions": functions,
            "classes": classes,
            "tree": tree,  # Include the Esprima AST
            "source_code": file_content
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"Esprima parsing failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during JS/TS structure extraction: {e}")
        return None
        

def insert_js_ts_docstrings(docstrings: Dict) -> str:
    """Inserts JSDoc comments into JavaScript/TypeScript code using Esprima AST."""
    try:
        source_code = docstrings['source_code']
        tree = docstrings['tree']
        # Use lines and columns for insertion with Esprima
        inserts = []
        for item_type in ['functions', 'classes']:
            for item in docstrings.get(item_type, []):
                docstring = item['docstring']
                if docstring and item.get('loc'): # Check if location info is available
                    formatted_comment = format_jsdoc_comment(docstring)
                    inserts.append((item['loc']['start']['line'], item['loc']['start']['column'], formatted_comment))

                    if item_type == 'classes':
                        for method in item.get('methods', []):
                            docstring = method['docstring']
                            if docstring and method.get('loc'):
                                formatted_comment = format_jsdoc_comment(docstring)
                                inserts.append((method['loc']['start']['line'], method['loc']['start']['column'], formatted_comment))

        # Sort inserts by line number, then column number (descending)
        inserts.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        lines = source_code.splitlines()

        for line_num, col_num, comment in inserts:
            lines.insert(line_num - 1,  # Adjust for 0-based indexing
                         " " * col_num + comment) # Insert at correct column

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error inserting docstrings into JS/TS code: {e}")
        return docstrings.get('source_code', '')

def format_jsdoc_comment(docstring: str) -> str:
    """Formats a docstring into a JSDoc comment block.

    Args:
        docstring: The docstring to format.

    Returns:
        The formatted JSDoc comment.
    """

    comment_lines = ['/**']
    for line in docstring.strip().split('\n'):
        comment_lines.append(f' * {line}')
    comment_lines.append(' */')
    return '\n'.join(comment_lines)


def insert_html_comments(file_content: str, docstrings: Dict) -> str:
    """Inserts comments into HTML code.

    Args:
        file_content: The original HTML code.
        docstrings: The docstrings to insert.

    Returns:
        The modified HTML code with inserted comments.
    """
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
                        if child.name in ['head', 'body', 'html']:
                            child.insert(0, comment)
                        else:
                            child.insert_before(comment)
                    traverse_and_insert(child)

        traverse_and_insert(soup)
        return str(soup)

    except Exception as e:
        logger.error(f"Error inserting comments into HTML code: {e}")
        return file_content


def insert_css_comments(file_content: str, docstrings: Dict) -> str:
    """Inserts comments into CSS code.

    Args:
        file_content: The original CSS code.
        docstrings: The docstrings to insert.

    Returns:
        The modified CSS code with inserted comments.
    """

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

async def process_file(session: aiohttp.ClientSession, file_path: str, skip_types: Set[str], output_file: str) -> None:
    """Processes a single file to generate and insert documentation.

    Args:
        session: The aiohttp client session.
        file_path: The path to the file.
        skip_types: Set of file extensions to skip.
        output_file: The output Markdown file.
    """

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

        documentation = await fetch_documentation(session, prompt)
        if not documentation:
            logger.error(f"Failed to generate documentation for '{file_path}'")
            return

        try:
            updated_code_structure = json.loads(documentation)
            if language in ['javascript', 'typescript']:
                updated_code_structure['tree'] = code_structure['tree']
                updated_code_structure['source_code'] = code_structure['source_code']
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse documentation JSON for '{file_path}': {e}")
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
            async with OUTPUT_LOCK:
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


async def process_all_files(file_paths: List[str], skip_types: Set[str], output_file: str) -> None:
    """Processes all files asynchronously to generate and insert documentation.

    Args:
        file_paths: A list of file paths to process.
        skip_types: Set of file extensions to skip.
        output_file: The output Markdown file.
    """
    global OUTPUT_LOCK
    OUTPUT_LOCK = asyncio.Lock()

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(process_file(session, file_path, skip_types, output_file))
            for file_path in file_paths
        ]

        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Files"):
            try:
                await f
            except Exception as e:
                logger.error(f"Error processing a file: {e}")


def main() -> None:
    """Main function to orchestrate documentation generation."""

    parser = argparse.ArgumentParser(
        description="Generate and insert comments/docstrings using OpenAI's GPT-4 API."
    )
    parser.add_argument(
        "repo_path",
        help="Path to the code repository"
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to config.json",
        default="config.json"
    )
    parser.add_argument(
        "--concurrency",
        help="Number of concurrent requests",
        type=int,
        default=5
    )
    parser.add_argument(
        "-o", "--output",
        help="Output Markdown file",
        default="output.md"
    )
    parser.add_argument(
        "--model",
        help="OpenAI model to use (default: gpt-4)",
        default="gpt-4"
    )
    parser.add_argument(
        "--skip-types",
        help="Comma-separated list of file extensions to skip",
        default=""
    )

    args = parser.parse_args()

    repo_path = args.repo_path
    config_path = args.config
    concurrency = args.concurrency
    output_file = args.output

    if not os.path.isdir(repo_path):
        logger.error(f"Invalid repository path: '{repo_path}' is not a directory.")
        sys.exit(1)

    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    excluded_files = set(DEFAULT_EXCLUDED_FILES)
    skip_types = set(DEFAULT_SKIP_TYPES)
    if args.skip_types:
        skip_types.update(args.skip_types.split(','))

    load_config(config_path, excluded_dirs, excluded_files, skip_types)

    file_paths = get_all_file_paths(repo_path, excluded_dirs, excluded_files)
    if not file_paths:
        logger.error("No files found to process.")
        sys.exit(1)

    logger.info(f"Starting documentation generation for {len(file_paths)} files.")

    open(output_file, 'w').close()  # Clear the output file

    global SEMAPHORE
    SEMAPHORE = asyncio.Semaphore(concurrency)

    global MODEL_NAME
    MODEL_NAME = args.model

    try:
        asyncio.run(process_all_files(file_paths, skip_types, output_file))
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)

    logger.info("Documentation generation completed successfully.")


if __name__ == "__main__":
    main()
