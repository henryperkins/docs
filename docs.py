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
from tree_sitter import Language, Parser

# Module Configuration
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
    logger.error("OPENAI_API_KEY is not set. Ensure it's present in the environment or in a .env file.")
    sys.exit(1)

try:
    if not os.path.exists('build/my-languages.so'):
        Language.build_library(
            'build/my-languages.so',
            [
                'vendor/tree-sitter-javascript',
                'vendor/tree-sitter-typescript/typescript',
            ]
        )
except Exception as e:
    logger.error(f"Failed to build Tree-sitter language library: {e}")
    sys.exit(1)

JAVASCRIPT_LANGUAGE = Language('build/my-languages.so', 'javascript')
TYPESCRIPT_LANGUAGE = Language('build/my-languages.so', 'typescript')

SEMAPHORE = None
OUTPUT_LOCK = None
MODEL_NAME = 'gpt-4' 

DEFAULT_EXCLUDED_DIRS = {'.git', '__pycache__', 'node_modules'}
DEFAULT_EXCLUDED_FILES = {'.DS_Store'}
DEFAULT_SKIP_TYPES = {'.json', '.md', '.txt', '.csv'} 

def get_language(ext: str) -> str:
    language_mapping = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.html': 'html',
        '.htm': 'html',
        '.css': 'css',
    }
    language = language_mapping.get(ext.lower(), 'plaintext')
    if language == 'plaintext':
        logger.debug(f"Unrecognized file extension '{ext}'. Skipping.")
    return language


def is_binary(file_path: str) -> bool:
    try:
        with open(file_path, 'rb') as file:
            chunk = file.read(1024)
            return b'\0' in chunk
    except Exception as e:
        logger.error(f"Error checking if file is binary '{file_path}': {e}")
        return True 

def load_config(config_path: str, excluded_dirs: Set[str], excluded_files: Set[str], skip_types: Set[str]) -> None:
    try:
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
            additional_dirs = config.get('excluded_dirs', [])
            additional_files = config.get('excluded_files', [])
            additional_skip_types = config.get('skip_types', [])
            
            if isinstance(additional_dirs, list):
                excluded_dirs.update(additional_dirs)
            else:
                logger.error(f"'excluded_dirs' should be a list in '{config_path}'.")
            
            if isinstance(additional_files, list):
                excluded_files.update(additional_files)
            else:
                logger.error(f"'excluded_files' should be a list in '{config_path}'.")
            
            if isinstance(additional_skip_types, list):
                skip_types.update(additional_skip_types)
            else:
                logger.error(f"'skip_types' should be a list in '{config_path}'.")
            
            logger.info(f"Loaded config from {config_path}.")
    except FileNotFoundError:
        logger.warning(f"Configuration file '{config_path}' not found. Using default exclusions.")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing configuration file '{config_path}': {e}")
    except Exception as e:
        logger.error(f"Error loading configuration file '{config_path}': {e}")


def get_all_file_paths(repo_path: str, excluded_dirs: Set[str], excluded_files: Set[str]) -> List[str]:
    file_paths = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [
            d for d in dirs if d not in excluded_dirs and not d.startswith('.')]
        for file in files:
            if file in excluded_files or file.startswith('.'):
                continue
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    logger.info(f"Collected {len(file_paths)} files from '{repo_path}'.")
    return file_paths

def is_valid_python_code(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in Python code: {e}")
        return False

def is_valid_extension(ext: str, skip_types: Set[str]) -> bool:
    return ext.lower() not in skip_types


def extract_python_structure(file_content: str) -> Optional[Dict]:
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

def extract_js_ts_structure(file_content: str, language: str) -> Optional[Dict]:
    try:
        parser = Parser()
        if language == 'javascript':
            parser.set_language(JAVASCRIPT_LANGUAGE)
        elif language == 'typescript':
            parser.set_language(TYPESCRIPT_LANGUAGE)
        else:
            logger.error(f"Unsupported language: {language}")
            return None

        tree = parser.parse(bytes(file_content, "utf8"))
        root_node = tree.root_node

        functions = []
        classes = []

        code_bytes = file_content.encode('utf8')

        def extract_function_info(node, existing_docstring):
            func_name_node = node.child_by_field_name('name') or node.child_by_field_name('property')
            func_name = func_name_node.text.decode('utf8') if func_name_node else 'anonymousFunction'
            params = []
            params_node = node.child_by_field_name('parameters')
            if params_node:
                for param in params_node.named_children:
                    param_info = extract_parameter_info(param)
                    params.append(param_info)
            return_type = "Any"
            return_type_node = None
            for child in node.children:
                if child.type == 'type_annotation':
                    return_type_node = child
                    break
            if return_type_node:
                return_type = code_bytes[return_type_node.start_byte:return_type_node.end_byte].decode('utf8').lstrip(':').strip()
            func_info = {
                "name": func_name,
                "args": params,
                "returns": {"type": return_type},
                "docstring": "",
                "existing_docstring": existing_docstring,
                "start_byte": node.start_byte,
                "end_byte": node.end_byte
            }
            return func_info

        def extract_parameter_info(param_node):
            param_name = code_bytes[param_node.start_byte:param_node.end_byte].decode('utf8')
            param_type = "Any"
            type_annotation_node = None
            for child in param_node.children:
                if child.type == 'type_annotation':
                    type_annotation_node = child
                    break
            if type_annotation_node:
                param_type = code_bytes[type_annotation_node.start_byte:type_annotation_node.end_byte].decode('utf8').lstrip(':').strip()
            return {"name": param_name, "type": param_type}

        def traverse(node, parent_class=None):
            existing_docstring = extract_preceding_jsdoc(node)

            if node.type in ['function_declaration', 'method_definition']:
                func_info = extract_function_info(node, existing_docstring)
                if parent_class:
                    parent_class['methods'].append(func_info)
                else:
                    functions.append(func_info)
            elif node.type == 'class_declaration':
                class_name_node = node.child_by_field_name('name')
                class_name = class_name_node.text.decode('utf8') if class_name_node else 'AnonymousClass'
                class_info = {
                    "name": class_name,
                    "methods": [],
                    "docstring": "",
                    "existing_docstring": existing_docstring,
                    "start_byte": node.start_byte,
                    "end_byte": node.end_byte
                }
                classes.append(class_info)
                body = node.child_by_field_name('body')
                if body:
                    for child in body.named_children:
                        traverse(child, parent_class=class_info)
            else:
                for child in node.named_children:
                    traverse(child, parent_class=parent_class)

        def extract_preceding_jsdoc(node):
            preceding_comments = []
            cursor = node.walk()
            while cursor.goto_prev_sibling():
                sibling = cursor.node
                if sibling.type == 'comment':
                    comment_text = code_bytes[sibling.start_byte:sibling.end_byte].decode('utf8')
                    if comment_text.strip().startswith('/**'):
                        preceding_comments.insert(0, comment_text.strip())
                    else:
                        break
                elif sibling.is_named:
                    break
                else:
                    continue
            existing_docstring = '\n'.join(preceding_comments)
            return existing_docstring

        traverse(root_node)

        return {
            "language": language,
            "functions": functions,
            "classes": classes,
            "tree": tree,
            "source_code": file_content
        }

    except Exception as e:
        logger.error(f"Error parsing {language} code: {e}")
        return None

def extract_html_structure(file_content: str) -> Optional[Dict]:
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
    """
    Generates a prompt for the OpenAI API to create documentation based on the code structure.

    Args:
        code_structure (Dict): The structured representation of the code.
        project_info (Optional[str]): Additional information about the project.
        style_guidelines (Optional[str]): Specific documentation style guidelines to follow.

    Returns:
        str: The prompt to send to the OpenAI API.
    """
    language = code_structure.get('language', 'code')
    json_structure = json.dumps(code_structure, indent=2)

    prompt_parts = [
        f"You are an expert {language} developer and technical writer.",
    ]

    if project_info:
        prompt_parts.append(f"The code belongs to a project that {project_info}.")

    if style_guidelines:
        prompt_parts.append(f"Please follow these documentation style guidelines: {style_guidelines}")

    # Include existing docstrings/comments to allow the AI to enhance them
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
    """
    Fetches generated documentation from the OpenAI API based on the prompt.

    Args:
        session (aiohttp.ClientSession): The aiohttp session for making requests.
        prompt (str): The prompt to send to the API.
        retry (int): Number of retry attempts.

    Returns:
        Optional[str]: Generated documentation in JSON format if successful, None otherwise.
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
                        # Enhanced response validation
                        if 'choices' in data and len(data['choices']) > 0 and 'message' in data['choices'][0]:
                            documentation = data['choices'][0]['message']['content'].strip()
                            logger.info(f"Generated documentation:\n{documentation}")
                            return documentation
                        else:
                            logger.error("Unexpected API response structure.")
                            return None
                    elif response.status in {429, 500, 502, 503, 504}:
                        # Handle rate limiting and server errors with exponential backoff
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
    """
    Inserts docstrings into Python code using the AST.

    Args:
        file_content (str): Original Python source code.
        docstrings (Dict): Generated docstrings to insert.

    Returns:
        str: Modified Python source code with docstrings inserted.
    """
    try:
        tree = ast.parse(file_content)
        parent_map = {}

        # Implement parent tracking
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent_map[child] = node

        # Map function and class names to docstrings
        func_doc_map = {func['name']: func['docstring'] for func in docstrings.get('functions', [])}
        class_doc_map = {cls['name']: cls['docstring'] for cls in docstrings.get('classes', [])}
        method_doc_map = {}
        for cls in docstrings.get('classes', []):
            for method in cls.get('methods', []):
                method_doc_map[(cls['name'], method['name'])] = method['docstring']

        class DocstringInserter(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                parent = parent_map.get(node)
                if isinstance(parent, ast.ClassDef):
                    # It's a method
                    key = (parent.name, node.name)
                    docstring = method_doc_map.get(key)
                    if docstring:
                        if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                            # Insert docstring if not present
                            node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
                        else:
                            # Replace existing docstring
                            node.body[0] = ast.Expr(value=ast.Constant(value=docstring))
                else:
                    # It's a top-level function
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

        # Validate the modified code
        try:
            ast.parse(new_code)
        except SyntaxError as e:
            logger.error(f"Syntax error in modified Python code: {e}")
            return file_content  # Return original content if there's an error

        return new_code
    except Exception as e:
        logger.error(f"Error inserting docstrings into Python code: {e}")
        return file_content  # Return original content if there's an error


def insert_js_ts_docstrings(docstrings: Dict) -> str:
    """
    Inserts JSDoc comments into JavaScript/TypeScript code using tree-sitter.

    Args:
        docstrings (Dict): Generated docstrings to insert, including the AST and source code.

    Returns:
        str: Modified source code with docstrings inserted.
    """
    try:
        source_code = docstrings['source_code']
        tree = docstrings['tree']
        code_bytes = source_code.encode('utf8')

        inserts = []

        for func in docstrings.get('functions', []):
            start_byte = func['start_byte']
            docstring = func['docstring']
            if docstring:
                formatted_comment = format_jsdoc_comment(docstring)
                inserts.append((start_byte, formatted_comment))

        for cls in docstrings.get('classes', []):
            start_byte = cls['start_byte']
            docstring = cls['docstring']
            if docstring:
                formatted_comment = format_jsdoc_comment(docstring)
                inserts.append((start_byte, formatted_comment))
            for method in cls.get('methods', []):
                start_byte = method['start_byte']
                docstring = method['docstring']
                if docstring:
                    formatted_comment = format_jsdoc_comment(docstring)
                    inserts.append((start_byte, formatted_comment))

        # Sort inserts by position in descending order to avoid offset issues during insertion
        inserts.sort(key=lambda x: x[0], reverse=True)

        new_code = source_code

        for start_byte, comment in inserts:
            # Convert byte position to character position
            char_pos = start_byte  # Assuming UTF-8 encoding and no multibyte characters for simplicity
            # Insert the comment before the function or class declaration
            new_code = new_code[:char_pos] + comment + '\n' + new_code[char_pos:]

        return new_code

    except Exception as e:
        logger.error(f"Error inserting docstrings into JS/TS code: {e}")
        return docstrings.get('source_code', '')  # Return original content if there's an error


def format_jsdoc_comment(docstring: str) -> str:
    """
    Formats a docstring into a JSDoc comment block.

    Args:
        docstring (str): The docstring to format.

    Returns:
        str: The formatted JSDoc comment.
    """
    comment_lines = ['/**']
    for line in docstring.strip().split('\n'):
        comment_lines.append(f' * {line}')
    comment_lines.append(' */')
    return '\n'.join(comment_lines)


def insert_html_comments(file_content: str, docstrings: Dict) -> str:
    """
    Inserts comments into HTML code using BeautifulSoup.

    Args:
        file_content (str): Original HTML source code.
        docstrings (Dict): Generated docstrings to insert.

    Returns:
        str: Modified HTML source code with comments inserted.
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
                            # Insert comment inside the element
                            child.insert(0, comment)
                        else:
                            # Insert comment before the element
                            child.insert_before(comment)
                    traverse_and_insert(child)

        traverse_and_insert(soup)
        return str(soup)

    except Exception as e:
        logger.error(f"Error inserting comments into HTML code: {e}")
        return file_content


def insert_css_comments(file_content: str, docstrings: Dict) -> str:
    """
    Inserts comments into CSS code.

    Args:
        file_content (str): Original CSS source code.
        docstrings (Dict): Generated docstrings to insert.

    Returns:
        str: Modified CSS source code with comments inserted.
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
    """
    Processes a single file: reads content, generates documentation, inserts it, and writes outputs.

    Args:
        session (aiohttp.ClientSession): HTTP session.
        file_path (str): Path to the source file.
        skip_types (Set[str]): Set of file extensions to skip.
        output_file (str): Output Markdown file.
    """
    try:
        _, ext = os.path.splitext(file_path)
        logger.debug(f"Processing file: {file_path} with extension: {ext}")

        # Validate file extension
        if not is_valid_extension(ext, skip_types):
            logger.info(f"Skipping file '{file_path}' due to extension '{ext}'")
            return

        # Check if the file is binary
        if is_binary(file_path):
            logger.info(f"Skipping binary file '{file_path}'")
            return

        language = get_language(ext)

        if language == 'plaintext':
            logger.info(f"Skipping file '{file_path}' with unrecognized language.")
            return

        # Read file content directly
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
        except Exception as e:
            logger.error(f"Failed to read '{file_path}': {e}")
            return

        # Extract code structure
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

        if code_structure is None:
            logger.error(f"Failed to extract structure from '{file_path}'")
            return

        # Generate prompt with existing docstrings/comments considered
        prompt = generate_documentation_prompt(code_structure)

        # Generate documentation using OpenAI API
        documentation = await fetch_documentation(session, prompt)
        if not documentation:
            logger.error(f"Failed to generate documentation for '{file_path}'")
            return

        # Parse the generated documentation JSON
        try:
            updated_code_structure = json.loads(documentation)
            if language in ['javascript', 'typescript']:
                updated_code_structure['tree'] = code_structure['tree']
                updated_code_structure['source_code'] = code_structure['source_code']
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse documentation JSON for '{file_path}': {e}")
            return

        # Insert documentation into code
        if language == 'python':
            new_content = insert_python_docstrings(content, updated_code_structure)
        elif language in ['javascript', 'typescript']:
            new_content = insert_js_ts_docstrings(updated_code_structure)
        elif language == 'html':
            new_content = insert_html_comments(content, updated_code_structure)
        elif language == 'css':
            new_content = insert_css_comments(content, updated_code_structure)
        else:
            new_content = content  # Fallback to original content

        # Validate modified code if possible
        if language == 'python':
            if not is_valid_python_code(new_content):
                logger.error(f"Modified Python code is invalid. Aborting insertion for '{file_path}'")
                return

        # Backup and write the modified content back to the file
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
            # Restore from backup
            if os.path.exists(backup_path):
                os.remove(file_path)
                os.rename(backup_path, file_path)
                logger.info(f"Restored original file from backup for '{file_path}'")
            return

        # Append the documented code to the output Markdown file
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
        return


async def process_all_files(file_paths: List[str], skip_types: Set[str], output_file: str) -> None:
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
    parser = argparse.ArgumentParser(
        description="Automatically generate and insert comments/docstrings into source files using OpenAI's GPT-4 API."
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

    open(output_file, 'w').close()

    global SEMAPHORE
    SEMAPHORE = asyncio.Semaphore(concurrency)

    global MODEL_NAME
    MODEL_NAME = args.model  # Get the model name from arguments

    try:
        asyncio.run(process_all_files(file_paths, skip_types, output_file))
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)

    logger.info("Documentation generation completed successfully.")


if __name__ == "__main__":
    main()
