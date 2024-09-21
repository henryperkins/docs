import argparse
import ast
import json
import logging
import os
import re
import sys
from typing import List, Set, Optional, Dict
import time
import asyncio
import aiohttp
import javalang
from dotenv import load_dotenv
from tqdm.asyncio import tqdm  # For asynchronous progress bar

# Load environment variables from .env file
load_dotenv()

# Configure logging to write to a file with rotation
from logging.handlers import RotatingFileHandler

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = RotatingFileHandler('docs_generation.log', maxBytes=10**6, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Retrieve the OpenAI API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    logger.error("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
    sys.exit(1)

# OpenAI API endpoint
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Mapping of file extensions to programming languages
LANGUAGE_MAPPING = {
    '.py': 'python',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.java': 'java',
    '.cpp': 'cpp',
    '.hpp': 'cpp',
    '.h': 'cpp',
    '.c': 'c',
    '.cs': 'csharp',
    '.rb': 'ruby',
    '.go': 'go',
    '.php': 'php',
    '.html': 'html',
    '.css': 'css',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.json': 'json',
    '.md': '',  # No language specified for Markdown
    # Add more mappings as needed
}

# Default directories to exclude from traversal
DEFAULT_EXCLUDED_DIRS = {
    'node_modules',
    '.git',
    '__pycache__',
    'venv',
    'dist',
    'build',
    '.venv',
    '.idea',
    '.vscode',
    '.turbo',
    '.next',
}

# Default files to exclude from processing
DEFAULT_EXCLUDED_FILES = {
    '.DS_Store',
    'Thumbs.db',
}

# Global semaphore and lock (to be initialized in main based on concurrency)
SEMAPHORE = None
OUTPUT_LOCK = None

def is_binary(file_path: str) -> bool:
    """
    Checks if a file is binary.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file is binary, False otherwise.
    """
    try:
        with open(file_path, 'rb') as file:
            chunk = file.read(1024)
            if b'\0' in chunk:
                return True
        return False
    except Exception as e:
        logging.error(f"Error checking if file is binary '{file_path}': {e}")
        return False

async def fetch_openai(session: aiohttp.ClientSession, prompt: str, language: str, semaphore: asyncio.Semaphore, retry: int = 3) -> Optional[str]:
    """
    Asynchronously fetches the generated Google-style docstrings in JSON format from OpenAI API.
    Implements retry mechanism for handling rate limits and connection errors.
    """
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",  # Ensure this is the correct model name
        "messages": [
            {"role": "system", "content": "You are an expert software developer specializing in writing detailed Google-style docstrings and comments."},
            {"role": "user", "content": f"Generate a detailed Google-style docstring for the following {language} code. Output the docstring in the following JSON format:\n\n{{\n  \"docstring\": {{\n    \"summary\": \"\",\n    \"extended_summary\": \"\",\n    \"args\": [{{\n      \"name\": \"\",\n      \"type\": \"\",\n      \"description\": \"\"\n    }}],\n    \"returns\": {{\n      \"type\": \"\",\n      \"description\": \"\"\n    }},\n    \"raises\": [{{\n      \"exception\": \"\",\n      \"description\": \"\"\n    }}],\n    \"examples\": [{{\n      \"code\": \"\",\n      \"explanation\": \"\"\n    }}],\n    \"notes\": \"\",\n    \"references\": [{{\n      \"text\": \"\",\n      \"link\": \"\"\n    }}]\n  }}\n}}"}
        ],
        "max_tokens": 1500,
        "temperature": 0.2
    }

    for attempt in range(1, retry + 1):
        try:
            async with semaphore:
                async with session.post(OPENAI_API_URL, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        documentation = data['choices'][0]['message']['content'].strip()
                        try:
                            doc_json = json.loads(documentation)
                            logger.info("Successfully generated GPT-4 Google-style JSON comments.")
                            return json.dumps(doc_json, indent=2)  # Ensure it's a JSON string
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decoding error: {e}")
                            return None
                    elif response.status in {429, 500, 502, 503, 504}:
                        logger.warning(f"API rate limit or server error (status {response.status}). Attempt {attempt}/{retry}.")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        error_text = await response.text()
                        logger.error(f"API request failed with status {response.status}: {error_text}")
                        return None
        except aiohttp.ClientError as e:
            logger.error(f"API connection error: {e}. Attempt {attempt}/{retry}.")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    logger.error("Failed to generate comments after multiple attempts.")
    return None

async def fetch_refined_documentation(session: aiohttp.ClientSession, doc_json: Dict, language: str, retry: int = 3) -> Optional[Dict]:
    """
    Uses GPT-4 to refine previously generated Google-style docstrings and comments in JSON format.

    Args:
        session (aiohttp.ClientSession): The aiohttp client session for the API request.
        doc_json (Dict): The initial docstring JSON generated by fetch_openai.
        language (str): The programming language of the code snippet.
        retry (int, optional): Retry attempts for rate limits/errors. Defaults to 3.

    Returns:
        Optional[Dict]: The refined documentation JSON if successful, otherwise None.
    """
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    feedback_prompt = f"""
    Refine the following Google-style docstring JSON for better clarity, correctness, and adherence to best practices. Ensure all sections are properly formatted according to Google's Python Style Guide.

    Original Docstring JSON:
    {json.dumps(doc_json, indent=2)}

    Provide the refined version of the docstring JSON.
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert software developer."},
            {"role": "user", "content": feedback_prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.2
    }

    for attempt in range(1, retry + 1):
        try:
            async with SEMAPHORE:
                async with session.post(OPENAI_API_URL, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        refined = data['choices'][0]['message']['content'].strip()
                        try:
                            refined_json = json.loads(refined)
                            logger.info("Successfully refined Google-style JSON comments.")
                            return refined_json
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decoding error during refinement: {e}")
                            return None
                    elif response.status in {429, 500, 502, 503, 504}:
                        logger.warning(f"API rate limit or server error during refinement (status {response.status}). Attempt {attempt}/{retry}.")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        error_text = await response.text()
                        logger.error(f"API request failed during refinement with status {response.status}: {error_text}")
                        return None
        except aiohttp.ClientError as e:
            logger.error(f"API connection error during refinement: {e}. Attempt {attempt}/{retry}.")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    logger.error("Failed to refine comments after multiple attempts.")
    return None

def generate_google_docstring_from_json(
    summary: str,
    extended_summary: str,
    args: List[dict],
    returns: dict,
    raises: List[dict],
    examples: List[dict],
    notes: str,
    references: List[dict]
) -> str:
    """
    Constructs a Google-style docstring from JSON data.

    Args:
        summary (str): The summary of the function or class.
        extended_summary (str): The extended description.
        args (List[dict]): List of arguments with their details.
        returns (dict): Return type and description.
        raises (List[dict]): Exceptions raised with descriptions.
        examples (List[dict]): Examples and explanations.
        notes (str): Additional notes.
        references (List[dict]): References with text and links.

    Returns:
        str: Formatted Google-style docstring.
    """
    docstring = f"{summary}\n\n"
    if extended_summary:
        docstring += f"{extended_summary}\n\n"
    if args:
        docstring += "Args:\n"
        for arg in args:
            docstring += f"    {arg['name']} ({arg['type']}): {arg['description']}\n"
        docstring += "\n"
    if returns:
        docstring += "Returns:\n"
        docstring += f"    {returns['type']}: {returns['description']}\n\n"
    if raises:
        docstring += "Raises:\n"
        for raise_info in raises:
            docstring += f"    {raise_info['exception']}: {raise_info['description']}\n"
        docstring += "\n"
    if examples:
        docstring += "Examples:\n"
        for example in examples:
            docstring += f"    {example['code']}\n    {example['explanation']}\n"
        docstring += "\n"
    if notes:
        docstring += f"Notes:\n    {notes}\n\n"
    if references:
        docstring += "References:\n"
        for ref in references:
            docstring += f"    {ref['text']}: {ref['link']}\n"
        docstring += "\n"
    return docstring.strip()


async def fetch_refined_documentation(session: aiohttp.ClientSession, doc_json: Dict, language: str, retry: int = 3) -> Optional[Dict]:
    """
    Uses GPT-4 to refine previously generated Google-style docstrings and comments in JSON format.

    Args:
        session (aiohttp.ClientSession): The aiohttp client session for the API request.
        doc_json (Dict): The initial docstring JSON generated by fetch_openai.
        language (str): The programming language of the code snippet.
        retry (int, optional): Retry attempts for rate limits/errors. Defaults to 3.

    Returns:
        Optional[Dict]: The refined documentation JSON if successful, otherwise None.
    """
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    feedback_prompt = f"""
    Refine the following Google-style docstring JSON for better clarity, correctness, and adherence to best practices. Ensure all sections are properly formatted according to Google's Python Style Guide.

    Original Docstring JSON:
    {json.dumps(doc_json, indent=2)}

    Provide the refined version of the docstring JSON.
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert software developer."},
            {"role": "user", "content": feedback_prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.2
    }

    for attempt in range(1, retry + 1):
        try:
            async with SEMAPHORE:
                async with session.post(OPENAI_API_URL, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        refined = data['choices'][0]['message']['content'].strip()
                        try:
                            refined_json = json.loads(refined)
                            logger.info("Successfully refined Google-style JSON comments.")
                            return refined_json
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decoding error during refinement: {e}")
                            return None
                    elif response.status in {429, 500, 502, 503, 504}:
                        logger.warning(f"API rate limit or server error during refinement (status {response.status}). Attempt {attempt}/{retry}.")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        error_text = await response.text()
                        logger.error(f"API request failed during refinement with status {response.status}: {error_text}")
                        return None
        except aiohttp.ClientError as e:
            logger.error(f"API connection error during refinement: {e}. Attempt {attempt}/{retry}.")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    logger.error("Failed to refine comments after multiple attempts.")
    return None

def get_language(file_extension: str) -> str:
    """
    Returns the corresponding programming language for the given file extension.
    """
    return LANGUAGE_MAPPING.get(file_extension, '')

def load_config(config_path: str, excluded_dirs: Set[str], excluded_files: Set[str]) -> None:
    """
    Loads exclusion configurations from a JSON file.
    Updates excluded_dirs and excluded_files based on the config.
    """
    if not os.path.exists(config_path):
        logger.warning(
            f"Configuration file '{config_path}' not found. Using default exclusions."
        )
        return
    try:
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
        excluded_dirs.update(config.get("excluded_dirs", []))
        excluded_files.update(config.get("excluded_files", []))
        logger.info(f"Loaded exclusions from '{config_path}'")
    except json.JSONDecodeError as json_err:
        logger.error(
            f"JSON decode error in configuration file '{config_path}': {json_err}"
        )
    except Exception as e:
        logger.error(f"Error loading configuration file '{config_path}': {e}")

async def extract_description(file_path: str, file_extension: str) -> Optional[Dict]:
    """
    Extracts a brief description of the file based on its content.
    Supports multiple programming languages.

    Args:
        file_path (str): Path to the source file.
        file_extension (str): File extension to determine the programming language.

    Returns:
        Optional[Dict]: A dictionary containing extracted components if successful, otherwise None.
    """
    language = get_language(file_extension)
    if language == 'python':
        return extract_python_description(file_path)
    elif language in ['javascript', 'typescript']:
        return await extract_javascript_description(file_path)
    elif language == 'java':
        return extract_java_description(file_path)
    elif language in ['cpp', 'c', 'hpp', 'h']:
        return extract_cpp_description(file_path)
    elif language == 'go':
        return extract_go_description(file_path)
    elif file_extension == '.md':
        try:
            with open(file_path, 'r', encoding='utf-8') as md_file:
                content = md_file.read()
            return {"summary": "Markdown Content", "extended_summary": content}
        except (FileNotFoundError, IOError) as e:
            logger.error(f"Could not read Markdown file '{file_path}': {e}")
            return None
    else:
        return extract_generic_description(file_path, file_extension)

def extract_python_description(file_path: str) -> Optional[Dict]:
    """
    Parses a Python file to extract class and function declarations along with docstrings.
    Returns a dictionary containing extracted components.

    Args:
        file_path (str): Path to the Python file.

    Returns:
        Optional[Dict]: Dictionary with classes and functions information if successful, otherwise None.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as python_file:
            content = python_file.read()
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Could not read Python file '{file_path}': {e}")
        return None

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        logger.error(f"Syntax error parsing Python file '{file_path}': {e}")
        return None

    classes = []
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or 'No description.'
            methods = []
            for body_item in node.body:
                if isinstance(body_item, ast.FunctionDef):
                    method_doc = ast.get_docstring(body_item) or 'No description.'
                    params = [arg.arg for arg in body_item.args.args if arg.arg != 'self']
                    methods.append({
                        "name": body_item.name,
                        "params": params,
                        "description": method_doc
                    })
            classes.append({
                "name": node.name,
                "description": doc,
                "methods": methods
            })
        elif isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node) or 'No description.'
            params = [arg.arg for arg in node.args.args]
            if 'self' in params:
                params.remove('self')
            functions.append({
                "name": node.name,
                "params": params,
                "description": doc
            })

    if not classes and not functions:
        return None

    return {
        "classes": classes,
        "functions": functions
    }

async def extract_javascript_description(file_path: str) -> Optional[Dict]:
    """
    Parses a JavaScript/TypeScript file to extract class and function declarations along with comments.
    Returns a dictionary containing extracted components.

    Args:
        file_path (str): Path to the JavaScript/TypeScript file.

    Returns:
        Optional[Dict]: Dictionary with classes and functions information if successful, otherwise None.
    """
    parser_script_path = os.path.join(os.path.dirname(__file__), 'js-ts-parser', 'parse.js')

    if not os.path.exists(parser_script_path):
        logger.error(f"Parser script not found at '{parser_script_path}'.")
        return None

    logger.info(f"Starting subprocess for JavaScript/TypeScript file: {file_path}")

    try:
        # Start the subprocess
        process = await asyncio.create_subprocess_exec(
            'node', parser_script_path, file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60.0)  # 60 seconds timeout
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.error(f"Parsing JavaScript/TypeScript file '{file_path}' timed out.")
            return None

        if process.returncode != 0:
            try:
                error_json = json.loads(stderr.decode().strip())
                error_detail = error_json.get('error', 'Unknown error.')
            except json.JSONDecodeError:
                error_detail = stderr.decode().strip()
            logger.error(f"Error parsing JavaScript/TypeScript file '{file_path}': {error_detail}")
            return None

        # Parse the JSON output
        descriptions = json.loads(stdout.decode())

        if not descriptions:
            return None

        return descriptions

    except Exception as e:
        logger.error(f"Unexpected error while parsing JavaScript/TypeScript file '{file_path}': {e}")
        return None

def extract_java_description(file_path: str) -> Optional[Dict]:
    """
    Parses a Java file to extract class and method declarations along with comments.
    Returns a dictionary containing extracted components.

    Args:
        file_path (str): Path to the Java file.

    Returns:
        Optional[Dict]: Dictionary with classes and methods information if successful, otherwise None.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as java_file:
            content = java_file.read()
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Could not read Java file '{file_path}': {e}")
        return None

    try:
        tree = javalang.parse.parse(content)
    except javalang.parser.JavaSyntaxError as e:
        logger.error(f"Syntax error parsing Java file '{file_path}': {e}")
        return None

    classes = []
    functions = []

    for path, node in tree.filter(javalang.tree.ClassDeclaration):
        doc = node.documentation or 'No description.'
        methods = []
        for member in node.body:
            if isinstance(member, javalang.tree.MethodDeclaration):
                method_doc = member.documentation or 'No description.'
                params = [param.name for param in member.parameters]
                methods.append({
                    "name": member.name,
                    "params": params,
                    "description": method_doc
                })
        classes.append({
            "name": node.name,
            "description": doc,
            "methods": methods
        })

    if not classes:
        return None

    return {
        "classes": classes,
        "functions": functions  # Java typically doesn't have standalone functions
    }

def extract_cpp_description(file_path: str) -> Optional[Dict]:
    """
    Parses a C++ file to extract class and function declarations along with comments.
    Returns a dictionary containing extracted components.

    Args:
        file_path (str): Path to the C++ file.

    Returns:
        Optional[Dict]: Dictionary with classes and functions information if successful, otherwise None.
    """
    # Comprehensive C++ parsing is complex. For this script, we'll perform basic extraction using regex.
    try:
        with open(file_path, 'r', encoding='utf-8') as cpp_file:
            content = cpp_file.read()
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Could not read C++ file '{file_path}': {e}")
        return None

    # Regex patterns for classes and functions
    class_pattern = re.compile(r'class\s+(\w+)\s*{([^}]*)};', re.MULTILINE | re.DOTALL)
    method_pattern = re.compile(r'(public|private|protected):\s*([^;{}]*)\s*;', re.MULTILINE | re.DOTALL)

    classes = []
    functions = []

    for class_match in class_pattern.finditer(content):
        class_name = class_match.group(1)
        class_body = class_match.group(2)
        class_doc = "No description."  # C++ does not have a standard for class-level comments
        methods = []
        for method_match in method_pattern.finditer(class_body):
            access_specifier = method_match.group(1)
            method_signature = method_match.group(2).strip()
            # Attempt to extract method name and parameters
            method_parts = method_signature.split('(')
            if len(method_parts) == 2:
                method_name = method_parts[0].split()[-1]
                params = method_parts[1].rstrip(')')
                params_list = [param.strip() for param in params.split(',') if param.strip()]
                methods.append({
                    "name": method_name,
                    "params": params_list,
                    "description": "No description."
                })
        classes.append({
            "name": class_name,
            "description": class_doc,
            "methods": methods
        })

    if not classes:
        return None

    return {
        "classes": classes,
        "functions": functions  # C++ functions are usually within classes
    }

def extract_go_description(file_path: str) -> Optional[Dict]:
    """
    Parses a Go file to extract struct and function declarations along with comments.
    Returns a dictionary containing extracted components.

    Args:
        file_path (str): Path to the Go file.

    Returns:
        Optional[Dict]: Dictionary with structs and functions information if successful, otherwise None.
    """
    # Basic Go parsing using regex. For more accurate parsing, consider using Go's parser via a subprocess.
    try:
        with open(file_path, 'r', encoding='utf-8') as go_file:
            content = go_file.read()
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Could not read Go file '{file_path}': {e}")
        return None

    struct_pattern = re.compile(r'type\s+(\w+)\s+struct\s*{([^}]*)}', re.MULTILINE | re.DOTALL)
    func_pattern = re.compile(r'func\s+(\w+)\s*\(([^)]*)\)\s*([\w\*]+)\s*{')

    structs = []
    functions = []

    for struct_match in struct_pattern.finditer(content):
        struct_name = struct_match.group(1)
        struct_body = struct_match.group(2)
        struct_doc = "No description."  # Go uses comments above declarations
        structs.append({
            "name": struct_name,
            "description": struct_doc,
            "fields": []  # Fields can be extracted similarly if needed
        })

    for func_match in func_pattern.finditer(content):
        func_name = func_match.group(1)
        params = func_match.group(2).strip().split(',')
        params = [param.strip().split(' ')[0] for param in params if param.strip()]
        return_type = func_match.group(3).strip()
        functions.append({
            "name": func_name,
            "params": params,
            "return_type": return_type,
            "description": "No description."
        })

    if not structs and not functions:
        return None

    return {
        "structs": structs,
        "functions": functions
    }

def extract_generic_description(file_path: str, file_extension: str) -> Optional[Dict]:
    """
    Provides a generic description for unsupported file types.

    Args:
        file_path (str): Path to the file.
        file_extension (str): Extension of the file.

    Returns:
        Optional[Dict]: Dictionary with generic components if applicable, otherwise None.
    """
    return {
        "summary": f"Documentation for {file_extension} files.",
        "extended_summary": f"No detailed descriptions available for files with extension '{file_extension}'."
    }

def generate_google_docstring(description: str, params: List[Dict] = [], return_info: Optional[Dict] = None) -> str:
    """
    Formats a description into a Google-style docstring.

    Args:
        description (str): The summary or description of the function/class.
        params (List[Dict], optional): A list of dictionaries containing parameter details. Defaults to [].
        return_info (Optional[Dict], optional): A dictionary containing return type and description. Defaults to None.

    Returns:
        str: The formatted Google-style docstring.
    """
    docstring = f"{description}\n\n"
    if params:
        docstring += "Args:\n"
        for param in params:
            name = param.get("name", "param")
            type_ = param.get("type", "type")
            description_ = param.get("description", "Description of the parameter.")
            docstring += f"    {name} ({type_}): {description_}\n"
        docstring += "\n"
    if return_info:
        type_ = return_info.get("type", "type")
        description_ = return_info.get("description", "Description of the return value.")
        docstring += "Returns:\n"
        docstring += f"    {type_}: {description_}\n"
    return docstring
    
def generate_java_doc(docstring: str) -> str:
    """
    Generates a JavaDoc comment block from the provided docstring.
    
    Args:
        docstring (str): The description to include in the JavaDoc.
    
    Returns:
        str: Formatted JavaDoc comment block.
    """
    formatted_docstring = docstring.replace('\n', '\n * ')
    return f"""/**
 * {formatted_docstring}
 */
"""

def generate_js_doc(docstring: str) -> str:
    """
    Generates a JSDoc comment block from the provided docstring.
    
    Args:
        docstring (str): The description to include in the JSDoc.
    
    Returns:
        str: Formatted JSDoc comment block.
    """
    formatted_doc = docstring.replace('\n', '\n * ')
    return f"""/**
 * {formatted_doc}
 */
"""

async def insert_comments_into_file(file_path: str, doc_json: dict, language: str) -> bool:
    """
    Inserts the generated JSON-formatted Google-style docstrings into the source file based on the language.
    Returns True if successful, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error reading file '{file_path}' for inserting comments: {e}")
        return False

    # Backup the original file
    backup_path = f"{file_path}.bak"
    try:
        with open(backup_path, 'w', encoding='utf-8') as backup_file:
            backup_file.writelines(lines)
        logger.info(f"Backup created for '{file_path}' at '{backup_path}'")
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error creating backup for file '{file_path}': {e}")
        return False

    # Insert comments based on language
    try:
        if language == 'python':
            updated_lines = insert_python_comments(lines, doc_json)
        elif language in ['javascript', 'typescript']:
            updated_lines = insert_js_ts_comments(lines, doc_json)
        elif language == 'java':
            updated_lines = insert_java_comments(lines, doc_json)
        elif language == 'cpp':
            updated_lines = insert_cpp_comments(lines, doc_json)
        elif language == 'go':
            updated_lines = insert_go_comments(lines, doc_json)
        else:
            logger.warning(f"Comment insertion not implemented for language '{language}'")
            return False

        # Write the updated lines back to the file
        async with OUTPUT_LOCK:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(updated_lines)
        logger.info(f"Comments inserted into '{file_path}' successfully.")
        return True

    except Exception as e:
        logger.error(f"Error inserting comments into file '{file_path}': {e}")
        return False

def insert_python_comments(lines: List[str], doc_json: Dict) -> List[str]:
    """
    Inserts Google-style docstrings into Python classes and functions using JSON data.

    Args:
        lines (List[str]): Original lines of the Python file.
        doc_json (Dict): The refined docstring JSON.

    Returns:
        List[str]: Updated lines with inserted docstrings.
    """
    updated_lines = []
    i = 0
    classes = doc_json.get("classes", [])
    functions = doc_json.get("functions", [])

    class_dict = {cls["name"]: cls for cls in classes}
    function_dict = {func["name"]: func for func in functions}

    while i < len(lines):
        line = lines[i]
        class_match = re.match(r'\s*class\s+(\w+)\s*(\(.*\))?:', line)
        func_match = re.match(r'\s*def\s+(\w+)\s*\((.*)\)\s*:', line)

        if class_match:
            class_name = class_match.group(1)
            updated_lines.append(line)
            # Insert docstring after class definition
            cls_info = class_dict.get(class_name, {})
            if cls_info:
                summary = cls_info.get("summary", "")
                extended_summary = cls_info.get("extended_summary", "")
                args = cls_info.get("args", [])
                returns = cls_info.get("returns", {})
                raises = cls_info.get("raises", [])
                examples = cls_info.get("examples", [])
                notes = cls_info.get("notes", "")
                references = cls_info.get("references", "")

                docstring = generate_google_docstring_from_json(
                    summary=summary,
                    extended_summary=extended_summary,
                    args=args,
                    returns=returns,
                    raises=raises,
                    examples=examples,
                    notes=notes,
                    references=references
                )
                if docstring:
                    docstring_formatted = f'    """\n    {docstring}\n    """\n'
                    updated_lines.append(docstring_formatted)
            i += 1
            continue

        if func_match:
            func_name = func_match.group(1)
            params = func_match.group(2).strip().split(',')
            params = [param.strip() for param in params if param.strip()]
            updated_lines.append(line)
            # Insert docstring after function definition
            func_info = function_dict.get(func_name, {})
            if func_info:
                summary = func_info.get("summary", "")
                extended_summary = func_info.get("extended_summary", "")
                args = func_info.get("args", [])
                returns = func_info.get("returns", {})
                raises = func_info.get("raises", [])
                examples = func_info.get("examples", [])
                notes = func_info.get("notes", "")
                references = func_info.get("references", "")

                docstring = generate_google_docstring_from_json(
                    summary=summary,
                    extended_summary=extended_summary,
                    args=args,
                    returns=returns,
                    raises=raises,
                    examples=examples,
                    notes=notes,
                    references=references
                )
                if docstring:
                    docstring_formatted = f'    """\n    {docstring}\n    """\n'
                    updated_lines.append(docstring_formatted)
            i += 1
            continue

        updated_lines.append(line)
        i += 1

    return updated_lines

def insert_js_ts_comments(lines: List[str], doc_json: dict) -> List[str]:
    """
    Inserts JSDoc comments into JavaScript/TypeScript classes and functions using JSON data.

    Args:
        lines (List[str]): The original lines of the JavaScript/TypeScript source file.
        doc_json (dict): The JSON data containing docstring information.

    Returns:
       List[str]: The updated lines with inserted JSDoc comments.
    """
    updated_lines = []
    i = 0
    comments = doc_json.get("docstring", {})
    class_docs = {cls['name']: cls for cls in comments.get("classes", [])}
    function_docs = {func['name']: func for func in comments.get("functions", [])}

    while i < len(lines):
        line = lines[i]
        class_match = re.match(r'\s*class\s+(\w+)\s*(extends\s+\w+\s*)?\{', line)
        func_match = re.match(r'\s*(function\s+)?(\w+)\s*\((.*)\)\s*\{', line)
        method_match = re.match(r'\s*(\w+)\s*\((.*)\)\s*\{', line)

        if class_match:
            class_name = class_match.group(1)
            updated_lines.append(line)
            # Insert JSDoc comment before class definition
            doc_info = class_docs.get(class_name, {})
            if doc_info:
                summary = doc_info.get("summary", "")
                extended_summary = doc_info.get("extended_summary", "")
                args = doc_info.get("args", [])
                returns = doc_info.get("returns", {})
                raises = doc_info.get("raises", [])
                examples = doc_info.get("examples", [])
                notes = doc_info.get("notes", "")
                references = doc_info.get("references", "")

                # Construct docstring from JSON data
                docstring = generate_google_docstring_from_json(
                    summary, extended_summary, args, returns, raises, examples, notes, references
                )
                js_doc = generate_js_doc(docstring)
                updated_lines.append(f"{js_doc}\n")
            i += 1
            continue

        if func_match:
            func_name = func_match.group(2)
            params = func_match.group(3).strip()
            updated_lines.append(line)
            # Insert JSDoc comment before function definition
            doc_info = function_docs.get(func_name, {})
            if doc_info:
                summary = doc_info.get("summary", "")
                extended_summary = doc_info.get("extended_summary", "")
                args = doc_info.get("args", [])
                returns = doc_info.get("returns", {})
                raises = doc_info.get("raises", [])
                examples = doc_info.get("examples", [])
                notes = doc_info.get("notes", "")
                references = doc_info.get("references", "")

                # Construct docstring from JSON data
                docstring = generate_google_docstring_from_json(
                    summary, extended_summary, args, returns, raises, examples, notes, references
                )
                js_doc = generate_js_doc(docstring)
                updated_lines.append(f"{js_doc}\n")
            i += 1
            continue

        if method_match:
            method_name = method_match.group(1)
            params = method_match.group(2).strip()
            updated_lines.append(line)
            # Insert JSDoc comment before method definition
            doc_info = function_docs.get(method_name, {})
            if doc_info:
                summary = doc_info.get("summary", "")
                extended_summary = doc_info.get("extended_summary", "")
                args = doc_info.get("args", [])
                returns = doc_info.get("returns", {})
                raises = doc_info.get("raises", [])
                examples = doc_info.get("examples", [])
                notes = doc_info.get("notes", "")
                references = doc_info.get("references", "")

                # Construct docstring from JSON data
                docstring = generate_google_docstring_from_json(
                    summary, extended_summary, args, returns, raises, examples, notes, references
                )
                js_doc = generate_js_doc(docstring)
                updated_lines.append(f"{js_doc}\n")
            i += 1
            continue

        updated_lines.append(line)
        i += 1

    return updated_lines

def insert_java_comments(lines: List[str], doc_json: dict) -> List[str]:
    """
    Inserts JavaDoc comments into Java classes and methods using JSON data.

    Args:
        lines (List[str]): The original lines of the Java source file.
        doc_json (dict): The JSON data containing docstring information.

    Returns:
        List[str]: The updated lines with inserted JavaDoc comments.
    """
    updated_lines = []
    i = 0
    comments = doc_json.get("docstring", {})
    class_docs = {cls['name']: cls for cls in comments.get("classes", [])}
    method_docs = {method['name']: method for method in comments.get("methods", [])}

    while i < len(lines):
        line = lines[i]
        class_match = re.match(r'\s*public\s+class\s+(\w+)', line)
        method_match = re.match(r'\s*public\s+(\w+)\s+(\w+)\s*\((.*)\)\s*\{', line)

        if class_match:
            class_name = class_match.group(1)
            # Insert JavaDoc comment before class definition
            doc_info = class_docs.get(class_name, {})
            if doc_info:
                summary = doc_info.get("summary", "")
                extended_summary = doc_info.get("extended_summary", "")
                args = doc_info.get("args", [])
                returns = doc_info.get("returns", {})
                raises = doc_info.get("raises", [])
                examples = doc_info.get("examples", [])
                notes = doc_info.get("notes", "")
                references = doc_info.get("references", "")

                # Construct docstring from JSON data
                docstring = generate_google_docstring_from_json(
                    summary, extended_summary, args, returns, raises, examples, notes, references
                )
                java_doc = generate_java_doc(docstring)
                updated_lines.append(f"{java_doc}\n")
            updated_lines.append(line)
            i += 1
            continue

        if method_match:
            return_type = method_match.group(1)
            method_name = method_match.group(2)
            params = method_match.group(3).strip()
            # Insert JavaDoc comment before method definition
            doc_info = method_docs.get(method_name, {})
            if doc_info:
                summary = doc_info.get("summary", "")
                extended_summary = doc_info.get("extended_summary", "")
                args = doc_info.get("args", [])
                returns = doc_info.get("returns", {})
                raises = doc_info.get("raises", [])
                examples = doc_info.get("examples", [])
                notes = doc_info.get("notes", "")
                references = doc_info.get("references", "")

                # Construct docstring from JSON data
                docstring = generate_google_docstring_from_json(
                    summary, extended_summary, args, returns, raises, examples, notes, references
                )
                java_doc = generate_java_doc(docstring, params=params)
                updated_lines.append(f"{java_doc}\n")
            updated_lines.append(line)
            i += 1
            continue

        updated_lines.append(line)
        i += 1

    return updated_lines

def insert_cpp_comments(lines: List[str], doc_json: Dict) -> List[str]:
    """
    Inserts comments into C++ classes and functions using JSON data.

    Args:
        lines (List[str]): Original lines of the C++ file.
        doc_json (Dict): The refined docstring JSON.

    Returns:
        List[str]: Updated lines with inserted comments.
    """
    updated_lines = []
    i = 0
    classes = doc_json.get("classes", [])
    functions = doc_json.get("functions", [])

    class_dict = {cls["name"]: cls for cls in classes}
    function_dict = {func["name"]: func for func in functions}

    while i < len(lines):
        line = lines[i]
        class_match = re.match(r'\s*class\s+(\w+)\s*{', line)
        func_match = re.match(r'\s*(\w+)\s+(\w+)\s*\((.*)\)\s*{', line)

        if class_match:
            class_name = class_match.group(1)
            # Insert comment before class definition
            cls_info = class_dict.get(class_name, {})
            if cls_info:
                summary = cls_info.get("summary", "")
                extended_summary = cls_info.get("extended_summary", "")
                args = cls_info.get("args", [])
                returns = cls_info.get("returns", {})
                raises = cls_info.get("raises", [])
                examples = cls_info.get("examples", [])
                notes = cls_info.get("notes", "")
                references = cls_info.get("references", "")

                docstring = generate_google_docstring_from_json(
                    summary=summary,
                    extended_summary=extended_summary,
                    args=args,
                    returns=returns,
                    raises=raises,
                    examples=examples,
                    notes=notes,
                    references=references
                )
                if docstring:
                    cpp_comment = f"// {docstring}\n"
                    updated_lines.append(cpp_comment)
            updated_lines.append(line)
            i += 1
            continue

        if func_match:
            return_type = func_match.group(1)
            func_name = func_match.group(2)
            params = func_match.group(3).strip().split(',')
            params = [param.strip().split(' ')[0] for param in params if param.strip()]
            # Insert comment before function definition
            func_info = function_dict.get(func_name, {})
            if func_info:
                summary = func_info.get("summary", "")
                extended_summary = func_info.get("extended_summary", "")
                args = func_info.get("args", [])
                returns = func_info.get("returns", {})
                raises = func_info.get("raises", [])
                examples = func_info.get("examples", [])
                notes = func_info.get("notes", "")
                references = func_info.get("references", "")

                docstring = generate_google_docstring_from_json(
                    summary=summary,
                    extended_summary=extended_summary,
                    args=args,
                    returns=returns,
                    raises=raises,
                    examples=examples,
                    notes=notes,
                    references=references
                )
                if docstring:
                    cpp_comment = f"// {docstring}\n"
                    updated_lines.append(cpp_comment)
            updated_lines.append(line)
            i += 1
            continue

        updated_lines.append(line)
        i += 1

    return updated_lines

def insert_go_comments(lines: List[str], doc_json: Dict) -> List[str]:
    """
    Inserts comments into Go structs and functions using JSON data.

    Args:
        lines (List[str]): Original lines of the Go file.
        doc_json (Dict): The refined docstring JSON.

    Returns:
        List[str]: Updated lines with inserted comments.
    """
    updated_lines = []
    i = 0
    structs = doc_json.get("structs", [])
    functions = doc_json.get("functions", [])

    struct_dict = {st["name"]: st for st in structs}
    function_dict = {fn["name"]: fn for fn in functions}

    while i < len(lines):
        line = lines[i]
        struct_match = re.match(r'\s*type\s+(\w+)\s+struct\s*{', line)
        func_match = re.match(r'\s*func\s+(\w+)\s*\((.*)\)\s*([\w\*]+)\s*{', line)

        if struct_match:
            struct_name = struct_match.group(1)
            # Insert comment before struct definition
            st_info = struct_dict.get(struct_name, {})
            if st_info:
                summary = st_info.get("summary", "")
                extended_summary = st_info.get("extended_summary", "")
                args = st_info.get("args", [])
                returns = st_info.get("returns", {})
                raises = st_info.get("raises", [])
                examples = st_info.get("examples", [])
                notes = st_info.get("notes", "")
                references = st_info.get("references", "")

                docstring = generate_google_docstring_from_json(
                    summary=summary,
                    extended_summary=extended_summary,
                    args=args,
                    returns=returns,
                    raises=raises,
                    examples=examples,
                    notes=notes,
                    references=references
                )
                if docstring:
                    go_comment = f"// {docstring}\n"
                    updated_lines.append(go_comment)
            updated_lines.append(line)
            i += 1
            continue

        if func_match:
            func_name = func_match.group(1)
            params = func_match.group(2).strip().split(',')
            params = [param.strip().split(' ')[0] for param in params if param.strip()]
            return_type = func_match.group(3).strip()
            # Insert comment before function definition
            fn_info = function_dict.get(func_name, {})
            if fn_info:
                summary = fn_info.get("summary", "")
                extended_summary = fn_info.get("extended_summary", "")
                args = fn_info.get("args", [])
                returns = fn_info.get("returns", {})
                raises = fn_info.get("raises", [])
                examples = fn_info.get("examples", [])
                notes = fn_info.get("notes", "")
                references = fn_info.get("references", "")

                docstring = generate_google_docstring_from_json(
                    summary=summary,
                    extended_summary=extended_summary,
                    args=args,
                    returns=returns,
                    raises=raises,
                    examples=examples,
                    notes=notes,
                    references=references
                )
                if docstring:
                    go_comment = f"// {docstring}\n"
                    updated_lines.append(go_comment)
            updated_lines.append(line)
            i += 1
            continue

        updated_lines.append(line)
        i += 1

    return updated_lines

async def insert_comments_into_file(file_path: str, doc_json: Dict, language: str) -> bool:
    """
    Inserts the generated JSON-formatted comments into the source file based on the language.
    Returns True if successful, False otherwise.

    Args:
        file_path (str): Path to the source file.
        doc_json (Dict): The refined docstring JSON.
        language (str): The programming language of the source file.

    Returns:
        bool: True if insertion is successful, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error reading file '{file_path}' for inserting comments: {e}")
        return False

    # Backup the original file
    backup_path = f"{file_path}.bak"
    try:
        with open(backup_path, 'w', encoding='utf-8') as backup_file:
            backup_file.writelines(lines)
        logger.info(f"Backup created for '{file_path}' at '{backup_path}'")
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error creating backup for file '{file_path}': {e}")
        return False

    # Insert comments based on language
    try:
        if language == 'python':
            updated_lines = insert_python_comments(lines, doc_json)
        elif language in ['javascript', 'typescript']:
            updated_lines = insert_js_ts_comments(lines, doc_json)
        elif language == 'java':
            updated_lines = insert_java_comments(lines, doc_json)
        elif language == 'cpp':
            updated_lines = insert_cpp_comments(lines, doc_json)
        elif language == 'go':
            updated_lines = insert_go_comments(lines, doc_json)
        else:
            logger.warning(f"Comment insertion not implemented for language '{language}'")
            return False

        # Write the updated lines back to the file
        async with OUTPUT_LOCK:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(updated_lines)
        logger.info(f"Comments inserted into '{file_path}' successfully.")
        return True

    except Exception as e:
        logger.error(f"Error inserting comments into file '{file_path}': {e}")
        return False

def get_all_file_paths(repo_path: str, excluded_dirs: Set[str], excluded_files: Set[str]) -> List[str]:
    """
    Collects all file paths in the repository, excluding specified directories and files.

    Args:
        repo_path (str): Path to the repository.
        excluded_dirs (Set[str]): Set of directory names to exclude.
        excluded_files (Set[str]): Set of file names to exclude.

    Returns:
        List[str]: List of file paths to process.
    """
    file_paths = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('.')]
        for file in files:
            file_path = os.path.join(root, file)
            if file in excluded_files or file.startswith('.'):
                logger.debug(f"Excluded file: {file_path}")
                continue
            file_paths.append(file_path)
    return file_paths

async def process_all_files(file_paths: List[str], skip_types: List[str], output_lock: asyncio.Lock) -> None:
    """
    Processes all files asynchronously, generating and inserting comments.
    Uses tqdm for a progress bar with human-readable time format.
    """
    semaphore = asyncio.Semaphore(5)  # Limit concurrency to 5

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(process_file(session, file_path, skip_types, output_lock, semaphore))
            for file_path in file_paths
        ]

        # Customize tqdm's bar_format for more readable output
        with tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Files", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
            for f in pbar:
                try:
                    await f
                except Exception as e:
                    logger.error(f"Error processing a file: {e}")
                    
async def process_file(session: aiohttp.ClientSession, file_path: str, skip_types: List[str], output_lock: asyncio.Lock, semaphore: asyncio.Semaphore) -> None:
    """
    Processes a single file by reading its content, generating Google-style docstrings,
    refining them, and inserting them into the source file.
    """
    _, ext = os.path.splitext(file_path)
    language = get_language(ext)

    # Skip file types that are not meant for commenting (json, css, md, etc.)
    if ext in skip_types:
        logger.info(f"Skipping file for commenting: {file_path} (type {ext})")
        return  # Skipping commenting for these types

    # Read the content of the file
    try:
        with open(file_path, 'r', encoding='utf-8') as content_file:
            content = content_file.read()
        logger.debug(f"Successfully read file: {file_path}")
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error reading file '{file_path}': {e}")
        return

    # Extract descriptions for classes and functions
    description = await extract_description(file_path, ext) 
    if not description:
        logger.info(f"No components found to comment in '{file_path}'.")
        return

    # Generate comments using OpenAI
    doc_json_str = await fetch_openai(session, content, language, semaphore) # Generate docstring for content
    if not doc_json_str:
        logger.error(f"Failed to generate comments for '{file_path}'.")
        return

    try:
        doc_json = json.loads(doc_json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error for file '{file_path}': {e}")
        return

    # Insert comments into the source file
    success = await insert_comments_into_file(file_path, doc_json, language)
    if not success:
        logger.error(f"Failed to insert comments into '{file_path}'.")
        return

def main():
    """
    Main entry point of the script.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate and insert docstrings/comments into source files.")
    parser.add_argument("repo_path", type=str, help="Path to the directory containing source files.")
    parser.add_argument("-c", "--config", type=str, default="config.json", help="Path to the configuration JSON file.")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent tasks.")
    parser.add_argument("-o", "--output", type=str, help="Path to the output Markdown file.")  # Add output argument
    args = parser.parse_args()

    if not os.path.isdir(args.repo_path):
        logger.error(f"The path '{args.repo_path}' is not a valid directory.")
        sys.exit(1)

    # Initialize exclusion sets
    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    excluded_files = set(DEFAULT_EXCLUDED_FILES)

    # Load configuration
    try:
        with open(args.config, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
        skip_types = config.get("skip_types", [])
        load_config(args.config, excluded_dirs, excluded_files) # Load exclusions from config
    except (FileNotFoundError, IOError, json.JSONDecodeError) as e:
        logger.error(f"Error loading configuration file '{args.config}': {e}")
        return

    # Gather all file paths
    file_paths = get_all_file_paths(args.repo_path, excluded_dirs, excluded_files)  # Get file paths here
    logger.info(f"Collected {len(file_paths)} files to process.")

    # Initialize an asyncio lock
    global OUTPUT_LOCK, SEMAPHORE
    OUTPUT_LOCK = asyncio.Lock()
    SEMAPHORE = asyncio.Semaphore(args.concurrency) # Initialize semaphore

    # Run the asyncio event loop
    try:
        asyncio.run(process_all_files(file_paths, skip_types, OUTPUT_LOCK))
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
