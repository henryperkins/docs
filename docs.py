#!/usr/bin/env python3

import os
import sys
import argparse
import asyncio
import aiohttp
import aiofiles
import ast
import re
import json
import logging
from typing import List, Optional
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Logging
logging.basicConfig(
    filename='docs_generation.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

# OpenAI API Configuration
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
openai_api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment or .env file

# Global variables for concurrency control
SEMAPHORE = None
OUTPUT_LOCK = None

DEFAULT_EXCLUDED_DIRS = ['.git', '__pycache__', 'node_modules']
DEFAULT_EXCLUDED_FILES = ['.DS_Store']

def get_language(ext: str) -> str:
    """
    Determines the programming language based on the file extension.

    Args:
        ext (str): File extension.

    Returns:
        str: Programming language.
    """
    language_mapping = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.go': 'go',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.rs': 'rust'
        # Add more mappings as needed
    }
    return language_mapping.get(ext.lower(), 'plaintext')

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
    except Exception as e:
        logger.error(f"Error checking if file is binary '{file_path}': {e}")
        return True  # Assume binary if there's an error
    return False

def load_config(config_path: str, excluded_dirs: set, excluded_files: set) -> None:
    """
    Loads additional exclusions from a configuration JSON file.

    Args:
        config_path (str): Path to the configuration file.
        excluded_dirs (set): Set of directories to exclude.
        excluded_files (set): Set of files to exclude.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
        excluded_dirs.update(config.get('excluded_dirs', []))
        excluded_files.update(config.get('excluded_files', []))
    except (FileNotFoundError, IOError, json.JSONDecodeError) as e:
        logger.error(f"Error loading configuration file '{config_path}': {e}")

def get_all_file_paths(repo_path: str, excluded_dirs: set, excluded_files: set) -> List[str]:
    """
    Collects all file paths in the repository, excluding specified directories and files.

    Args:
        repo_path (str): Path to the repository.
        excluded_dirs (set): Set of directories to exclude.
        excluded_files (set): Set of files to exclude.

    Returns:
        List[str]: List of file paths.
    """
    file_paths = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for file in files:
            if file in excluded_files:
                continue
            file_paths.append(os.path.join(root, file))
    return file_paths

async def extract_description(file_path: str, ext: str) -> Optional[str]:
    """
    Asynchronously extracts descriptions for classes and functions from the given file.
    Returns a JSON-formatted description.

    Args:
        file_path (str): Path to the source file.
        ext (str): File extension indicating the programming language.

    Returns:
        Optional[str]: JSON-formatted description or None if extraction fails.
    """
    language = get_language(ext)
    if language == 'python':
        return await extract_python_description(file_path)
    elif language in ['javascript', 'typescript']:
        return await extract_js_ts_description(file_path, language)
    elif language == 'java':
        return await extract_java_description(file_path)
    elif language == 'cpp' or language == 'c':
        return await extract_cpp_description(file_path)
    elif language == 'go':
        return await extract_go_description(file_path)
    else:
        logger.warning(f"Extraction not implemented for language '{language}'.")
        return None

async def extract_python_description(file_path: str) -> Optional[str]:
    """
    Extracts descriptions from a Python file using AST.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        Optional[str]: JSON-formatted description or None if extraction fails.
    """
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        tree = ast.parse(content, filename=file_path)
    except SyntaxError as e:
        logger.error(f"Syntax error parsing Python file '{file_path}': {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading Python file '{file_path}': {e}")
        return None

    classes = []
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_doc = ast.get_docstring(node) or "No description."
            classes.append({
                "name": node.name,
                "docstring": class_doc
            })
        elif isinstance(node, ast.FunctionDef):
            func_doc = ast.get_docstring(node) or "No description."
            args = []
            for arg in node.args.args:
                if arg.arg != 'self':
                    arg_type = "Any"
                    if arg.annotation:
                        arg_type = ast.unparse(arg.annotation)
                    args.append({
                        "name": arg.arg,
                        "type": arg_type
                    })
            returns = {}
            if node.returns:
                returns = {
                    "type": ast.unparse(node.returns)
                }
            functions.append({
                "name": node.name,
                "docstring": func_doc,
                "args": args,
                "returns": returns
            })

    descriptions = {
        "language": "python",
        "classes": classes,
        "functions": functions
    }

    return json.dumps(descriptions, indent=2)

async def extract_js_ts_description(file_path: str, language: str) -> Optional[str]:
    """
    Extracts descriptions from JavaScript/TypeScript files using regex.

    Args:
        file_path (str): Path to the JavaScript/TypeScript source file.
        language (str): 'javascript' or 'typescript'.

    Returns:
        Optional[str]: JSON-formatted description or None if extraction fails.
    """
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
    except Exception as e:
        logger.error(f"Error reading {language} file '{file_path}': {e}")
        return None

    classes = []
    functions = []

    # Regex patterns for classes and functions
    class_pattern = re.compile(r'class\s+(\w+)')
    function_pattern = re.compile(r'(?:function\s+)?(\w+)\s*\((.*?)\)\s*\{')

    for match in class_pattern.finditer(content):
        class_name = match.group(1)
        classes.append({
            "name": class_name,
            "docstring": "No description."
        })

    for match in function_pattern.finditer(content):
        func_name = match.group(1)
        params = match.group(2).split(',') if match.group(2).strip() else []
        args = []
        for param in params:
            param = param.strip()
            if param:
                args.append({
                    "name": param,
                    "type": "Any"
                })
        functions.append({
            "name": func_name,
            "docstring": "No description.",
            "args": args,
            "returns": {}
        })

    descriptions = {
        "language": language,
        "classes": classes,
        "functions": functions
    }

    return json.dumps(descriptions, indent=2)

async def extract_java_description(file_path: str) -> Optional[str]:
    """
    Extracts descriptions from Java files using regex.

    Args:
        file_path (str): Path to the Java source file.

    Returns:
        Optional[str]: JSON-formatted description or None if extraction fails.
    """
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
    except Exception as e:
        logger.error(f"Error reading Java file '{file_path}': {e}")
        return None

    classes = []
    methods = []

    # Regex patterns for classes and methods
    class_pattern = re.compile(r'public\s+class\s+(\w+)')
    method_pattern = re.compile(r'public\s+[\w<>\[\]]+\s+(\w+)\s*\((.*?)\)\s*\{')

    for match in class_pattern.finditer(content):
        class_name = match.group(1)
        classes.append({
            "name": class_name,
            "docstring": "No description."
        })

    for match in method_pattern.finditer(content):
        method_name = match.group(1)
        params = match.group(2).split(',') if match.group(2).strip() else []
        args = []
        for param in params:
            param = param.strip()
            if param:
                param_parts = param.strip().split()
                if len(param_parts) >= 2:
                    param_type = ' '.join(param_parts[:-1])
                    param_name = param_parts[-1]
                    args.append({
                        "name": param_name,
                        "type": param_type
                    })
                else:
                    args.append({
                        "name": param,
                        "type": "Any"
                    })
        methods.append({
            "name": method_name,
            "docstring": "No description.",
            "args": args,
            "returns": {}
        })

    descriptions = {
        "language": "java",
        "classes": classes,
        "methods": methods
    }

    return json.dumps(descriptions, indent=2)

async def extract_cpp_description(file_path: str) -> Optional[str]:
    """
    Extracts descriptions from C/C++ files using regex.

    Args:
        file_path (str): Path to the C/C++ source file.

    Returns:
        Optional[str]: JSON-formatted description or None if extraction fails.
    """
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
    except Exception as e:
        logger.error(f"Error reading C/C++ file '{file_path}': {e}")
        return None

    classes = []
    functions = []

    # Regex patterns for classes and functions
    class_pattern = re.compile(r'class\s+(\w+)')
    function_pattern = re.compile(r'[\w:<>\*&]+\s+(\w+)\s*\((.*?)\)\s*(const)?\s*\{')

    for match in class_pattern.finditer(content):
        class_name = match.group(1)
        classes.append({
            "name": class_name,
            "docstring": "No description."
        })

    for match in function_pattern.finditer(content):
        func_name = match.group(1)
        params = match.group(2).split(',') if match.group(2).strip() else []
        args = []
        for param in params:
            param = param.strip()
            if param:
                param_parts = param.strip().split()
                if len(param_parts) >= 2:
                    param_type = ' '.join(param_parts[:-1])
                    param_name = param_parts[-1]
                    args.append({
                        "name": param_name,
                        "type": param_type
                    })
                else:
                    args.append({
                        "name": param,
                        "type": "Any"
                    })
        functions.append({
            "name": func_name,
            "docstring": "No description.",
            "args": args,
            "returns": {}
        })

    descriptions = {
        "language": "cpp",
        "classes": classes,
        "functions": functions
    }

    return json.dumps(descriptions, indent=2)

async def extract_go_description(file_path: str) -> Optional[str]:
    """
    Extracts descriptions from Go files using regex.

    Args:
        file_path (str): Path to the Go source file.

    Returns:
        Optional[str]: JSON-formatted description or None if extraction fails.
    """
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
    except Exception as e:
        logger.error(f"Error reading Go file '{file_path}': {e}")
        return None

    structs = []
    functions = []

    # Regex patterns for structs and functions
    struct_pattern = re.compile(r'type\s+(\w+)\s+struct\s*\{')
    function_pattern = re.compile(r'func\s+(\w+)\s*\((.*?)\)\s*(\w+)?\s*\{')

    for match in struct_pattern.finditer(content):
        struct_name = match.group(1)
        structs.append({
            "name": struct_name,
            "docstring": "No description."
        })

    for match in function_pattern.finditer(content):
        func_name = match.group(1)
        params = match.group(2).split(',') if match.group(2).strip() else []
        return_type = match.group(3) if match.group(3) else "Any"
        args = []
        for param in params:
            param = param.strip()
            if param:
                param_parts = param.strip().split()
                if len(param_parts) >= 2:
                    param_type = ' '.join(param_parts[1:])
                    param_name = param_parts[0]
                    args.append({
                        "name": param_name,
                        "type": param_type
                    })
                else:
                    args.append({
                        "name": param,
                        "type": "Any"
                    })
        functions.append({
            "name": func_name,
            "docstring": "No description.",
            "args": args,
            "returns": {
                "type": return_type
            }
        })

    descriptions = {
        "language": "go",
        "structs": structs,
        "functions": functions
    }

    return json.dumps(descriptions, indent=2)

async def fetch_openai(session: aiohttp.ClientSession, content: str, language: str, retry: int = 3) -> Optional[str]:
    """
    Asynchronously fetches the generated documentation from OpenAI API.

    Args:
        session (aiohttp.ClientSession): The aiohttp session for making requests.
        content (str): The code content to generate documentation for.
        language (str): The programming language of the code.
        retry (int): Number of retry attempts.

    Returns:
        Optional[str]: Generated documentation if successful, None otherwise.
    """
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    Generate detailed Google-style docstrings/comments for the following {language} code. Ensure that each docstring includes descriptions of all parameters and return types where applicable.

    ```{language}
    {content}
    ```

    Please output the docstrings/comments in Markdown format.
    """

    payload = {
        "model": "gpt-4o-mini”,
        "messages": [
            {"role": "system", "content": "You are an expert software developer specializing in writing detailed Google-style docstrings and comments."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.2
    }

    for attempt in range(1, retry + 1):
        try:
            async with SEMAPHORE:
                async with session.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 200:
                        data = await response.json()
                        documentation = data['choices'][0]['message']['content'].strip()
                        logger.info("Successfully generated documentation from OpenAI.")
                        return documentation
                    elif response.status in {429, 500, 502, 503, 504}:
                        logger.warning(f"API rate limit or server error (status {response.status}). Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds.")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        error_text = await response.text()
                        logger.error(f"API request failed with status {response.status}: {error_text}")
                        return None
        except asyncio.TimeoutError:
            logger.error(f"Request timed out. Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds.")
            await asyncio.sleep(2 ** attempt)
        except aiohttp.ClientError as e:
            logger.error(f"API connection error: {e}. Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds.")
            await asyncio.sleep(2 ** attempt)

    logger.error("Failed to generate documentation after multiple attempts.")
    return None

async def fetch_refined_documentation(session: aiohttp.ClientSession, documentation: str, retry: int = 3) -> Optional[str]:
    """
    Asynchronously refines the generated documentation using OpenAI API.

    Args:
        session (aiohttp.ClientSession): The aiohttp session for making requests.
        documentation (str): The initial generated documentation.
        retry (int): Number of retry attempts.

    Returns:
        Optional[str]: Refined documentation if successful, None otherwise.
    """
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    Please refine the following documentation to ensure clarity, correctness, and adherence to Google-style guidelines. Make sure all code examples are accurate.

    {documentation}
    """

    payload = {
        "model": "gpt-4o-mini”,
        "messages": [
            {"role": "system", "content": "You are an expert technical writer and software developer specializing in refining and improving documentation."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.2
    }

    for attempt in range(1, retry + 1):
        try:
            async with SEMAPHORE:
                async with session.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 200:
                        data = await response.json()
                        refined_doc = data['choices'][0]['message']['content'].strip()
                        logger.info("Successfully refined documentation using OpenAI.")
                        return refined_doc
                    elif response.status in {429, 500, 502, 503, 504}:
                        logger.warning(f"API rate limit or server error (status {response.status}). Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds.")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        error_text = await response.text()
                        logger.error(f"API request failed with status {response.status}: {error_text}")
                        return None
        except asyncio.TimeoutError:
            logger.error(f"Request timed out. Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds.")
            await asyncio.sleep(2 ** attempt)
        except aiohttp.ClientError as e:
            logger.error(f"API connection error: {e}. Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds.")
            await asyncio.sleep(2 ** attempt)

    logger.error("Failed to refine documentation after multiple attempts.")
    return None

async def process_file(session: aiohttp.ClientSession, file_path: str, skip_types: List[str], output_file: str) -> None:
    """
    Processes a single file by reading its content, generating enhanced documentation,
    and writing the output to the specified markdown file.

    Args:
        session (aiohttp.ClientSession): The aiohttp session for making requests.
        file_path (str): Path to the source file.
        skip_types (List[str]): List of file extensions to skip.
        output_file (str): Path to the output markdown file.
    """
    _, ext = os.path.splitext(file_path)
    logger.debug(f"Processing file: {file_path} with extension: {ext}")

    # Skip file types that are not meant for commenting (json, css, md, etc.)
    if ext in skip_types:
        logger.info(f"Skipping file: {file_path} (type {ext})")
        return

    # Check if the file is binary
    if is_binary(file_path):
        logger.warning(f"Skipped binary file: {file_path}")
        return

    # Read the content of the file
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as content_file:
            content = await content_file.read()
        logger.debug(f"Successfully read file: {file_path}")
    except UnicodeDecodeError as e:
        logger.error(f"UnicodeDecodeError for file '{file_path}': {e}")
        return
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error reading file '{file_path}': {e}")
        return

    # Generate enhanced documentation for the file
    language = get_language(ext)
    logger.info(f"Generating documentation for file: {file_path}")
    enhanced_documentation = await fetch_openai(session, content, language)
    if not enhanced_documentation:
        logger.error(f"Failed to generate documentation for '{file_path}'.")
        return

    refined_documentation = await fetch_refined_documentation(session, enhanced_documentation)
    if not refined_documentation:
        refined_documentation = enhanced_documentation  # Fallback to original documentation

    # Write the file content and enhanced documentation to the output markdown file
    try:
        async with OUTPUT_LOCK:
            async with aiofiles.open(output_file, 'a', encoding='utf-8') as md_file:
                await md_file.write(f"## {file_path}\n\n")
                await md_file.write(f"```{language}\n{content}\n```\n\n")
                await md_file.write(f"### Generated Documentation:\n{refined_documentation}\n\n")
        logger.info(f"Documentation for '{file_path}' written to {output_file}")
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error writing to output file '{output_file}': {e}")

async def process_all_files(file_paths: List[str], skip_types: List[str], output_lock: asyncio.Lock, output_file: str) -> None:
    """
    Processes all files asynchronously, generating and inserting comments.

    Args:
        file_paths (List[str]): List of file paths to process.
        skip_types (List[str]): List of file extensions to skip.
        output_lock (asyncio.Lock): Lock to manage concurrent file writes.
        output_file (str): Path to the output markdown file.
    """
    global OUTPUT_LOCK
    OUTPUT_LOCK = output_lock

    async with aiohttp.ClientSession() as session:
        tasks = []
        for file_path in file_paths:
            task = asyncio.create_task(process_file(session, file_path, skip_types, output_file))
            tasks.append(task)
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Files"):
            await f

def main() -> None:
    """
    Main entry point of the script.
    """
    parser = argparse.ArgumentParser(
        description="Automatically generate and insert Google-style comments/docstrings into source files using GPT-4."
    )
    parser.add_argument(
        "repo_path",
        help="Path to the local repository"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.json",
        help="Path to configuration file for additional exclusions"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent API requests"
    )
    parser.add_argument(
        "-o", "--output",
        default="output.md",
        help="Output Markdown file to write generated documentation"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.repo_path):
        logger.error(f"The path '{args.repo_path}' is not a valid directory.")
        sys.exit(1)

    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    excluded_files = set(DEFAULT_EXCLUDED_FILES)
    skip_types = []

    # Load exclusions from config file if provided
    if args.config:
        load_config(args.config, excluded_dirs, excluded_files)

    # Collect all file paths
    file_paths = get_all_file_paths(args.repo_path, excluded_dirs, excluded_files)

    logger.info(f"Collected {len(file_paths)} files to process.")

    global SEMAPHORE
    SEMAPHORE = asyncio.Semaphore(args.concurrency)
    output_lock = asyncio.Lock()

    # Clear the output file if it exists
    if os.path.exists(args.output):
        open(args.output, 'w').close()

    # Run the asynchronous processing
    start_time = asyncio.get_event_loop().time()
    try:
        asyncio.run(process_all_files(file_paths, skip_types, output_lock, args.output))
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
        sys.exit(1)
    end_time = asyncio.get_event_loop().time()

    logger.info(f"Documentation generation completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
