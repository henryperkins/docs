# utils.py

import os
import sys
import json
import logging
import aiohttp
import asyncio
import re
import subprocess
from dotenv import load_dotenv
from typing import Any, Set, List, Optional, Dict, Tuple
from jsonschema import validate, ValidationError
import aiofiles

# Load environment variables
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("documentation_generation.log"),
        logging.StreamHandler(sys.stdout)
    ],
)
logger = logging.getLogger(__name__)

# ----------------------------
# Constants
# ----------------------------

DEFAULT_EXCLUDED_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', '.idea', 'scripts'}
DEFAULT_EXCLUDED_FILES = {".DS_Store"}
DEFAULT_SKIP_TYPES = {".json", ".md", ".txt", ".csv", ".lock"}

LANGUAGE_MAPPING = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".go": "go",
    ".cpp": "cpp",
    ".c": "cpp",
    ".java": "java",
}

# ----------------------------
# Language and File Utilities
# ----------------------------

def get_language(ext: str) -> str:
    """
    Determines the programming language based on file extension.

    Args:
        ext (str): File extension.

    Returns:
        str: Corresponding programming language.
    """
    language = LANGUAGE_MAPPING.get(ext.lower(), "plaintext")
    logger.debug(f"Detected language for extension '{ext}': {language}")
    return language

def is_valid_extension(ext: str, skip_types: Set[str]) -> bool:
    """
    Checks if a file extension is valid (not in the skip list).

    Args:
        ext (str): File extension.
        skip_types (Set[str]): Set of file extensions to skip.

    Returns:
        bool: True if valid, False otherwise.
    """
    is_valid = ext.lower() not in skip_types
    logger.debug(f"Extension '{ext}' is valid: {is_valid}")
    return is_valid

def is_binary(file_path: str) -> bool:
    """
    Checks if a file is binary.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if binary, False otherwise.
    """
    try:
        with open(file_path, "rb") as file:
            return b"\0" in file.read(1024)
    except Exception as e:
        logger.error(f"Error checking if file is binary '{file_path}': {e}")
        return True

def get_all_file_paths(repo_path: str, excluded_dirs: Set[str], excluded_files: Set[str], skip_types: Set[str]) -> List[str]:
    """
    Retrieves all file paths in the repository, excluding specified directories and files.

    Args:
        repo_path (str): Path to the repository.
        excluded_dirs (Set[str]): Set of directories to exclude.
        excluded_files (Set[str]): Set of files to exclude.
        skip_types (Set[str]): Set of file extensions to skip.

    Returns:
        List[str]: List of file paths.
    """
    file_paths = []
    normalized_excluded_dirs = {os.path.normpath(os.path.join(repo_path, d)) for d in excluded_dirs}

    for root, dirs, files in os.walk(repo_path, topdown=True):
        # Exclude directories
        dirs[:] = [d for d in dirs if os.path.normpath(os.path.join(root, d)) not in normalized_excluded_dirs]

        for file in files:
            # Exclude files
            if file in excluded_files:
                continue
            file_ext = os.path.splitext(file)[1]
            # Skip specified file types
            if file_ext in skip_types:
                continue
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
    logger.debug(f"Collected {len(file_paths)} files from '{repo_path}'.")
    return file_paths

# ----------------------------
# Configuration Management
# ----------------------------

def load_json_schema(schema_path: str) -> Optional[dict]:
    """
    Loads a JSON schema from the specified path.

    Args:
        schema_path (str): Path to the JSON schema file.

    Returns:
        Optional[dict]: Loaded JSON schema or None if failed.
    """
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        logger.debug(f"Successfully loaded JSON schema from '{schema_path}'.")
        return schema
    except FileNotFoundError:
        logger.error(f"JSON schema file '{schema_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{schema_path}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading JSON schema from '{schema_path}': {e}")
        return None

def load_function_schema(schema_path: str) -> dict:
    """Loads a function schema from a specified path.

    Args:
        schema_path (str): Path to the schema file.

    Returns:
        dict: The function schema loaded from the file."""
    logger.debug(f"Attempting to load function schema from '{schema_path}'.")
    schema = load_json_schema(schema_path)
    if not isinstance(schema, dict):
        logger.critical(f"Function schema should be a JSON object with a 'functions' key. Found type: {type(schema)}")
        sys.exit(1)
    if 'functions' not in schema:
        logger.critical(f"Function schema missing 'functions' key.")
        sys.exit(1)
    return schema

def load_config(config_path: str, excluded_dirs: Set[str], excluded_files: Set[str], skip_types: Set[str]) -> Tuple[str, str]:
    """
    Loads additional configurations from a config.json file.

    Args:
        config_path (str): Path to the config.json file.
        excluded_dirs (Set[str]): Set to update with excluded directories.
        excluded_files (Set[str]): Set to update with excluded files.
        skip_types (Set[str]): Set to update with file types to skip.

    Returns:
        Tuple[str, str]: Project information and style guidelines.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        project_info = config.get("project_info", "")
        style_guidelines = config.get("style_guidelines", "")
        excluded_dirs.update(config.get("excluded_dirs", []))
        excluded_files.update(config.get("excluded_files", []))
        skip_types.update(config.get("skip_types", []))
        logger.debug(f"Loaded configuration from '{config_path}'.")
        return project_info, style_guidelines
    except FileNotFoundError:
        logger.error(f"Config file '{config_path}' not found.")
        return "", ""
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{config_path}': {e}")
        return "", ""
    except Exception as e:
        logger.error(f"Unexpected error loading config file '{config_path}': {e}")
        return "", ""

# ----------------------------
# OpenAI REST API Interaction
# ----------------------------

async def fetch_documentation(
    session: aiohttp.ClientSession,
    prompt: str,
    semaphore: asyncio.Semaphore,
    deployment_name: str,
    function_schema: Dict[str, Any],
    azure_api_key: str,
    azure_endpoint: str,
    azure_api_version: str,
    retry: int = 3
) -> Optional[Dict[str, Any]]:
    """Fetches documentation using Azure OpenAI REST API with function calling."""
    logger.debug(f"Fetching documentation using REST API for deployment: {deployment_name}")

    url = f"{azure_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={azure_api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_api_key,
    }

    for attempt in range(1, retry + 1):
        async with semaphore:
            try:
                payload = {
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "functions": function_schema["functions"],
                    "function_call": {"name": "generate_documentation"},
                }

                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"API Response: {json.dumps(data, indent=2)}")

                        if "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            message = choice["message"]

                            if "function_call" in message:
                                function_call = message["function_call"]
                                if function_call["name"] == "generate_documentation":
                                    arguments = function_call["arguments"]
                                    try:
                                        documentation = json.loads(arguments)
                                        logger.debug("Received documentation via function_call.")
                                        return documentation
                                    except json.JSONDecodeError as e:
                                        logger.error(f"Error decoding JSON from function_call arguments: {e}")
                                        logger.error(f"Arguments Content: {arguments}")
                                        return None
                                else:
                                    logger.error(f"Unexpected function called: {function_call['name']}")
                                    return None
                            else:
                                logger.error("No function_call found in the response.")
                                return None
                        else:
                            logger.error("No choices found in the response.")
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(f"Request failed with status {response.status}: {error_text}")
                        if attempt < retry:
                            wait_time = 2 ** attempt
                            logger.info(f"Retrying after {wait_time} seconds... (Attempt {attempt}/{retry})")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error("All retry attempts failed.")
                            return None

            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}", exc_info=True)
                if attempt < retry:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying after {wait_time} seconds... (Attempt {attempt}/{retry})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("All retry attempts failed.")
                    return None

    logger.error("All retry attempts failed.")
    return None

# ----------------------------
# Code Formatting and Cleanup
# ----------------------------

def clean_unused_imports(code: str, file_path: str) -> str:
    """Cleans unused imports from the given code string.

    Args:
        code (str): The code from which to remove unused imports.
        file_path (str): The path to the file whose imports are being cleaned.

    Returns:
        str: The cleaned code with unused imports removed."""
    logger.debug('Cleaning unused imports in the given code.')
    try:
        command = [
            'autoflake',
            '--remove-all-unused-imports',
            '--remove-unused-variables',
            '--stdin-display-name',
            file_path,
            '-',
        ]
        logger.debug(f'Running command: {" ".join(command)}')
        process = subprocess.run(
            command,
            input=code,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        cleaned_code = process.stdout
        logger.debug('Successfully cleaned code with Autoflake.')
        return cleaned_code
    except subprocess.CalledProcessError as e:
        logger.error(f'Autoflake failed: {e.stderr}')
        return code
    except Exception as e:
        logger.error(f'Unexpected error during Autoflake processing: {e}')
        return code

def format_with_black(code: str) -> str:
    """
    Formats the given Python code using Black.

    Args:
        code (str): The Python code to format.

    Returns:
        str: The formatted Python code.
    """
    try:
        process = subprocess.run(
            ['black', '--quiet', '-'],
            input=code.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        formatted_code = process.stdout.decode('utf-8')
        logger.debug('Successfully formatted code with Black.')
        return formatted_code
    except subprocess.CalledProcessError as e:
        logger.error(f'Black formatting failed: {e.stderr.decode("utf-8")}')
        return code  # Return unformatted code if Black fails
    except Exception as e:
        logger.error(f'Unexpected error during Black formatting: {e}')
        return code  # Return unformatted code if any other error occurs

def run_flake8(file_path: str) -> Optional[str]:
    """
    Runs Flake8 on the specified file and returns the output if there are linting issues.

    Args:
        file_path (str): The path to the file to lint.

    Returns:
        Optional[str]: The Flake8 output if issues are found, else None.
    """
    try:
        process = subprocess.run(
            ['flake8', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # Do not raise exception on linting errors
        )
        if process.stdout:
            logger.debug(f'Flake8 issues found in {file_path}:\n{process.stdout}')
            return process.stdout.strip()
        return None
    except Exception as e:
        logger.error(f'Error running Flake8 on {file_path}: {e}')
        return None

# ----------------------------
# JavaScript/TypeScript Utilities
# ----------------------------

def run_node_script(script_name: str, input_data: dict) -> Optional[dict]:
    """
    Runs a Node.js script and returns the output.

    Args:
        script_name (str): Name of the script to run.
        input_data (dict): Input data to pass to the script.

    Returns:
        Optional[dict]: The output from the script if successful, None otherwise.
    """
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'scripts', script_name)
        logger.debug(f"Running Node.js script: {script_path}")

        input_json = json.dumps(input_data)
        result = subprocess.run(
            ["node", script_path],
            input=input_json,
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"Successfully ran {script_path}")
        output = json.loads(result.stdout)
        return output
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error(f"Node.js script {script_name} not found.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error running {script_name}: {e}")
        return None

def run_node_insert_docstrings(script_name: str, input_data: dict) -> Optional[str]:
    """
    Runs a Node.js script to insert docstrings and returns the modified code.

    Args:
        script_name (str): Name of the script to run.
        input_data (dict): Input data to pass to the script.

    Returns:
        Optional[str]: The modified code if successful, None otherwise.
    """
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'scripts', script_name)
        logger.debug(f"Running Node.js script: {script_path}")

        input_json = json.dumps(input_data)
        result = subprocess.run(
            ["node", script_path],
            input=input_json,
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"Successfully ran {script_path}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error(f"Node.js script {script_name} not found.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error running {script_name}: {e}")
        return None

# ----------------------------
# Documentation Generation
# ----------------------------

def generate_documentation_prompt(
    file_name: str,
    code_structure: Dict[str, Any],
    project_info: Optional[str],
    style_guidelines: Optional[str],
    language: str,
) -> str:
    prompt = (
        "You are an expert software engineer tasked with generating comprehensive documentation for the following code structure."
    )
    if project_info:
        prompt += f"\n\nProject Information:\n{project_info}"
    if style_guidelines:
        prompt += f"\n\nStyle Guidelines:\n{style_guidelines}"
    prompt += f"\n\nFile Name: {file_name}"
    prompt += f"\nLanguage: {language}"
    prompt += f"\n\nCode Structure:\n{json.dumps(code_structure, indent=2)}"
    prompt += """
    
Instructions:
Using the code structure provided, generate detailed documentation in JSON format that matches the following schema:

{
  "summary": "A detailed summary of the file.",
  "changes_made": ["List of changes made to the file."],
  "functions": [
    {
      "name": "Function name",
      "docstring": "Detailed description of the function, including its purpose and any important details.",
      "args": ["List of argument names"],
      "async": true or false
    }
    // More functions...
  ],
  "classes": [
    {
      "name": "Class name",
      "docstring": "Detailed description of the class.",
      "methods": [
        {
          "name": "Method name",
          "docstring": "Detailed description of the method.",
          "args": ["List of argument names"],
          "async": true or false,
          "type": "Method type (e.g., 'instance', 'class', 'static')"
        }
        // More methods...
      ]
    }
    // More classes...
  ]
}

Ensure that:
- **All 'docstring' fields contain comprehensive descriptions.**
- The 'args' lists contain the argument names of each function or method.
- The 'async' fields correctly indicate whether the function or method is asynchronous.
- The 'type' field for methods specifies if it's an 'instance', 'class', or 'static' method.

Please output only the JSON object that strictly follows the above schema, without any additional commentary or explanations.
"""
    return prompt


async def write_documentation_report(documentation: Optional[Dict[str, Any]], language: str, file_path: str, repo_root: str, new_content: str) -> str:
    """Asynchronously writes a documentation report to a specified file path.

    Args:
        documentation (Optional[Dict[str, Any]]): The documentation content to write.
        language (str): The language in which documentation is written.
        file_path (str): File path to write the report to.
        repo_root (str): Root path of the repository.
        new_content (str): New content to include in the report.

    Returns:
        str: The documentation content as a string.
    """
    'Writes a documentation report for a file in the specified language.'
    try:
        def sanitize_text(text: str) -> str:
            """Sanitizes the given text for usage in documentation or logging.

            Args:
                text (str): The text to sanitize.

            Returns:
                str: Sanitized version of the input text."""
            'Sanitizes and normalizes a string of text for safe processing.'
            if not text:
                return ''
            lines = text.strip().splitlines()
            sanitized_lines = [line.strip() for line in lines if line.strip()]
            return '\n'.join(sanitized_lines)

        relative_path = os.path.relpath(file_path, repo_root)
        file_header = f'# File: {relative_path}\n\n'
        documentation_content = file_header
        summary = documentation.get('summary', '') if documentation else ''
        summary = sanitize_text(summary)
        if summary:
            summary_section = f'## Summary\n\n{summary}\n'
            documentation_content += summary_section
        changes = documentation.get('changes_made', []) if documentation else []
        changes = [sanitize_text(change) for change in changes if change.strip()]
        if changes:
            changes_formatted = '\n'.join((f'- {change}' for change in changes))
            changes_section = f'## Changes Made\n\n{changes_formatted}\n'
            documentation_content += changes_section
        functions = documentation.get('functions', []) if documentation else []
        if functions:
            functions_section = '## Functions\n\n'
            functions_section += '| Function | Arguments | Description | Async |\n'
            functions_section += '|----------|-----------|-------------|-------|\n'
            for func in functions:
                func_name = func.get('name', 'N/A')
                func_args = ', '.join(func.get('args', []))
                func_doc = sanitize_text(func.get('docstring', 'No description provided.'))
                first_line_doc = func_doc.splitlines()[0]
                func_async = 'Yes' if func.get('async', False) else 'No'
                functions_section += f'| `{func_name}` | `{func_args}` | {first_line_doc} | {func_async} |\n'
            documentation_content += functions_section + '\n'
        classes = documentation.get('classes', []) if documentation else []
        if classes:
            classes_section = '## Classes\n\n'
            for cls in classes:
                cls_name = cls.get('name', 'N/A')
                cls_doc = sanitize_text(cls.get('docstring', 'No description provided.'))
                if cls_doc:
                    classes_section += f'### Class: `{cls_name}`\n\n{cls_doc}\n\n'
                else:
                    classes_section += f'### Class: `{cls_name}`\n\n'
                methods = cls.get('methods', [])
                if methods:
                    classes_section += '| Method | Arguments | Description | Async | Type |\n'
                    classes_section += '|--------|-----------|-------------|-------|------|\n'
                    for method in methods:
                        method_name = method.get('name', 'N/A')
                        method_args = ', '.join(method.get('args', []))
                        method_doc = sanitize_text(method.get('docstring', 'No description provided.'))
                        first_line_method_doc = method_doc.splitlines()[0] if method_doc else 'No description provided.'
                        method_async = 'Yes' if method.get('async', False) else 'No'
                        method_type = method.get('type', 'N/A')
                        classes_section += f'| `{method_name}` | `{method_args}` | {first_line_method_doc} | {method_async} | {method_type} |\n'
                    classes_section += '\n'
            documentation_content += classes_section
        code_content = new_content.strip()
        code_block = f'```{language}\n{code_content}\n```\n\n---\n'
        documentation_content += code_block
        return documentation_content
    except Exception as e:
        logger.error(f"Error generating documentation for '{file_path}': {e}", exc_info=True)
        return ''

def generate_table_of_contents(content: str) -> str:
    """
    Generates a table of contents from markdown headings.

    Args:
        content (str): Markdown content.

    Returns:
        str: Markdown-formatted table of contents.
    """
    toc = []
    for line in content.splitlines():
        if line.startswith("#"):
            level = line.count("#")
            title = line.lstrip("#").strip()
            anchor = re.sub(r'[^a-zA-Z0-9\s]', '', title).replace(' ', '-').lower()
            toc.append(f"{'  ' * (level - 1)}- [{title}](#{anchor})")
    return "\n".join(toc)

# ----------------------------
# Schema Validation
# ----------------------------

def validate_schema(schema: dict):
    """
    Validates the loaded schema against a predefined schema.

    Args:
        schema (dict): The schema to validate.
    """
    predefined_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "changes_made": {"type": "array", "items": {"type": "string"}},
            "functions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "args": {"type": "array", "items": {"type": "string"}},
                        "docstring": {"type": "string"},
                        "async": {"type": "boolean"}
                    },
                    "required": ["name", "args", "docstring", "async"]
                }
            },
            "classes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "docstring": {"type": "string"},
                        "methods": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "args": {"type": "array", "items": {"type": "string"}},
                                    "docstring": {"type": "string"},
                                    "async": {"type": "boolean"},
                                    "type": {"type": "string"}
                                },
                                "required": ["name", "args", "docstring", "async", "type"]
                            }
                        }
                    },
                    "required": ["name", "docstring", "methods"]
                }
            }
        },
        "required": ["summary", "changes_made", "functions", "classes"]
    }
    try:
        validate(instance=schema, schema=predefined_schema)
        logger.debug("Documentation schema is valid.")
    except ValidationError as ve:
        logger.critical(f"Schema validation error: {ve.message}")
        sys.exit(1)

# ----------------------------
# EOF
# ----------------------------
