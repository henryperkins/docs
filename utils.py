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

async def clean_unused_imports_async(code: str, file_path: str) -> str:
    """
    Asynchronously removes unused imports and variables from the provided code using autoflake.

    Args:
        code (str): The source code to clean.
        file_path (str): The file path used for display purposes in autoflake.

    Returns:
        str: The cleaned code with unused imports and variables removed.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            'autoflake', '--remove-all-unused-imports', '--remove-unused-variables', '--stdin-display-name', file_path, '-',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=code.encode())
        
        if process.returncode != 0:
            logger.error(f"Autoflake failed:\n{stderr.decode()}")
            return code
        
        return stdout.decode()
    except Exception as e:
        logger.error(f'Error running Autoflake: {e}')
        return code

# Example using asyncio subprocesses
async def format_with_black_async(code: str) -> str:
    process = await asyncio.create_subprocess_exec(
        'black', '--quiet', '-',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate(input=code.encode())
    if process.returncode == 0:
        return stdout.decode()
    else:
        logger.error(f"Black formatting failed: {stderr.decode()}")
        return code


async def run_flake8_async(file_path: str) -> Optional[str]:
    """
    Asynchronously runs Flake8 on the specified file to check for style violations.

    Args:
        file_path (str): The path to the file to be checked.

    Returns:
        Optional[str]: The output from Flake8 if there are violations, otherwise None.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            'flake8', file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            return stdout.decode() + stderr.decode()
        
        return None
    except Exception as e:
        logger.error(f'Error running Flake8: {e}')
        return None

# ----------------------------
# JavaScript/TypeScript Utilities
# ----------------------------

async def run_node_script_async(script_path: str, input_json: str) -> Optional[str]:
    try:
        process = await asyncio.create_subprocess_exec(
            'node', script_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=input_json.encode())
        if process.returncode != 0:
            logger.error(f"Node script error:\n{stderr.decode()}")
            return None
        return stdout.decode()
    except Exception as e:
        logger.error(f'Error running Node script: {e}')
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
def get_threshold(metric: str, key: str, default: int) -> int:
    return int(os.getenv(f"{metric.upper()}_{key.upper()}_THRESHOLD", default))

def generate_all_badges(
    complexity: Optional[int] = None, 
    halstead: Optional[Dict[str, Any]] = None, 
    mi: Optional[float] = None
) -> str:
    badges = []
    
    # Cyclomatic Complexity
    if complexity is not None:
        low_threshold = get_threshold('complexity', 'low', 10)
        medium_threshold = get_threshold('complexity', 'medium', 20)
        color = "green" if complexity < low_threshold else "yellow" if complexity < medium_threshold else "red"
        complexity_badge = f'![Complexity: {complexity}](https://img.shields.io/badge/Complexity-{complexity}-{color}?style=flat-square)'
        badges.append(complexity_badge)
    
    # Halstead Metrics
    if halstead:
        try:
            volume = halstead['volume']
            difficulty = halstead['difficulty']
            effort = halstead['effort']
            
            volume_low = get_threshold('halstead_volume', 'low', 100)
            volume_medium = get_threshold('halstead_volume', 'medium', 500)
            volume_color = "green" if volume < volume_low else "yellow" if volume < volume_medium else "red"
            
            difficulty_low = get_threshold('halstead_difficulty', 'low', 10)
            difficulty_medium = get_threshold('halstead_difficulty', 'medium', 20)
            difficulty_color = "green" if difficulty < difficulty_low else "yellow" if difficulty < difficulty_medium else "red"
            
            effort_low = get_threshold('halstead_effort', 'low', 500)
            effort_medium = get_threshold('halstead_effort', 'medium', 1000)
            effort_color = "green" if effort < effort_low else "yellow" if effort < effort_medium else "red"
            
            volume_badge = f'![Halstead Volume: {volume}](https://img.shields.io/badge/Volume-{volume}-{volume_color}?style=flat-square)'
            difficulty_badge = f'![Halstead Difficulty: {difficulty}](https://img.shields.io/badge/Difficulty-{difficulty}-{difficulty_color}?style=flat-square)'
            effort_badge = f'![Halstead Effort: {effort}](https://img.shields.io/badge/Effort-{effort}-{effort_color}?style=flat-square)'
            
            badges.extend([volume_badge, difficulty_badge, effort_badge])
        except KeyError as e:
            print(f"Missing Halstead metric: {e}. Halstead badges will not be generated.")
    
    # Maintainability Index
    if mi is not None:
        high_threshold = get_threshold('maintainability_index', 'high', 80)
        medium_threshold = get_threshold('maintainability_index', 'medium', 50)
        color = "green" if mi > high_threshold else "yellow" if mi > medium_threshold else "red"
        mi_badge = f'![Maintainability Index: {mi}](https://img.shields.io/badge/Maintainability-{mi}-{color}?style=flat-square)'
        badges.append(mi_badge)
    
    return ' '.join(badges)


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
      "async": true or false,
      "complexity": "Cyclomatic complexity score"
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
          "type": "Method type (e.g., 'instance', 'class', 'static')",
          "complexity": "Cyclomatic complexity score"
        }
        // More methods...
      ]
    }
    // More classes...
  ]
}

Ensure that:
- **All 'docstring' fields contain comprehensive and meaningful descriptions.**
- The 'args' lists contain the argument names of each function or method.
- The 'async' fields correctly indicate whether the function or method is asynchronous.
- The 'type' field for methods specifies if it's an 'instance', 'class', or 'static' method.
- The 'complexity' field provides the cyclomatic complexity score.

**Do not omit any 'docstring' fields. Provide detailed descriptions for each.**

Please output only the JSON object that strictly follows the above schema, without any additional commentary or explanations.
"""
    return prompt

def truncate_description(description: str, max_length: int = 100) -> str:
    """Truncates the description to a specified maximum length."""
    return (description[:max_length] + '...') if len(description) > max_length else description

# Usage within write_documentation_report
first_line_doc = truncate_description(func_doc)

async def write_documentation_report(
    documentation: Optional[Dict[str, Any]], 
    language: str, 
    file_path: str, 
    repo_root: str, 
    new_content: str
) -> str:
    try:
        relative_path = os.path.relpath(file_path, repo_root)
        file_header = f'# File: {relative_path}\n\n'
        documentation_content = file_header

        # Summary Section
        summary = documentation.get('summary', '') if documentation else ''
        summary = sanitize_text(summary)
        if summary:
            summary_section = f'## Summary\n\n{summary}\n'
            documentation_content += summary_section

        # Changes Made Section
        changes = documentation.get('changes_made', []) if documentation else []
        changes = [sanitize_text(change) for change in changes if change.strip()]
        if changes:
            changes_formatted = '\n'.join((f'- {change}' for change in changes))
            changes_section = f'## Changes Made\n\n{changes_formatted}\n'
            documentation_content += changes_section

        # Generate overall badges
        halstead = documentation.get('halstead') if documentation else {}
        mi = documentation.get('maintainability_index') if documentation else None
        complexity = max(
            (func.get('complexity', 0) for func in documentation.get('functions', [])),
            *(method.get('complexity', 0) for cls in documentation.get('classes', []) for method in cls.get('methods', [])),
            default=0
        )
        overall_badges = generate_all_badges(complexity, halstead, mi)
        if overall_badges:
            documentation_content += f"{overall_badges}\n\n"

        # Functions Section
        functions = documentation.get('functions', []) if documentation else []
        if functions:
            functions_section = '## Functions\n\n'
            functions_section += '| Function | Arguments | Description | Async | Complexity |\n'
            functions_section += '|----------|-----------|-------------|-------|------------|\n'
            for func in functions:
                func_name = func.get('name', 'N/A')
                func_args = ', '.join(func.get('args', []))
                func_doc = sanitize_text(func.get('docstring', 'No description provided.'))
                first_line_doc = truncate_description(func_doc)
                func_async = 'Yes' if func.get('async', False) else 'No'
                func_complexity = func.get('complexity', 0)
                complexity_badge = generate_all_badges(func_complexity, {}, 0)
                functions_section += f'| `{func_name}` | `{func_args}` | {first_line_doc} | {func_async} | {complexity_badge} |\n'
            documentation_content += functions_section + '\n'

        # Classes Section
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
                    classes_section += '| Method | Arguments | Description | Async | Type | Complexity |\n'
                    classes_section += '|--------|-----------|-------------|-------|------|------------|\n'
                    for method in methods:
                        method_name = method.get('name', 'N/A')
                        method_args = ', '.join(method.get('args', []))
                        method_doc = sanitize_text(method.get('docstring', 'No description provided.'))
                        first_line_method_doc = truncate_description(method_doc)
                        method_async = 'Yes' if method.get('async', False) else 'No'
                        method_type = method.get('type', 'N/A')
                        method_complexity = method.get('complexity', 0)
                        complexity_badge = generate_all_badges(method_complexity, {}, 0)
                        classes_section += (
                            f'| `{method_name}` | `{method_args}` | {first_line_method_doc} | '
                            f'{method_async} | {method_type} | {complexity_badge} |\n'
                        )
                    classes_section += '\n'
            documentation_content += classes_section

        # Source Code Block
        code_content = new_content.strip()
        code_block = f'```{language}\n{code_content}\n```\n\n---\n'
        documentation_content += code_block

        # Generate and prepend Table of Contents
        toc = generate_table_of_contents(documentation_content)
        documentation_content = toc + "\n\n" + documentation_content

        # Write to file asynchronously
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(documentation_content)
        logger.info(f"Documentation written to '{file_path}' successfully.")
        return documentation_content

    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e}")
        return ''
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return ''
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return ''
    
def sanitize_text(text: str) -> str:
        """
        Sanitizes text for Markdown formatting.

        Args:
            text (str): The text to sanitize.

        Returns:
            str: Sanitized text.
        """
        markdown_special_chars = ['*', '_', '`', '~', '<', '>', '#']
        for char in markdown_special_chars:
            text = text.replace(char, f"\\{char}")
        return text.replace('|', '\\|').replace('\n', ' ').strip()

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
                        "async": {"type": "boolean"},
                        "complexity": {"type": "integer"}
                    },
                    "required": ["name", "args", "docstring", "async", "complexity"]
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
                                    "type": {"type": "string"},
                                    "complexity": {"type": "integer"}
                                },
                                "required": ["name", "args", "docstring", "async", "type", "complexity"]
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