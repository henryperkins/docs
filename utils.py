# utils.py

import os
import sys
import json
import logging
import aiohttp
import asyncio
import subprocess
from dotenv import load_dotenv
from typing import Any, Set, List, Optional, Dict, Tuple
from jsonschema import validate, ValidationError

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
# Code Formatting and Cleanup
# ----------------------------

def format_table(headers: list, rows: list) -> str:
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        table += "| " + " | ".join(row) + " |\n"
    return table

def truncate_description(description: str, max_length: int = 100) -> str:
    return (description[:max_length] + '...') if len(description) > max_length else description


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
