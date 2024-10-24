"""
utils.py

This module provides utility functions for handling language and file operations, configuration management, code formatting, and cleanup. It includes functions for loading JSON schemas, managing file paths, and running code formatters like autoflake, black, and flake8.
"""

import os
import sys
import json
import logging
import asyncio
import subprocess
import pathspec
from dotenv import load_dotenv
from typing import Any, Set, List, Optional, Dict, Tuple
from jsonschema import Draft7Validator, ValidationError, SchemaError
import aiohttp # Import aiohttp

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

DEFAULT_COMPLEXITY_THRESHOLDS = {"low": 10, "medium": 20, "high": 30}

DEFAULT_HALSTEAD_THRESHOLDS = {
    "volume": {"low": 100, "medium": 500, "high": 1000},
    "difficulty": {"low": 10, "medium": 20, "high": 30},
    "effort": {"low": 500, "medium": 1000, "high": 2000}
}

DEFAULT_MAINTAINABILITY_THRESHOLDS = {"low": 50, "medium": 70, "high": 85}

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

def get_threshold(metric: str, key: str, default: int) -> int:
    """
    Retrieves the threshold value for a given metric and key from environment variables.

    Args:
        metric (str): The metric name.
        key (str): The threshold key (e.g., 'low', 'medium', 'high').
        default (int): The default value if the environment variable is not set or invalid.

    Returns:
        int: The threshold value.
    """
    try:
        return int(os.getenv(f"{metric.upper()}_{key.upper()}_THRESHOLD", default))
    except ValueError:
        logger.error(f"Invalid environment variable for {metric.upper()}_{key.upper()}_THRESHOLD")
        return default
    
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

def should_process_file(file_path: str, skip_types: Set[str]) -> bool:  # Add skip_types argument
    """Determines if a file should be processed."""

    if not os.path.exists(file_path):
        return False  # Don't process non-existent files

    _, ext = os.path.splitext(file_path)

    # Combine all skip conditions
    if (
        os.path.islink(file_path) or
        any(part in file_path for part in ['node_modules', '.bin', '.git',  '__pycache__', 'build', 'dist']) or # Add .git and other common directories
        file_path.endswith('.d.ts') or
        ext in {'.flake8', '.gitignore', '.env', '.pyc', '.pyo', '.pyd', '.git', '.d.ts'} or
        ext in skip_types or not ext or is_binary(file_path)
    ):
        logger.debug(f"Skipping file '{file_path}'")
        return False

    return True

def load_gitignore(repo_path: str) -> pathspec.PathSpec:
    """
    Loads .gitignore patterns into a PathSpec object.
    
    Args:
        repo_path (str): Path to the repository root
        
    Returns:
        pathspec.PathSpec: Compiled gitignore patterns
    """
    gitignore_path = os.path.join(repo_path, '.gitignore')
    patterns = []
    
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
    
    return pathspec.PathSpec.from_lines(
        pathspec.patterns.GitWildMatchPattern, patterns
    )

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
    
    # Add common node_modules patterns to excluded dirs
    node_modules_patterns = {
        'node_modules',
        '.bin',
        'node_modules/.bin',
        '**/node_modules/**/.bin',
        '**/node_modules/**/node_modules'
    }
    normalized_excluded_dirs.update({os.path.normpath(os.path.join(repo_path, d)) for d in node_modules_patterns})

    for root, dirs, files in os.walk(repo_path, topdown=True):
        # Skip node_modules and other excluded directories
        if any(excluded in root for excluded in ['node_modules', '.bin']):
            dirs[:] = []  # Skip processing subdirectories
            continue

        # Exclude directories
        dirs[:] = [
            d for d in dirs 
            if os.path.normpath(os.path.join(root, d)) not in normalized_excluded_dirs
            and not any(excluded in d for excluded in ['node_modules', '.bin'])
        ]

        for file in files:
            # Skip excluded files
            if file in excluded_files:
                continue
                
            # Skip specified file types
            file_ext = os.path.splitext(file)[1]
            if file_ext in skip_types:
                continue

            # Skip symlinks
            full_path = os.path.join(root, file)
            if os.path.islink(full_path):
                continue

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

def load_function_schema(schema_path: str) -> Dict[str, Any]:
    """Loads and validates the function schema."""
    # Removed redundant call to validate_schema
    schema = load_json_schema(schema_path) # Load the schema
    if schema is None:
        raise ValueError("Invalid or missing schema file.")
    if "functions" not in schema:
        raise ValueError("Schema missing 'functions' key.")
    Draft7Validator.check_schema(schema) # Validate the schema
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

async def format_with_black_async(code: str) -> str:
    """
    Asynchronously formats the provided code using Black.

    Args:
        code (str): The source code to format.

    Returns:
        str: The formatted code.
    """
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
    """
    Runs a Node.js script asynchronously.

    Args:
        script_path (str): Path to the Node.js script.
        input_json (str): JSON string to pass as input to the script.

    Returns:
        Optional[str]: The output from the script if successful, None otherwise.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            'node', script_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate(input=input_json.encode())

        if proc.returncode != 0:
            logger.error(f"Node.js script '{script_path}' failed: {stderr.decode()}")
            return None

        return stdout.decode()
    except FileNotFoundError:
        logger.error("Node.js is not installed or not in PATH. Please install Node.js.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while running Node.js script '{script_path}': {e}")
        return None
        
async def run_node_insert_docstrings(script_name: str, input_data: dict) -> Optional[str]:
    """Runs a Node.js script to insert docstrings asynchronously."""
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'scripts', script_name)
        logger.debug(f"Running Node.js script: {script_path}")

        input_json = json.dumps(input_data)

        proc = await asyncio.create_subprocess_exec(
            'node', script_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await proc.communicate(input=input_json.encode())

        if proc.returncode != 0:
            logger.error(f"Error running {script_name}: {stderr.decode()}")
            return None

        logger.debug(f"Successfully ran {script_path}")
        return stdout.decode()

    except FileNotFoundError:
        logger.error(f"Node.js script {script_name} not found.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error running {script_name}: {e}")
        return None


# ----------------------------
# Documentation Generation
# ----------------------------

def calculate_project_metrics(successful_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates aggregate project metrics."""

    if not successful_results:
        logger.warning("No successful results to calculate project metrics.")  # Add warning
        return {  # Return default metrics
            "maintainability_index": 0,
            "complexity": 0,
            "halstead": {
                "volume": 0,
                "difficulty": 0,
                "effort": 0,
            },
        }

    total_files = len(successful_results)
    metric_sums = {  # Use a dictionary to store sums
        "maintainability_index": 0,
        "complexity": 0,
        "halstead_volume": 0,
        "halstead_difficulty": 0,
        "halstead_effort": 0,
    }

    for result in successful_results:
        metrics = result.get("metrics", {selectedText})
        for metric_name in metric_sums:
            try:
                value = float(metrics.get(metric_name.replace("halstead_", "halstead."))) # Handle nested Halstead metrics
                metric_sums[metric_name] += value
            except (TypeError, ValueError):
                logger.warning(f"Invalid value for {metric_name} in file {result.get('file_path', 'unknown')}: {metrics.get(metric_name)}")

    # Calculate averages and aggregate metrics
    avg_maintainability = metric_sums["maintainability_index"] / total_files if total_files else 0
    
    return {
        "maintainability_index": avg_maintainability,
        "complexity": metric_sums["complexity"],
        "halstead": {
            "volume": metric_sums["halstead_volume"],
            "difficulty": metric_sums["halstead_difficulty"] / total_files if total_files else 0,
            "effort": metric_sums["halstead_effort"],
        },
        # ... (Add other aggregate metrics here)
    }

def validate_schema(schema_path: str) -> Optional[Dict[str, Any]]:
    """
    Validates a JSON schema and loads it if valid.

    Args:
        schema_path (str): Path to the JSON schema file.

    Returns:
        Optional[Dict[str, Any]]: The loaded schema if valid, None otherwise.
    """
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        try:
            Draft7Validator.check_schema(schema)
            logger.debug("Schema is valid.")
            return schema
        except (SchemaError, ValidationError) as e:
            logger.error(f"Invalid schema: {e}")
            return None
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading schema file: {e}")
        return None
    
# ----------------------------
# EOF
# ----------------------------
