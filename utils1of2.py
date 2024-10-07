# utils.py

import os
import sys
import json
import logging
import aiohttp
import asyncio
import re
import subprocess
import black
from dotenv import load_dotenv
from typing import Any, Set, List, Optional, Dict, Tuple
import tempfile
from bs4 import BeautifulSoup, Comment
from jsonschema import validate, ValidationError
from logging.handlers import RotatingFileHandler
from openai import OpenAIError, APIError, APIConnectionError, RateLimitError

# ----------------------------
# Configuration and Setup
# ----------------------------

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("documentation_generation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ----------------------------
# Constants
# ----------------------------

DEFAULT_EXCLUDED_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".idea"}
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
# OpenAI API Configuration
# ----------------------------

# Determine whether to use Azure OpenAI based on environment variable
use_azure = os.getenv("USE_AZURE", "false").lower() == "true"

if use_azure:
    # Azure OpenAI configuration
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("ENDPOINT_URL")
    API_VERSION = os.getenv("API_VERSION")
    DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, API_VERSION, DEPLOYMENT_NAME]):
        logger.error("Azure OpenAI environment variables are not set properly.")
        sys.exit(1)

    from openai import AsyncAzureOpenAI

    aclient = AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=API_VERSION
    )
else:
    # OpenAI configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    from openai import AsyncOpenAI

    aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# Language and File Utilities
# ----------------------------

def get_language(ext: str) -> str:
    """Determines the programming language based on file extension."""
    language = LANGUAGE_MAPPING.get(ext.lower(), "plaintext")
    logger.debug(f"Detected language for extension '{ext}': {language}")
    return language

def is_valid_extension(ext: str, skip_types: Set[str]) -> bool:
    """Checks if a file extension is valid (not in the skip list)."""
    is_valid = ext.lower() not in skip_types
    logger.debug(f"Extension '{ext}' is valid: {is_valid}")
    return is_valid

def is_binary(file_path: str) -> bool:
    """Checks if a file is binary."""
    try:
        with open(file_path, "rb") as file:
            return b"\0" in file.read(1024)
    except Exception as e:
        logger.error(f"Error checking if file is binary '{file_path}': {e}")
        return True

def get_all_file_paths(repo_path: str, excluded_dirs: Set[str], excluded_files: Set[str], skip_types: Set[str]) -> List[str]:
    """Retrieves all file paths in the repository, excluding specified directories and files."""
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
        logger.debug(f"Successfully loaded JSON schema from '{schema_path}'")
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
    """
    Loads the function schema.

    Args:
        schema_path (str): Path to the function schema JSON file.

    Returns:
        dict: Function schema.
    """
    logger.debug(f"Attempting to load function schema from '{schema_path}'")
    schema = load_json_schema(schema_path)
    if not schema:
        logger.critical(f"Failed to load function schema from '{schema_path}'. Exiting.")
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
        logger.debug(f"Loaded configuration from '{config_path}'")
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
# OpenAI API Interaction
# ----------------------------

async def fetch_documentation(
    session: aiohttp.ClientSession,
    prompt: str,
    semaphore: asyncio.Semaphore,
    model_name: str,
    function_schema: dict,
    retry: int = 3,
    use_azure: bool = False
) -> Optional[dict]:
    """
    Fetches documentation from the OpenAI or Azure OpenAI API based on the provided prompt.

    Args:
        session (aiohttp.ClientSession): The HTTP session.
        prompt (str): The prompt to send to the API.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrency.
        model_name (str): The OpenAI model or Azure deployment ID.
        function_schema (dict): The function schema for structured responses.
        retry (int, optional): Number of retry attempts on failure. Defaults to 3.
        use_azure (bool, optional): Whether to use Azure OpenAI API. Defaults to False.

    Returns:
        Optional[dict]: The documentation as a dictionary if successful, else None.
    """
    for attempt in range(1, retry + 1):
        async with semaphore:
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that generates code documentation."},
                    {"role": "user", "content": prompt},
                ]

                if use_azure:
                    # Use 'model' parameter for Azure OpenAI
                    response = await aclient.chat.completions.create(
                        model=model_name,  # model_name is your deployment ID
                        messages=messages,
                        functions=[function_schema],
                        function_call="auto",
                    )
                else:
                    # Use 'model' parameter for OpenAI API
                    response = await aclient.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        functions=[function_schema],
                        function_call="auto",
                    )

                logger.debug(f"API Response: {response}")
                choice = response.choices[0]
                message = choice.message
                if message.get("function_call"):
                    arguments = message["function_call"].get("arguments")
                    try:
                        documentation = json.loads(arguments)
                        logger.debug("Received documentation via function_call.")
                        return documentation
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON from function_call arguments: {e}")
                        logger.error(f"Arguments Content: {arguments}")
                        return None
                else:
                    logger.error("No function_call found in the response.")
                    return None
            except APIError as e:
                logger.error(f"OpenAI API returned an API Error: {e}")
                if attempt < retry:
                    logger.info(f"Retrying... (Attempt {attempt}/{retry})")
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error("All retry attempts failed.")
                    return None
            except APIConnectionError as e:
                logger.error(f"Failed to connect to OpenAI API: {e}")
                if attempt < retry:
                    logger.info(f"Retrying... (Attempt {attempt}/{retry})")
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error("All retry attempts failed.")
                    return None
            except RateLimitError as e:
                logger.error(f"OpenAI API request exceeded rate limit: {e}")
                if attempt < retry:
                    logger.info(f"Retrying after rate limit error... (Attempt {attempt}/{retry})")
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error("All retry attempts failed due to rate limit.")
                    return None
            except OpenAIError as e:
                logger.error(f"An OpenAI error occurred: {e}")
                return None
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}", exc_info=True)
                return None
    return None

# ----------------------------
# Rest of Your Functions
# ----------------------------

# Include all other functions from your utils.py here, fully implemented, with no placeholders or incomplete code.

# ----------------------------
# EOF
# ----------------------------
