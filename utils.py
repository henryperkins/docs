# utils.py

import os
import sys
import json
import fnmatch
import black
import logging
import aiohttp
import asyncio
import re
import subprocess
from dotenv import load_dotenv
from typing import Any, Set, List, Optional, Dict, Tuple
import tempfile  # For JS/TS extraction
import astor  # For Python docstring insertion
from bs4 import BeautifulSoup, Comment  # For HTML and CSS functions
import tinycss2  # For CSS functions
import openai
from jsonschema import validate, ValidationError
from openai import OpenAIError  # For OpenAI exception handling

# ----------------------------
# Configuration and Setup
# ----------------------------

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("documentation_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# Constants
# ----------------------------

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_EXCLUDED_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', '.idea'}
DEFAULT_EXCLUDED_FILES = {'.DS_Store'}
DEFAULT_SKIP_TYPES = {'.json', '.md', '.txt', '.csv', '.lock'}

LANGUAGE_MAPPING = {
    '.py': 'python',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.html': 'html',
    '.htm': 'html',
    '.css': 'css',
}

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
        with open(file_path, 'rb') as file:
            return b'\0' in file.read(1024)
    except Exception as e:
        logger.error(f"Error checking binary file '{file_path}': {e}")
        return True

def get_all_file_paths(repo_path: str, excluded_dirs: Set[str], excluded_files: Set[str], skip_types: Set[str]) -> List[str]:
    """Retrieves all file paths in the repository, excluding specified directories and files."""
    file_paths = []
    normalized_excluded_dirs = {os.path.normpath(os.path.join(repo_path, d)) for d in excluded_dirs}

    for root, dirs, files in os.walk(repo_path):
        normalized_root = os.path.normpath(root)
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
        with open(schema_path, 'r', encoding='utf-8') as f:
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

def load_function_schema() -> dict:
    """
    Loads the function schema from 'function_schema.json'.
    
    Returns:
        dict: Function schema.
    """
    schema_path = os.getenv('FUNCTION_SCHEMA_PATH', 'function_schema.json')
    schema = load_json_schema(schema_path)
    if not schema:
        logger.critical(f"Failed to load function schema from '{schema_path}'. Exiting.")
        sys.exit(1)
    return schema

# Initialize function_schema at module load
function_schema = load_function_schema()

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
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        project_info = config.get('project_info', '')
        style_guidelines = config.get('style_guidelines', '')
        excluded_dirs.update(config.get('excluded_dirs', []))
        excluded_files.update(config.get('excluded_files', []))
        skip_types.update(config.get('skip_types', []))
        logger.debug(f"Loaded configuration from '{config_path}'")
        return project_info, style_guidelines
    except FileNotFoundError:
        logger.error(f"Config file '{config_path}' not found.")
        return '', ''
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{config_path}': {e}")
        return '', ''
    except Exception as e:
        logger.error(f"Unexpected error loading config file '{config_path}': {e}")
        return '', ''

def extract_json_from_response(response: str) -> Optional[dict]:
    """Extracts JSON content from the model's response.

    Attempts multiple methods to extract JSON:
    1. Function calling format.
    2. JSON enclosed in triple backticks.
    3. Entire response as JSON.

    Args:
        response (str): The raw response string from the model.

    Returns:
        Optional[dict]: The extracted JSON as a dictionary, or None if extraction fails.
    """
    # First, try to extract JSON using the function calling format
    try:
        response_json = json.loads(response)
        if "function_call" in response_json and "arguments" in response_json["function_call"]:
            return json.loads(response_json["function_call"]["arguments"])
    except json.JSONDecodeError:
        pass  # Fallback to other extraction methods

    # Try to find JSON enclosed in triple backticks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # As a last resort, attempt to use the entire response if it's valid JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None
    
# ----------------------------
# OpenAI API Interaction
# ----------------------------

def call_openai_api(prompt: str, model: str, functions: List[dict], function_call: Optional[dict] = None) -> Optional[dict]:
    """
    Centralized function to call the OpenAI API.
    
    Args:
        prompt (str): The prompt to send.
        model (str): The OpenAI model to use.
        functions (List[dict]): List of function schemas.
        function_call (Optional[dict]): Function call parameters.
    
    Returns:
        Optional[dict]: The API response or None if failed.
    """
    openai.api_key = OPENAI_API_KEY
    if not OPENAI_API_KEY:
        logger.critical("OPENAI_API_KEY not set. Please set it in your environment or .env file.")
        sys.exit(1)
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an assistant that generates documentation."},
                {"role": "user", "content": prompt}
            ],
            functions=functions,
            function_call=function_call
        )
        logger.debug("OpenAI API call successful.")
        return response
    except OpenAIError as e:
        logger.error(f"OpenAI API Error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI API: {e}")
        return None
    
# ----------------------------
# Code Formatting and Cleanup
# ----------------------------

def format_with_black(code: str) -> str:
    """
    Formats the given Python code using Black.
    
    Args:
        code (str): The Python code to format.
    
    Returns:
        str: The formatted Python code.
    """
    try:
        formatted_code = black.format_str(code, mode=black.FileMode())
        logger.debug("Successfully formatted code with Black.")
        return formatted_code
    except black.NothingChanged:
        logger.debug("No changes made by Black; code is already formatted.")
        return code
    except Exception as e:
        logger.error(f"Error formatting code with Black: {e}")
        return code  # Return the original code if formatting fails

def clean_unused_imports(code: str) -> str:
    """
    Removes unused imports from Python code using autoflake.
    
    Args:
        code (str): The Python code to clean.
    
    Returns:
        str: The cleaned Python code.
    """
    try:
        cleaned_code = subprocess.check_output(
            ['autoflake', '--remove-all-unused-imports', '--stdout'],
            input=code.encode('utf-8'),
            stderr=subprocess.STDOUT
        )
        logger.debug("Successfully removed unused imports with autoflake.")
        return cleaned_code.decode('utf-8')
    except subprocess.CalledProcessError as e:
        logger.error(f"Autoflake failed: {e.output.decode('utf-8')}")
        return code  # Return original code if autoflake fails
    except FileNotFoundError:
        logger.error("Autoflake is not installed. Please install it using 'pip install autoflake'.")
        return code
    except Exception as e:
        logger.error(f"Error cleaning imports with autoflake: {e}")
        return code

def check_with_flake8(file_path: str) -> bool:
    """
    Checks Python code compliance using flake8 and attempts to fix issues if found.
    
    Args:
        file_path (str): Path to the Python file to check.
    
    Returns:
        bool: True if the code passes flake8 checks after fixes, False otherwise.
    """
    logger.debug(f"Entering check_with_flake8 with file_path={file_path}")
    result = subprocess.run(["flake8", file_path], capture_output=True, text=True)
    if result.returncode == 0:
        logger.debug(f"No flake8 issues in {file_path}")
        return True
    else:
        logger.error(f"flake8 issues in {file_path}:\n{result.stdout}")
        # Attempt to auto-fix with autoflake and black
        try:
            logger.info(f"Attempting to auto-fix flake8 issues in {file_path}")
            subprocess.run(['autoflake', '--remove-all-unused-imports', '--in-place', file_path], check=True)
            subprocess.run(['black', '--quiet', file_path], check=True)
            # Re-run flake8 to confirm
            result = subprocess.run(["flake8", file_path], capture_output=True, text=True)
            if result.returncode == 0:
                logger.debug(f"No flake8 issues after auto-fix in {file_path}")
                return True
            else:
                logger.error(f"flake8 issues remain after auto-fix in {file_path}:\n{result.stdout}")
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Auto-fix failed for {file_path}: {e}", exc_info=True)
            return False

def run_flake8(file_path: str) -> Optional[str]:
    """
    Runs flake8 on the specified file and returns the output.

    Parameters:
        file_path (str): Path to the Python file to check.

    Returns:
        Optional[str]: The flake8 output if any issues are found, else None.
    """
    try:
        result = subprocess.run(
            ["flake8", file_path],
            capture_output=True,
            text=True,
            check=False  # Do not raise exception on non-zero exit
        )
        if result.stdout:
            return result.stdout.strip()
        return None
    except Exception as e:
        logger.error(f"Error running flake8 on '{file_path}': {e}", exc_info=True)
        return None

def run_node_script(script_path: str, input_code: str) -> Optional[Dict[str, any]]:
    """
    Runs a Node.js script that outputs JSON (e.g., extract_structure.js) and returns the parsed JSON.

    Parameters:
        script_path (str): Path to the Node.js script.
        input_code (str): The code to process.

    Returns:
        Optional[Dict[str, Any]]: The JSON output from the script if successful, None otherwise.
    """
    try:
        logger.debug(f"Running Node.js script: {script_path}")
        result = subprocess.run(
            ['node', 'scripts/extract_structure.js'],
            input=input_code,
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"Successfully ran {script_path}")
        output_json = json.loads(result.stdout)
        return output_json
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON output from {script_path}: {e}")
        return None
    except FileNotFoundError:
        logger.error(f"Node.js script {script_path} not found.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error running {script_path}: {e}")
        return None

def run_node_insert_docstrings(script_path: str, input_code: str) -> Optional[str]:
    """
    Runs the insert_docstrings.js script and returns the modified code.

    Parameters:
        script_path (str): Path to the insert_docstrings.js script.
        input_code (str): The code to process.

    Returns:
        Optional[str]: The modified code if successful, None otherwise.
    """
    try:
        logger.debug(f"Running Node.js script: {script_path}")
        result = subprocess.run(
            ['node', 'scripts/insert_docstrings.js'],
            input=input_code,
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"Successfully ran {script_path}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error(f"Node.js script {script_path} not found.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error running {script_path}: {e}")
        return None
    
async def fetch_documentation(
    session: aiohttp.ClientSession,
    prompt: str,
    semaphore: asyncio.Semaphore,
    model_name: str,
    function_schema: dict,
    retry: int = 3
) -> Optional[dict]:
    """
    Fetches documentation from OpenAI's API with optional retries.

    Parameters:
        session (aiohttp.ClientSession): The HTTP session.
        prompt (str): The prompt to send to the AI.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrency.
        model_name (str): The AI model to use.
        function_schema (dict): The function schema for structured responses.
        retry (int): Number of retry attempts.

    Returns:
        Optional[dict]: The structured documentation or None if failed.
    """
    for attempt in range(1, retry + 1):
        async with semaphore:
            try:
                headers = {
                    'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
                    'Content-Type': 'application/json'
                }
                payload = {
                    'model': model_name,
                    'messages': [
                        {'role': 'system', 'content': 'You are a helpful assistant that generates code documentation.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    'functions': [function_schema],
                    'function_call': 'auto'
                }
                # Log the payload for debugging
                logger.debug(f"API Payload: {json.dumps(payload, indent=2)}")
                async with session.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload) as resp:
                    response_text = await resp.text()
                    if resp.status != 200:
                        logger.error(f'OpenAI API request failed with status {resp.status}: {response_text}')
                        continue
                    response = await resp.json()
                    logger.debug(f"API Response: {json.dumps(response, indent=2)}")
                    choice = response.get('choices', [])[0]
                    message = choice.get('message', {})
                    if 'function_call' in message:
                        arguments = message['function_call'].get('arguments', '{}')
                        try:
                            documentation = json.loads(arguments)
                            logger.debug('Received documentation via function_call.')
                            return documentation
                        except json.JSONDecodeError as e:
                            logger.error(f'Error decoding JSON from function_call arguments: {e}')
                            logger.error(f'Arguments Content: {arguments}')
                            return None
                    else:
                        logger.error('No function_call found in the response.')
                        return None
            except Exception as e:
                logger.error(f'Error fetching documentation from OpenAI API: {e}')
                if attempt < retry:
                    logger.info(f'Retrying... (Attempt {attempt}/{retry})')
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error('All retry attempts failed.')
                    return None
    return None


def generate_documentation_prompt(
    file_name: str,
    code_structure: Dict[str, Any],
    project_info: Optional[str],
    style_guidelines: Optional[str],
    language: str
) -> str:
    """
    Generates a documentation prompt based on the code structure and other details.

    Parameters:
        file_name (str): The name of the file.
        code_structure (Dict[str, Any]]): The structure of the code.
        project_info (Optional[str]): Information about the project.
        style_guidelines (Optional[str]): Style guidelines for the documentation.
        language (str): Programming language.

    Returns:
        str: The generated prompt.
    """
    prompt = (
        'You are an experienced software developer tasked with generating comprehensive documentation for a specific file in a codebase.'
    )
    if project_info:
        prompt += f'\n\n**Project Information:**\n{project_info}'
    if style_guidelines:
        prompt += f'\n\n**Style Guidelines:**\n{style_guidelines}'

    prompt += f'\n\n**File Name:** {file_name}'
    prompt += f'\n\n**Language:** {language}'
    prompt += (
        f'\n\n**Code Structure:**\n```json\n{json.dumps(code_structure, indent=2)}\n```'
    )
    prompt += """
    
    **Instructions:** Based on the above code structure, generate the following documentation sections specifically for this file:
    1. **Summary:** A detailed summary of this file, including its purpose, key components, and how it integrates with the overall project.
    2. **Changes Made:** A comprehensive list of changes or updates made to this file.
    3. **Functions:** Detailed documentation for each function, including its purpose, arguments, return values, and whether it is asynchronous.
    4. **Classes:** Detailed documentation for each class, including its purpose, methods, and any inheritance details.
    
    **Please ensure that the documentation is clear, detailed, and adheres to the provided style guidelines.**"""
    return prompt
