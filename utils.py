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
from bs4 import BeautifulSoup, Comment
from jsonschema import validate, ValidationError
from logging.handlers import RotatingFileHandler
import openai
from openai.error import OpenAIError, APIError, APIConnectionError, RateLimitError

# ----------------------------
# Configuration and Setup
# ----------------------------

# Load environment variables from .env file
load_dotenv()

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

def configure_openai(use_azure: bool, deployment_name: str):
    """
    Configures the OpenAI client for either Azure OpenAI or standard OpenAI API.

    Args:
        use_azure (bool): Flag indicating whether to use Azure OpenAI.
        deployment_name (str): Deployment name for Azure OpenAI or model name for standard OpenAI.
    """
    if use_azure:
        AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("ENDPOINT_URL")
        API_VERSION = os.getenv("API_VERSION")

        if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, API_VERSION, deployment_name]):
            logger.critical(
                "Azure OpenAI environment variables are not set properly. "
                "Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, API_VERSION, and DEPLOYMENT_NAME."
            )
            sys.exit(1)

        # Configure the OpenAI library for Azure
        openai.api_type = "azure"
        openai.api_key = AZURE_OPENAI_API_KEY
        openai.api_base = AZURE_OPENAI_ENDPOINT
        openai.api_version = API_VERSION
        logger.debug("Configured OpenAI for Azure.")
        logger.debug(f"api_type: {openai.api_type}")
        logger.debug(f"api_base: {openai.api_base}")
        logger.debug(f"api_version: {openai.api_version}")
    else:
        # OpenAI configuration
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        if not OPENAI_API_KEY:
            logger.critical(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it in your environment or .env file."
            )
            sys.exit(1)

        openai.api_key = OPENAI_API_KEY
        logger.debug("Configured OpenAI for standard API.")
        logger.debug(f"api_key: {'***' * 4} (hidden)")

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
    """
    Loads the function schema.

    Args:
        schema_path (str): Path to the function schema JSON file.

    Returns:
        dict: Function schema.
    """
    logger.debug(f"Attempting to load function schema from '{schema_path}'.")
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
# Model Validation
# ----------------------------

def validate_model_name(model_name: str, use_azure: bool = False) -> bool:
    """
    Validates the provided model name against a list of supported OpenAI models.

    Args:
        model_name (str): The name of the OpenAI model to validate.
        use_azure (bool, optional): Flag indicating whether Azure OpenAI is being used.
                                     Defaults to False.

    Returns:
        bool: True if the model name is valid and supported, False otherwise.
    """
    # Define a list of supported models for the standard OpenAI API
    supported_models = [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-32k",
        # Add other supported models as needed
    ]
    
    if use_azure:
        # When using Azure OpenAI, model validation is different because
        # deployment names are used instead of model names.
        # Here, we assume that deployment names are correctly configured
        # and skip model name validation.
        logger.debug("Azure OpenAI is enabled; skipping model name validation.")
        return True
    else:
        # Validate the model name against the supported models list
        if model_name in supported_models:
            logger.debug(f"Model name '{model_name}' is valid and supported.")
            return True
        else:
            logger.error(f"Model name '{model_name}' is not supported.")
            return False

# ----------------------------
# OpenAI API Interaction
# ----------------------------

async def fetch_documentation(
    session: aiohttp.ClientSession,
    prompt: str,
    semaphore: asyncio.Semaphore,
    deployment_name: str,
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
        deployment_name (str): Deployment name for Azure OpenAI or model name for standard OpenAI.
        function_schema (dict): The function schema for structured responses.
        retry (int, optional): Number of retry attempts on failure. Defaults to 3.
        use_azure (bool, optional): Whether to use Azure OpenAI API. Defaults to False.

    Returns:
        Optional[dict]: The documentation as a dictionary if successful, else None.
    """
    logger.debug(f"Fetching documentation for deployment/model: {deployment_name}, use_azure: {use_azure}")
    for attempt in range(1, retry + 1):
        async with semaphore:
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that generates code documentation."},
                    {"role": "user", "content": prompt},
                ]

                if use_azure:
                    # Use 'deployment_id' parameter for Azure OpenAI
                    logger.debug("Using Azure OpenAI deployment.")
                    response = await openai.ChatCompletion.acreate(
                        deployment_id=deployment_name,  # deployment_name is your deployment ID
                        messages=messages,
                        functions=[function_schema],
                        function_call="auto",
                    )
                else:
                    # Use 'model' parameter for OpenAI API
                    logger.debug("Using standard OpenAI model.")
                    response = await openai.ChatCompletion.acreate(
                        model=deployment_name,  # For standard OpenAI, 'deployment_name' acts as 'model'
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
            except RateLimitError as e:
                logger.error(f"OpenAI API rate limit exceeded: {e}")
                if attempt < retry:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying after {wait_time} seconds... (Attempt {attempt}/{retry})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("All retry attempts failed due to rate limiting.")
                    return None
            except APIConnectionError as e:
                logger.error(f"Failed to connect to OpenAI API: {e}")
                if attempt < retry:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying after {wait_time} seconds... (Attempt {attempt}/{retry})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("All retry attempts failed due to connection errors.")
                    return None
            except APIError as e:
                logger.error(f"OpenAI API returned an API Error: {e}")
                if attempt < retry:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying after {wait_time} seconds... (Attempt {attempt}/{retry})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("All retry attempts failed due to API errors.")
                    return None
            except OpenAIError as e:
                logger.error(f"An OpenAI error occurred: {e}")
                return None
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}", exc_info=True)
                return None
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

    Raises:
        Exception: If Black cannot format the code.
    """
    try:
        formatted_code = black.format_str(code, mode=black.Mode())
        logger.debug('Successfully formatted code with Black.')
        return formatted_code
    except black.NothingChanged:
        logger.debug('No changes made by Black; code is already formatted.')
        return code
    except Exception as e:
        logger.error(f'Error formatting code with Black: {e}')
        raise e  # Raise exception to be handled by the caller

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
            ["autoflake", "--remove-all-unused-imports", "--stdout", "-"],
            input=code.encode("utf-8"),
            stderr=subprocess.STDOUT,
        )
        logger.debug("Successfully removed unused imports with autoflake.")
        return cleaned_code.decode("utf-8")
    except subprocess.CalledProcessError as e:
        logger.error(f"Autoflake failed: {e.output.decode('utf-8')}")
        return code  # Return original code if autoflake fails
    except FileNotFoundError:
        logger.error("Autoflake is not installed. Please install it using 'pip install autoflake'.")
        return code
    except Exception as e:
        logger.error(f"Error cleaning imports with autoflake: {e}")
        return code

def run_flake8(file_path: str) -> Optional[str]:
    """
    Runs flake8 on the specified file and returns the output.

    Args:
        file_path (str): Path to the Python file to check.

    Returns:
        Optional[str]: The flake8 output if any issues are found, else None.
    """
    try:
        result = subprocess.run(
            ["flake8", file_path],
            capture_output=True,
            text=True,
            check=False,  # Do not raise exception on non-zero exit
        )
        if result.stdout:
            return result.stdout.strip()
        return None
    except Exception as e:
        logger.error(f"Error running flake8 on '{file_path}': {e}", exc_info=True)
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
    """
    Creates a tailored documentation prompt based on various parameters like file name and code structure,
    aligning with project info and guidelines.

    Args:
        file_name (str): The name of the file for which documentation is being generated.
        code_structure (Dict[str, Any]): The structure of the code in the file, typically in JSON format.
        project_info (Optional[str]): Information about the project, if available.
        style_guidelines (Optional[str]): Documentation style guidelines to follow.
        language (str): The programming language of the file.

    Returns:
        str: A comprehensive prompt for generating documentation.
    """
    prompt = (
        "You are an experienced software developer tasked with generating comprehensive documentation "
        "for a specific file in a codebase."
    )
    if project_info:
        prompt += f"\n\n**Project Information:**\n{project_info}"
    if style_guidelines:
        prompt += f"\n\n**Style Guidelines:**\n{style_guidelines}"
    prompt += f"\n\n**File Name:** {file_name}"
    prompt += f"\n\n**Language:** {language}"
    prompt += f"\n\n**Code Structure:**\n```json\n{json.dumps(code_structure, indent=2)}\n```"
    prompt += """
**Instructions:** Based on the above code structure, generate the following documentation sections specifically for this file:
1. **Overview:** A high-level overview of the module or class, explaining its purpose, responsibilities, and integration within the project.
2. **Summary:** A detailed summary of this file, including its purpose, key components, and how it integrates with the overall project.
3. **Changes Made:** A comprehensive list of changes or updates made to this file.
4. **Functions:** Provide detailed documentation for each function, including its purpose, parameters (`@param`), return values (`@returns` or `@return`), and whether it is asynchronous.
5. **Classes:** Provide detailed documentation for each class, including its purpose, methods, inheritance details (`@extends` or `@implements`), and any interfaces it implements. Also, provide documentation for each method within the class.
"""
    return prompt

async def write_documentation_report(
    documentation: Optional[Dict[str, Any]],
    language: str,
    file_path: str,
    repo_root: str,
    new_content: str,
) -> str:
    """
    Generates the documentation report content for a single file.

    Args:
        documentation (Optional[Dict[str, Any]]): Documentation data generated by OpenAI.
        language (str): Programming language of the file.
        file_path (str): Path to the file.
        repo_root (str): Root path of the repository.
        new_content (str): Content of the file after processing.

    Returns:
        str: Markdown-formatted documentation for the file.
    """
    try:
        def sanitize_text(text: str) -> str:
            """Removes excessive newlines and whitespace from the text."""
            if not text:
                return ""
            lines = text.strip().splitlines()
            sanitized_lines = [line.strip() for line in lines if line.strip()]
            return "\n".join(sanitized_lines)

        relative_path = os.path.relpath(file_path, repo_root)
        file_header = f"# File: {relative_path}\n\n"
        documentation_content = file_header

        # Summary Section
        summary = documentation.get("summary", "") if documentation else ""
        summary = sanitize_text(summary)
        if summary:
            summary_section = f"## Summary\n\n{summary}\n"
            documentation_content += summary_section

        # Changes Made Section
        changes = documentation.get("changes_made", []) if documentation else []
        changes = [sanitize_text(change) for change in changes if change.strip()]
        if changes:
            changes_formatted = "\n".join(f"- {change}" for change in changes)
            changes_section = f"## Changes Made\n\n{changes_formatted}\n"
            documentation_content += changes_section

        # Functions Section
        functions = documentation.get("functions", []) if documentation else []
        if functions:
            functions_section = "## Functions\n\n"
            functions_section += "| Function | Arguments | Description | Async |\n"
            functions_section += "|----------|-----------|-------------|-------|\n"
            for func in functions:
                func_name = func.get("name", "N/A")
                func_args = ", ".join(func.get("args", []))
                func_doc = sanitize_text(func.get("docstring", ""))
                first_line_doc = (
                    func_doc.splitlines()[0] if func_doc else "No description provided."
                )
                func_async = "Yes" if func.get("async", False) else "No"
                functions_section += f"| `{func_name}` | `{func_args}` | {first_line_doc} | {func_async} |\n"
            documentation_content += functions_section + "\n"

        # Classes Section
        classes = documentation.get("classes", []) if documentation else []
        if classes:
            classes_section = "## Classes\n\n"
            for cls in classes:
                cls_name = cls.get("name", "N/A")
                cls_doc = sanitize_text(cls.get("docstring", ""))
                if cls_doc:
                    classes_section += f"### Class: `{cls_name}`\n\n{cls_doc}\n\n"
                else:
                    classes_section += f"### Class: `{cls_name}`\n\n"

                methods = cls.get("methods", [])
                if methods:
                    classes_section += "| Method | Arguments | Description | Async | Type |\n"
                    classes_section += "|--------|-----------|-------------|-------|------|\n"
                    for method in methods:
                        method_name = method.get("name", "N/A")
                        method_args = ", ".join(method.get("args", []))
                        method_doc = sanitize_text(method.get("docstring", ""))
                        first_line_method_doc = (
                            method_doc.splitlines()[0] if method_doc else "No description provided."
                        )
                        method_async = "Yes" if method.get("async", False) else "No"
                        method_type = method.get("type", "N/A")
                        classes_section += f"| `{method_name}` | `{method_args}` | {first_line_method_doc} | {method_async} | {method_type} |\n"
                    classes_section += "\n"
            documentation_content += classes_section

        # Code Block
        code_content = new_content.strip()
        code_block = f"```{language}\n{code_content}\n```\n\n---\n"
        documentation_content += code_block

        return documentation_content
    except Exception as e:
        logger.error(
            f"Error generating documentation for '{file_path}': {e}", exc_info=True
        )
        return ""

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
