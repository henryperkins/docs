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
import tinycss2
import argparse
from jsonschema import validate, ValidationError
from logging.handlers import RotatingFileHandler
import openai
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

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
        with open(file_path, "rb") as file:
            return b"\0" in file.read(1024)
    except Exception as e:
        logger.error(f"Error checking binary file '{file_path}': {e}")
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
    try:
        response_json = json.loads(response)
        if (
            "function_call" in response_json
            and "arguments" in response_json["function_call"]
        ):
            return json.loads(response_json["function_call"]["arguments"])
    except json.JSONDecodeError:
        pass
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None

# ----------------------------
# OpenAI API Interaction
# ----------------------------

def call_openai_api(prompt: str, model: str, functions: List[dict], function_call: Optional[dict] = None, use_azure: bool = False) -> Optional[dict]:
    """
    Centralized function to call the OpenAI API or Azure OpenAI API.

    Args:
        prompt (str): The prompt to send.
        model (str): The OpenAI model to use, or deployment ID for Azure.
        functions (List[dict]): List of function schemas.
        function_call (Optional[dict]): Function call parameters.
        use_azure (bool): Whether to use Azure OpenAI API instead of regular OpenAI API.

    Returns:
        Optional[dict]: The API response or None if failed.
    """
    if not check_api_keys(use_azure):
        return None

    try:
        messages = [
            {"role": "system", "content": "You are an assistant that generates documentation."},
            {"role": "user", "content": prompt},
        ]

        if use_azure:
            # Use 'engine' parameter for Azure OpenAI
            response = openai.ChatCompletion.create(
                engine=model,  # 'model' is the deployment ID in Azure
                messages=messages,
                functions=functions,
                function_call=function_call,
            )
        else:
            # Use 'model' parameter for OpenAI API
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                functions=functions,
                function_call=function_call,
            )

        logger.debug("API call successful.")
        return response
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API Error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling API: {e}")
        return None



logger = logging.getLogger(__name__)

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
                    # Use 'engine' parameter for Azure OpenAI
                    response = openai.ChatCompletion.create(
                        engine=model_name,  # model_name is your deployment ID
                        messages=messages,
                        functions=[function_schema],
                        function_call="auto",
                    )
                else:
                    # Use 'model' parameter for OpenAI API
                    response = openai.ChatCompletion.create(
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
            subprocess.run(["autoflake", "--remove-all-unused-imports", "--in-place", file_path], check=True)
            subprocess.run(["black", "--quiet", file_path], check=True)
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
        except FileNotFoundError as e:
            logger.error(f"Required tool not found: {e}")
            return False
        except Exception as e:
            logger.error(f"Error auto-fixing flake8 issues for {file_path}: {e}", exc_info=True)
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
            check=False,  # Do not raise exception on non-zero exit
        )
        if result.stdout:
            return result.stdout.strip()
        return None
    except Exception as e:
        logger.error(f"Error running flake8 on '{file_path}': {e}", exc_info=True)
        return None

import os

def run_node_script(script_path: str, input_code: str) -> Optional[Dict[str, Any]]:
    try:
        # Adjust the path to point to 'scripts/acorn_parser.js'
        script_full_path = os.path.join(os.path.dirname(__file__), 'scripts', 'acorn_parser.js')
        logger.debug(f"Running Node.js script: {script_full_path}")
        result = subprocess.run(
            ["node", script_full_path],
            input=input_code,
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"Successfully ran {script_full_path}")
        output_json = json.loads(result.stdout)
        return output_json
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_full_path}: {e.stderr}")
        return None
    # ... rest of the code remains unchanged


def run_node_insert_docstrings(script_path: str, input_code: str) -> Optional[str]:
    """
    Runs a Node.js script to insert docstrings and returns the modified code.

    Parameters:
        script_path (str): Path to the Node.js script.
        input_code (str): JSON string containing the code and documentation.

    Returns:
        Optional[str]: The modified code if successful, None otherwise.
    """
    try:
        logger.debug(f"Running Node.js script: {script_path}")
        result = subprocess.run(["node", script_path], input=input_code, capture_output=True, text=True, check=True)
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

def generate_documentation_prompt(
    file_name: str,
    code_structure: Dict[str, Any],
    project_info: Optional[str],
    style_guidelines: Optional[str],
    language: str,
) -> str:
    """
    Creates a tailored documentation prompt based on various parameters like file name and code structure, aligning with project info and guidelines.

    Args:
        file_name (str): The name of the file for which documentation is being generated.
        code_structure (Dict[str, Any]): The structure of the code in the file, typically in JSON format.
        project_info (Optional[str]): Information about the project, if available.
        style_guidelines (Optional[str]): Documentation style guidelines to follow.
        language (str): The programming language of the file.

    Returns:
        str: A comprehensive prompt for generating documentation.
    """
    prompt = "You are an experienced software developer tasked with generating comprehensive documentation for a specific file in a codebase."
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
    4. **Functions:** Provide a JSDoc (for JavaScript/TypeScript) or Javadoc (for Java) comment for each function, including its purpose, parameters (`@param`), return values (`@returns` or `@return`), and whether it is asynchronous.
    5. **Classes:** Provide a JSDoc/Javadoc comment for each class, including its purpose, methods, inheritance details (`@extends` or `@implements`), and any interfaces it implements. Also, provide JSDoc/Javadoc comments for each method within the class.
    """
    return prompt

# ----------------------------
# Schema Validation (Optional)
# ----------------------------

def validate_schema(schema: dict):
    """Validates the loaded schema against a predefined schema."""
    predefined_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "parameters": {"type": "object"}
            },
            "required": ["name", "description", "parameters"]
        }
    }
    try:
        validate(instance=schema, schema=predefined_schema)
        logger.debug("Function schema is valid.")
    except ValidationError as ve:
        logger.critical(f"Schema validation error: {ve.message}")
        sys.exit(1)

# ----------------------------
# Initialize function_schema.json
# ----------------------------

async def process_all_files(
    session: aiohttp.ClientSession,
    file_paths: List[str],
    skip_types: Set[str],
    semaphore: asyncio.Semaphore,
    model_name: str,
    function_schema: dict,
    repo_root: str,
    project_info: Optional[str],
    style_guidelines: Optional[str],
    safe_mode: bool,
    output_file: str,
):
    """
    Processes all files: extracts structure, fetches documentation, and inserts docstrings.

    Args:
        session (aiohttp.ClientSession): The HTTP session.
        file_paths (List[str]): List of file paths to process.
        skip_types (Set[str]): Set of file types to skip.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrency.
        model_name (str): The AI model to use.
        function_schema (dict): The function schema for structured responses.
        repo_root (str): Root directory of the repository.
        project_info (Optional[str]): Project information.
        style_guidelines (Optional[str]): Style guidelines.
        safe_mode (bool): Whether to run in safe mode.
        output_file (str): Output Markdown file.
    """
    for file_path in file_paths:
        try:
            file_ext = os.path.splitext(file_path)[1]
            language = get_language(file_ext)
            if is_binary(file_path):
                logger.info(f"Skipping binary file: {file_path}")
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()

            # Optionally format the code
            code = format_with_black(code)
            code = clean_unused_imports(code)

            # Extract code structure using Node.js script (assuming extract_structure.js exists)
            structure = run_node_script("extract_structure.js", code)
            if not structure:
                logger.error(f"Failed to extract structure from {file_path}")
                continue

            # Generate prompt
            file_name = os.path.basename(file_path)
            prompt = generate_documentation_prompt(
                file_name=file_name,
                code_structure=structure,
                project_info=project_info,
                style_guidelines=style_guidelines,
                language=language
            )

            # Fetch documentation from API
            documentation = await fetch_documentation(
                session=session,
                prompt=prompt,
                semaphore=semaphore,
                model_name=model_name,
                function_schema=function_schema,
                use_azure=args.use_azure  # Ensure 'args' is accessible or passed appropriately
            )

            if not documentation:
                logger.error(f"Failed to fetch documentation for {file_path}")
                continue

            # Insert docstrings using Node.js script (assuming insert_docstrings.js exists)
            input_json = json.dumps({
                "code": code,
                "documentation": documentation
            })
            modified_code = run_node_insert_docstrings("insert_docstrings.js", input_json)
            if not modified_code:
                logger.error(f"Failed to insert docstrings into {file_path}")
                continue

            if not safe_mode:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(modified_code)
                logger.info(f"Updated file with docstrings: {file_path}")
            else:
                logger.info(f"Safe mode enabled. Skipping file modification: {file_path}")

            # Optionally, append to output Markdown
            with open(output_file, "a", encoding="utf-8") as out_f:
                out_f.write(f"# Documentation for {file_name}\n\n")
                out_f.write(f"## Summary\n{documentation.get('summary', 'No summary provided.')}\n\n")
                out_f.write(f"## Changes Made\n" + "\n".join(documentation.get('changes_made', [])) + "\n\n")
                out_f.write(f"## Functions\n")
                for func in documentation.get('functions', []):
                    out_f.write(f"### {func['name']}\n{func['docstring']}\n\n")
                out_f.write(f"## Classes\n")
                for cls in documentation.get('classes', []):
                    out_f.write(f"### {cls['name']}\n{cls['docstring']}\n\n")
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {e}", exc_info=True)


# ----------------------------
# Initialize function_schema.json
# ----------------------------

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Documentation generation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
