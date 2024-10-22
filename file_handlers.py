"""
file_handlers.py

This module contains asynchronous functions for processing individual files, extracting code structures, generating documentation via Azure OpenAI API calls, and inserting docstrings into the code. It also manages backups and uses the ContextManager to maintain persistent context across files.
"""

import os
import shutil
import logging
import aiofiles
import aiohttp
import json
import asyncio
import jsonschema
from typing import Set, List, Dict, Any, Optional
from jsonschema import validate, ValidationError  # Add these imports
from language_functions import get_handler
from language_functions.base_handler import BaseHandler
from utils import (
    is_binary,
    get_language,
    is_valid_extension,
    clean_unused_imports_async,
    format_with_black_async,
    run_flake8_async,
)
from write_documentation_report import (
    generate_documentation_prompt,
    write_documentation_report,
    sanitize_filename,
    generate_table_of_contents,
)
from context_manager import ContextManager  # Import ContextManager

logger = logging.getLogger(__name__)

# Initialize the ContextManager
context_manager = ContextManager(max_entries=100)


async def extract_code_structure(
    content: str, file_path: str, language: str, handler: BaseHandler
) -> Optional[Dict[str, Any]]:
    """
    Asynchronously extracts the code structure from the given content using the specified handler.

    Args:
        content (str): The source code content.
        file_path (str): Path to the source file.
        language (str): Programming language of the source code.
        handler (BaseHandler): The handler object for the specific language.

    Returns:
        Optional[Dict[str, Any]]: A dictionary representing the code structure or None if extraction fails.
    """
    logger.debug(f"Extracting code structure for '{file_path}' (language: {language})")
    try:
        structure = await asyncio.to_thread(handler.extract_structure, content, file_path)
        if not structure:
            logger.warning(f"No structure extracted from '{file_path}'")
            return None
        return structure
    except Exception as e:
        logger.error(f"Error extracting structure from '{file_path}': {e}", exc_info=True)
        return None


async def backup_and_write_new_content(file_path: str, new_content: str) -> None:
    """
    Creates a backup of the original file and writes new content to it.

    Args:
        file_path (str): Path to the file to update.
        new_content (str): The new content to write to the file.
    """
    backup_path = f"{file_path}.bak"
    try:
        if os.path.exists(backup_path):
            os.remove(backup_path)
            logger.debug(f"Removed existing backup at '{backup_path}'.")
        await asyncio.to_thread(shutil.copy, file_path, backup_path)
        logger.debug(f"Backup created at '{backup_path}'.")
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(new_content)
        logger.info(f"Inserted documentation into '{file_path}'.")
    except Exception as e:
        logger.error(f"Error writing to '{file_path}': {e}", exc_info=True)
        # Attempt to restore from backup
        if os.path.exists(backup_path):
            try:
                await asyncio.to_thread(shutil.copy, backup_path, file_path)
                os.remove(backup_path)
                logger.info(f"Restored original file from backup for '{file_path}'.")
            except Exception as restore_error:
                logger.error(f"Failed to restore backup for '{file_path}': {restore_error}", exc_info=True)
                

def validate_ai_response(response: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and potentially reformats the AI's response to match the required schema.

    Args:
        response (Dict[str, Any]): The AI's response.
        schema (Dict[str, Any]): The expected schema.

    Returns:
        Dict[str, Any]: The validated and potentially reformatted response.
    """
    try:
        validate(instance=response, schema=schema)
        return response
    except ValidationError as validation_error:
        logger.error(f"AI response does not match schema: {validation_error}")
        # Here you could implement logic to try to correct the response
        # For now, we'll just return None to indicate failure
        return None

async def fetch_documentation_rest(
    session: aiohttp.ClientSession,
    prompt: str,
    semaphore: asyncio.Semaphore,
    deployment_name: str,
    function_schema: Dict[str, Any],
    azure_api_key: str,
    azure_endpoint: str,
    azure_api_version: str,
    retry: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Fetches documentation from Azure OpenAI API using the provided prompt and schema.
    
    ... (rest of the docstring)
    """
    logger.debug(f"Fetching documentation using REST API for deployment: {deployment_name}")

    url = f"{azure_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={azure_api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_api_key,
    }

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "functions": function_schema.get("functions", []),
        "function_call": {"name": "generate_documentation"}
    }

    async def handle_api_error(response: aiohttp.ClientResponse) -> None:
        """Handle API errors and log appropriate messages."""
        error_text = await response.text()
        try:
            error_json = json.loads(error_text)
            error_message = error_json.get("error", {}).get("message", "Unknown error")
            logger.error(f"Azure OpenAI API request failed: {response.status} - {error_message}")
        except json.JSONDecodeError:
            logger.error(f"API request failed with non-JSON response: {response.status} - {error_text}")

    for attempt in range(retry):
        try:
            async with semaphore:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"API Response: {data}")

                        if "choices" in data and data["choices"]:
                            choice = data["choices"][0]
                            message = choice.get("message")
                            
                            if message and "function_call" in message:
                                function_call = message["function_call"]
                                if function_call.get("name") == "generate_documentation":
                                    arguments = function_call.get("arguments")
                                    try:
                                        documentation = json.loads(arguments)
                                        # Validate the AI's response against the schema
                                        validated_documentation = validate_ai_response(
                                            documentation, 
                                            function_schema["functions"][0]["parameters"]
                                        )
                                        if validated_documentation:
                                            logger.debug("Successfully validated documentation response")
                                            return validated_documentation
                                        else:
                                            logger.error("AI response validation failed")
                                            # Add more context to the prompt for retry
                                            payload["messages"][0]["content"] = (
                                                prompt + "\n\nPlease ensure your response exactly matches "
                                                "the provided schema and includes all required fields."
                                            )
                                            continue
                                    except json.JSONDecodeError as e:
                                        logger.error(f"Error decoding JSON: {e}")
                                        logger.error(f"Arguments Content: {arguments}")
                                else:
                                    logger.error(f"Unexpected function called: {function_call.get('name')}")
                            else:
                                logger.error("No function_call found in the response.")
                        else:
                            logger.error("No choices found in the API response.")
                    elif response.status == 429:  # Rate limit hit
                        retry_after = response.headers.get("Retry-After", str(2 ** attempt))
                        wait_time = int(retry_after)
                        logger.warning(f"Rate limited. Retrying after {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        await handle_api_error(response)

        except aiohttp.ClientError as e:
            logger.error(f"Network error during API request: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during API request: {e}", exc_info=True)

        if attempt < retry - 1:
            wait_time = 2 ** attempt
            logger.warning(f"Retrying after {wait_time} seconds... (Attempt {attempt + 1}/{retry})")
            await asyncio.sleep(wait_time)

    logger.error("All retry attempts to fetch documentation failed.")
    return None


async def handle_api_error(response):
    error_text = await response.text()
    try:
        error_json = json.loads(error_text)
        error_message = error_json.get("error", {}).get("message", "Unknown error")
        logger.error(f"Azure OpenAI API request failed: {response.status} - {error_message}")
    except json.JSONDecodeError:
        logger.error(f"API request failed with non-JSON response: {response.status} - {error_text}")


def validate_documentation(documentation: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validates the given documentation against the provided JSON schema.

    Args:
        documentation (Dict[str, Any]): The documentation to be validated.
        schema (Dict[str, Any]): The JSON schema to validate against.

    Returns:
        Optional[Dict[str, Any]]: The validated documentation if it passes the schema validation, 
        otherwise None if validation fails.
    """
    try:
        jsonschema.validate(instance=documentation, schema=schema)
        return documentation
    except jsonschema.exceptions.ValidationError as e:
        logger.error(f"Documentation validation failed: {e}")
        # You might want to add logic here to attempt to fix the documentation
        return None


async def process_file(
    session: aiohttp.ClientSession,
    file_path: str,
    skip_types: Set[str],
    semaphore: asyncio.Semaphore,
    deployment_name: str,
    function_schema: Dict[str, Any],
    repo_root: str,
    project_info: str,
    style_guidelines: str,
    safe_mode: bool,
    azure_api_key: str,
    azure_endpoint: str,
    azure_api_version: str,
    output_dir: str,
) -> Optional[str]:
    """
    Processes a single file to extract its structure and generate documentation.

    Args:
        session (aiohttp.ClientSession): The HTTP session for making requests.
        file_path (str): Path to the file to process.
        skip_types (Set[str]): Set of file extensions to skip.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent API requests.
        deployment_name (str): The Azure OpenAI deployment name.
        function_schema (Dict[str, Any]): The schema defining functions.
        repo_root (str): Root directory of the repository.
        project_info (str): Information about the project.
        style_guidelines (str): Documentation style guidelines.
        safe_mode (bool): If True, no files will be modified.
        azure_api_key (str): The API key for Azure OpenAI.
        azure_endpoint (str): The endpoint URL for the Azure OpenAI service.
        azure_api_version (str): The API version to use.
        output_dir (str): Directory to save documentation files.

    Returns:
        Optional[str]: The content of the documentation report or None if processing fails.
    """
    logger.debug(f"Processing file: {file_path}")
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None

        # Get file extension and check if we should process this file
        _, ext = os.path.splitext(file_path)
        
        # Additional skip types for config and hidden files
        extra_skip_types = {'.flake8', '.gitignore', '.env', '.pyc', '.pyo', '.pyd', '.git'}
        if ext in extra_skip_types or ext in skip_types or not ext:
            logger.debug(f"Skipping file '{file_path}' due to extension: {ext}")
            return None

        # Check if file is binary
        if is_binary(file_path):
            logger.debug(f"Skipping binary file: {file_path}")
            return None

        # Get language and handler
        language = get_language(ext)
        logger.debug(f"Detected language for '{file_path}': {language}")

        if language == "plaintext":
            logger.debug(f"Skipping plaintext file: {file_path}")
            return None

        handler = get_handler(language, function_schema)
        if not handler:
            logger.debug(f"No handler available for language: {language}")
            return None

        # Read file content
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            logger.debug(f"Successfully read content from {file_path}")
        except UnicodeDecodeError:
            logger.warning(f"Skipping file due to encoding issues: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to read '{file_path}': {e}", exc_info=True)
            return None

        # Extract code structure
        try:
            code_structure = await extract_code_structure(content, file_path, language, handler)
            if not code_structure:
                logger.warning(f"Could not extract code structure from '{file_path}'")
                return None

            # Add to context manager
            critical_info = extract_critical_info(code_structure, file_path)
            context_manager.add_context(critical_info)

            # Generate prompt with context
            persistent_context = "\n".join(context_manager.get_context())
            prompt = f"""
[Context Start]
{persistent_context}
[Context End]

{generate_documentation_prompt(
    file_name=os.path.basename(file_path),
    code_structure=code_structure,
    project_info=project_info,
    style_guidelines=style_guidelines,
    language=language,
    function_schema=function_schema
)}
""".strip()

            # Generate documentation
            documentation = await fetch_documentation_rest(
                session=session,
                prompt=prompt,
                semaphore=semaphore,
                deployment_name=deployment_name,
                function_schema=function_schema,
                azure_api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                azure_api_version=azure_api_version,
            )

            if not documentation:
                logger.error(f"Failed to generate documentation for '{file_path}'")
                return None

            # Update documentation with code structure metrics
            documentation.update({
                "halstead": code_structure.get("halstead", {}),
                "maintainability_index": code_structure.get("maintainability_index"),
                "variables": code_structure.get("variables", []),
                "constants": code_structure.get("constants", []),
                "changes_made": documentation.get("changes_made", [])
            })

            # Update complexity metrics
            function_complexity = {
                func["name"]: func.get("complexity", 0)
                for func in code_structure.get("functions", [])
            }
            for func in documentation.get("functions", []):
                func["complexity"] = function_complexity.get(func["name"], 0)

            class_complexity = {}
            for cls in code_structure.get("classes", []):
                class_name = cls["name"]
                methods_complexity = {
                    method["name"]: method.get("complexity", 0)
                    for method in cls.get("methods", [])
                }
                class_complexity[class_name] = methods_complexity

            for cls in documentation.get("classes", []):
                class_name = cls["name"]
                methods_complexity = class_complexity.get(class_name, {})
                for method in cls.get("methods", []):
                    method["complexity"] = methods_complexity.get(method["name"], 0)

            # Process documentation
            if not safe_mode:
                try:
                    new_content = await asyncio.to_thread(
                        handler.insert_docstrings,
                        content,
                        documentation
                    )

                    if language.lower() == "python":
                        new_content = await clean_unused_imports_async(new_content, file_path)
                        new_content = await format_with_black_async(new_content)

                    is_valid = await asyncio.to_thread(
                        handler.validate_code,
                        new_content,
                        file_path
                    )

                    if is_valid:
                        await backup_and_write_new_content(file_path, new_content)
                        logger.info(f"Documentation inserted into '{file_path}'")
                    else:
                        logger.error(f"Code validation failed for '{file_path}'")
                except Exception as e:
                    logger.error(f"Error processing documentation for '{file_path}': {e}", exc_info=True)

            # Generate documentation report
            file_content = await write_documentation_report(
                documentation=documentation,
                language=language,
                file_path=file_path,
                repo_root=repo_root,
                output_dir=output_dir,
            )

            logger.info(f"Finished processing '{file_path}'")
            return file_content

        except Exception as e:
            logger.error(f"Error processing '{file_path}': {e}", exc_info=True)
            return None

    except Exception as e:
        logger.error(f"Unexpected error processing '{file_path}': {e}", exc_info=True)
        return None

async def process_all_files(
    session: aiohttp.ClientSession,
    file_paths: List[str],
    skip_types: Set[str],
    semaphore: asyncio.Semaphore,
    deployment_name: str,
    function_schema: Dict[str, Any],
    repo_root: str,
    project_info: Optional[str],
    style_guidelines: Optional[str],
    safe_mode: bool = False,
    output_file: str = "output.md",
    azure_api_key: str = "",
    azure_endpoint: str = "",
    azure_api_version: str = "",
    output_dir: str = "documentation",
) -> None:
    logger.info("Starting process of all files.")
    tasks = [
        process_file(
            session=session,
            file_path=file_path,
            skip_types=skip_types,
            semaphore=semaphore,
            deployment_name=deployment_name,
            function_schema=function_schema,
            repo_root=repo_root,
            project_info=project_info,
            style_guidelines=style_guidelines,
            safe_mode=safe_mode,
            azure_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            output_dir=output_dir,
        )
        for file_path in file_paths
    ]

    documentation_contents = []
    for f in asyncio.as_completed(tasks):
        try:
            file_content = await f
            if file_content:
                documentation_contents.append(file_content)
        except Exception as e:
            logger.error(f"Error processing a file: {e}", exc_info=True)

    logger.info("Completed processing all files.")

    final_content = "\n\n".join(documentation_contents)

    if final_content:
        toc = generate_table_of_contents(final_content)
        report_content = (
            "# Documentation Generation Report\n\n## Table of Contents\n\n" + toc + "\n\n" + final_content
        )

        try:
            async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
                await f.write(report_content)
            logger.info(f"Documentation report written to '{output_file}'")
        except Exception as e:
            logger.error(f"Error writing final documentation to '{output_file}': {e}", exc_info=True)

    else:
        logger.warning("No documentation was generated.")

    logger.info("Documentation generation process completed.")


def extract_critical_info(code_structure: Dict[str, Any], file_path: str) -> str:
    """
    Extracts critical information from the code structure to be used as persistent context.

    Args:
        code_structure (Dict[str, Any]): The extracted code structure.
        file_path (str): Path to the source file.

    Returns:
        str: A formatted string containing critical context information.
    """
    info_lines = [f"File: {file_path}"]
    # Extract function signatures
    functions = code_structure.get("functions", [])
    for func in functions:
        signature = f"def {func.get('name')}({', '.join(func.get('args', []))}):"
        doc = func.get("docstring", "").split("\n")[0]
        info_lines.append(f"{signature}  # {doc}")

    # Extract class definitions
    classes = code_structure.get("classes", [])
    for cls in classes:
        class_info = f"class {cls.get('name')}:"
        doc = cls.get("docstring", "").split("\n")[0]
        info_lines.append(f"{class_info}  # {doc}")
        # Include methods
        for method in cls.get("methods", []):
            method_signature = f"    def {method.get('name')}({', '.join(method.get('args', []))}):"
            method_doc = method.get("docstring", "").split("\n")[0]
            info_lines.append(f"{method_signature}  # {method_doc}")

    # Extract important variables
    variables = code_structure.get("variables", [])
    for var in variables:
        var_info = f"{var.get('name')} = "
        var_type = var.get("type", "Unknown")
        var_desc = var.get("description", "").split("\n")[0]
        info_lines.append(f"{var_info}  # Type: {var_type}, {var_desc}")

    critical_info = "\n".join(info_lines)
    return critical_info
