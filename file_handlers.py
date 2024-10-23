"""
file_handlers.py

This module contains asynchronous functions for processing individual files, extracting code structures, generating documentation via Azure OpenAI API calls, and inserting docstrings into the code. It also manages backups and uses the ContextManager to maintain persistent context across files.
"""
import os
import shutil
import logging
import aiohttp
import json
import aiofiles
import asyncio
import jsonschema
from typing import Set, List, Dict, Any, Optional, Tuple
from jsonschema import validate, ValidationError
from language_functions import get_handler
from language_functions.base_handler import BaseHandler
from utils import (
    is_binary,
    get_language,
    clean_unused_imports_async,
    format_with_black_async,
)
from write_documentation_report import generate_documentation_prompt, write_documentation_report
from context_manager import ContextManager

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
    session, file_path, skip_types, semaphore, deployment_name,
    function_schema, repo_root, project_info, style_guidelines, safe_mode,
    azure_api_key, azure_endpoint, azure_api_version, output_dir
) -> Optional[str]:
    """Main file processing function."""

    if not should_process_file(file_path, skip_types):
        return None

    content, language, handler = await _prepare_file(file_path, function_schema, skip_types)
    if not all([content, language, handler]):  # Check if all are not None/False
        return None

    code_structure = await _extract_code_structure(content, file_path, language, handler)
    if not code_structure:
        return None

    documentation = await _generate_documentation(
        session, semaphore, deployment_name, function_schema,
        repo_root, project_info, style_guidelines, code_structure, file_path,
        azure_api_key, azure_endpoint, azure_api_version
    )
    if not documentation:
        return None

    _update_documentation_metrics(documentation, code_structure)

    if not safe_mode:
        await _insert_and_validate_documentation(handler, content, documentation, file_path, language)

    return await _write_documentation_report(documentation, language, file_path, repo_root, output_dir)



async def _prepare_file(file_path: str, function_schema: Dict[str, Any], skip_types: Set[str]) -> tuple[Optional[str], Optional[str], Optional[BaseHandler]]:
    """Reads file content, gets language and handler."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None, None, None

    _, ext = os.path.splitext(file_path)
    extra_skip_types = {'.flake8', '.gitignore', '.env', '.pyc', '.pyo', '.pyd', '.git', '.d.ts'}
    if ext in extra_skip_types or ext in skip_types or not ext or "node_modules" in file_path:
        logger.debug(f"Skipping file '{file_path}' due to extension/location: {ext}")
        return None, None, None

    if is_binary(file_path):
        logger.debug(f"Skipping binary file: {file_path}")
        return None, None, None

    language = get_language(ext)
    if language == "plaintext":
        logger.debug(f"Skipping plaintext file: {file_path}")
        return None, None, None

    handler = get_handler(language, function_schema)
    if not handler:
        logger.debug(f"No handler available for language: {language}")
        return None, None, None

    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
        logger.debug(f"Successfully read content from {file_path}")
        return content, language, handler
    except UnicodeDecodeError:
        logger.warning(f"Skipping file due to encoding issues: {file_path}")
        return None, None, None
    except Exception as e:
        logger.error(f"Failed to read '{file_path}': {e}", exc_info=True)
        return None, None, None


async def _extract_code_structure(content: str, file_path: str, language: str, handler: BaseHandler) -> Optional[Dict[str, Any]]:
    """Extracts code structure and adds context."""

    try:
        code_structure = await extract_code_structure(content, file_path, language, handler)  # Assuming this function is defined elsewhere
        if not code_structure:
            return None

        try:
            critical_info = extract_critical_info(code_structure, file_path)  # Assuming this function is defined elsewhere
            context_manager.add_context(critical_info)
        except Exception as e:
            logger.error(f"Error extracting critical info: {e}", exc_info=True)
            critical_info = f"File: {file_path}\n# Failed to extract detailed information"
            context_manager.add_context(critical_info)
        return code_structure

    except Exception as e:
        logger.error(f"Error extracting structure: {e}", exc_info=True)
        return None



async def _generate_documentation(
    session, semaphore, deployment_name, function_schema,
    repo_root, project_info, style_guidelines, code_structure, file_path,
    azure_api_key, azure_endpoint, azure_api_version
) -> Optional[Dict[str, Any]]:
    """Generates documentation from Azure OpenAI."""
    try:
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
    language=get_language(os.path.splitext(file_path)[1]),
    function_schema=function_schema
)}
        """.strip()

        documentation = await fetch_documentation_rest(  # Assuming this function is defined elsewhere
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
        return documentation

    except Exception as e:
        logger.error(f"Error generating documentation: {e}", exc_info=True)
        return None


def _update_documentation_metrics(documentation: Dict[str, Any], code_structure: Dict[str, Any]) -> None:
    """Updates documentation with metrics from code structure."""

    documentation.update({
        "halstead": code_structure.get("halstead", {}),
        "maintainability_index": code_structure.get("maintainability_index"),
        "variables": code_structure.get("variables", []),
        "constants": code_structure.get("constants", []),
        "changes_made": documentation.get("changes_made", [])
    })

    function_complexity = {func["name"]: func.get("complexity", 0) for func in code_structure.get("functions", [])}
    for func in documentation.get("functions", []):
        func["complexity"] = function_complexity.get(func["name"], 0)

    class_complexity = {}
    for cls in code_structure.get("classes", []):
        methods_complexity = {method["name"]: method.get("complexity", 0) for method in cls.get("methods", [])}
        class_complexity[cls["name"]] = methods_complexity

    for cls in documentation.get("classes", []):
        methods_complexity = class_complexity.get(cls["name"], {})
        for method in cls.get("methods", []):
            method["complexity"] = methods_complexity.get(method["name"], 0)


async def _insert_and_validate_documentation(handler: BaseHandler, content: str, documentation: Dict[str, Any], file_path: str, language: str) -> None:
    """Inserts documentation and validates the updated code."""
    try:
        new_content = await asyncio.to_thread(handler.insert_docstrings, content, documentation)

        if language.lower() == "python":
            new_content = await clean_unused_imports_async(new_content, file_path)
            new_content = await format_with_black_async(new_content)

        is_valid = await asyncio.to_thread(handler.validate_code, new_content, file_path)

        if is_valid:
            await backup_and_write_new_content(file_path, new_content)  # Assuming this function is defined elsewhere
            logger.info(f"Documentation inserted into '{file_path}'")
        else:
            logger.error(f"Code validation failed for '{file_path}'")

    except Exception as e:
        logger.error(f"Error processing documentation: {e}", exc_info=True)


async def _write_documentation_report(documentation: Dict[str, Any], language: str, file_path: str, repo_root: str, output_dir: str) -> Optional[str]:
    """Writes the documentation report to a file."""
    try:
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
        logger.error(f"Error generating report: {e}", exc_info=True)
        return None

def should_process_file(file_path: str, skip_types: Set[str]) -> bool:
    """
    Checks if a file should be processed based on path, extension and skip types.
    """

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    _, ext = os.path.splitext(file_path)

    # Combines all skip conditions for clarity
    if (
        os.path.islink(file_path) or
        any(part in file_path for part in ['node_modules', '.bin']) or
        file_path.endswith('.d.ts') or
        any(excluded in file_path for excluded in {'.git', '__pycache__', 'node_modules', '.bin', 'build', 'dist'}) or
        ext in {'.flake8', '.gitignore', '.env', '.pyc', '.pyo', '.pyd', '.git', '.d.ts'} or
        ext in skip_types or not ext
    ):
        logger.debug(f"Skipping file '{file_path}' due to extension/location: {ext}")
        return False

    return True

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
        if isinstance(func, dict):  # Check if func is a dictionary
            signature = f"def {func.get('name', 'unknown')}({', '.join(func.get('args', []))}):"
            doc = func.get('docstring', '').split('\n')[0]
            info_lines.append(f"{signature}  # {doc}")

    # Extract class definitions
    classes = code_structure.get("classes", [])
    for cls in classes:
        if isinstance(cls, dict):  # Check if cls is a dictionary
            class_info = f"class {cls.get('name', 'unknown')}:"
            doc = cls.get('docstring', '').split('\n')[0]
            info_lines.append(f"{class_info}  # {doc}")
            
            # Include methods
            for method in cls.get('methods', []):
                if isinstance(method, dict):  # Check if method is a dictionary
                    method_signature = f"    def {method.get('name', 'unknown')}({', '.join(method.get('args', []))}):"
                    method_doc = method.get('docstring', '').split('\n')[0]
                    info_lines.append(f"{method_signature}  # {method_doc}")

    # Extract important variables
    variables = code_structure.get("variables", [])
    for var in variables:
        if isinstance(var, dict):  # Check if var is a dictionary
            var_info = f"{var.get('name', 'unknown')} = "
            var_type = var.get('type', 'Unknown')
            var_desc = var.get('description', '').split('\n')[0]
            info_lines.append(f"{var_info}  # Type: {var_type}, {var_desc}")
        elif isinstance(var, str):  # Handle string variables
            info_lines.append(f"{var} = ")  # Just add the variable name if it's a string

    critical_info = "\n".join(info_lines)
    return critical_info
