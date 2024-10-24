# file_handlers.py
import os
import shutil
import logging
import aiohttp
import json
import aiofiles
import asyncio
import jsonschema
from typing import Set, List, Dict, Any, Optional, Tuple
from pathlib import Path
import textwrap
from jsonschema import validate, ValidationError
from language_functions import get_handler
from language_functions.base_handler import BaseHandler
from utils import (
    is_binary,
    get_language,
    clean_unused_imports_async,
    format_with_black_async,
    should_process_file,
    prepare_file,
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


def validate_ai_response(response: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]]:
    """
    Validates the AI response against the provided schema.

    Args:
        response (Dict[str, Any]): The AI response to validate.
        schema (Dict[str, Any]): The schema to validate against.

    Returns:
        Dict[str, Any]: The validated response.
    """
    try:
        validate(instance=response, schema=schema)
        return response
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return {}


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
    Fetches documentation from the Azure OpenAI API.

    Args:
        session (aiohttp.ClientSession): The aiohttp session.
        prompt (str): The prompt to send to the API.
        semaphore (asyncio.Semaphore): The semaphore to limit concurrency.
        deployment_name (str): The deployment name.
        function_schema (Dict[str, Any]): The function schema.
        azure_api_key (str): The Azure API key.
        azure_endpoint (str): The Azure endpoint.
        azure_api_version (str): The Azure API version.
        retry (int): The number of retry attempts.

    Returns:
        Optional[Dict[str, Any]]: The documentation response or None if it fails.
    """
    url = f"{azure_endpoint}/openai/deployments/{deployment_name}/completions?api-version={azure_api_version}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {azure_api_key}",
    }
    payload = {
        "prompt": prompt,
        "max_tokens": 1500,
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1,
        "stop": None,
    }

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
        validate(instance=documentation, schema=schema)
        return documentation
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
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
    project_id: str,
) -> Optional[Dict[str, Any]]:
    """Processes a single file through the documentation pipeline."""

    if not should_process_file(file_path, skip_types):
        return None

    try:
        content, language, handler = await prepare_file(file_path, function_schema, skip_types)
        if not all([content, language, handler]):
            return None

        code_structure = await extract_code_structure(content, file_path, language, handler)
        if not code_structure:
            return None

        # Generate documentation prompt
        context = context_manager.get_relevant_context(file_path) if context_manager.context_entries else ""
        prompt = generate_documentation_prompt(
            file_name=Path(file_path).name,
            code_structure=code_structure,
            project_info=f"{project_info}\n\nContext:\n{context}",
            style_guidelines=style_guidelines,
            language=language,
            function_schema=function_schema,
        )

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
            logger.error(f"Failed to generate documentation for {file_path}")
            return None

        # Update documentation with metrics and structure
        documentation.update(
            {
                "file_path": str(Path(file_path).relative_to(repo_root)),
                "language": language,
                "metrics": code_structure.get("metrics", {}),
                "structure": code_structure,
            }
        )

        # Write documentation to files (add project_id)
        result = await write_documentation_report(
            documentation=documentation,
            language=language,
            file_path=file_path,
            repo_root=repo_root,
            output_dir=output_dir,
            project_id=project_id,
        )

        return result

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
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
    """
    Processes all files and generates documentation.

    Args:
        session (aiohttp.ClientSession): The aiohttp session.
        file_paths (List[str]): List of file paths to process.
        skip_types (Set[str]): Set of file types to skip.
        semaphore (asyncio.Semaphore): The semaphore to limit concurrency.
        deployment_name (str): The deployment name.
        function_schema (Dict[str, Any]): The function schema.
        repo_root (str): The root directory of the repository.
        project_info (Optional[str]): The project information.
        style_guidelines (Optional[str]): The style guidelines.
        safe_mode (bool): Whether to run in safe mode.
        output_file (str): The output file for the documentation report.
        azure_api_key (str): The Azure API key.
        azure_endpoint (str): The Azure endpoint.
        azure_api_version (str): The Azure API version.
        output_dir (str): The output directory for the documentation.

    Returns:
        None
    """
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
            project_id="project_id",  # Placeholder, replace with actual project_id
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


async def _insert_and_validate_documentation(handler: BaseHandler, content: str, documentation: Dict[str, Any], file_path: str, language: str) -> None:
    """
    Inserts documentation and validates the updated code.

    Args:
        handler (BaseHandler): The handler object for the specific language.
        content (str): The original content of the file.
        documentation (Dict[str, Any]): The generated documentation.
        file_path (str): Path to the source file.
        language (str): Programming language of the source code.
    """
    try:
        new_content = await asyncio.to_thread(handler.insert_docstrings, content, documentation)

        if language.lower() == "python":
            new_content = await clean_unused_imports_async(new_content, file_path)
            new_content = await format_with_black_async(new_content)

        is_valid = await asyncio.to_thread(handler.validate_code, new_content, file_path)

        if is_valid:
            await backup_and_write_new_content(file_path, new_content)
            logger.info(f"Documentation inserted into '{file_path}'")
        else:
            logger.error(f"Code validation failed for '{file_path}'")

    except Exception as e:
        logger.error(f"Error processing documentation: {e}", exc_info=True)


async def _write_documentation_report(documentation: Dict[str, Any], language: str, file_path: str, repo_root: str, output_dir: str, project_id: str) -> Optional[str]:
    """
    Writes the documentation report to a file.

    Args:
        documentation (Dict[str, Any]): The generated documentation.
        language (str): Programming language of the source code.
        file_path (str): Path to the source file.
        repo_root (str): The root directory of the repository.
        output_dir (str): The output directory for the documentation.
        project_id (str): The project ID.

    Returns:
        Optional[str]: The file content if successful, otherwise None.
    """
    try:
        file_content = await write_documentation_report(
            documentation=documentation,
            language=language,
            file_path=file_path,
            repo_root=repo_root,
            output_dir=output_dir,
            project_id=project_id,
        )
        logger.info(f"Finished processing '{file_path}'")
        return file_content

    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        return None


def _update_documentation_metrics(documentation: Dict[str, Any], code_structure: Dict[str, Any]) -> None:
    """
    Updates documentation with metrics from code structure.

    Args:
        documentation (Dict[str, Any]): The generated documentation.
        code_structure (Dict[str, Any]): The extracted code structure.
    """
    class_complexity = code_structure.get("class_complexity", {})
    for cls in documentation.get("classes", []):
        methods_complexity = class_complexity.get(cls["name"], {})
        for method in cls.get("methods", []):
            method["complexity"] = methods_complexity.get(method["name"], 0)