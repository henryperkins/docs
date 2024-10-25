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
)
from write_documentation_report import generate_documentation_prompt, write_documentation_report
from context_manager import ContextManager
from fastapi import HTTPException
from datetime import datetime
from time import perf_counter
from metrics import MetricsAnalyzer, MetricsResult, calculate_code_metrics
from metrics_utils import MetricsThresholds, get_metric_severity

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
    session, prompt, semaphore, deployment_name, function_schema,
    azure_api_key, azure_endpoint, azure_api_version, retry=3
):
    """Fetches documentation, correctly handling function calls and retries."""
    url = f"{azure_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={azure_api_version}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {azure_api_key}",
    }

    for attempt in range(retry):
        try:
            async with semaphore:
                payload = {
                    "messages": prompt,
                    "max_tokens": 1500,
                    "temperature": 0.7,  # Adjust as needed
                    "top_p": 0.9,       # Adjust as needed
                    "n": 1,
                    "stop": None,
                    "functions": function_schema["functions"],
                    "function_call": {"name": "generate_documentation"}
                }

                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"API Response: {data}")

                        choice = data.get("choices", [{}])[0]
                        message = choice.get("message", {})

                        if "function_call" in message:
                            function_call = message["function_call"]
                            arguments = function_call.get("arguments", "{}")
                            try:
                                function_args = json.loads(arguments)
                                validated_args = validate_ai_response(
                                    function_args,
                                    function_schema["functions"][0]["parameters"]
                                )
                                if validated_args:
                                    logger.debug("Function call successful")
                                    return validated_args
                                else:
                                    raise ValueError("AI response validation failed")
                            except (json.JSONDecodeError, ValueError) as e:
                                logger.error(f"Error processing function call arguments: {e}")
                                if attempt < retry - 1:
                                    prompt[-1]["content"] += "\n\nPlease ensure the `generate_documentation` function call arguments are valid JSON and match the provided schema."
                                    await asyncio.sleep(2**attempt)  # Exponential backoff
                                    continue
                                else:
                                    raise  # Re-raise after retries
                        else:
                            logger.error("Model did not make the expected function call. Returning the raw response (if any) for debugging.")
                            return message.get("content")  # Return content for debugging

                    elif response.status == 429:  # Rate limit hit
                        retry_after = int(response.headers.get("Retry-After", 1)) * (attempt +1) # Increase backoff with attempts
                        logger.warning(f"Rate limited (Attempt {attempt+1}). Retrying after {retry_after} seconds...")
                        await asyncio.sleep(retry_after)
                        continue
                    elif response.status == 401:
                        raise HTTPException(status_code=401, detail="Unauthorized. Check your API key and endpoint.")
                    else:
                        error_text = await response.text()
                        logger.error(f"Azure OpenAI API request failed with status {response.status}: {error_text}")
                        raise HTTPException(status_code=response.status, detail=error_text)

        except aiohttp.ClientError as e:
            logger.error(f"Network error during API request (Attempt {attempt+1}): {e}", exc_info=True)
            if attempt < retry - 1:
                await asyncio.sleep(2**attempt) # Exponential backoff
                continue
            else:
                raise  # Re-raise after retries

        except Exception as e:
            logger.error(f"Unexpected error during API request (Attempt {attempt+1}): {e}", exc_info=True)
            if attempt < retry - 1:
                await asyncio.sleep(2**attempt) # Exponential backoff
                continue
            else:
                raise # Re-raise after retries


    logger.error("All retry attempts to fetch documentation failed.")
    return None

async def handle_api_error(response: aiohttp.ClientResponse):
    """Handles errors from the Azure OpenAI API."""
    try:
        response_text = await response.text()
        logger.error(f"Azure OpenAI API request failed with status {response.status}: {response_text}")

        # More specific error handling can be added here based on response.status
        raise HTTPException(status_code=response.status, detail=response_text)

    except Exception as e:
        logger.error(f"Error handling API error: {e}", exc_info=True)
        raise # Re-raise the exception after logging

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


class FileProcessingResult:
    """Class to store file processing results"""
    def __init__(self, 
                 file_path: str, 
                 success: bool, 
                 metrics_result: Optional[MetricsResult] = None,
                 error: Optional[str] = None,
                 documentation: Optional[Dict[str, Any]] = None):
        self.file_path = file_path
        self.success = success
        self.metrics_result = metrics_result
        self.error = error
        self.documentation = documentation
        self.timestamp = datetime.now()
        
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
) -> Optional[FileProcessingResult]:
    """Enhanced process_file with better metrics handling and error tracking"""
    
    start_time = perf_counter()
    metrics_analyzer = MetricsAnalyzer()
    
    if not should_process_file(file_path, skip_types):
        return None

    try:
        # Prepare file and get initial content
        content, language, handler = await _prepare_file(file_path, function_schema, skip_types)
        if not all([content, language, handler]):
            return FileProcessingResult(
                file_path=file_path,
                success=False,
                error="Failed to prepare file"
            )

        # Calculate metrics with enhanced error handling
        try:
            metrics_result = await calculate_code_metrics(content, file_path, language)
            metrics_analyzer.add_result(metrics_result)
            
            if not metrics_result.success:
                logger.warning(f"Metrics calculation partial failure for {file_path}: {metrics_result.error}")
            
        except Exception as e:
            logger.error(f"Metrics calculation failed for {file_path}: {e}", exc_info=True)
            metrics_result = MetricsResult(
                file_path=file_path,
                timestamp=datetime.now(),
                execution_time=perf_counter() - start_time,
                success=False,
                error=str(e)
            )

        # Extract code structure with metrics
        code_structure = await _extract_code_structure(
            content=content,
            file_path=file_path,
            language=language,
            handler=handler,
            metrics=metrics_result.metrics if metrics_result and metrics_result.success else None
        )

        if not code_structure:
            return FileProcessingResult(
                file_path=file_path,
                success=False,
                metrics_result=metrics_result,
                error="Failed to extract code structure"
            )

        # Generate documentation with metrics information
        context = context_manager.get_relevant_context(file_path)
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
            return FileProcessingResult(
                file_path=file_path,
                success=False,
                metrics_result=metrics_result,
                error="Failed to generate documentation"
            )

        # Enhance documentation with metrics
        documentation.update({
            "file_path": str(Path(file_path).relative_to(repo_root)),
            "language": language,
            "metrics": metrics_result.metrics if metrics_result and metrics_result.success else {},
            "metrics_summary": metrics_analyzer.get_summary(),
            "metrics_analysis": {
                "problematic_files": metrics_analyzer.get_problematic_files(),
                "execution_time": perf_counter() - start_time,
                "severity_levels": {
                    metric: get_metric_severity(metric, value, MetricsThresholds())
                    for metric, value in (metrics_result.metrics or {}).items()
                }
            },
            "structure": code_structure,
        })

        # Write documentation
        result = await write_documentation_report(
            documentation=documentation,
            language=language,
            file_path=file_path,
            repo_root=repo_root,
            output_dir=output_dir,
            project_id=project_id,
        )

        return FileProcessingResult(
            file_path=file_path,
            success=True,
            metrics_result=metrics_result,
            documentation=result
        )

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
        return FileProcessingResult(
            file_path=file_path,
            success=False,
            metrics_result=metrics_result if 'metrics_result' in locals() else None,
            error=str(e)
        )
    
async def _prepare_file(file_path: str, function_schema: Dict[str, Any], skip_types: Set[str]) -> Tuple[Optional[str], Optional[str], Optional[BaseHandler]]:
    """Reads file content, gets language, and retrieves the appropriate handler."""

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None, None, None

    _, ext = os.path.splitext(file_path)

    # Enhanced skip logic (moved from should_process_file)
    if not should_process_file(file_path, skip_types):
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


async def _extract_code_structure(content: str, file_path: str, language: str, handler: BaseHandler, metrics: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        structure = await asyncio.to_thread(handler.extract_structure, content, file_path, metrics) 
        if not structure:
            logger.warning(f"No structure extracted from '{file_path}'")
            return None

        try:
            critical_info = extract_critical_info(structure, file_path)
            context_manager.add_context(critical_info)
        except Exception as e:
            logger.error(f"Error extracting critical info: {e}", exc_info=True)
            critical_info = f"File: {file_path}\n# Failed to extract detailed information"
            context_manager.add_context(critical_info)

        return structure

    except Exception as e:
        logger.error(f"Error extracting structure: {e}", exc_info=True)
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
    """Extracts critical information from the code structure."""
    info_lines = [f"File: {file_path}"]

    functions = code_structure.get("functions", [])
    for func in functions:
        if isinstance(func, dict):
            # Extract argument names as strings
            arg_names = [arg.get("name", "unknown") for arg in func.get("args", [])]
            signature = f"def {func.get('name', 'unknown')}({', '.join(arg_names)}):"
            doc = func.get('docstring', '').split('\n')[0]
            info_lines.append(f"{signature}  # {doc}")

    classes = code_structure.get("classes", [])
    for cls in classes:
        if isinstance(cls, dict):
            class_info = f"class {cls.get('name', 'unknown')}:"
            doc = cls.get('docstring', '').split('\n')[0]
            info_lines.append(f"{class_info}  # {doc}")

            for method in cls.get('methods', []):
                if isinstance(method, dict):
                    # Extract argument names for methods
                    method_arg_names = [arg.get("name", "unknown") for arg in method.get("args", [])]
                    method_signature = f"    def {method.get('name', 'unknown')}({', '.join(method_arg_names)}):"
                    method_doc = method.get('docstring', '').split('\n')[0]
                    info_lines.append(f"{method_signature}  # {method_doc}")

    variables = code_structure.get("variables", [])
    for var in variables:
        if isinstance(var, dict):
            var_info = f"{var.get('name', 'unknown')} = "
            var_type = var.get('type', 'Unknown')
            var_desc = var.get('description', '').split('\n')[0]
            info_lines.append(f"{var_info}  # Type: {var_type}, {var_desc}")


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