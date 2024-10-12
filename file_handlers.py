# file_handlers.py

import os
import logging
import shutil
import aiofiles
import aiohttp
import json
import asyncio
from typing import Optional, Set, List, Dict, Any
from tqdm.asyncio import tqdm
from language_functions import get_handler  # Only import get_handler
from language_functions.base_handler import BaseHandler
from utils import (
    is_binary,
    get_language,
    is_valid_extension,
    generate_documentation_prompt,
    write_documentation_report,
    generate_table_of_contents,
    clean_unused_imports,
    format_with_black,
    run_flake8
)

logger = logging.getLogger(__name__)

# Initialize Sentry SDK if DSN is provided
SENTRY_DSN = os.getenv('SENTRY_DSN')
if SENTRY_DSN:
    import sentry_sdk
    sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=1.0)
    logger.info('Sentry SDK initialized.')
else:
    logger.info('Sentry DSN not provided. Sentry SDK will not be initialized.')

async def extract_code_structure(
    content: str,
    file_path: str,
    language: str,
    handler: BaseHandler
) -> Optional[Dict[str, Any]]:
    """Extracts code structure based on language using the appropriate handler."""
    logger.debug(f"Extracting code structure for '{file_path}' (language: {language})")
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, handler.extract_structure, content, file_path)
    except Exception as e:
        logger.error(f"Error extracting structure from '{file_path}': {e}", exc_info=True)
        return None

async def backup_and_write_new_content(file_path: str, new_content: str) -> None:
    """Creates a backup of the file and writes the new content."""
    backup_path = f'{file_path}.bak'
    try:
        if os.path.exists(backup_path):
            os.remove(backup_path)
            logger.debug(f"Removed existing backup at '{backup_path}'.")
        shutil.copy(file_path, backup_path)
        logger.debug(f"Backup created at '{backup_path}'.")
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(new_content)
        logger.info(f"Inserted documentation into '{file_path}'.")
    except Exception as e:
        logger.error(f"Error writing to '{file_path}': {e}", exc_info=True)
        if os.path.exists(backup_path):
            shutil.copy(backup_path, file_path)
            os.remove(backup_path)
            logger.info(f"Restored original file from backup for '{file_path}'.")

async def fetch_documentation_rest(
    session: aiohttp.ClientSession,
    prompt: str,
    semaphore: asyncio.Semaphore,
    deployment_name: str,
    function_schema: Dict[str, Any],
    azure_api_key: str,
    azure_endpoint: str,
    azure_api_version: str,
    retry: int = 3
) -> Optional[Dict[str, Any]]:
    """Fetches documentation using Azure OpenAI REST API with function calling."""
    logger.debug(f"Fetching documentation using REST API for deployment: {deployment_name}")

    url = f"{azure_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={azure_api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_api_key,
    }

    for attempt in range(1, retry + 1):
        async with semaphore:
            try:
                payload = {
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "functions": function_schema["functions"],
                    "function_call": {"name": "generate_documentation"},
                }

                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"API Response: {data}")

                        if "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            message = choice["message"]

                            if "function_call" in message:
                                function_call = message["function_call"]
                                if function_call["name"] == "generate_documentation":
                                    arguments = function_call["arguments"]
                                    try:
                                        documentation = json.loads(arguments)
                                        logger.debug("Received documentation via function_call.")
                                        return documentation
                                    except json.JSONDecodeError as e:
                                        logger.error(f"Error decoding JSON from function_call arguments: {e}")
                                        logger.error(f"Arguments Content: {arguments}")
                                        return None
                                else:
                                    logger.error(f"Unexpected function called: {function_call['name']}")
                                    return None
                            else:
                                logger.error("No function_call found in the response.")
                                return None
                        else:
                            logger.error("No choices found in the response.")
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(f"Request failed with status {response.status}: {error_text}")
                        if attempt < retry:
                            wait_time = 2 ** attempt
                            logger.info(f"Retrying after {wait_time} seconds... (Attempt {attempt}/{retry})")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error("All retry attempts failed.")
                            return None

            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}", exc_info=True)
                if attempt < retry:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying after {wait_time} seconds... (Attempt {attempt}/{retry})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("All retry attempts failed.")
                    return None

    logger.error("All retry attempts failed.")
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
    azure_api_version: str
) -> Optional[str]:
    """
    Processes a single file: extracts structure, generates documentation, inserts documentation, validates, and returns the documentation content.
    """
    logger.debug(f'Processing file: {file_path}')
    try:
        _, ext = os.path.splitext(file_path)
        if not is_valid_extension(ext, skip_types) or is_binary(file_path):
            logger.debug(f"Skipping file '{file_path}' due to invalid extension or binary content.")
            return None

        language = get_language(ext)
        logger.debug(f"Detected language for '{file_path}': {language}")

        handler: Optional[BaseHandler] = get_handler(language, function_schema)
        if handler is None:
            logger.warning(f'Unsupported language: {language}')
            return None

        logger.info(f'Processing file: {file_path}')

        # Read file content asynchronously
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            logger.debug(f"File content for '{file_path}' read successfully.")
        except Exception as e:
            logger.error(f"Failed to read '{file_path}': {e}", exc_info=True)
            return None

        documentation = None

        # Extract code structure and generate documentation
        try:
            code_structure = await extract_code_structure(content, file_path, language, handler)
            if not code_structure:
                logger.warning(f"Could not extract code structure from '{file_path}'")
            else:
                logger.debug(f"Extracted code structure for '{file_path}': {code_structure}")
                prompt = generate_documentation_prompt(
                    file_name=os.path.basename(file_path),
                    code_structure=code_structure,
                    project_info=project_info,
                    style_guidelines=style_guidelines,
                    language=language
                )
                documentation = await fetch_documentation_rest(
                    session=session,
                    prompt=prompt,
                    semaphore=semaphore,
                    deployment_name=deployment_name,
                    function_schema=function_schema,
                    azure_api_key=azure_api_key,
                    azure_endpoint=azure_endpoint,
                    azure_api_version=azure_api_version
                )
                if not documentation:
                    logger.error(f"Failed to generate documentation for '{file_path}'.")
        except Exception as e:
            logger.error(f"Error during code structure extraction or documentation generation for '{file_path}': {e}", exc_info=True)

        new_content = content

        if documentation:
            try:
                loop = asyncio.get_event_loop()

                # Insert docstrings using the handler
                new_content = await loop.run_in_executor(None, handler.insert_docstrings, content, documentation)

                if language.lower() == 'python':
                    new_content = clean_unused_imports(new_content, file_path)
                    new_content = format_with_black(new_content)

                if not safe_mode:
                    # Validate code
                    is_valid = await loop.run_in_executor(None, handler.validate_code, new_content, file_path)
                    if is_valid:
                        # Backup original file and write new content
                        await backup_and_write_new_content(file_path, new_content)
                        logger.info(f"Documentation inserted into '{file_path}'")
                    else:
                        logger.error(f"Code validation failed for '{file_path}'.")
                else:
                    logger.info(f"Safe mode active. Skipping file modification for '{file_path}'")
            except Exception as e:
                logger.error(f"Error processing code documentation for '{file_path}': {e}", exc_info=True)
                new_content = content

        # Generate documentation report content for the file
        file_content = await write_documentation_report(
            documentation=documentation,
            language=language,
            file_path=file_path,
            repo_root=repo_root,
            new_content=new_content
        )
        logger.info(f"Finished processing '{file_path}'")
        return file_content
    except Exception as e:
        logger.error(f"Error processing file '{file_path}': {e}", exc_info=True)
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
    output_file: str = 'output.md',
    azure_api_key: str = '',
    azure_endpoint: str = '',
    azure_api_version: str = ''
) -> None:
    """
    Processes multiple files for documentation.
    """
    logger.info('Starting process of all files.')
    tasks = []
    for file_path in file_paths:
        task = asyncio.create_task(
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
                azure_api_version=azure_api_version
            )
        )
        tasks.append(task)

    documentation_contents = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='Processing Files'):
        try:
            file_content = await f
            if file_content:
                documentation_contents.append(file_content)
        except Exception as e:
            logger.error(f'Error processing a file: {e}', exc_info=True)
            if 'sentry_sdk' in globals():
                sentry_sdk.capture_exception(e)

    logger.info('Completed processing all files.')

    # Combine all documentation contents
    final_content = '\n\n'.join(documentation_contents)

    if final_content:
        # Generate Table of Contents
        toc = generate_table_of_contents(final_content)

        # Create the final report content
        report_content = '# Documentation Generation Report\n\n## Table of Contents\n\n' + toc + '\n\n' + final_content

        # Write the report to the output file
        try:
            async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                await f.write(report_content)
            logger.info(f"Documentation report written to '{output_file}'")
        except Exception as e:
            logger.error(f"Error writing final documentation to '{output_file}': {e}", exc_info=True)
            if 'sentry_sdk' in globals():
                sentry_sdk.capture_exception(e)
    else:
        logger.warning("No documentation was generated.")

    # Optional: Run Flake8 on all processed Python files to capture any remaining issues
    logger.info('Running Flake8 on processed files for final linting.')
    for file_path in file_paths:
        _, ext = os.path.splitext(file_path)
        if ext.lower() in {'.py'}:
            flake8_output = run_flake8(file_path)
            if flake8_output:
                logger.warning(f'Flake8 issues found in {file_path}:\n{flake8_output}')
    logger.info('Flake8 linting completed.')