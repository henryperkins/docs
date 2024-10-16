import os
import logging
import aiofiles
import aiohttp
import json
import asyncio
from typing import Set, List, Dict, Any
from language_functions import get_handler
from language_functions.base_handler import BaseHandler
from utils import (
    is_binary,
    get_language,
    is_valid_extension,
    clean_unused_imports_async,
    format_with_black_async,
    run_flake8_async
)
from write_documentation_report import generate_documentation_prompt, generate_table_of_contents, write_documentation_report

logger = logging.getLogger(__name__)

# Initialize Sentry SDK if DSN is provided
SENTRY_DSN = os.getenv('SENTRY_DSN')
if SENTRY_DSN:
    import sentry_sdk
    sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=1.0)
    logger.info('Sentry SDK initialized.')
else:
    logger.info('Sentry DSN not provided. Sentry SDK will not be initialized.')

async def extract_code_structure(content: str, file_path: str, language: str, handler: BaseHandler) -> Optional[Dict[str, Any]]:
    logger.debug(f"Extracting code structure for '{file_path}' (language: {language})")
    try:
        structure = await asyncio.to_thread(handler.extract_structure, content, file_path)
        if structure is None:
            logger.error(f"Extracted structure is None for '{file_path}'")
            return None
        return structure
    except Exception as e:
        logger.error(f"Error extracting structure from '{file_path}': {e}", exc_info=True)
        return None

async def backup_and_write_new_content(file_path: str, new_content: str) -> None:
    backup_path = f'{file_path}.bak'
    try:
        if os.path.exists(backup_path):
            os.remove(backup_path)
            logger.debug(f"Removed existing backup at '{backup_path}'.")
        await asyncio.to_thread(shutil.copy, file_path, backup_path)
        logger.debug(f"Backup created at '{backup_path}'.")
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(new_content)
        logger.info(f"Inserted documentation into '{file_path}'.")
    except Exception as e:
        logger.error(f"Error writing to '{file_path}': {e}", exc_info=True)
        if os.path.exists(backup_path):
            await asyncio.to_thread(shutil.copy, backup_path, file_path)
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
    logger.debug(f"Fetching documentation using REST API for deployment: {deployment_name}")

    url = f"{azure_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={azure_api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_api_key,
    }

    for attempt in range(1, retry + 1):
        try:
            async with semaphore, session.post(url, headers=headers, json={
                "messages": [{"role": "user", "content": prompt}],
                "functions": function_schema["functions"],
                "function_call": {"name": "generate_documentation"},
            }) as response:
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
                                    continue
                        logger.error("No function_call found in the response.")
                    else:
                        logger.error("No choices found in the response.")
                else:
                    error_text = await response.text()
                    logger.error(f"Request failed with status {response.status}: {error_text}")

            if attempt < retry:
                wait_time = min(2 ** attempt, 16)
                logger.info(f"Retrying after {wait_time} seconds... (Attempt {attempt}/{retry})")
                await asyncio.sleep(wait_time)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            if attempt < retry:
                wait_time = min(2 ** attempt, 16)
                logger.info(f"Retrying after {wait_time} seconds... (Attempt {attempt}/{retry})")
                await asyncio.sleep(wait_time)

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

        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            logger.debug(f"File content for '{file_path}' read successfully.")
        except Exception as e:
            logger.error(f"Failed to read '{file_path}': {e}", exc_info=True)
            return None

        documentation = None
        code_structure = None

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
                else:
                    documentation['halstead'] = code_structure.get('halstead', {})
                    documentation['maintainability_index'] = code_structure.get('maintainability_index', None)

                    function_complexity = {}
                    for func in code_structure.get('functions', []):
                        function_complexity[func['name']] = func.get('complexity', 0)

                    for cls in code_structure.get('classes', []):
                        class_name = cls['name']
                        for method in cls.get('methods', []):
                            full_method_name = f"{class_name}.{method['name']}"
                            function_complexity[full_method_name] = method.get('complexity', 0)

                    for func in documentation.get('functions', []):
                        func_name = func['name']
                        func['complexity'] = function_complexity.get(func_name, 0)

                    for cls in documentation.get('classes', []):
                        class_name = cls['name']
                        for method in cls.get('methods', []):
                            full_method_name = f"{class_name}.{method['name']}"
                            method['complexity'] = function_complexity.get(full_method_name, 0)

        except Exception as e:
            logger.error(f"Error during code structure extraction or documentation generation for '{file_path}': {e}", exc_info=True)

        new_content = content

        if documentation and not safe_mode:
            try:
                new_content = await asyncio.to_thread(handler.insert_docstrings, content, documentation)

                if language.lower() == 'python':
                    new_content = await clean_unused_imports_async(new_content, file_path)
                    new_content = await format_with_black_async(new_content)

                is_valid = await asyncio.to_thread(handler.validate_code, new_content, file_path)
                if is_valid:
                    await backup_and_write_new_content(file_path, new_content)
                    logger.info(f"Documentation inserted into '{file_path}'")
                else:
                    logger.error(f"Code validation failed for '{file_path}'.")
            except Exception as e:
                logger.error(f"Error processing code documentation for '{file_path}': {e}", exc_info=True)
                new_content = content

        file_content = await write_documentation_report(
            documentation=documentation or {},
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
    logger.info('Starting process of all files.')
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
            azure_api_version=azure_api_version
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
            logger.error(f'Error processing a file: {e}', exc_info=True)
            if 'sentry_sdk' in globals():
                sentry_sdk.capture_exception(e)

    logger.info('Completed processing all files.')

    final_content = '\n\n'.join(documentation_contents)

    if final_content:
        toc = generate_table_of_contents(final_content)
        report_content = '# Documentation Generation Report\n\n## Table of Contents\n\n' + toc + '\n\n' + final_content

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

    logger.info('Running Flake8 on processed files for final linting.')
    for file_path in file_paths:
        _, ext = os.path.splitext(file_path)
        if ext.lower() in {'.py'}:
            flake8_output = await run_flake8_async(file_path)
            if flake8_output:
                logger.warning(f'Flake8 issues found in {file_path}:\n{flake8_output}')
    logger.info('Flake8 linting completed.')