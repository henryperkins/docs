import os
import logging
import shutil
import aiofiles
import aiohttp
import asyncio
from typing import Any, Set, Optional, Dict, List
from tqdm.asyncio import tqdm  # For progress bar

from language_functions.base_handler import BaseHandler
from language_functions import (
    PythonHandler,
    JSTsHandler,
    GoHandler,
    CppHandler,
    HTMLHandler,
    CSSHandler,
)
from utils import (
    is_binary,
    get_language,
    is_valid_extension,
    generate_documentation_prompt,
    fetch_documentation,
    extract_code_structure,
    backup_and_write_new_content,
    write_documentation_report,
    generate_table_of_contents,
)

# Initialize Sentry
import sentry_sdk
sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",  # Replace with your actual Sentry DSN
    traces_sample_rate=1.0,
)

logger = logging.getLogger(__name__)

async def extract_code_structure(
    content: str, file_path: str, language: str, handler: BaseHandler
) -> Optional[dict]:
    """Extracts code structure based on language using the appropriate handler."""
    logger.debug(f"Extracting code structure for '{file_path}' (language: {language})")
    try:
        loop = asyncio.get_event_loop()
        # Run the potentially blocking extract_structure in a thread pool
        return await loop.run_in_executor(None, handler.extract_structure, content, file_path)
    except Exception as e:
        logger.error(f"Error extracting structure from '{file_path}': {e}", exc_info=True)
        return None

async def backup_and_write_new_content(file_path: str, new_content: str) -> None:
    """Creates a backup of the file and writes the new content."""
    backup_path = f"{file_path}.bak"
    try:
        if os.path.exists(backup_path):
            os.remove(backup_path)
            logger.debug(f"Removed existing backup at '{backup_path}'.")
        shutil.copy(file_path, backup_path)
        logger.debug(f"Backup created at '{backup_path}'.")
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(new_content)
        logger.info(f"Inserted documentation into '{file_path}'.")
    except Exception as e:
        logger.error(f"Error writing to '{file_path}': {e}", exc_info=True)
        if os.path.exists(backup_path):
            shutil.copy(backup_path, file_path)
            os.remove(backup_path)
            logger.info(f"Restored original file from backup for '{file_path}'.")


async def process_file(
    session: aiohttp.ClientSession,
    file_path: str,
    skip_types: Set[str],
    semaphore: asyncio.Semaphore,
    model_name: str,
    function_schema: dict,
    repo_root: str,
    project_info: str,
    style_guidelines: str,
    safe_mode: bool,
    use_azure: bool = False,
) -> Optional[str]:
    """Processes a single file: extracts structure, generates documentation, inserts documentation, validates, and returns the documentation content."""
    logger.debug(f"Processing file: {file_path}")
    try:
        _, ext = os.path.splitext(file_path)
        if not is_valid_extension(ext, skip_types) or is_binary(file_path):
            logger.debug(f"Skipping file '{file_path}' due to invalid extension or binary content.")
            return None

        language = get_language(ext)
        logger.debug(f"Detected language for '{file_path}': {language}")

        # Create a language handler based on the detected language
        handler: Optional[BaseHandler] = None
        if language == "python":
            handler = PythonHandler(function_schema)
        elif language in ["javascript", "typescript"]:
            handler = JSTsHandler(function_schema)
        elif language == "go":
            handler = GoHandler(function_schema)
        elif language == "cpp":
            handler = CppHandler(function_schema)
        elif language == "html":
            handler = HTMLHandler(function_schema)
        elif language == "css":
            handler = CSSHandler(function_schema)

        if handler is None:
            logger.warning(f"Unsupported language: {language}")
            return None

        logger.info(f"Processing file: {file_path}")
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            logger.debug(f"File content for '{file_path}' read successfully.")
        except Exception as e:
            logger.error(f"Failed to read '{file_path}': {e}", exc_info=True)
            return None

        documentation = None
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
                    language=language,
                )
                documentation = await fetch_documentation(
                    session=session,
                    prompt=prompt,
                    semaphore=semaphore,
                    model_name=model_name,
                    function_schema=function_schema,
                    use_azure=use_azure,  # Pass use_azure to fetch_documentation
                )
                if not documentation:
                    logger.error(f"Failed to generate documentation for '{file_path}'.")
        except Exception as e:
            logger.error(
                f"Error during code structure extraction or documentation generation for '{file_path}': {e}",
                exc_info=True,
            )

        new_content = content  # Default to original content
        if documentation:
            try:
                loop = asyncio.get_event_loop()
                # Insert docstrings in a thread to avoid blocking
                new_content = await loop.run_in_executor(None, handler.insert_docstrings, content, documentation)
                if not safe_mode:
                    # Validate the new content
                    is_valid = await loop.run_in_executor(None, handler.validate_code, new_content, file_path)
                    if is_valid:
                        await backup_and_write_new_content(file_path, new_content)
                        logger.info(f"Documentation inserted into '{file_path}'")
                    else:
                        logger.error(f"Code validation failed for '{file_path}'.")
                else:
                    logger.info(f"Safe mode active. Skipping file modification for '{file_path}'")
            except Exception as e:
                logger.error(f"Error processing code documentation for '{file_path}': {e}", exc_info=True)
                new_content = content  # Revert to original content on error

        file_content = await write_documentation_report(
            documentation=documentation,
            language=language,
            file_path=file_path,
            repo_root=repo_root,
            new_content=new_content,
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
    model_name: str,
    function_schema: dict,
    repo_root: str,
    project_info: Optional[str],
    style_guidelines: Optional[str],
    safe_mode: bool = False,
    output_file: str = "output.md",
    use_azure: bool = False,
) -> None:
    """Processes multiple files for documentation."""
    logger.info("Starting process of all files.")
    tasks = []
    for file_path in file_paths:
        task = asyncio.create_task(
            process_file(
                session=session,
                file_path=file_path,
                skip_types=skip_types,
                semaphore=semaphore,
                model_name=model_name,
                function_schema=function_schema,
                repo_root=repo_root,
                project_info=project_info,
                style_guidelines=style_guidelines,
                safe_mode=safe_mode,
                use_azure=use_azure,
            )
        )
        tasks.append(task)

    documentation_contents = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Files"):
        try:
            file_content = await f
            if file_content:
                documentation_contents.append(file_content)
        except Exception as e:
            logger.error(f"Error processing a file: {e}", exc_info=True)
            # Capture exception in Sentry
            sentry_sdk.capture_exception(e)

    logger.info("Completed processing all files.")

    final_content = "\n\n".join(documentation_contents)
    toc = generate_table_of_contents(final_content)
    report_content = (
        "# Documentation Generation Report\n\n## Table of Contents\n\n"
        + toc
        + "\n\n"
        + final_content
    )

    try:
        async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
            await f.write(report_content)
        logger.info(f"Documentation report written to '{output_file}'")
    except Exception as e:
        logger.error(
            f"Error writing final documentation to '{output_file}': {e}", exc_info=True
        )
        # Capture exception in Sentry
        sentry_sdk.capture_exception(e)