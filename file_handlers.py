import os
import sys
import json
import logging
import ast
import astor
import shutil
from typing import Set, List, Optional, Dict
import aiofiles
import aiohttp
import asyncio
from tqdm.asyncio import tqdm
import subprocess
from pathlib import Path
import tempfile

from language_functions import (
    extract_python_structure,
    insert_python_docstrings,
    is_valid_python_code,
    extract_js_ts_structure,
    insert_js_ts_docstrings,
    extract_html_structure,
    insert_html_comments,
    extract_css_structure,
    insert_css_docstrings,
)

from utils import (
    load_config,
    is_binary,
    get_language,
    get_all_file_paths,
    is_valid_extension,
    OPENAI_API_KEY,
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    DEFAULT_SKIP_TYPES,
    generate_documentation_prompt,
    fetch_documentation_with_retries,
    function_schema,
    format_with_black,
    clean_unused_imports,
    check_with_flake8,
    run_flake8,
    run_node_script,
    run_node_insert_docstrings,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of logs

# Create formatter with module, function, and line number
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(funcName)s:%(lineno)d:%(message)s')

# Create file handler which logs debug and higher level messages
file_handler = logging.FileHandler('file_handlers.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Change to DEBUG for more verbosity on console
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


async def main():
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(10)
        prompt = "Your prompt here"
        model_name = "gpt-4"
        result = await fetch_documentation_with_retries(
            session=session,
            prompt=prompt,
            semaphore=semaphore,
            model_name=model_name,
            function_schema=function_schema
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main())


async def insert_docstrings_for_file(js_ts_file: str, documentation_file: str) -> None:
    logger.debug(f"Entering insert_docstrings_for_file with js_ts_file={js_ts_file}, documentation_file={documentation_file}")
    process = await asyncio.create_subprocess_exec(
        'node',
        'insert_docstrings.js',
        js_ts_file,
        documentation_file,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        logger.error(f"Error inserting docstrings into {js_ts_file}: {stderr.decode().strip()}")
    else:
        logger.info(stdout.decode().strip())
    logger.debug("Exiting insert_docstrings_for_file")


async def process_file(
    session: aiohttp.ClientSession,
    file_path: str,
    skip_types: Set[str],
    output_file: str,
    semaphore: asyncio.Semaphore,
    output_lock: asyncio.Lock,
    model_name: str,
    function_schema: dict,
    repo_root: str,
    project_info: Optional[str] = None,
    style_guidelines: Optional[str] = None,
    safe_mode: bool = False,
) -> None:
    logger.debug(f"Entering process_file with file_path={file_path}")
    summary = ""
    changes = []
    
    try:
        # Check if file extension is valid or binary, and get language type
        _, ext = os.path.splitext(file_path)
        if not is_valid_extension(ext, skip_types) or is_binary(file_path):
            logger.debug(f"Skipping file '{file_path}' due to invalid extension or binary content.")
            return

        language = get_language(ext)
        if language == "plaintext":
            logger.debug(f"Skipping unsupported language in '{file_path}'.")
            return

        logger.info(f"Processing file: {file_path}")

        # Read the file content
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            logger.debug(f"Read content from '{file_path}'.")
        except Exception as e:
            logger.error(f"Failed to read '{file_path}': {e}", exc_info=True)
            return

        # Extract code structure based on the language
        code_structure = await extract_code_structure(content, file_path, language)
        if not code_structure:
            logger.warning(f"Could not extract code structure from '{file_path}'")
            return
        
        # Generate the documentation prompt and log it
        prompt = generate_documentation_prompt(
            code_structure=code_structure,
            project_info=project_info,
            style_guidelines=style_guidelines,
            language=language
        )
        logger.debug(f"Generated prompt for '{file_path}': {prompt}")

        # Fetch documentation from OpenAI
        documentation = await fetch_documentation_with_retries(
            session=session,
            prompt=prompt,
            semaphore=semaphore,
            model_name=model_name,
            function_schema=function_schema
        )
        if not documentation:
            logger.error(f"Failed to generate documentation for '{file_path}'.")
            return

        summary, changes, new_content = await process_code_documentation(
            content, documentation, language, file_path
        )

        if safe_mode:
            logger.info(f"Safe mode active. Skipping file modification for '{file_path}'")
        else:
            await backup_and_write_new_content(file_path, new_content)

        # Write the documentation report
        await write_documentation_report(output_file, summary, changes, new_content, language, output_lock, file_path, repo_root)
        
        logger.info(f"Successfully processed and documented '{file_path}'")
    
    except Exception as e:
        logger.error(f"Error processing file '{file_path}': {e}", exc_info=True)


async def extract_code_structure(content: str, file_path: str, language: str) -> Optional[dict]:
    """
    Extracts the structure of the code based on the language.
    """
    try:
        if language == "python":
            return extract_python_structure(content)
        elif language in ["javascript", "typescript"]:
            structure_output = run_node_script('extract_structure.js', content)
            if not structure_output:
                logger.warning(f"Could not extract code structure from '{file_path}'")
                return None
            return structure_output
        elif language == "html":
            return extract_html_structure(content)
        elif language == "css":
            return extract_css_structure(content)
        else:
            logger.warning(f"Language '{language}' not supported for structured extraction.")
            return None
    except Exception as e:
        logger.error(f"Error extracting structure from '{file_path}': {e}", exc_info=True)
        return None


async def process_code_documentation(content: str, documentation: dict, language: str, file_path: str) -> tuple[str, list, str]:
    """
    Inserts the docstrings or comments into the code based on the documentation.
    """
    summary = documentation.get("summary", "")
    changes = documentation.get("changes", [])
    new_content = content

    try:
        if language == "python":
            new_content = insert_python_docstrings(content, documentation)
            if not is_valid_python_code(new_content):
                logger.error(f"Modified Python code is invalid. Aborting insertion for '{file_path}'")
        elif language in ["javascript", "typescript"]:
            new_content = run_node_insert_docstrings('insert_docstrings.js', content)
            new_content = format_with_black(new_content)
        elif language == "html":
            new_content = insert_html_comments(content, documentation)
        elif language == "css":
            new_content = insert_css_docstrings(content, documentation)
        
        logger.debug(f"Processed {language} file '{file_path}'.")
        return summary, changes, new_content
    
    except Exception as e:
        logger.error(f"Error processing {language} file '{file_path}': {e}", exc_info=True)
        return summary, changes, content


async def backup_and_write_new_content(file_path: str, new_content: str) -> None:
    """
    Creates a backup of the file and writes the new content.
    """
    backup_path = f"{file_path}.bak"
    try:
        if os.path.exists(backup_path):
            os.remove(backup_path)
        shutil.copy(file_path, backup_path)
        logger.debug(f"Backup created at '{backup_path}'")

        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(new_content)
        logger.info(f"Inserted documentation into '{file_path}'")
    
    except Exception as e:
        logger.error(f"Error writing to '{file_path}': {e}", exc_info=True)
        if backup_path and os.path.exists(backup_path):
            shutil.copy(backup_path, file_path)
            os.remove(backup_path)
            logger.info(f"Restored original file from backup for '{file_path}'")


async def write_documentation_report(
    output_file: str, summary: str, changes: list, new_content: str, language: str,
    output_lock: asyncio.Lock, file_path: str, repo_root: str
) -> None:
    """
    Writes the summary, changes, and new content to the output markdown report.
    """
    try:
        relative_path = os.path.relpath(file_path, repo_root)
        async with output_lock:
            async with aiofiles.open(output_file, "a", encoding="utf-8") as f:
                header = f"# File: {relative_path}\n\n"
                summary_section = f"## Summary\n\n{summary}\n\n"
                changes_section = f"## Changes Made\n\n" + "\n".join(f"- {change}" for change in changes) + "\n\n"
                code_block = f"```{language}\n{new_content}\n```\n\n"
                await f.write(header)
                await f.write(summary_section)
                await f.write(changes_section)
                await f.write(code_block)
    except Exception as e:
        logger.error(f"Error writing documentation for '{file_path}': {e}", exc_info=True)

