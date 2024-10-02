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
        _, ext = os.path.splitext(file_path)
        if not is_valid_extension(ext, skip_types) or is_binary(file_path):
            logger.debug(f"Skipping file '{file_path}' due to extension '{ext}' or binary content.")
            return

        language = get_language(ext)
        if language == "plaintext":
            logger.debug(f"Skipping unsupported language in '{file_path}'.")
            return

        logger.info(f"Processing file: {file_path}")

        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            logger.debug(f"Read content from '{file_path}'.")
        except Exception as e:
            logger.error(f"Failed to read '{file_path}': {e}", exc_info=True)
            return

        # Attempt to extract the structure of the code
        try:
            if language == "python":
                code_structure = extract_python_structure(content)
            elif language in ["javascript", "typescript"]:
                # Use the Node.js script to extract structure
                structure_output = run_node_script('extract_structure.js', content)
                if not structure_output:
                    logger.warning(f"Could not extract code structure from '{file_path}'")
                    return
                code_structure = structure_output
            elif language == "html":
                code_structure = extract_html_structure(content)
            elif language == "css":
                code_structure = extract_css_structure(content)
            else:
                logger.warning(f"Language '{language}' not supported for structured extraction.")
                return

            if not code_structure:
                logger.warning(f"Could not extract code structure from '{file_path}'")
                return

            logger.debug(f"Extracted code structure from '{file_path}': {code_structure}")

        except Exception as e:
            logger.error(f"Error extracting structure from '{file_path}': {e}", exc_info=True)
            return

        # Generate the documentation using the updated prompt
        prompt = generate_documentation_prompt(
            code_structure=code_structure,
            project_info=project_info,
            style_guidelines=style_guidelines,
            language=language
        )
        logger.debug(f"Generated prompt for '{file_path}': {prompt}")

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

        summary = documentation.get("summary", "")
        changes = documentation.get("changes", [])

        documentation["source_code"] = content  # Always include the source code

        # Compute relative path
        relative_path = os.path.relpath(file_path, repo_root)

        # Insert docstrings or comments based on language
        if language == "python":
            new_content = insert_python_docstrings(content, documentation)
            # Validate Python code if applicable
            if not is_valid_python_code(new_content):
                logger.error(f"Modified Python code is invalid. Aborting insertion for '{file_path}'")
                return
            # Check with flake8
            # Write to a temporary file for flake8 checking
            temp_file_path = None  # Initialize before try
            try:
                with tempfile.NamedTemporaryFile('w', delete=False, suffix='.py') as temp_file:
                    temp_file.write(new_content)
                    temp_file_path = temp_file.name
                # Check with flake8
                if not check_with_flake8(temp_file_path):
                    logger.warning(f"flake8 compliance failed for '{file_path}'. Continuing processing.")
                # Optionally, capture the flake8 issues for documentation
                try:
                    with open(temp_file_path, 'r') as f:
                        code_content = f.read()
                    # You can include flake8 issues in the markdown output if desired
                    flake8_issues = run_flake8(file_path)
                except Exception as e:
                    logger.error(f"Error reading '{temp_file_path}': {e}", exc_info=True)
                finally:
                    # Remove the temporary file after checking
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                logger.debug(f"Processed Python file '{file_path}'.")
        except Exception as e:
            logger.error(f"Error processing Python file '{file_path}': {e}", exc_info=True)
            return
        elif language in ["javascript", "typescript"]:
            try:
                # Insert docstrings using the Node.js script
                new_content = run_node_insert_docstrings('insert_docstrings.js', content)
                if not new_content:
                    logger.warning(f"Could not insert docstrings into '{file_path}'")
                    return
                # Optionally, format the modified code using Black or Prettier
                new_content = format_with_black(new_content)  # If desired, use a JS formatter like Prettier instead
                # Clean unused imports if applicable (for JS/TS, you might use a different tool)
                # For demonstration, using clean_unused_imports (which is for Python)
                # Consider implementing a similar function for JS/TS if needed
                new_content = clean_unused_imports(new_content)  # Not applicable for JS/TS
                # Check with flake8 if necessary or use a JS linter like eslint
                # For demonstration, skipping flake8 for JS/TS
                logger.debug(f"Processed {language} file '{file_path}'.")
            except Exception as e:
                logger.error(f"Error processing {language} file '{file_path}': {e}", exc_info=True)
                return
        elif language == "html":
            try:
                new_content = insert_html_comments(content, documentation)
                # (Optional) Integrate HTML formatters here
                logger.debug(f"Processed HTML file '{file_path}'.")
            except Exception as e:
                logger.error(f"Error processing HTML file '{file_path}': {e}", exc_info=True)
                return
        elif language == "css":
            try:
                if "rules" not in documentation:
                    logger.error(f"Documentation for '{file_path}' lacks 'rules'. Skipping insertion.")
                    new_content = content
                else:
                    new_content = insert_css_docstrings(content, documentation)
                    # (Optional) Integrate CSS formatters here
                    logger.debug(f"Processed CSS file '{file_path}'.")
            except Exception as e:
                logger.error(f"Error processing CSS file '{file_path}': {e}", exc_info=True)
                return
        else:
            new_content = content

        # Safe mode: Do not overwrite the original file, just log the changes
        if safe_mode:
            logger.info(f"Safe mode active. Skipping file modification for '{file_path}'")
        else:
            # Backup and write new content
            backup_path = f"{file_path}.bak"  # Initialize backup_path here
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
                # Restore from backup if write fails
                if backup_path and os.path.exists(backup_path):
                    shutil.copy(backup_path, file_path)
                    os.remove(backup_path)
                    logger.info(f"Restored original file from backup for '{file_path}'")
                return

    # After inserting docstrings and formatting
    try:
        async with output_lock:
            async with aiofiles.open(output_file, "a", encoding="utf-8") as f:
                header = f"# File: {relative_path}\n\n"  # Use relative path
                summary_section = f"## Summary\n\n{summary}\n\n"
                changes_section = (
                    "## Changes Made\n\n" + "\n".join(f"- {change}" for change in changes) + "\n\n"
                )
                code_block = f"```{language}\n{new_content}\n```\n\n"

                # Add flake8 issues if any
                if flake8_output:
                    flake8_section = f"## flake8 Issues\n\n```\n{flake8_output}\n```\n\n"
                else:
                    flake8_section = ""

                await f.write(header)
                await f.write(summary_section)
                await f.write(changes_section)
                await f.write(code_block)
                await f.write(flake8_section)
        logger.info(f"Successfully processed and documented '{file_path}'")
    except Exception as e:
        logger.error(f"Error writing documentation for '{file_path}': {e}", exc_info=True)
except Exception as e:
    logger.error(f"Error processing file '{file_path}': {e}", exc_info=True)
