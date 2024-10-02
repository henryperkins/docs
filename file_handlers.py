# file_handlers.py

import os
import fnmatch
import json
import logging
import ast
import astor
import shutil
from typing import Set, List, Optional, Dict, Tuple
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
    fetch_documentation,
    function_schema,
    format_with_black,
    clean_unused_imports,
    check_with_flake8,
    run_flake8,
    run_node_script,
    run_node_insert_docstrings,
    call_openai_function,
    load_json_schema,
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
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

async def main():
    # Load the function schema from the JSON schema file
    schema_path = 'function_schema.json'  # Ensure the path is correct
    function_schema = load_json_schema(schema_path)
    
    if not function_schema:
        logger.critical(f"Failed to load function schema from '{schema_path}'. Exiting.")
        return

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(10)  # Adjust concurrency as needed
        
        # Define a meaningful prompt based on your documentation needs
        prompt = (
            "You are an experienced software developer tasked with generating comprehensive documentation "
            "for the following Python code structure."
        )
        
        model_name = "gpt-4"
        
        try:
            documentation = await fetch_documentation(
                session=session,
                prompt=prompt,
                semaphore=semaphore,
                model_name=model_name,
                function_schema=function_schema
            )
            
            if documentation:
                # Pretty-print the documentation for readability
                formatted_doc = json.dumps(documentation, indent=2)
                logger.info("Received Documentation:")
                logger.info(formatted_doc)
            else:
                logger.warning("No documentation was returned from the API.")
        
        except Exception as e:
            logger.error(f"An error occurred while fetching documentation: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())

async def insert_docstrings_for_file(js_ts_file: str, documentation_file: str) -> None:
    """
    Inserts docstrings into JS/TS files by running an external Node.js script.

    Parameters:
        js_ts_file (str): Path to the JS/TS source file.
        documentation_file (str): Path to the documentation JSON file.
    """
    logger.debug(f"Entering insert_docstrings_for_file with js_ts_file={js_ts_file}, documentation_file={documentation_file}")
    try:
        process = await asyncio.create_subprocess_exec(
            'node',
            'scripts/insert_docstrings.js',
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
    except Exception as e:
        logger.error(f"Exception while inserting docstrings into {js_ts_file}: {e}", exc_info=True)
    finally:
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
    """
    Processes a single file: extracts structure, generates documentation, inserts documentation,
    validates, backs up, writes changes, and logs the process.

    Parameters:
        session (aiohttp.ClientSession): The HTTP session for API calls.
        file_path (str): Path to the file to process.
        skip_types (Set[str]): Set of file extensions to skip.
        output_file (str): Path to the output Markdown report.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent API calls.
        output_lock (asyncio.Lock): Lock to synchronize writing to the output report.
        model_name (str): OpenAI model to use.
        function_schema (dict): JSON schema for OpenAI function calling.
        repo_root (str): Root directory of the repository.
        project_info (Optional[str]): Information about the project.
        style_guidelines (Optional[str]): Style guidelines for documentation.
        safe_mode (bool): If True, do not modify files.
    """
    logger.debug(f"Entering process_file with file_path={file_path}")
    summary = ""
    changes = []
    new_content = ""

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
            logger.debug(f"File content for {file_path}:\n{content[:500]}...")  # Log first 500 characters for brevity
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
        logger.debug(f"Generated prompt for '{file_path}': {prompt[:500]}...")  # Log first 500 characters for brevity

        # Fetch documentation from OpenAI using function calling
        documentation = await fetch_documentation(
            session=session,
            prompt=prompt,
            semaphore=semaphore,
            model_name=model_name,
            function_schema=function_schema
        )
        if not documentation:
            logger.error(f"Failed to generate documentation for '{file_path}'.")
            return

        # Extract summary and changes from documentation
        summary = documentation.get("summary", "").strip()
        changes = documentation.get("changes_made", [])

        if not summary and not changes:
            logger.warning(f"No documentation details provided for '{file_path}'. Skipping insertion.")
            return

        # Insert documentation into code based on language
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

    Parameters:
        content (str): The source code.
        file_path (str): Path to the source file.
        language (str): Programming language.

    Returns:
        Optional[dict]: Extracted code structure or None.
    """
    try:
        if language == "python":
            logger.debug(f"Extracting Python structure for file '{file_path}'")
            return extract_python_structure(content)
        elif language in ["javascript", "typescript"]:
            logger.debug(f"Extracting JS/TS structure for file '{file_path}'")
            return await extract_js_ts_structure(file_path, content, language)
        elif language == "html":
            logger.debug(f"Extracting HTML structure for file '{file_path}'")
            return extract_html_structure(content)
        elif language == "css":
            logger.debug(f"Extracting CSS structure for file '{file_path}'")
            return extract_css_structure(content)
        else:
            logger.warning(f"Unsupported language for structure extraction: {language}")
            return None
    except Exception as e:
        logger.error(f"Error extracting structure from '{file_path}': {e}", exc_info=True)
        return None


async def process_code_documentation(content: str, documentation: dict, language: str, file_path: str) -> tuple[str, list, str]:
    """
    Inserts the docstrings or comments into the code based on the documentation.

    Parameters:
        content (str): The original source code.
        documentation (dict): Documentation details from AI.
        language (str): Programming language.
        file_path (str): Path to the source file.

    Returns:
        tuple[str, list, str]: Summary, list of changes, and modified code.
    """
    summary = documentation.get("summary", "")
    changes = documentation.get("changes_made", [])
    new_content = content

    try:
        if language == "python":
            new_content = insert_python_docstrings(content, documentation)
            if not is_valid_python_code(new_content):
                logger.error(f"Modified Python code is invalid. Aborting insertion for '{file_path}'")
        elif language in ["javascript", "typescript"]:
            new_content = insert_js_ts_docstrings(content, documentation)
        elif language == "html":
            new_content = insert_html_comments(content, documentation)
        elif language == "css":
            new_content = insert_css_docstrings(content, documentation)
        else:
            logger.warning(f"Unsupported language '{language}' for documentation insertion.")
        
        logger.debug(f"Processed {language} file '{file_path}'.")
        return summary, changes, new_content

    except Exception as e:
        logger.error(f"Error processing {language} file '{file_path}': {e}", exc_info=True)
        return summary, changes, content

async def backup_and_write_new_content(file_path: str, new_content: str) -> None:
    """
    Creates a backup of the file and writes the new content.

    Parameters:
        file_path (str): Path to the original file.
        new_content (str): Modified content to write.
    """
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
        # Attempt to restore from backup
        if os.path.exists(backup_path):
            shutil.copy(backup_path, file_path)
            os.remove(backup_path)
            logger.info(f"Restored original file from backup for '{file_path}'.")

async def write_documentation_report(
    output_file: str,
    summary: str,
    changes: list,
    new_content: str,
    language: str,
    output_lock: asyncio.Lock,
    file_path: str,
    repo_root: str
) -> None:
    """
    Writes the summary, changes, and new content to the output markdown report.

    Parameters:
        output_file (str): Path to the output Markdown file.
        summary (str): Summary of the documentation.
        changes (list): List of changes made.
        new_content (str): Modified source code with inserted documentation.
        language (str): Programming language.
        output_lock (asyncio.Lock): Lock for synchronizing file writes.
        file_path (str): Path to the source file.
        repo_root (str): Root path of the repository.
    """
    try:
        # Determine the relative path of the file for documentation
        relative_path = os.path.relpath(file_path, repo_root)

        async with output_lock:
            async with aiofiles.open(output_file, 'a', encoding='utf-8') as f:
                # Create the header and sections for the documentation
                header = f'# File: {relative_path}\n\n'
                summary_section = f'## Summary\n\n{summary}\n\n'

                changes_section = '## Changes Made\n\n'
                if changes:
                    changes_section += '\n'.join(f'- {change}' for change in changes) + '\n\n'
                else:
                    changes_section += 'No changes were made to this file.\n\n'

                # Extract code structure from the new content
                structure = await extract_code_structure(new_content, file_path, language)

                # Build function table
                function_table_header = '| Function | Arguments | Description |\n|----------|-----------|-------------|\n'
                function_table_rows = ''
                if not structure.get('functions'):
                    function_table_rows = '| No functions are defined in this file. | | |\n'
                else:
                    for func in structure['functions']:
                        func_name = func.get('name', 'Unnamed Function')
                        func_args = ', '.join(func.get('args', []))
                        func_doc = func.get('docstring', 'No description provided.')
                        func_type = 'async ' if func.get('async', False) else ''
                        function_table_rows += f"| `{func_type}{func_name}` | `{func_args}` | {func_doc.splitlines()[0]} |\n"

                # Build class table
                class_table_header = '## Classes\n\n'
                class_table_rows = ''
                if not structure.get('classes'):
                    class_table_rows = 'No classes are defined in this file.\n\n'
                else:
                    for cls in structure['classes']:
                        cls_name = cls.get('name', 'Unnamed Class')
                        class_table_rows += f'### `{cls_name}`\n\n'
                        if cls.get('methods'):
                            class_table_rows += '#### Methods:\n\n'
                            for method in cls['methods']:
                                method_name = method.get('name', 'Unnamed Method')
                                method_args = ', '.join(method.get('args', []))
                                method_doc = method.get('docstring', 'No description provided.')
                                method_type = 'async ' if method.get('async', False) else ''
                                class_table_rows += f"- **`{method_type}{method_name}({method_args})`**: {method_doc.splitlines()[0]}\n"
                        else:
                            class_table_rows += 'No methods defined in this class.\n\n'

                # Include the code block with syntax highlighting
                code_block = f'```{language}\n{new_content}\n```\n\n---\n\n'

                # Write all sections to the output file
                await f.write(header)
                await f.write(summary_section)
                await f.write(changes_section)
                await f.write('## Functions\n\n')
                await f.write(function_table_header)
                await f.write(function_table_rows)
                await f.write(class_table_header)
                await f.write(class_table_rows)
                await f.write(code_block)

    except Exception as e:
        logger.error(f"Error writing documentation for '{file_path}': {e}", exc_info=True)


async def process_all_files(
    session: aiohttp.ClientSession,
    file_paths: List[str],
    skip_types: Set[str],
    output_file: str,
    semaphore: asyncio.Semaphore,
    output_lock: asyncio.Lock,
    model_name: str,
    function_schema: dict,
    repo_root: str,
    project_info: str,
    style_guidelines: str,
    safe_mode: bool = False,
) -> None:
    logger.info("Starting process of all files.")
    tasks = []
    
    for file_path in file_paths:
        # Call process_file for each file asynchronously
        task = process_file(
            session=session,
            file_path=file_path,
            skip_types=skip_types,
            output_file=output_file,
            semaphore=semaphore,
            output_lock=output_lock,
            model_name=model_name,
            function_schema=function_schema,
            repo_root=repo_root,
            project_info=project_info,
            style_guidelines=style_guidelines,
            safe_mode=safe_mode
        )
        tasks.append(task)
    
    # Use tqdm for progress tracking
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await f
    
    logger.info("Completed processing all files.")
