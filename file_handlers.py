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
    semaphore: asyncio.Semaphore,
    model_name: str,
    function_schema: dict,
    repo_root: str,
    project_info: Optional[str] = None,
    style_guidelines: Optional[str] = None,
    safe_mode: bool = False
) -> str:
    """
    Processes a single file: extracts structure, generates documentation, inserts documentation,
    validates, backs up, writes changes, and returns the documentation content.

    Parameters:
        session (aiohttp.ClientSession): The HTTP session for API calls.
        file_path (str): Path to the file to process.
        skip_types (Set[str]): Set of file extensions to skip.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent API calls.
        model_name (str): OpenAI model to use.
        function_schema (dict): JSON schema for OpenAI function calling.
        repo_root (str): Root directory of the repository.
        project_info (Optional[str]): Information about the project.
        style_guidelines (Optional[str]): Style guidelines for documentation.
        safe_mode (bool): If True, do not modify files.

    Returns:
        str: The documentation content for this file.
    """
    logger.debug(f'Entering process_file with file_path={file_path}')
    summary = ''
    changes = []
    new_content = ''
    documentation_content = ''
    try:
        _, ext = os.path.splitext(file_path)
        if not is_valid_extension(ext, skip_types) or is_binary(file_path):
            logger.debug(
                f"Skipping file '{file_path}' due to invalid extension or binary content."
            )
            return ''
        language = get_language(ext)
        if language == 'plaintext':
            logger.debug(f"Skipping unsupported language in '{file_path}'.")
            return ''
        logger.info(f'Processing file: {file_path}')
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            logger.debug(f'File content for {file_path}:\n{content[:500]}...')
        except Exception as e:
            logger.error(f"Failed to read '{file_path}': {e}", exc_info=True)
            return ''
        code_structure = await extract_code_structure(content, file_path, language)
        if not code_structure:
            logger.warning(f"Could not extract code structure from '{file_path}'")
            return ''
        prompt = generate_documentation_prompt(
            code_structure=code_structure,
            project_info=project_info,
            style_guidelines=style_guidelines,
            language=language
        )
        logger.debug(f"Generated prompt for '{file_path}': {prompt[:500]}...")
        documentation = await fetch_documentation(
            session=session,
            prompt=prompt,
            semaphore=semaphore,
            model_name=model_name,
            function_schema=function_schema
        )
        if not documentation:
            logger.error(f"Failed to generate documentation for '{file_path}'.")
            return ''
        summary, changes, new_content = await process_code_documentation(
            content, documentation, language, file_path
        )
        if safe_mode:
            logger.info(f"Safe mode active. Skipping file modification for '{file_path}'")
        else:
            await backup_and_write_new_content(file_path, new_content)
        documentation_content = await write_documentation_report(
            summary, changes, new_content, language, file_path, repo_root
        )
        logger.info(f"Successfully processed and documented '{file_path}'")
        return documentation_content
    except Exception as e:
        logger.error(f"Error processing file '{file_path}': {e}", exc_info=True)
        return ''



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
    summary: str,
    changes: List[str],
    new_content: str,
    language: str,
    file_path: str,
    repo_root: str
) -> str:
    """
    Generates the documentation report content for a single file.

    Parameters:
        summary (str): Summary of the documentation.
        changes (List[str]): List of changes made.
        new_content (str): Modified source code with inserted documentation.
        language (str): Programming language.
        file_path (str): Path to the source file.
        repo_root (str): Root directory of the repository.

    Returns:
        str: The documentation content for the file.
    """
    try:
        # Determine the relative path of the file for documentation
        relative_path = os.path.relpath(file_path, repo_root)
        file_header = f'# File: {relative_path}\n\n'

        # Prepare the documentation sections
        summary_section = f'## Summary\n\n{summary.strip()}\n\n'
        changes_section = '## Changes Made\n\n'
        if changes:
            changes_section += '\n'.join(f'- {change.strip()}' for change in changes) + '\n\n'
        else:
            changes_section += 'No changes were made to this file.\n\n'

        # Extract code structure from the new content
        structure = await extract_code_structure(new_content, file_path, language)

        # Build the functions section
        function_section = ''
        if structure.get('functions'):
            function_section += '## Functions\n\n'
            function_table_header = '| Function | Arguments | Description |\n|----------|-----------|-------------|\n'
            function_table_rows = ''
            for func in structure['functions']:
                func_name = func.get('name', 'Unnamed Function')
                func_args = ', '.join(func.get('args', []))
                func_doc = func.get('docstring', 'No description provided.')
                func_type = 'async ' if func.get('async', False) else ''
                function_table_rows += f"| `{func_type}{func_name}` | `{func_args}` | {func_doc.splitlines()[0]} |\n"
            function_section += function_table_header + function_table_rows + '\n'
        else:
            function_section += '## Functions\n\nNo functions are defined in this file.\n\n'

        # Build the classes section
        class_section = ''
        if structure.get('classes'):
            class_section += '## Classes\n\n'
            for cls in structure['classes']:
                cls_name = cls.get('name', 'Unnamed Class')
                class_section += f'### Class: `{cls_name}`\n\n'
                class_doc = cls.get('docstring', 'No description provided.')
                class_section += f"{class_doc}\n\n"

                if cls.get('methods'):
                    class_section += '#### Methods:\n\n'
                    method_table_header = '| Method | Arguments | Description |\n|--------|-----------|-------------|\n'
                    method_table_rows = ''
                    for method in cls['methods']:
                        method_name = method.get('name', 'Unnamed Method')
                        method_args = ', '.join(method.get('args', []))
                        method_doc = method.get('docstring', 'No description provided.')
                        method_type = 'async ' if method.get('async', False) else ''
                        method_table_rows += f"| `{method_type}{method_name}` | `{method_args}` | {method_doc.splitlines()[0]} |\n"
                    class_section += method_table_header + method_table_rows + '\n'
                else:
                    class_section += 'No methods defined in this class.\n\n'
        else:
            class_section += '## Classes\n\nNo classes are defined in this file.\n\n'

        # Include the code block with syntax highlighting
        code_block = f'```{language}\n{new_content}\n```\n\n---\n\n'

        # Combine all sections
        file_content = (
            file_header +
            summary_section +
            changes_section +
            function_section +
            class_section +
            code_block
        )

        return file_content

    except Exception as e:
        logger.error(f"Error generating documentation for '{file_path}': {e}", exc_info=True)
        return ''

def generate_table_of_contents(markdown_content: str) -> str:
    """
    Generates a markdown table of contents from the given markdown content.

    Parameters:
        markdown_content (str): The markdown content to generate the TOC from.

    Returns:
        str: The generated table of contents in markdown format.
    """
    import re

    toc = []
    seen_anchors = set()
    for line in markdown_content.split('\n'):
        match = re.match(r'^(#{1,6})\s+(.*)', line)
        if match:
            level = len(match.group(1))  # Heading level (1 to 6)
            title = match.group(2).strip()
            # Generate an anchor link
            anchor = re.sub(r'[^\w\s\-]', '', title).lower()
            anchor = re.sub(r'\s+', '-', anchor)
            # Ensure unique anchors
            original_anchor = anchor
            counter = 1
            while anchor in seen_anchors:
                anchor = f"{original_anchor}-{counter}"
                counter += 1
            seen_anchors.add(anchor)
            indent = '  ' * (level - 1)
            toc.append(f'{indent}- [{title}](#{anchor})')
    return '\n'.join(toc)


async def process_all_files(
    session: aiohttp.ClientSession,
    file_paths: List[str],
    skip_types: Set[str],
    semaphore: asyncio.Semaphore,
    model_name: str,
    function_schema: dict,
    repo_root: str,
    project_info: str,
    style_guidelines: str,
    safe_mode: bool = False,
    output_file: str = 'output.md'
) -> None:
    logger.info('Starting process of all files.')
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
                safe_mode=safe_mode
            )
        )
        tasks.append(task)

    documentation_contents = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        file_content = await f
        if file_content:
            documentation_contents.append(file_content)

    logger.info('Completed processing all files.')

    # After processing all files, combine the contents
    final_content = '\n\n'.join(documentation_contents)

    # Generate TOC
    toc = generate_table_of_contents(final_content)

    # Build the final report with TOC
    report_content = '# Documentation Generation Report\n\n## Table of Contents\n\n' + toc + '\n\n' + final_content

    # Write to the output file
    try:
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            await f.write(report_content)
        logger.info(f'Documentation report written to {output_file}')
    except Exception as e:
        logger.error(f"Error writing final documentation to '{output_file}': {e}", exc_info=True)
