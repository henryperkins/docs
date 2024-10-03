# file_handlers.py

import os
import fnmatch
import json
import logging
import ast
import astor
import shutil
from typing import Any, Set, List, Optional, Dict, Tuple
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
    call_openai_api,
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

async def extract_code_structure(
    content: str, file_path: str, language: str, function_schema: dict = None
) -> Optional[dict]:
    """Extracts code structure based on language."""
    logger.debug(f"Extracting code structure for '{file_path}' (language: {language})")
    try:
        if language == "python":
            return extract_python_structure(content)
        elif language in ["javascript", "typescript"]:
            return await extract_js_ts_structure(file_path, content, language, function_schema)
        elif language == "html":
            return extract_html_structure(content)
        elif language == "css":
            return extract_css_structure(content)
        elif language == "json":
            logger.debug(f"Processing JSON file: {file_path}")
            try:
                # Attempt to parse the JSON content
                json.loads(content)
                logger.debug(f"Successfully parsed JSON content from {file_path}")
                # You might want to return a specific structure for JSON files here
                return {"type": "json", "content": content}
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from {file_path}: {e}")
                return None
        else:
            logger.warning(f"Unsupported language for structure extraction: {language}")
            return None
    except Exception as e:
        logger.error(
            f"Error extracting structure from '{file_path}': {e}", exc_info=True
        )
        return None

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
) -> Optional[str]:
    """
    Processes a single file: extracts structure, generates documentation, inserts documentation, validates, and returns the documentation content to be added to the report.

    Parameters:
        session (aiohttp.ClientSession): The HTTP session.
        file_path (str): Path to the source file.
        skip_types (Set[str]): Set of file extensions to skip.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrency.
        model_name (str): The AI model to use.
        function_schema (dict): The function schema for structured responses.
        repo_root (str): Root directory of the repository.
        project_info (Optional[str]): Information about the project.
        style_guidelines (Optional[str]): Style guidelines for the documentation.
        safe_mode (bool): If True, do not modify files.

    Returns:
        Optional[str]: The documentation content for the file or None if failed.
    """
    logger.debug(f'Processing file: {file_path}')
    try:
        _, ext = os.path.splitext(file_path)
        if not is_valid_extension(ext, skip_types) or is_binary(file_path):
            logger.debug(f"Skipping file '{file_path}' due to invalid extension or binary content.")
            return ''

        language = get_language(ext)
        logger.debug(f"Detected language for '{file_path}': {language}")
        if language == 'plaintext':
            logger.debug(f"Skipping unsupported language in '{file_path}'.")
            return ''

        logger.info(f'Processing file: {file_path}')
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            logger.debug(f'File content for {file_path} read successfully.')
        except Exception as e:
            logger.error(f"Failed to read '{file_path}': {e}", exc_info=True)
            return ''

        code_structure = await extract_code_structure(content, file_path, language, function_schema)
        if not code_structure or (not code_structure.get('functions') and not code_structure.get('classes')):
            logger.warning(f"Could not extract code structure from '{file_path}'")
            return ''

        logger.debug(f'Extracted code structure for {file_path}: {code_structure}')
        prompt = generate_documentation_prompt(
            file_name=os.path.basename(file_path),
            code_structure=code_structure,
            project_info=project_info,
            style_guidelines=style_guidelines,
            language=language
        )
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

        logger.debug(f"Received documentation for '{file_path}': {documentation}")

        # Extracting documentation components
        summary = documentation.get('summary', '')
        changes = documentation.get('changes_made', [])
        functions = documentation.get('functions', [])
        classes = documentation.get('classes', [])

        # Process code documentation (insert docstrings/comments)
        new_content = await process_code_documentation(
            content, documentation, language, file_path
        )

        if not safe_mode:
            await backup_and_write_new_content(file_path, new_content)
            logger.info(f"Documentation inserted into '{file_path}'")
        else:
            logger.info(f"Safe mode active. Skipping file modification for '{file_path}'")

        # Generate documentation report content
        file_content = await write_documentation_report(
            summary=summary,
            changes=changes,
            functions=functions,
            classes=classes,
            language=language,
            file_path=file_path,
            repo_root=repo_root,
            new_content=new_content
        )

        logger.info(f"Successfully processed and documented '{file_path}'")
        return file_content

    except Exception as e:
        logger.error(f"Error processing file '{file_path}': {e}", exc_info=True)
        return ''

async def process_code_documentation(
    content: str,
    documentation: Dict[str, Any],
    language: str,
    file_path: str
) -> str:
    """
    Processes the code documentation by inserting docstrings/comments based on the language.
    
    Parameters:
        content (str): The original source code.
        documentation (Dict[str, Any]]: Documentation details from AI.
        language (str): Programming language.
        file_path (str): Path to the source file.
    
    Returns:
        str: The modified source code with inserted documentation.
    """
    logger.debug(f"Processing documentation for '{file_path}' in language '{language}'")
    try:
        if language == 'python':
            modified_code = insert_python_docstrings(content, documentation)
            # Optionally format and clean the code
            modified_code = format_with_black(modified_code)
            modified_code = clean_unused_imports(modified_code)
            if not check_with_flake8(file_path):
                logger.warning(f"Flake8 issues remain after formatting and cleaning in '{file_path}'")
        elif language in ['javascript', 'typescript']:
            modified_code = insert_js_ts_docstrings(content, documentation)
            # Optionally, format the JS/TS code using Prettier or similar tools
            # This can be integrated as needed
        elif language == 'html':
            modified_code = insert_html_comments(content, documentation)
        elif language == 'css':
            modified_code = insert_css_docstrings(content, documentation)
        else:
            logger.warning(f"Unsupported language '{language}'. Skipping documentation insertion.")
            modified_code = content
        return modified_code
    except Exception as e:
        logger.error(f"Error processing documentation for '{file_path}': {e}", exc_info=True)
        return content

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
    functions: List[dict],
    classes: List[dict],
    language: str,
    file_path: str,
    repo_root: str,
    new_content: str
) -> str:
    """
    Generates the documentation report content for a single file.

    Parameters:
        summary (str): Summary of the documentation.
        changes (List[str]): List of changes made.
        functions (List[dict]): List of functions documented.
        classes (List[dict]): List of classes documented.
        language (str): Programming language.
        file_path (str): Path to the source file.
        repo_root (str): Root directory of the repository.
        new_content (str): Modified source code with inserted documentation.

    Returns:
        str: The documentation content for the file.
    """
    try:
        relative_path = os.path.relpath(file_path, repo_root)
        file_header = f'# File: {relative_path}\n\n'
        summary_section = f'## Summary\n\n{summary.strip()}\n\n'
        changes_section = '## Changes Made\n\n'
        if changes:
            changes_section += '\n'.join(f'- {change.strip()}' for change in changes) + '\n\n'
        else:
            changes_section += 'No changes were made to this file.\n\n'

        functions_section = ''
        if functions:
            functions_section += '## Functions\n\n'
            functions_section += '| Function | Arguments | Description | Async |\n'
            functions_section += '|----------|-----------|-------------|-------|\n'
            for func in functions:
                func_name = func.get('name', 'N/A')
                func_args = ', '.join(func.get('args', []))
                func_doc = func.get('docstring') or 'No description provided.'
                # Ensure func_doc is a string before calling splitlines()
                first_line_doc = func_doc.splitlines()[0] if isinstance(func_doc, str) else 'No description provided.'
                func_async = 'Yes' if func.get('async', False) else 'No'
                functions_section += f'| `{func_name}` | `{func_args}` | {first_line_doc} | {func_async} |\n'
            functions_section += '\n'
        else:
            functions_section += '## Functions\n\nNo functions are defined in this file.\n\n'

        classes_section = ''
        if classes:
            classes_section += '## Classes\n\n'
            for cls in classes:
                cls_name = cls.get('name', 'N/A')
                cls_doc = cls.get('docstring') or 'No description provided.'
                classes_section += f'### Class: `{cls_name}`\n\n{cls_doc}\n\n'

                methods = cls.get('methods', [])
                if methods:
                    classes_section += '| Method | Arguments | Description | Async | Type |\n'
                    classes_section += '|--------|-----------|-------------|-------|------|\n'
                    for method in methods:
                        method_name = method.get('name', 'N/A')
                        method_args = ', '.join(method.get('args', []))
                        method_doc = method.get('docstring') or 'No description provided.'
                        first_line_method_doc = method_doc.splitlines()[0] if isinstance(method_doc, str) else 'No description provided.'
                        method_async = 'Yes' if method.get('async', False) else 'No'
                        method_type = method.get('type', 'N/A')
                        classes_section += f'| `{method_name}` | `{method_args}` | {first_line_method_doc} | {method_async} | {method_type} |\n'
                    classes_section += '\n'
                else:
                    classes_section += 'No methods defined in this class.\n\n'
        else:
            classes_section += '## Classes\n\nNo classes are defined in this file.\n\n'

        code_block = f'```{language}\n{new_content}\n```\n\n---\n\n'

        # Combine all sections
        documentation_content = (
            file_header +
            summary_section +
            changes_section +
            functions_section +
            classes_section +
            code_block
        )

        return documentation_content

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
