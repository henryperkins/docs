import os
import json
import logging
import shutil
from typing import Any, Set, List, Optional, Dict
import aiofiles
import aiohttp
import asyncio
from tqdm.asyncio import tqdm
import subprocess

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
    is_valid_extension,
    generate_documentation_prompt,
    fetch_documentation,
    function_schema,
    format_with_black,
    clean_unused_imports,
    check_with_flake8,
    run_node_script,
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
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Remove unnecessary imports
# Remove 'fnmatch', 'ast', 'astor', 'Path', 'tempfile' as they are not used

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
    Processes a single file: extracts structure, generates documentation, inserts documentation,
    validates, and returns the documentation content to be added to the report.

    Parameters:
        ... (existing parameters)

    Returns:
        Optional[str]: The documentation content for the file or None if failed.
    """
    logger.debug(f'Processing file: {file_path}')
    try:
        _, ext = os.path.splitext(file_path)
        if not is_valid_extension(ext, skip_types) or is_binary(file_path):
            logger.debug(f"Skipping file '{file_path}' due to invalid extension or binary content.")
            return None

        language = get_language(ext)
        logger.debug(f"Detected language for '{file_path}': {language}")
        if language == 'plaintext':
            logger.debug(f"Skipping unsupported language in '{file_path}'.")
            return None

        logger.info(f'Processing file: {file_path}')
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            logger.debug(f'File content for {file_path} read successfully.')
        except Exception as e:
            logger.error(f"Failed to read '{file_path}': {e}", exc_info=True)
            return None

        # Initialize documentation to None
        documentation = None

        try:
            code_structure = await extract_code_structure(content, file_path, language, function_schema)
            if not code_structure:
                logger.warning(f"Could not extract code structure from '{file_path}'")
            else:
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
        except Exception as e:
            logger.error(f"Error during code structure extraction or documentation generation for '{file_path}': {e}", exc_info=True)

        # Process code documentation (insert docstrings/comments)
        if documentation:
            try:
                new_content = await process_code_documentation(
                    content, documentation, language, file_path
                )
                if not safe_mode:
                    await backup_and_write_new_content(file_path, new_content)
                    logger.info(f"Documentation inserted into '{file_path}'")
                else:
                    logger.info(f"Safe mode active. Skipping file modification for '{file_path}'")
            except Exception as e:
                logger.error(f"Error processing code documentation for '{file_path}': {e}", exc_info=True)
                new_content = content  # Use original content if processing fails
        else:
            new_content = content  # Use original content if documentation generation failed

        # Generate documentation report content
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
        documentation (Dict[str, Any]]): Documentation details from AI.
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
            # Optionally, you can format the JS/TS code using Prettier or similar tools.
            # E.g., modified_code = format_with_prettier(modified_code)
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
    documentation: Optional[Dict[str, Any]],
    language: str,
    file_path: str,
    repo_root: str,
    new_content: str,
) -> str:
    """
    Generates the documentation report content for a single file.

    Parameters:
        documentation (Optional[Dict[str, Any]]): The documentation details or None if unavailable.
        language (str): The programming language of the file.
        file_path (str): The path to the file for which the report is generated.
        repo_root (str): The root of the repository.
        new_content (str): The content of the file after documentation.

    Returns:
        str: The documentation content for the file.
    """
    try:
        # Helper function to sanitize text
        def sanitize_text(text: str) -> str:
            """Removes excessive newlines and whitespace from the text."""
            if not text:
                return ""
            lines = text.strip().splitlines()
            sanitized_lines = [line.strip() for line in lines if line.strip()]
            return '\n'.join(sanitized_lines)

        relative_path = os.path.relpath(file_path, repo_root)
        file_header = f"# File: {relative_path}\n\n"
        documentation_content = file_header

        # Process summary
        summary = documentation.get("summary", "") if documentation else ""
        summary = sanitize_text(summary)
        if summary:
            summary_section = f"## Summary\n\n{summary}\n"
            documentation_content += summary_section

        # Process changes
        changes = documentation.get("changes_made", []) if documentation else []
        changes = [sanitize_text(change) for change in changes if change.strip()]
        if changes:
            changes_formatted = "\n".join(f"- {change}" for change in changes)
            changes_section = f"## Changes Made\n\n{changes_formatted}\n"
            documentation_content += changes_section

        # Process functions
        functions = documentation.get("functions", []) if documentation else []
        if functions:
            functions_section = "## Functions\n\n"
            functions_section += "| Function | Arguments | Description | Async |\n"
            functions_section += "|----------|-----------|-------------|-------|\n"
            for func in functions:
                func_name = func.get("name", "N/A")
                func_args = ", ".join(func.get("args", []))
                func_doc = sanitize_text(func.get("docstring", ""))
                first_line_doc = func_doc.splitlines()[0] if func_doc else "No description provided."
                func_async = "Yes" if func.get("async", False) else "No"
                functions_section += f"| `{func_name}` | `{func_args}` | {first_line_doc} | {func_async} |\n"
            functions_section += "\n"
            documentation_content += functions_section

        # Process classes
        classes = documentation.get("classes", []) if documentation else []
        if classes:
            classes_section = "## Classes\n\n"
            for cls in classes:
                cls_name = cls.get("name", "N/A")
                cls_doc = sanitize_text(cls.get("docstring", ""))
                if cls_doc:
                    classes_section += f"### Class: `{cls_name}`\n\n{cls_doc}\n\n"
                else:
                    classes_section += f"### Class: `{cls_name}`\n\n"
                methods = cls.get("methods", [])
                if methods:
                    classes_section += "| Method | Arguments | Description | Async | Type |\n"
                    classes_section += "|--------|-----------|-------------|-------|------|\n"
                    for method in methods:
                        method_name = method.get("name", "N/A")
                        method_args = ", ".join(method.get("args", []))
                        method_doc = sanitize_text(method.get("docstring", ""))
                        first_line_method_doc = method_doc.splitlines()[0] if method_doc else "No description provided."
                        method_async = "Yes" if method.get("async", False) else "No"
                        method_type = method.get("type", "N/A")
                        classes_section += f"| `{method_name}` | `{method_args}` | {first_line_method_doc} | {method_async} | {method_type} |\n"
                    classes_section += "\n"
            classes_section += "\n"
            documentation_content += classes_section

        # Append code block
        code_content = new_content.strip()
        code_block = f"```{language}\n{code_content}\n```\n\n---\n"
        documentation_content += code_block

        return documentation_content
    except Exception as e:
        logger.error(
            f"Error generating documentation for '{file_path}': {e}", exc_info=True
        )
        return ""

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

