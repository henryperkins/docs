# file_handlers.py

import os
import logging
from typing import List, Set, Dict, Any, Optional
import aiofiles
from utils import (
    call_openai_api,
    extract_json_from_response,
    generate_documentation_prompt,
    run_node_script,
)
from language_functions import insert_docstrings
import asyncio

logger = logging.getLogger(__name__)


async def process_file(
    session: aiohttp.ClientSession,
    file_path: str,
    skip_types: Set[str],
    semaphore: asyncio.Semaphore,
    model_name: str,
    function_schema: dict,
    repo_root: str,
    project_info: Optional[str],
    style_guidelines: Optional[str],
    safe_mode: bool,
) -> Optional[str]:
    """
    Processes a single file: extracts structure, generates documentation, inserts docstrings.

    Parameters:
        session (aiohttp.ClientSession): The HTTP session for API calls.
        file_path (str): Path to the file to process.
        skip_types (Set[str]): Set of file extensions to skip.
        semaphore (asyncio.Semaphore): Semaphore to control concurrency.
        model_name (str): OpenAI model name.
        function_schema (dict): Schema for function calling.
        repo_root (str): Root path of the repository.
        project_info (Optional[str]): Project information.
        style_guidelines (Optional[str]): Style guidelines for documentation.
        safe_mode (bool): If True, do not modify files.

    Returns:
        Optional[str]: The documentation content or None if failed.
    """
    ext = os.path.splitext(file_path)[1]
    language = get_language(ext)
    if language == "plaintext":
        logger.debug(f"Skipping unsupported file type: {file_path}")
        return None

    if is_binary(file_path):
        logger.debug(f"Skipping binary file: {file_path}")
        return None

    logger.info(f"Processing file: {file_path} (Language: {language})")

    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            code = await f.read()
    except Exception as e:
        logger.error(f"Failed to read file '{file_path}': {e}")
        return None

    # Extract code structure based on language
    if language == "java":
        # Use Java parser
        from parser.java_parser import extract_structure
        code_structure = extract_structure(code)
    else:
        # For other languages, implement appropriate structure extraction
        code_structure = {"structure": "Not implemented yet"}

    if not code_structure:
        logger.error(f"Failed to extract structure from '{file_path}'")
        return None

    # Generate documentation prompt
    file_name = os.path.basename(file_path)
    prompt = generate_documentation_prompt(
        file_name=file_name,
        code_structure=code_structure,
        project_info=project_info,
        style_guidelines=style_guidelines,
        language=language,
    )

    # Fetch documentation from OpenAI
    documentation = await fetch_documentation(
        session=session,
        prompt=prompt,
        semaphore=semaphore,
        model_name=model_name,
        function_schema=function_schema,
    )

    if not documentation:
        logger.error(f"Failed to generate documentation for '{file_path}'")
        return None

    # Insert docstrings/comments into code
    modified_code = insert_docstrings(code, documentation, language)

    if not safe_mode:
        # Write the modified code back to the file
        try:
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(modified_code)
            logger.info(f"Inserted docstrings/comments into '{file_path}'")
        except Exception as e:
            logger.error(f"Failed to write to file '{file_path}': {e}")
            return None
    else:
        logger.info(f"Safe mode enabled. Skipping file modification for '{file_path}'")

    # Generate documentation report entry
    documentation_content = await write_documentation_report(
        documentation=documentation,
        language=language,
        file_path=file_path,
        repo_root=repo_root,
        new_content=modified_code,
    )

    return documentation_content


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
    safe_mode: bool,
    output_file: str,
):
    """
    Processes all files asynchronously and writes the documentation report.

    Parameters:
        session (aiohttp.ClientSession): The HTTP session for API calls.
        file_paths (List[str]): List of file paths to process.
        skip_types (Set[str]): Set of file extensions to skip.
        semaphore (asyncio.Semaphore): Semaphore to control concurrency.
        model_name (str): OpenAI model name.
        function_schema (dict): Schema for function calling.
        repo_root (str): Root path of the repository.
        project_info (Optional[str]): Project information.
        style_guidelines (Optional[str]): Style guidelines for documentation.
        safe_mode (bool): If True, do not modify files.
        output_file (str): Path to the output Markdown file.
    """
    logger.debug("Starting process_all_files")
    documentation_entries = []
    for file_path in file_paths:
        doc_content = await process_file(
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
        )
        if doc_content:
            documentation_entries.append(doc_content)

    # Compile documentation entries into the output file
    try:
        async with aiofiles.open(output_file, "a", encoding="utf-8") as f:
            for entry in documentation_entries:
                await f.write(entry + "\n\n")
        logger.info(f"Documentation report written to '{output_file}'")
    except Exception as e:
        logger.error(f"Failed to write documentation report to '{output_file}': {e}")


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
        documentation (Optional[Dict[str, Any]]): The documentation data.
        language (str): Programming language of the file.
        file_path (str): Path to the file.
        repo_root (str): Root path of the repository.
        new_content (str): The modified code with inserted docstrings/comments.

    Returns:
        str: The formatted documentation content.
    """
    try:
        def sanitize_text(text: str) -> str:
            """Removes excessive newlines and whitespace from the text."""
            if not text:
                return ""
            lines = text.strip().splitlines()
            sanitized_lines = [line.strip() for line in lines if line.strip()]
            return "\n".join(sanitized_lines)

        relative_path = os.path.relpath(file_path, repo_root)
        file_header = f"# File: {relative_path}\n\n"
        documentation_content = file_header

        # Overview
        overview = documentation.get("overview", "") if documentation else ""
        overview = sanitize_text(overview)
        if overview:
            overview_section = f"## Overview\n\n{overview}\n"
            documentation_content += overview_section

        # Summary
        summary = documentation.get("summary", "") if documentation else ""
        summary = sanitize_text(summary)
        if summary:
            summary_section = f"## Summary\n\n{summary}\n"
            documentation_content += summary_section

        # Changes Made
        changes = documentation.get("changes_made", []) if documentation else []
        changes = [sanitize_text(change) for change in changes if change.strip()]
        if changes:
            changes_formatted = "\n".join(f"- {change}" for change in changes)
            changes_section = f"## Changes Made\n\n{changes_formatted}\n"
            documentation_content += changes_section

        # Functions
        functions = documentation.get("functions", []) if documentation else []
        if functions:
            functions_section = "## Functions\n\n"
            functions_section += "| Function | Arguments | Description | Async | Parameters | Returns |\n"
            functions_section += "|----------|-----------|-------------|-------|------------|---------|\n"
            for func in functions:
                func_name = func.get("name", "N/A")
                func_args = ", ".join(func.get("args", []))
                func_doc = sanitize_text(func.get("docstring", ""))
                func_params = sanitize_text(func.get("parameters", ""))
                func_returns = sanitize_text(func.get("returns", ""))
                func_async = "Yes" if func.get("async", False) else "No"
                functions_section += f"""| `{func_name}` | `{func_args}` | {func_doc} | {func_async} | {func_params} | {func_returns} |\n"""
            functions_section += "\n"
            documentation_content += functions_section

        # Classes
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
                        classes_section += f"""| `{method_name}` | `{method_args}` | {first_line_method_doc} | {method_async} | {method_type} |\n"""
                    classes_section += "\n"
            documentation_content += classes_section

        # Example Usage
        example_usage = documentation.get("example_usage", "") if documentation else ""
        example_usage = sanitize_text(example_usage)
        if example_usage:
            example_section = f"## Example Usage\n\n```{language}\n{example_usage}\n```\n"
            documentation_content += example_section

        # Code Block
        code_content = new_content.strip()
        code_block = f"```{language}\n{code_content}\n```\n\n---\n"
        documentation_content += code_block

        return documentation_content
    except Exception as e:
        logger.error(
            f"Error generating documentation for '{file_path}': {e}", exc_info=True
        )
        return ""
