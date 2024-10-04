# file_handlers.py

import os
import logging
import asyncio
from typing import List, Set, Dict, Any, Optional
from utils import (
    fetch_documentation,
    generate_documentation_prompt,
    call_openai_api,
    run_flake8,
)
from language_functions import insert_docstrings
import aiofiles

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
    Processes a single file: extracts structure, fetches documentation, inserts docstrings/comments.

    Parameters:
        session (aiohttp.ClientSession): The HTTP session.
        file_path (str): Path to the file.
        skip_types (Set[str]): Set of file extensions to skip.
        semaphore (asyncio.Semaphore): Semaphore to control concurrency.
        model_name (str): OpenAI model name.
        function_schema (dict): Function schema for structured responses.
        repo_root (str): Root path of the repository.
        project_info (Optional[str]): Project information.
        style_guidelines (Optional[str]): Style guidelines for documentation.
        safe_mode (bool): If True, do not modify files.

    Returns:
        Optional[str]: Documentation content or None if failed.
    """
    try:
        ext = os.path.splitext(file_path)[1]
        language = get_language(ext)
        if language == "plaintext":
            logger.info(f"Skipping non-code file: {file_path}")
            return None

        # Read file content
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()

        # Extract code structure
        if language == "java":
            # Use Java-specific structure extraction
            structure = extract_java_structure(content)
        elif language in ["javascript", "typescript"]:
            # Use JS/TS-specific structure extraction
            structure = extract_js_ts_structure(content, language)
        elif language == "html":
            structure = extract_html_structure(content)
        elif language == "css":
            structure = extract_css_structure(content)
        elif language == "python":
            structure = extract_python_structure(content)
        else:
            logger.warning(f"No structure extraction implemented for language '{language}'. Skipping.")
            return None

        if not structure:
            logger.warning(f"Could not extract structure from file: {file_path}")
            return None

        # Generate documentation prompt
        file_name = os.path.basename(file_path)
        prompt = generate_documentation_prompt(
            file_name=file_name,
            code_structure=structure,
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
            logger.error(f"Failed to fetch documentation
