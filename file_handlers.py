# file_handlers.py

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
    get_all_file_paths,
    OPENAI_API_KEY,
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    DEFAULT_SKIP_TYPES,
    function_schema,
    format_with_black,
)

logger = logging.getLogger(__name__)

# Helper Functions
def is_valid_extension(ext: str, skip_types: Set[str]) -> bool:
    """Checks if the file extension is not in the skip list."""
    return ext.lower() not in skip_types

def is_binary(file_path: str) -> bool:
    """Checks if a file is binary."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking if file is binary '{file_path}': {e}")
        return False

def get_language(ext: str) -> str:
    """Determines the programming language based on file extension."""
    ext = ext.lower()
    if ext in ['.py']:
        return "python"
    elif ext in ['.js', '.jsx', '.ts', '.tsx']:
        return "javascript" if ext in ['.js', '.jsx'] else "typescript"
    elif ext in ['.html', '.htm']:
        return "html"
    elif ext in ['.css']:
        return "css"
    else:
        return "plaintext"

def generate_documentation_prompt(code_structure: dict, project_info: Optional[str], style_guidelines: Optional[str], language: str) -> str:
    """
    Generates a prompt for the OpenAI API based on the code structure.
    
    Parameters:
        code_structure (dict): The extracted structure of the code.
        project_info (Optional[str]): Information about the project.
        style_guidelines (Optional[str]): Documentation style guidelines.
        language (str): The programming language of the code.
    
    Returns:
        str: The generated prompt.
    """
    prompt = "You are an experienced software developer tasked with generating comprehensive documentation for a codebase."
    if project_info:
        prompt += f"\n\n**Project Information:** {project_info}"
    if style_guidelines:
        prompt += f"\n\n**Style Guidelines:** {style_guidelines}"
    prompt += f"\n\n**Language:** {language.capitalize()}"
    prompt += f"\n\n**Code Structure:**\n```json\n{json.dumps(code_structure, indent=2)}\n```"
    prompt += "\n\n**Instructions:** Based on the above code structure, generate the following documentation sections:\n1. **Summary:** A detailed summary of the codebase.\n2. **Changes Made:** A comprehensive list of changes or updates made to the code.\n\n**Please ensure that the documentation is clear, detailed, and adheres to the provided style guidelines.**"
    return prompt
    
async def fetch_documentation(session: aiohttp.ClientSession, prompt: str, semaphore: asyncio.Semaphore, model_name: str, function_schema: dict) -> Optional[dict]:
    """
    Fetches documentation from OpenAI's API based on the provided prompt using function calling.
    
    Parameters:
        session (aiohttp.ClientSession): The aiohttp client session.
        prompt (str): The prompt to send to the API.
        semaphore (asyncio.Semaphore): Semaphore to control concurrency.
        model_name (str): The OpenAI model to use.
        function_schema (dict): The JSON schema for the function call.
    
    Returns:
        Optional[dict]: The generated documentation, or None if failed.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates code documentation."},
            {"role": "user", "content": prompt}
        ],
        "functions": [function_schema],
        "function_call": "auto"  # Let the model decide which function to call
    }

    async with semaphore:
        try:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as resp:
                response_text = await resp.text()
                logger.debug(f"API Response Status: {resp.status}")
                logger.debug(f"API Response Body: {response_text}")

                if resp.status != 200:
                    logger.error(f"OpenAI API request failed with status {resp.status}: {response_text}")
                    return None

                response = await resp.json()
                logger.debug(f"Parsed JSON Response: {json.dumps(response, indent=2)}")
                choice = response.get("choices", [])[0]
                message = choice.get('message', {})

                # Check for function_call
                if 'function_call' in message:
                    arguments = message['function_call'].get('arguments', '{}')
                    try:
                        documentation = json.loads(arguments)
                        logger.debug("Received documentation via function_call.")
                        return documentation
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON from function_call arguments: {e}")
                        logger.error(f"Arguments Content: {arguments}")
                        return None
                else:
                    # Fallback: Extract documentation directly from content
                    content = message.get('content', '')
                    if content:
                        logger.debug("No function_call detected. Attempting to extract documentation from content.")
                        try:
                            # Initialize documentation dictionary
                            documentation = {
                                "summary": "",
                                "changes": [],
                                "functions": [],
                                "classes": []
                            }
                            lines = content.split('\n')
                            current_section = None
                            current_item = {}
                            for line in lines:
                                if line.startswith("**Summary:**"):
                                    current_section = "summary"
                                    documentation["summary"] = line.replace("**Summary:**", "").strip()
                                elif line.startswith("**Changes Made:**"):
                                    current_section = "changes"
                                elif line.startswith("**Function Documentation:**"):
                                    current_section = "functions"
                                elif line.startswith("**Class Documentation:**"):
                                    current_section = "classes"
                                elif current_section == "changes" and line.startswith("- "):
                                    change = line.replace("- ", "").strip()
                                    documentation["changes"].append(change)
                                elif current_section == "functions" and line.startswith("- **Function Name:**"):
                                    if current_item:
                                        documentation["functions"].append(current_item)
                                        current_item = {}
                                    current_item["name"] = line.replace("- **Function Name:**", "").strip()
                                elif current_section == "functions" and line.startswith("  - **Description:**"):
                                    current_item["description"] = line.replace("  - **Description:**", "").strip()
                                elif current_section == "functions" and line.startswith("  - **Parameters:**"):
                                    params = []
                                    # Expect parameters to be listed below
                                    continue  # Parameters will be handled in the next lines
                                elif current_section == "functions" and line.startswith("  - **Returns:**"):
                                    current_item["returns"] = line.replace("  - **Returns:**", "").strip()
                                elif current_section == "classes" and line.startswith("- **Class Name:**"):
                                    if current_item:
                                        documentation["classes"].append(current_item)
                                        current_item = {}
                                    current_item["name"] = line.replace("- **Class Name:**", "").strip()
                                elif current_section == "classes" and line.startswith("  - **Description:**"):
                                    current_item["description"] = line.replace("  - **Description:**", "").strip()
                                elif current_section == "classes" and line.startswith("  - **Methods:**"):
                                    methods = []
                                    current_item["methods"] = methods
                                    continue  # Methods will be handled in the next lines
                                elif current_section == "classes" and line.startswith("    - **Method Name:**"):
                                    method = {}
                                    method["name"] = line.replace("    - **Method Name:**", "").strip()
                                    current_item["methods"].append(method)
                                elif current_section == "classes" and line.startswith("      - **Description:**"):
                                    method["description"] = line.replace("      - **Description:**", "").strip()
                                elif current_section == "classes" and line.startswith("      - **Parameters:**"):
                                    method["parameters"] = []
                                    continue  # Parameters will be handled in the next lines
                                elif current_section == "classes" and line.startswith("      - **Returns:**"):
                                    method["returns"] = line.replace("      - **Returns:**", "").strip()
                            # Append the last item if exists
                            if current_item:
                                if current_section == "functions":
                                    documentation["functions"].append(current_item)
                                elif current_section == "classes":
                                    documentation["classes"].append(current_item)
                            logger.debug("Extracted documentation from content.")
                            return documentation
                        except Exception as e:
                            logger.error(f"Error parsing documentation content: {e}")
                            logger.error(f"Content Received: {content}")
                            return None
                    else:
                        logger.error("No content found in the API response.")
                        return None
        except Exception as e:
            logger.error(f"Error fetching documentation from OpenAI API: {e}")
            return None

async def fetch_documentation_with_retries(session: aiohttp.ClientSession, prompt: str, semaphore: asyncio.Semaphore, model_name: str, function_schema: dict, max_retries: int = 3, backoff_factor: int = 2) -> Optional[dict]:
    """
    Fetches documentation from OpenAI's API with retry mechanism.
    
    Parameters:
        session (aiohttp.ClientSession): The aiohttp client session.
        prompt (str): The prompt to send to the API.
        semaphore (asyncio.Semaphore): Semaphore to control concurrency.
        model_name (str): The OpenAI model to use.
        function_schema (dict): The JSON schema for the function call.
        max_retries (int): Maximum number of retries.
        backoff_factor (int): Factor by which the wait time increases after each retry.
    
    Returns:
        Optional[dict]: The generated documentation, or None if failed.
    """
    for attempt in range(1, max_retries + 1):
        documentation = await fetch_documentation(session, prompt, semaphore, model_name, function_schema)
        if documentation:
            return documentation
        else:
            wait_time = backoff_factor ** attempt
            logger.warning(f"Retrying in {wait_time} seconds... (Attempt {attempt}/{max_retries})")
            await asyncio.sleep(wait_time)
    logger.error("Max retries exceeded. Documentation generation failed.")
    return None
    
async def insert_docstrings_for_file(js_ts_file, documentation_file):
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
        
# Processing Functions
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
    Processes a single file: extracts code structure, generates documentation, inserts docstrings/comments,
    and ensures code compliance (e.g., PEP8 for Python).

    Parameters:
        session (aiohttp.ClientSession): The aiohttp client session.
        file_path (str): Path to the file to process.
        skip_types (Set[str]): Set of file extensions to skip.
        output_file (str): Path to the output Markdown file.
        semaphore (asyncio.Semaphore): Semaphore to control concurrency.
        output_lock (asyncio.Lock): Lock to manage asynchronous writes to the output file.
        model_name (str): OpenAI model to use (e.g., 'gpt-4').
        function_schema (dict): JSON schema for the function call.
        repo_root (str): Root directory of the repository.
        project_info (Optional[str]): Information about the project.
        style_guidelines (Optional[str]): Documentation style guidelines to follow.
        safe_mode (bool): If True, do not modify original files.

    Returns:
        None
    """
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
            logger.error(f"Failed to read '{file_path}': {e}")
            return

        # Attempt to extract the structure of the code
        try:
            if language == "python":
                code_structure = extract_python_structure(content)
            elif language in ["javascript", "typescript"]:
                code_structure = await extract_js_ts_structure(file_path, content, language)
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
            # Format with Black
            new_content = format_with_black(new_content)
            # Write to a temporary file for flake8 checking
            try:
                with tempfile.NamedTemporaryFile('w', delete=False, suffix='.py') as temp_file:
                    temp_file.write(new_content)
                    temp_file_path = temp_file.name
                # Check with flake8
                if not check_with_flake8(temp_file_path):
                    logger.error(f"flake8 compliance failed for '{file_path}'. Skipping file.")
                    os.remove(temp_file_path)  # Remove the temporary file
                    return
            finally:
                # Remove the temporary file after checking
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            logger.debug(f"Processed Python file '{file_path}'.")
        elif language in ["javascript", "typescript"]:
            new_content = insert_js_ts_docstrings(content, documentation)
            # (Optional) Integrate JavaScript/TypeScript formatters like Prettier here
            logger.debug(f"Processed {language} file '{file_path}'.")
        elif language == "html":
            new_content = insert_html_comments(content, documentation)
            # (Optional) Integrate HTML formatters here
            logger.debug(f"Processed HTML file '{file_path}'.")
        elif language == "css":
            if "rules" not in documentation:
                logger.error(f"Documentation for '{file_path}' lacks 'rules'. Skipping insertion.")
                new_content = content
            else:
                new_content = insert_css_docstrings(content, documentation)
                # (Optional) Integrate CSS formatters here
                logger.debug(f"Processed CSS file '{file_path}'.")
        else:
            new_content = content

        # Safe mode: Do not overwrite the original file, just log the changes
        if safe_mode:
            logger.info(f"Safe mode active. Skipping file modification for '{file_path}'")
        else:
            # Backup and write new content
            try:
                backup_path = f"{file_path}.bak"
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
                if os.path.exists(backup_path):
                    shutil.copy(backup_path, file_path)
                    os.remove(backup_path)
                    logger.info(f"Restored original file from backup for '{file_path}'")
                return

        # Output the final results to the markdown file
        try:
            async with output_lock:
                async with aiofiles.open(output_file, "a", encoding="utf-8") as f:
                    header = f"# File: {relative_path}\n\n"  # Use relative path
                    summary_section = f"## Summary\n\n{summary}\n\n"
                    changes_section = (
                        "## Changes Made\n\n" + "\n".join(f"- {change}" for change in changes) + "\n\n"
                    )
                    code_block = f"```{language}\n{new_content}\n```\n\n"
                    await f.write(header)
                    await f.write(summary_section)
                    await f.write(changes_section)
                    await f.write(code_block)
            logger.info(f"Successfully processed and documented '{file_path}'")
        except Exception as e:
            logger.error(f"Error writing documentation for '{file_path}': {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error writing documentation")

async def process_all_files(
    file_paths: List[str],
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
    Processes all files asynchronously.

    Parameters:
        file_paths (List[str]): List of file paths to process.
        skip_types (Set[str]): Set of file extensions to skip.
        output_file (str): Path to the output Markdown file.
        semaphore (asyncio.Semaphore): Semaphore to control concurrency.
        output_lock (asyncio.Lock): Lock to manage asynchronous writes to the output file.
        model_name (str): OpenAI model to use (e.g., 'gpt-4').
        function_schema (dict): JSON schema for the function call.
        repo_root (str): Root directory of the repository.
        project_info (Optional[str]): Information about the project.
        style_guidelines (Optional[str]): Documentation style guidelines to follow.
        safe_mode (bool): If True, do not modify original files.

    Returns:
        None
    """
    tasks = []
    async with aiohttp.ClientSession() as session:
        for file_path in file_paths:
            tasks.append(
                process_file(
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
            )
        logger.debug(f"Created {len(tasks)} tasks for processing files.")
        # Use gather with return_exceptions=True to handle individual task errors without cancelling all
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing file '{file_paths[idx]}': {result}", exc_info=True)
            else:
                logger.debug(f"Completed processing file '{file_paths[idx]}'.")
