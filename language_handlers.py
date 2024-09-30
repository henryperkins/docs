# language_handlers.py
from typing import Set, List, Optional, Dict
import os
import sys
import json
import logging
import ast
import astor
import tinycss2
from bs4 import BeautifulSoup, Comment
import shutil
from utils import (
    load_config,
    get_all_file_paths,
    OPENAI_API_KEY,
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    DEFAULT_SKIP_TYPES,
    function_schema,
    is_binary,
    is_valid_extension,
    get_language,
    fetch_documentation,
    fetch_summary,
)
import aiofiles
import aiohttp
import asyncio
from tqdm.asyncio import tqdm
import subprocess
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

# Define or import format_with_black and check_with_flake8
def format_with_black(code: str) -> str:
    """Formats Python code using Black."""
    try:
        import black
        formatted_code = black.format_str(code, mode=black.FileMode())
        return formatted_code
    except ImportError:
        logger.warning("Black is not installed. Skipping formatting.")
        return code
    except Exception as e:
        logger.error(f"Black formatting failed: {e}")
        return code

def check_with_flake8(file_path: str) -> bool:
    """Checks Python code with Flake8."""
    try:
        result = subprocess.run(['flake8', file_path], capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            logger.error(f"flake8 errors in {file_path}:\n{result.stdout}")
            return False
    except FileNotFoundError:
        logger.error("flake8 is not installed or not found in PATH.")
        return False
    except Exception as e:
        logger.error(f"Error running flake8 on {file_path}: {e}")
        return False

# Python handlers
def extract_python_structure(file_content: str) -> dict:
    """Extracts the structure of Python code."""
    try:
        tree = ast.parse(file_content)
        parent_map = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent_map[child] = node

        functions = []
        classes = []

        def get_node_source(node):
            try:
                return ast.unparse(node)
            except AttributeError:
                return astor.to_source(node).strip()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "description": ast.get_docstring(node) or "",
                    "parameters": [],
                    "returns": {"type": "Any", "description": ""},
                    "decorators": [],
                    "examples": []
                }
                # Extract parameters
                for arg in node.args.args:
                    arg_type = "Any"
                    if arg.annotation:
                        arg_type = get_node_source(arg.annotation)
                    param_info = {
                        "name": arg.arg,
                        "type": arg_type,
                        "description": ""
                    }
                    func_info["parameters"].append(param_info)
                # Extract return type
                if node.returns:
                    func_info["returns"]["type"] = get_node_source(node.returns)
                # Extract decorators
                for decorator in node.decorator_list:
                    func_info["decorators"].append(get_node_source(decorator))
                parent = parent_map.get(node)
                if isinstance(parent, ast.ClassDef):
                    class_name = parent.name
                    class_obj = next(
                        (cls for cls in classes if cls["name"] == class_name), None
                    )
                    if not class_obj:
                        class_obj = {
                            "name": class_name,
                            "description": ast.get_docstring(parent) or "",
                            "bases": [get_node_source(base) for base in parent.bases],
                            "decorators": [],
                            "methods": [],
                            "examples": []
                        }
                        # Extract class decorators
                        for decorator in parent.decorator_list:
                            class_obj["decorators"].append(get_node_source(decorator))
                        classes.append(class_obj)
                    class_obj["methods"].append(func_info)
                else:
                    functions.append(func_info)
            elif isinstance(node, ast.ClassDef):
                class_exists = any(cls["name"] == node.name for cls in classes)
                if not class_exists:
                    class_info = {
                        "name": node.name,
                        "description": ast.get_docstring(node) or "",
                        "bases": [get_node_source(base) for base in node.bases],
                        "decorators": [],
                        "methods": [],
                        "examples": []
                    }
                    # Extract class decorators
                    for decorator in node.decorator_list:
                        class_info["decorators"].append(get_node_source(decorator))
                    classes.append(class_info)
        return {"language": "python", "functions": functions, "classes": classes}
    except Exception as e:
        logger.error(f"Error parsing Python code: {e}")
        return {}

def insert_python_docstrings(file_content: str, documentation: dict) -> str:
    """Inserts docstrings into Python code, including parameters and return descriptions."""
    try:
        tree = ast.parse(file_content)
        parent_map = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent_map[child] = node

        func_doc_map = {func["name"]: func for func in documentation.get("functions", [])}
        class_doc_map = {cls["name"]: cls for cls in documentation.get("classes", [])}
        method_doc_map = {}
        for cls in documentation.get("classes", []):
            for method in cls.get("methods", []):
                method_doc_map[(cls["name"], method["name"])] = method

        class DocstringInserter(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                parent = parent_map.get(node)
                if isinstance(parent, ast.ClassDef):
                    key = (parent.name, node.name)
                    doc_info = method_doc_map.get(key)
                else:
                    doc_info = func_doc_map.get(node.name)
                if doc_info:
                    # Construct docstring
                    description = doc_info.get('description', '')
                    param_docs = ''
                    for param in doc_info.get('parameters', []):
                        param_docs += f":param {param['type']} {param['name']}: {param['description']}\n"
                    return_doc = ''
                    returns = doc_info.get('returns', {})
                    if returns.get('type'):
                        return_doc = f":return {returns['type']}: {returns.get('description', '')}\n"
                    full_docstring = description.strip() + '\n\n' + param_docs + return_doc
                    # Insert docstring if not already present
                    if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                        if hasattr(ast, "Constant"):  # Python 3.8+
                            docstring_node = ast.Expr(value=ast.Constant(value=full_docstring.strip()))
                        else:
                            docstring_node = ast.Expr(value=ast.Str(s=full_docstring.strip()))
                        node.body.insert(0, docstring_node)
                return node

            def visit_ClassDef(self, node):
                self.generic_visit(node)
                class_doc = class_doc_map.get(node.name)
                if class_doc:
                    description = class_doc.get('description', '')
                    # Insert docstring if not already present
                    if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                        if hasattr(ast, "Constant"):
                            docstring_node = ast.Expr(value=ast.Constant(value=description.strip()))
                        else:
                            docstring_node = ast.Expr(value=ast.Str(s=description.strip()))
                        node.body.insert(0, docstring_node)
                return node

        inserter = DocstringInserter()
        new_tree = inserter.visit(tree)
        ast.fix_missing_locations(new_tree)  # Ensure the AST is properly fixed
        new_code = astor.to_source(new_tree)
        return new_code
    except Exception as e:
        logger.error(f"Error inserting docstrings into Python code: {e}")
        return file_content

def is_valid_python_code(code: str) -> bool:
    """Checks if the given code is valid Python code."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

async def extract_js_ts_structure(file_path: str, file_content: str, language: str) -> dict:
    """
    Extracts the structure of JavaScript or TypeScript code using a Node.js script.

    Parameters:
        file_path (str): The path to the file being processed.
        file_content (str): The content of the file (not used in this function but kept for consistency).
        language (str): The programming language ('javascript' or 'typescript').

    Returns:
        dict: A dictionary containing the extracted structure of the code.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            'node', 'extract_structure.js', file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if stderr:
            logger.error(f"Error from Node.js script for '{file_path}': {stderr.decode().strip()}")
            return {}

        try:
            structure = json.loads(stdout.decode())
            # Ensure the structure aligns with the updated schema
            # For example, replace 'docstring' with 'description' if necessary
            # This assumes the Node.js script has been updated accordingly
            return structure
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error for '{file_path}': {e}")
            return {}

    except FileNotFoundError:
        logger.error("Node.js or extract_structure.js not found. Ensure Node.js is installed and the script path is correct.")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error while extracting JS/TS structure from '{file_path}': {e}")
        return {}

def insert_js_ts_docstrings(content: str, documentation: dict) -> str:
    """
    Inserts JSDoc comments into JavaScript or TypeScript code.

    Parameters:
        content (str): The original source code.
        documentation (dict): The documentation dictionary containing 'functions' and 'classes'.

    Returns:
        str: The updated source code with inserted docstrings/comments.
    """
    try:
        if not content:
            logger.error("No source code provided.")
            return ""

        # Ensure that 'source_code' is included in the documentation
        documentation['source_code'] = content

        # Prepare the input for the Node.js script
        input_data = json.dumps(documentation)

        # Run the Node.js script to insert docstrings
        process = subprocess.Popen(
            ['node', 'insert_docstrings.js'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(input=input_data)

        if stderr:
            logger.error(f"Error from insert_docstrings.js: {stderr.strip()}")
            return content  # Return the original content if there's an error

        return stdout.strip()

    except FileNotFoundError:
        logger.error("Node.js or insert_docstrings.js not found. Ensure Node.js is installed and the script path is correct.")
        return content
    except Exception as e:
        logger.error(f"Error inserting JS/TS docstrings: {e}")
        return content

def extract_html_structure(file_content: str) -> dict:
    """Extracts the structure of HTML code."""
    try:
        soup = BeautifulSoup(file_content, "html.parser")
        elements = []

        def traverse(node):
            for child in node.children:
                if isinstance(child, str):
                    continue
                if child.name:
                    element_info = {
                        "tag": child.name,
                        "attributes": dict(child.attrs),
                        "text": child.get_text(strip=True),
                        "description": "",  # Replaced 'docstring' with 'description'
                    }
                    elements.append(element_info)
                    traverse(child)

        traverse(soup)
        return {"language": "html", "elements": elements}

    except Exception as e:
        logger.error(f"Error parsing HTML code: {e}")
        return {}

def insert_html_comments(file_content: str, documentation: dict) -> str:
    """Inserts comments into HTML code, preventing duplicates."""
    try:
        soup = BeautifulSoup(file_content, "html.parser")
        elements = documentation.get("elements", [])

        element_map = {}
        for element in elements:
            key = (
                element["tag"],
                tuple(sorted(element["attributes"].items())),
                element["text"],
            )
            element_map[key] = element.get("description", "")

        def traverse_and_insert(node):
            for child in node.children:
                if isinstance(child, str):
                    continue
                if child.name:
                    key = (
                        child.name,
                        tuple(sorted(child.attrs.items())),
                        child.get_text(strip=True),
                    )
                    description = element_map.get(key)
                    if description:
                        # Check for existing comment with the same description
                        existing_comment = child.find_previous_sibling(string=lambda text: isinstance(text, Comment) and description in text)
                        if not existing_comment:
                            comment = Comment(f" {description} ")
                            child.insert_before(comment)
                    traverse_and_insert(child)

        traverse_and_insert(soup)
        return str(soup)

    except Exception as e:
        logger.error(f"Error inserting comments into HTML code: {e}")
        return file_content

def extract_css_structure(file_content: str) -> dict:
    """Extracts the structure of CSS code."""
    try:
        stylesheet = tinycss2.parse_stylesheet(file_content, skip_whitespace=True)
        rules = []
        for rule in stylesheet:
            if rule.type == "qualified-rule":
                selector = tinycss2.serialize(rule.prelude).strip()
                declarations = tinycss2.serialize(rule.content).strip()
                rules.append({"selector": selector, "declarations": declarations, "description": ""})
            elif rule.type == "at-rule":
                # Handle at-rules (e.g., @media, @import)
                at_keyword = rule.at_keyword
                prelude = tinycss2.serialize(rule.prelude).strip()
                content = tinycss2.serialize(rule.content).strip() if rule.content else None
                rules.append({"selector": f"@{at_keyword} {prelude}", "declarations": content or "", "description": ""})
        return {"language": "css", "rules": rules}
    except Exception as e:
        logger.error(f"Error parsing CSS code: {e}")
        return {}

def insert_css_docstrings(content: str, documentation: dict) -> str:
    """
    Inserts comments into CSS code based on the provided documentation.

    Parameters:
        content (str): The original CSS code.
        documentation (dict): The documentation data containing 'rules'.

    Returns:
        str: The updated CSS code with inserted comments.
    """
    try:
        lines = content.split('\n')
        updated_lines = []
        for line in lines:
            stripped_line = line.strip()
            # Check if the line is a selector
            if stripped_line.endswith('{'):
                selector = stripped_line[:-1].strip()
                # Find the corresponding rule in the documentation
                for rule in documentation.get('rules', []):
                    if selector == rule['selector']:
                        # Insert the comment above the rule
                        description = rule.get('description', '')
                        if description:  # Only add comment if description exists
                            comment = f"/* {description} */"
                            updated_lines.append(comment)
                        break
                updated_lines.append(line)
            else:
                updated_lines.append(line)
        return '\n'.join(updated_lines)
    except Exception as e:
        logger.error(f"Error inserting comments into CSS code: {e}")
        return content

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
            logger.debug(f"Skipping file '{file_path}'.")
            return

        language = get_language(ext)
        if language == "plaintext":
            logger.debug(f"Skipping unsupported language in '{file_path}'.")
            return
    except Exception as e:
        logger.error(f"An error occurred while processing file '{file_path}': {e}")
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
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

        except Exception as e:
            logger.error(f"Error extracting structure from '{file_path}': {e}", exc_info=True)
            return

        # Generate the documentation using the updated prompt
        prompt = generate_documentation_prompt(
            code_structure=code_structure,
            project_info=project_info,
            style_guidelines=style_guidelines
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
        elif language in ["javascript", "typescript"]:
            new_content = insert_js_ts_docstrings(content, documentation)
            # (Optional) Integrate JavaScript/TypeScript formatters like Prettier here
        elif language == "html":
            new_content = insert_html_comments(content, documentation)
            # (Optional) Integrate HTML formatters here
        elif language == "css":
            if "rules" not in documentation:
                logger.error(f"Documentation for '{file_path}' lacks 'rules'. Skipping insertion.")
                new_content = content
            else:
                new_content = insert_css_docstrings(content, documentation)
                # (Optional) Integrate CSS formatters here
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
                logger.info(f"Backup created at '{backup_path}'")

                async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                    await f.write(new_content)
                logger.info(f"Inserted documentation into '{file_path}'")
            except Exception as e:
                logger.error(f"Error writing to '{file_path}': {e}", exc_info=True)
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
        # Use gather with return_exceptions=True to handle individual task errors without cancelling all
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing file '{file_paths[idx]}': {result}", exc_info=True)