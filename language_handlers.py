import os
import sys
import json
import logging
import ast
import astor
import tinycss2
import re
from bs4 import BeautifulSoup, Comment
from utils import (
    is_valid_extension,
    get_language,
    generate_documentation_prompt,
    fetch_documentation,
    is_binary,
    function_schema,
)
import aiofiles
import aiohttp
import asyncio
from typing import Set, List, Optional, Dict
from tqdm.asyncio import tqdm
import subprocess

logger = logging.getLogger(__name__)

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
                    "args": [],
                    "returns": {"type": "Any"},
                    "decorators": [],
                    "docstring": ast.get_docstring(node) or "",
                }
                for arg in node.args.args:
                    arg_type = "Any"
                    if arg.annotation:
                        arg_type = get_node_source(arg.annotation)
                    func_info["args"].append({"name": arg.arg, "type": arg_type})
                if node.returns:
                    func_info["returns"]["type"] = get_node_source(node.returns)
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
                            "bases": [get_node_source(base) for base in parent.bases],
                            "methods": [],
                            "decorators": [],
                            "docstring": ast.get_docstring(parent) or "",
                        }
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
                        "bases": [get_node_source(base) for base in node.bases],
                        "methods": [],
                        "decorators": [],
                        "docstring": ast.get_docstring(node) or "",
                    }
                    for decorator in node.decorator_list:
                        class_info["decorators"].append(get_node_source(decorator))
                    classes.append(class_info)
        return {"language": "python", "functions": functions, "classes": classes}
    except Exception as e:
        logger.error(f"Error parsing Python code: {e}")
        return {}


def insert_python_docstrings(file_content: str, docstrings: dict) -> str:
    """Inserts docstrings into Python code, preventing duplicates."""

    try:
        tree = ast.parse(file_content)
        parent_map = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent_map[child] = node

        func_doc_map = {
            func["name"]: func["docstring"] for func in docstrings.get("functions", [])
        }
        class_doc_map = {
            cls["name"]: cls["docstring"] for cls in docstrings.get("classes", [])
        }
        method_doc_map = {}
        for cls in docstrings.get("classes", []):
            for method in cls.get("methods", []):
                method_doc_map[(cls["name"], method["name"])] = method["docstring"]

        class DocstringInserter(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                parent = parent_map.get(node)
                if isinstance(parent, ast.ClassDef):
                    key = (parent.name, node.name)
                    docstring = method_doc_map.get(key)
                else:
                    docstring = func_doc_map.get(node.name)
                if docstring:
                    if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                        if hasattr(ast, "Constant"):  # Python 3.8+
                            docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                        else:
                            docstring_node = ast.Expr(value=ast.Str(s=docstring))
                        node.body.insert(0, docstring_node)
                return node

            def visit_ClassDef(self, node):
                self.generic_visit(node)
                docstring = class_doc_map.get(node.name)
                if docstring:
                    if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                        if hasattr(ast, "Constant"):
                            docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                        else:
                            docstring_node = ast.Expr(value=ast.Str(s=docstring))
                        node.body.insert(0, docstring_node)
                return node


        inserter = DocstringInserter()
        new_tree = inserter.visit(tree)
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


async def extract_js_ts_structure(file_path: str, file_content: str, language: str) -> dict: # Add language argument
    """Extracts the structure of JavaScript/TypeScript code."""
    try:
        process = subprocess.Popen(
            ["node", "extract_structure.js", file_path], # Path to your Node.js script
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(input=file_content)

        if stderr:
            logger.error(f"Error from Node.js script: {stderr}")
            return {}

        try:
            structure = json.loads(stdout)
            structure['source_code'] = file_content
            structure['language'] = language # Include the language
            return structure
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            return {}

    except FileNotFoundError:
        logger.error("Node.js or extract_structure.js not found. Make sure Node.js is installed and the script path is correct.")
        return {}
    except Exception as e:
        logger.error(f"Error extracting JS/TS structure: {e}")
        return {}



def insert_js_ts_docstrings(docstrings: dict) -> str:
    """Inserts JSDoc comments into JavaScript/TypeScript code."""
    try:
        process = subprocess.Popen(
            ["node", "insert_docstrings.js"], # Path to your Node.js script
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout, stderr = process.communicate(input=json.dumps(docstrings))

        if stderr:
            logger.error(f"Error from Node.js script: {stderr}")
            return docstrings.get("source_code", "")

        return stdout.strip()

    except FileNotFoundError:
        logger.error("Node.js or insert_docstrings.js not found. Make sure Node.js is installed and the script path is correct.")
        return docstrings.get("source_code", "")
    except Exception as e:
        logger.error(f"Error inserting JS/TS docstrings: {e}")
        return docstrings.get("source_code", "")



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
                        "docstring": "",
                    }
                    elements.append(element_info)
                    traverse(child)

        traverse(soup)
        return {"language": "html", "elements": elements}

    except Exception as e:
        logger.error(f"Error parsing HTML code: {e}")
        return {}


def insert_html_comments(file_content: str, docstrings: dict) -> str:
    """Inserts comments into HTML code, preventing duplicates."""
    try:
        soup = BeautifulSoup(file_content, "html.parser")
        elements = docstrings.get("elements", [])

        element_map = {}
        for element in elements:
            key = (
                element["tag"],
                tuple(sorted(element["attributes"].items())),
                element["text"],
            )
            element_map[key] = element["docstring"]

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
                    docstring = element_map.get(key)
                    if docstring:
                        # Check for existing comment with the same docstring
                        existing_comment = child.find_previous_sibling(string=lambda text: isinstance(text, Comment) and docstring in text)
                        if not existing_comment:
                            comment = Comment(f" {docstring} ")
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
                rules.append({"selector": selector, "declarations": declarations, "docstring": ""})
            elif rule.type == "at-rule":
                # Handle at-rules (e.g., @media, @import)
                at_keyword = rule.at_keyword
                prelude = tinycss2.serialize(rule.prelude).strip()
                content = tinycss2.serialize(rule.content).strip() if rule.content else None
                rules.append({"at_keyword": at_keyword, "prelude": prelude, "content": content, "docstring": ""})
        return {"language": "css", "rules": rules}
    except Exception as e:
        logger.error(f"Error parsing CSS code: {e}")
        return {}



def insert_css_comments(file_content: str, docstrings: dict) -> str:
    """Inserts comments into CSS code, preventing duplicates."""
    try:
        stylesheet = tinycss2.parse_stylesheet(file_content, skip_whitespace=True)
        rules = docstrings.get("rules", [])
        rule_map = {rule["selector"]: rule["docstring"] for rule in rules if "selector" in rule} # Use selector as key

        modified_content = ""
        inserted_selectors = set()

        for rule in stylesheet:
            if rule.type == "qualified-rule":
                selector = tinycss2.serialize(rule.prelude).strip()
                if selector not in inserted_selectors:
                    docstring = rule_map.get(selector)
                    if docstring:
                        # Check if a comment already exists *before* inserting a new one
                        existing_comment = rule.prev  # Get the immediately preceding node
                        if existing_comment is None or existing_comment.type != 'comment' or docstring.strip() not in existing_comment.serialize().strip():
                            modified_content += f"/* {docstring} */\n"
                        inserted_selectors.add(selector)
                modified_content += tinycss2.serialize(rule).strip() + "\n"

            # Handle at-rules
            elif rule.type == "at-rule":
                modified_content += tinycss2.serialize(rule).strip() + "\n"

            # Handle comments
            elif rule.type == "comment":
                modified_content += tinycss2.serialize(rule).strip() + "\n"
            else:
                modified_content += tinycss2.serialize(rule).strip() + "\n"


        return modified_content

    except Exception as e:
        logger.error(f"Error inserting comments into CSS code: {e}")
        return file_content


async def process_file(
    session, file_path, skip_types, output_file, semaphore, output_lock, model_name
):
    """Processes a single file."""
    try:
        _, ext = os.path.splitext(file_path)
        if not is_valid_extension(ext, skip_types) or is_binary(file_path):
            logger.debug(f"Skipping file '{file_path}'.")
            return

        language = get_language(ext)
        if language == "plaintext":
            logger.debug(f"Skipping unsupported language in '{file_path}'.")
            return

        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
        except Exception as e:
            logger.error(f"Failed to read '{file_path}': {e}")
            return

        if language == "python":
            code_structure = extract_python_structure(content)
        elif language in ["javascript", "typescript"]:
            code_structure = await extract_js_ts_structure(file_path, content, language) # Pass language here
        elif language == "html":
            code_structure = extract_html_structure(content)
        elif language == "css":
            code_structure = extract_css_structure(content)
        else:
            logger.warning(
                f"Language '{language}' not supported for structured extraction."
            )
            return

        if not code_structure or (language in ["javascript", "typescript"] and not (code_structure.get("functions") or code_structure.get("classes"))):
            if language in ["javascript", "typescript"]:
                logger.warning(f"Failed to extract JS/TS structure from '{file_path}'. Likely parsing errors. Skipping documentation generation.")
            else:
                logger.error(f"Failed to extract structure from '{file_path}'.")
            return

        prompt = generate_documentation_prompt(code_structure, model_name)

        documentation = await fetch_documentation(session, prompt, semaphore, model_name, function_schema)
        if not documentation:
            logger.error(f"Failed to generate documentation for '{file_path}'")
            return

        summary = documentation.get("summary", "")
        changes = documentation.get("changes", [])

        if language == "python":
            new_content = insert_python_docstrings(content, documentation)
        elif language in ["javascript", "typescript"]:
            new_content = insert_js_ts_docstrings(documentation)
        elif language == "html":
            new_content = insert_html_comments(content, documentation)
        elif language == "css":
            new_content = insert_css_comments(content, documentation)
        else:
            new_content = content

        if language == "python":
            if not is_valid_python_code(new_content):
                logger.error(
                    f"Modified Python code is invalid. Aborting insertion for '{file_path}'"
                )
                return

        try:
            backup_path = file_path + ".bak"
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.rename(file_path, backup_path)
            logger.info(f"Backup created at '{backup_path}'")

            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(new_content)
            logger.info(f"Inserted comments into '{file_path}'")
        except Exception as e:
            logger.error(f"Error writing to '{file_path}': {e}")
            if os.path.exists(backup_path):
                os.remove(file_path)
                os.rename(backup_path, file_path)
                logger.info(f"Restored original file from backup for '{file_path}'")
            return


        try:
            async with output_lock:
                async with aiofiles.open(output_file, "a", encoding="utf-8") as f:
                    header = f"# File: {file_path}\n\n"
                    summary_section = f"## Summary\n\n{summary}\n\n"
                    changes_section = (
                        "## Changes Made\n\n"
                        + "\n".join(f"- {change}" for change in changes)
                        + "\n\n"
                    )
                    code_block = f"```{language}\n{new_content}\n```\n\n"
                    await f.write(header)
                    await f.write(summary_section)
                    await f.write(changes_section)
                    await f.write(code_block)
            logger.info(f"Successfully processed and documented '{file_path}'")
        except Exception as e:
            logger.error(f"Error writing documentation for '{file_path}': {e}")

    except Exception as e:
        logger.error(f"Unexpected error processing '{file_path}': {e}")

async def process_all_files(
    file_paths: List[str],
    skip_types: Set[str],
    output_file: str,
    semaphore: asyncio.Semaphore,
    output_lock: asyncio.Lock,
    model_name: str,
) -> None:
    """Processes all files asynchronously to generate and insert documentation."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(
                process_file(
                    session, file_path, skip_types, output_file, semaphore, output_lock, model_name
                )
            )
            for file_path in file_paths
        ]

        for f in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Processing Files"
        ):
            try:
                await f
            except Exception as e:
                logger.error(f"Error processing a file: {e}")