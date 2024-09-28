import os
import sys
import json
import logging
import esprima  # Changed from subprocess to esprima
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
from typing import Set, List, Optional
from tqdm.asyncio import tqdm
import typescript as ts

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
    """Inserts docstrings into Python code based on the provided documentation."""
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
                    if hasattr(ast, "Constant"):  # Python 3.8+
                        docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                    else:
                        docstring_node = ast.Expr(value=ast.Str(s=docstring))
                    if not (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(
                            node.body[0].value, (ast.Str, ast.Constant)
                        )
                    ):
                        node.body.insert(0, docstring_node)
                    else:
                        node.body[0] = docstring_node
                return node

            def visit_ClassDef(self, node):
                self.generic_visit(node)
                docstring = class_doc_map.get(node.name)
                if docstring:
                    if hasattr(ast, "Constant"):  # Python 3.8+
                        docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                    else:
                        docstring_node = ast.Expr(value=ast.Str(s=docstring))
                    if not (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(
                            node.body[0].value, (ast.Str, ast.Constant)
                        )
                    ):
                        node.body.insert(0, docstring_node)
                    else:
                        node.body[0] = docstring_node
                return node

        inserter = DocstringInserter()
        new_tree = inserter.visit(tree)
        new_code = astor.to_source(new_tree)

        try:
            ast.parse(new_code)
        except SyntaxError as e:
            logger.error(f"Syntax error in modified Python code: {e}")
            return file_content
        finally:
            logger.debug("Finished parsing new code.")

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


async def extract_js_ts_structure(file_path: str, file_content: str) -> dict:
    """Extracts the structure of JavaScript/TypeScript code using TypeScript compiler."""
    try:
        # Use TypeScript compiler API to parse the TypeScript or JavaScript code
        source_file = ts.createSourceFile(
            file_path, file_content, ts.ScriptTarget.Latest, True
        )

        functions = []
        classes = []

        def visit(node, parent_class=None):
            node_type = node.kind

            if node_type == ts.SyntaxKind.FunctionDeclaration:
                func_info = {
                    "name": node.name.text if node.name else "anonymous",
                    "docstring": "",  # To be filled by the model
                    "args": [param.text for param in node.parameters],
                    "returns": {"type": "Any"},
                    "decorators": [],  # TypeScript doesn't support decorators natively
                }
                if parent_class:
                    parent_class["methods"].append(func_info)
                else:
                    functions.append(func_info)
            elif node_type == ts.SyntaxKind.ClassDeclaration:
                class_info = {
                    "name": node.name.text,
                    "docstring": "",  # To be filled by the model
                    "bases": [
                        heritage.getText() for heritage in node.heritageClauses
                    ]
                    if node.heritageClauses
                    else [],
                    "decorators": [],  # TypeScript doesn't support decorators natively
                    "methods": [],
                }
                classes.append(class_info)
                for member in node.members:
                    visit(member, parent_class=class_info)
            elif node_type == ts.SyntaxKind.MethodDeclaration:
                func_info = {
                    "name": node.name.text,
                    "docstring": "",  # To be filled by the model
                    "args": [param.text for param in node.parameters],
                    "returns": {"type": "Any"},
                    "decorators": [],  # TypeScript doesn't support decorators natively
                }
                if parent_class:
                    parent_class["methods"].append(func_info)

        for statement in source_file.statements:
            visit(statement)

        return {
            "language": "typescript",
            "functions": functions,
            "classes": classes,
            "source_code": file_content,
        }

    except Exception as e:
        logger.error(f"Error during JS/TS structure extraction: {e}")
        return {
            "language": "typescript",
            "functions": [],
            "classes": [],
            "source_code": file_content,
            "docstring": "",
        }
        

def insert_js_ts_docstrings(docstrings: dict) -> str:
    """Inserts JSDoc comments into JavaScript/TypeScript code."""
    source_code = docstrings.get("source_code")
    if not source_code:
        logger.error("No source code found in docstrings dictionary. Skipping docstring insertion.")
        return ""  # Or you could return an appropriate default value

    inserts = []

    # Function to map locations to positions in the code
    def get_position(loc):
        lines = source_code.splitlines()
        start_line = loc["start"]["line"] - 1
        start_column = loc["start"]["column"]
        pos = sum(len(line) + 1 for line in lines[:start_line]) + start_column
        return pos

    for item_type in ["functions", "classes"]:
        for item in docstrings.get(item_type, []):
            docstring = item["docstring"]
            if docstring and item.get("name"):
                # Find the location based on the function or class name
                # This simplistic approach assumes unique names
                pattern = re.escape(item["name"])
                match = re.search(r"\b" + pattern + r"\b", source_code)
                if match:
                    position = match.start()
                    formatted_comment = format_jsdoc_comment(docstring)
                    inserts.append((position, formatted_comment))

                    if item_type == "classes":
                        for method in item.get("methods", []):
                            method_docstring = method["docstring"]
                            if method_docstring and method.get("name"):
                                method_pattern = re.escape(method["name"])
                                method_match = re.search(
                                    r"\b" + method_pattern + r"\b", source_code
                                )
                                if method_match:
                                    method_position = method_match.start()
                                    method_comment = format_jsdoc_comment(
                                        method_docstring
                                    )
                                    inserts.append((method_position, method_comment))

    # Sort inserts by position descending to avoid shifting positions
    inserts.sort(key=lambda x: x[0], reverse=True)
    code = source_code
    for position, comment in inserts:
        code = code[:position] + comment + "\n" + code[position:]

    return code

def format_jsdoc_comment(docstring: str) -> str:
    """Formats a docstring into a JSDoc comment block."""
    comment_lines = ["/**"]
    for line in docstring.strip().split("\n"):
        comment_lines.append(f" * {line}")
    comment_lines.append(" */")
    return "\n".join(comment_lines)


# HTML handlers
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
    """Inserts comments into HTML code."""
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
                        comment = Comment(f" {docstring} ")
                        child.insert_before(comment)
                    traverse_and_insert(child)

        traverse_and_insert(soup)
        return str(soup)

    except Exception as e:
        logger.error(f"Error inserting comments into HTML code: {e}")
        return file_content


# CSS handlers
def extract_css_structure(file_content: str) -> dict:
    """Extracts the structure of CSS code."""
    try:
        rules = tinycss2.parse_stylesheet(file_content)
        style_rules = []

        for rule in rules:
            if rule.type == "qualified-rule":
                prelude = tinycss2.serialize(rule.prelude).strip()
                content = tinycss2.serialize(rule.content).strip()
                rule_info = {
                    "selector": prelude,
                    "docstring": "",  # To be filled by the model
                    "declarations": content,
                }
                style_rules.append(rule_info)

        return {"language": "css", "rules": style_rules}

    except Exception as e:
        logger.error(f"Error parsing CSS code: {e}")
        return {}


def insert_css_comments(file_content: str, docstrings: dict) -> str:
    """Inserts comments into CSS code."""
    try:
        rules = tinycss2.parse_stylesheet(file_content, skip_whitespace=True)
        style_rules = docstrings.get("rules", [])
        rule_map = {}
        for rule in style_rules:
            key = rule["selector"]
            docstring = rule.get("docstring", "")
            if key in rule_map:
                rule_map[key] += f"\n{docstring}"
            else:
                rule_map[key] = docstring

        modified_content = ""
        inserted_selectors = set()

        for rule in rules:
            if rule.type == "qualified-rule":
                selector = tinycss2.serialize(rule.prelude).strip()
                if selector not in inserted_selectors:
                    docstring = rule_map.get(selector)
                    if docstring:
                        modified_content += f"/* {docstring} */\n"
                    inserted_selectors.add(selector)
                # Serialize the rule content
                content = tinycss2.serialize(rule.content).strip()
                modified_content += f"{selector} {{\n{content}\n}}\n"
            elif rule.type == "comment":
                # Serialize comments directly
                modified_content += f"/*{rule.value}*/\n"
            elif rule.type == "at-rule":
                # Handle at-rules like @media, @keyframes
                at_rule_name = rule.lower_at_keyword
                at_rule_prelude = tinycss2.serialize(rule.prelude).strip()
                if rule.content:
                    content = tinycss2.serialize(rule.content).strip()
                    modified_content += f"@{at_rule_name} {at_rule_prelude} {{\n{content}\n}}\n"
                else:
                    modified_content += f"@{at_rule_name} {at_rule_prelude};\n"
            else:
                # For other rule types, serialize them directly
                modified_content += tinycss2.serialize(rule).strip() + "\n"

        return modified_content

    except Exception as e:
        logger.error(f"Error inserting comments into CSS code: {e}")
        return file_content

async def process_file(
    session: aiohttp.ClientSession,
    file_path: str,
    skip_types: Set[str],
    output_file: str,
    semaphore: asyncio.Semaphore,
    output_lock: asyncio.Lock,
    model_name: str,
) -> None:
    """Processes a single file to generate and insert documentation, summaries, and change lists."""
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
            code_structure = await extract_js_ts_structure(file_path, content)
        elif language == "html":
            code_structure = extract_html_structure(content)
        elif language == "css":
            code_structure = extract_css_structure(content)
        else:
            logger.warning(
                f"Language '{language}' not supported for structured extraction."
            )
            return

        # Check for empty structures, especially for JS/TS due to potential parsing errors
        if not code_structure or (language in ["javascript", "typescript"] and not (code_structure.get("functions") or code_structure.get("classes"))):
            if language in ["javascript", "typescript"]:
                logger.warning(f"Failed to extract JS/TS structure from '{file_path}'. Likely parsing errors. Skipping documentation generation.")
            else:
                logger.error(f"Failed to extract structure from '{file_path}'.")
            return

        prompt = generate_documentation_prompt(code_structure)

        # Fetch documentation using the updated fetch_documentation function
        documentation = await fetch_documentation(
            session, prompt, semaphore, model_name, function_schema
        )
        if not documentation:
            logger.error(f"Failed to generate documentation for '{file_path}'")
            return

        # Extract the summary and changes
        summary = documentation.get("summary", "")
        changes = documentation.get("changes", [])

        # Proceed with inserting the documentation into the code
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
        
# Process all files
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
