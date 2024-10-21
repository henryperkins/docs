"""
write_documentation_report.py

This module provides functions for generating documentation reports in Markdown format. It includes utilities for creating badges, formatting tables, generating summaries, and writing the final documentation report to a file.
"""

import aiofiles
import re
import json
import os
import textwrap
import logging
import sys
from typing import Optional, Dict, Any, List

# Import default thresholds and get_threshold function from utils
from utils import (
    DEFAULT_COMPLEXITY_THRESHOLDS,
    DEFAULT_HALSTEAD_THRESHOLDS,
    DEFAULT_MAINTAINABILITY_THRESHOLDS,
    get_threshold  # Import get_threshold function
)

logger = logging.getLogger(__name__)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def generate_badge(metric_name: str, value: float, thresholds: Dict[str, int], logo: str = None) -> str:
    """
    Generates a dynamic badge for a given metric.

    Args:
        metric_name (str): The name of the metric.
        value (float): The metric value.
        thresholds (Dict[str, int]): The thresholds for determining badge color.
        logo (str, optional): The logo to include in the badge. Defaults to None.

    Returns:
        str: The Markdown string for the badge.
    """
    low, medium, high = thresholds["low"], thresholds["medium"], thresholds["high"]
    color = "green" if value <= low else "yellow" if value <= medium else "red"
    badge_label = metric_name.replace("_", " ").title()
    logo_part = f"&logo={logo}" if logo else ""
    return f"![{badge_label}](https://img.shields.io/badge/{badge_label}-{value:.2f}-{color}?style=flat-square{logo_part})"


def generate_all_badges(
    complexity: Optional[int] = None,
    halstead: Optional[dict] = None,
    mi: Optional[float] = None
) -> str:
    """
    Generates badges for all metrics, using dynamic thresholds.

    Args:
        complexity (Optional[int]): The complexity value.
        halstead (Optional[dict]): The Halstead metrics.
        mi (Optional[float]): The maintainability index.

    Returns:
        str: A string containing all generated badges.
    """
    # Fetch thresholds using get_threshold function
    thresholds = {
        "complexity": {
            "low": get_threshold("complexity", "low", DEFAULT_COMPLEXITY_THRESHOLDS["low"]),
            "medium": get_threshold("complexity", "medium", DEFAULT_COMPLEXITY_THRESHOLDS["medium"]),
            "high": get_threshold("complexity", "high", DEFAULT_COMPLEXITY_THRESHOLDS["high"]),
        },
        "halstead_volume": {
            "low": get_threshold("halstead_volume", "low", DEFAULT_HALSTEAD_THRESHOLDS["volume"]["low"]),
            "medium": get_threshold("halstead_volume", "medium", DEFAULT_HALSTEAD_THRESHOLDS["volume"]["medium"]),
            "high": get_threshold("halstead_volume", "high", DEFAULT_HALSTEAD_THRESHOLDS["volume"]["high"]),
        },
        "halstead_difficulty": {
            "low": get_threshold("halstead_difficulty", "low", DEFAULT_HALSTEAD_THRESHOLDS["difficulty"]["low"]),
            "medium": get_threshold("halstead_difficulty", "medium", DEFAULT_HALSTEAD_THRESHOLDS["difficulty"]["medium"]),
            "high": get_threshold("halstead_difficulty", "high", DEFAULT_HALSTEAD_THRESHOLDS["difficulty"]["high"]),
        },
        "halstead_effort": {
            "low": get_threshold("halstead_effort", "low", DEFAULT_HALSTEAD_THRESHOLDS["effort"]["low"]),
            "medium": get_threshold("halstead_effort", "medium", DEFAULT_HALSTEAD_THRESHOLDS["effort"]["medium"]),
            "high": get_threshold("halstead_effort", "high", DEFAULT_HALSTEAD_THRESHOLDS["effort"]["high"]),
        },
        "maintainability_index": {
            "low": get_threshold("maintainability_index", "low", DEFAULT_MAINTAINABILITY_THRESHOLDS["low"]),
            "medium": get_threshold("maintainability_index", "medium", DEFAULT_MAINTAINABILITY_THRESHOLDS["medium"]),
            "high": get_threshold("maintainability_index", "high", DEFAULT_MAINTAINABILITY_THRESHOLDS["high"]),
        },
    }

    badges = []

    if complexity is not None:
        badges.append(generate_badge("Complexity", complexity, thresholds["complexity"], logo="codeClimate"))

    if halstead:
        badges.append(generate_badge("Halstead Volume", halstead["volume"], thresholds["halstead_volume"], logo="stackOverflow"))
        badges.append(generate_badge("Halstead Difficulty", halstead["difficulty"], thresholds["halstead_difficulty"], logo="codewars"))
        badges.append(generate_badge("Halstead Effort", halstead["effort"], thresholds["halstead_effort"], logo="atlassian"))

    if mi is not None:
        badges.append(generate_badge("Maintainability Index", mi, thresholds["maintainability_index"], logo="codeclimate"))

    return " ".join(badges)


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """
    Formats a table in Markdown.

    Args:
        headers (List[str]): The table headers.
        rows (List[List[str]]): The table rows.

    Returns:
        str: The formatted Markdown table.
    """
    # Sanitize headers
    headers = [sanitize_text(header) for header in headers]

    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        # Sanitize each cell
        sanitized_row = [sanitize_text(cell) for cell in row]
        table += "| " + " | ".join(sanitized_row) + " |\n"
    return table


def truncate_description(description: str, max_length: int = 100) -> str:
    """
    Truncates a description to a maximum length.

    Args:
        description (str): The description to truncate.
        max_length (int, optional): The maximum length. Defaults to 100.

    Returns:
        str: The truncated description.
    """
    return (description[:max_length] + '...') if len(description) > max_length else description


def sanitize_text(text: str) -> str:
    """
    Sanitizes text for Markdown by escaping special characters.

    Args:
        text (str): The text to sanitize.

    Returns:
        str: The sanitized text.
    """
    markdown_special_chars = ['\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '#', '+', '-', '.', '!', '|']
    for char in markdown_special_chars:
        text = text.replace(char, f"\\{char}")
    return text.replace('\n', ' ').strip()


def generate_table_of_contents(content: str) -> str:
    """
    Generates a table of contents from Markdown headers.

    Args:
        content (str): The Markdown content.

    Returns:
        str: The generated table of contents.
    """
    toc = []
    for line in content.splitlines():
        if line.startswith("#"):
            level = line.count("#")
            title = line.lstrip("#").strip()
            anchor = re.sub(r'[^a-zA-Z0-9\s-]', '', title)
            anchor = re.sub(r'\s+', '-', anchor).lower()
            anchor = re.sub(r'-+', '-', anchor).strip('-')
            toc.append(f"{'  ' * (level - 1)}- [{title}](#{anchor})")
    return "\n".join(toc)


def format_methods(methods: List[Dict[str, Any]]) -> str:
    """
    Formats method information into a Markdown table.

    Args:
        methods (List[Dict[str, Any]]): The methods to format.

    Returns:
        str: The formatted Markdown table.
    """
    headers = ["Method Name", "Complexity", "Async", "Docstring"]
    rows = [
        [
            method.get("name", "N/A"),
            str(method.get("complexity", 0)),
            str(method.get("async", False)),
            truncate_description(sanitize_text(method.get("docstring", "")))
        ]
        for method in methods
    ]
    return format_table(headers, rows)


def format_classes(classes: List[Dict[str, Any]]) -> str:
    """
    Formats class information into a Markdown table.

    Args:
        classes (List[Dict[str, Any]]): The classes to format.

    Returns:
        str: The formatted Markdown table.
    """
    headers = ["Class Name", "Docstring"]
    rows = [
        [
            cls.get("name", "N/A"),
            truncate_description(sanitize_text(cls.get("docstring", "")))
        ]
        for cls in classes
    ]
    class_table = format_table(headers, rows)
    
    method_tables = []
    for cls in classes:
        if cls.get("methods"):
            method_tables.append(f"#### Methods for {cls.get('name')}\n")
            method_tables.append(format_methods(cls.get("methods", [])))
    
    return class_table + "\n\n" + "\n".join(method_tables)


def format_functions(functions: List[Dict[str, Any]]) -> str:
    """
    Formats function information into a Markdown table.

    Args:
        functions (List[Dict[str, Any]]): The functions to format.

    Returns:
        str: The formatted Markdown table or a message if no functions are found.
    """
    if not functions:
        return "No functions found."

    headers = ["Function Name", "Complexity", "Async", "Docstring"]
    rows = [
        [
            func.get("name", "N/A"),
            str(func.get("complexity", 0)),
            str(func.get("async", False)),
            truncate_description(sanitize_text(func.get("docstring", "")))
        ]
        for func in functions
    ]
    return format_table(headers, rows)


def generate_summary(variables: List[Dict[str, Any]], constants: List[Dict[str, Any]]) -> str:
    """
    Generates a summary with tooltips for variables and constants.

    Args:
        variables (List[Dict[str, Any]]): The variables to include in the summary.
        constants (List[Dict[str, Any]]): The constants to include in the summary.

    Returns:
        str: The generated summary.
    """
    total_vars = len(variables)
    total_consts = len(constants)

    # Create tooltip content for variables
    var_tooltip = ""
    if total_vars > 0:
        var_tooltip = "\n".join([f"- {sanitize_text(var.get('name', 'N/A'))}" for var in variables])
        total_vars_display = f'<span title="{var_tooltip}">{total_vars}</span>'
    else:
        total_vars_display = str(total_vars)

    # Create tooltip content for constants
    const_tooltip = ""
    if total_consts > 0:
        const_tooltip = "\n".join([f"- {sanitize_text(const.get('name', 'N/A'))}" for const in constants])
        total_consts_display = f'<span title="{const_tooltip}">{total_consts}</span>'
    else:
        total_consts_display = str(total_consts)

    summary = f"### **Summary**\n\n- **Total Variables:** {total_vars_display}\n- **Total Constants:** {total_consts_display}\n"
    return summary


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a filename by replacing invalid characters.

    Args:
        filename (str): The filename to sanitize.

    Returns:
        str: The sanitized filename.
    """
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def generate_documentation_prompt(file_name, code_structure, project_info, style_guidelines, language, function_schema):
    """
    Generates a prompt for documentation generation.

    Args:
        file_name (str): The name of the file.
        code_structure (dict): The code structure.
        project_info (str): Information about the project.
        style_guidelines (str): The style guidelines.
        language (str): The programming language.
        function_schema (dict): The function schema.

    Returns:
        str: The generated prompt.
    """
    docstring_format = function_schema["functions"][0]["parameters"]["properties"]["docstring_format"]["enum"][0]
    prompt = f"""
    You are a code documentation generator.

    Project Info:
    {project_info}

    Style Guidelines:
    {style_guidelines}

    Use the {docstring_format} style for docstrings.

    Given the following code structure of the {language} file '{file_name}', generate detailed documentation according to the specified schema.

    Code Structure:
    {json.dumps(code_structure, indent=2)}

    Schema:
    {json.dumps(function_schema["functions"][0]["parameters"], indent=2)}

    Ensure that the output is a JSON object that follows the schema exactly, including all required fields.
    """
    return textwrap.dedent(prompt).strip()


async def write_documentation_report(
    documentation: Optional[dict],
    language: str,
    file_path: str,
    repo_root: str,
    output_dir: str
) -> Optional[str]:
    """
    Writes a documentation report to a Markdown file.

    Args:
        documentation (Optional[dict]): The documentation data.
        language (str): The programming language.
        file_path (str): The path to the source file.
        repo_root (str): The root directory of the repository.
        output_dir (str): The directory to save the documentation file.

    Returns:
        Optional[str]: The content of the documentation report or None if an error occurs.
    """
    try:
        if not documentation:
            logger.warning(f"No documentation to write for '{file_path}'")
            return None

        relative_path = os.path.relpath(file_path, repo_root)
        safe_file_name = sanitize_filename(relative_path.replace(os.sep, '_'))
        doc_file_path = os.path.join(output_dir, f"{safe_file_name}.md")

        file_header = f"# File: {relative_path}\n\n"
        documentation_content = file_header

        # Generate and add badges
        badges = generate_all_badges(
            complexity=documentation.get('complexity'),
            halstead=documentation.get('halstead', {}),
            mi=documentation.get('maintainability_index')
        )
        documentation_content += badges + "\n\n"

        # Generate Table of Contents placeholder (will be updated later)
        documentation_content += "# Table of Contents\n\n"

        # Add Summary
        summary = generate_summary(documentation.get('variables', []), documentation.get('constants', []))
        documentation_content += summary + "\n"

        # Add Changes Made
        changes_made = documentation.get('changes_made', [])
        if changes_made:
            documentation_content += f"## Changes Made\n\n"
            for change in changes_made:
                documentation_content += f"- {sanitize_text(change)}\n"
            documentation_content += "\n"

        # Add Classes and Methods
        classes = documentation.get('classes', [])
        if classes:
            documentation_content += "## Classes\n\n"
            documentation_content += format_classes(classes)
            documentation_content += "\n"

        # Add Functions
        functions = documentation.get('functions', [])
        if functions:
            documentation_content += "## Functions\n\n"
            documentation_content += format_functions(functions)
            documentation_content += "\n"

        # Add Source Code
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            source_code = await file.read()
        documentation_content += f"## Source Code\n\n```{language}\n{source_code}\n```\n"

        # Generate Table of Contents
        toc = generate_table_of_contents(documentation_content)
        # Replace the placeholder with the actual TOC
        documentation_content = documentation_content.replace("# Table of Contents\n\n", f"# Table of Contents\n\n{toc}\n\n", 1)

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        async with aiofiles.open(doc_file_path, 'w', encoding='utf-8') as f:
            await f.write(documentation_content)
        logger.info(f"Documentation written to '{doc_file_path}' successfully.")
        return documentation_content

    except Exception as e:
        logger.error(f"Error writing documentation report for '{file_path}': {e}", exc_info=True)
        return None