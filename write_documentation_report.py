"""
write_documentation_report.py

This module provides functions for generating documentation reports in both JSON (for frontend) 
and Markdown formats. It includes utilities for creating badges, formatting tables, generating 
summaries, and structuring documentation data.
"""

import aiofiles
import re
import json
import os
import textwrap
import logging
import sys
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

from utils import (
    DEFAULT_COMPLEXITY_THRESHOLDS,
    DEFAULT_HALSTEAD_THRESHOLDS,
    DEFAULT_MAINTAINABILITY_THRESHOLDS,
    get_threshold
)

logger = logging.getLogger(__name__)


def generate_badge(metric_name: str, value: float, thresholds: Dict[str, int], logo: str = None) -> str:
    """Generates a Markdown badge for a given metric."""
    low, medium, high = thresholds["low"], thresholds["medium"], thresholds["high"]
    color = "green" if value <= low else "yellow" if value <= medium else "red"
    badge_label = metric_name.replace("_", " ").title()
    logo_part = f"&logo={logo}" if logo else ""
    return f"![{badge_label}](https://img.shields.io/badge/{badge_label}-{value:.2f}-{color}?style=flat-square{logo_part})"


def generate_all_badges(metrics: Dict[str, Any]) -> str:
    """Generates badges for all metrics."""

    complexity = metrics.get("complexity")
    halstead = metrics.get("halstead")
    mi = metrics.get("maintainability_index")

    badges = []

    if complexity is not None:
        badges.append(generate_badge("Complexity", complexity, DEFAULT_COMPLEXITY_THRESHOLDS, logo="codeClimate"))

    if halstead:
        badges.append(generate_badge("Halstead Volume", halstead["volume"], DEFAULT_HALSTEAD_THRESHOLDS["volume"], logo="stackOverflow"))
        badges.append(generate_badge("Halstead Difficulty", halstead["difficulty"], DEFAULT_HALSTEAD_THRESHOLDS["difficulty"], logo="codewars"))
        badges.append(generate_badge("Halstead Effort", halstead["effort"], DEFAULT_HALSTEAD_THRESHOLDS["effort"], logo="atlassian"))

    if mi is not None:
        badges.append(generate_badge("Maintainability Index", mi, DEFAULT_MAINTAINABILITY_THRESHOLDS, logo="codeclimate"))

    return " ".join(badges)


def format_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Formats data into a Markdown table."""
    headers = [sanitize_text(str(header)) for header in headers]
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        sanitized_row = [sanitize_text(str(cell)) for cell in row]
        table += "| " + " | ".join(sanitized_row) + " |\n"
    return table


def truncate_description(description: str, max_length: int = 100) -> str:
    """Truncates a description to a maximum length."""
    if len(description) > max_length:
        return description[:max_length] + "..."
    return description


def sanitize_text(text: str) -> str:
    """Sanitizes text for Markdown."""
    escaped_text = re.escape(text)
    return escaped_text.replace('\n', ' ').replace('\r', '').strip()



def generate_table_of_contents(content: str) -> str:
    """Generates a table of contents from Markdown headers."""
    headers = re.findall(r"^(#{1,6})\s+(.*)", content, re.MULTILINE)
    toc = []
    for header in headers:
        level = len(header[0])
        title = header[1]
        link = title.lower().replace(" ", "-").replace(".", "").replace(",", "")
        toc.append(f"{'  ' * (level - 1)}- [{title}](#{link})")
    return "\n".join(toc)


def format_methods(methods: List[Dict[str, Any]]) -> str:
    """Formats method information into a Markdown table."""
    headers = ["Method", "Description", "Arguments", "Complexity"]
    rows = []
    for method in methods:
        name = method.get("name", "unknown")
        doc = truncate_description(method.get("docstring", ""))
        args = ", ".join(method.get("args", []))
        complexity = method.get("complexity", "N/A")
        rows.append([name, doc, args, complexity])
    return format_table(headers, rows)


def format_classes(classes: List[Dict[str, Any]]) -> str:
    """Formats class information into a Markdown table."""
    headers = ["Class", "Description", "Methods"]
    rows = []
    for cls in classes:
        name = cls.get("name", "unknown")
        doc = truncate_description(cls.get("docstring", ""))
        methods = format_methods(cls.get("methods", []))
        rows.append([name, doc, methods])
    return format_table(headers, rows)


def format_functions(functions: List[Dict[str, Any]]) -> str:
    """Formats function information into a Markdown table."""
    headers = ["Function", "Description", "Arguments", "Complexity"]
    rows = []
    for func in functions:
        name = func.get("name", "unknown")
        doc = truncate_description(func.get("docstring", ""))
        args = ", ".join(func.get("args", []))
        complexity = func.get("complexity", "N/A")
        rows.append([name, doc, args, complexity])
    return format_table(headers, rows)


def generate_summary(documentation: Dict[str, Any]) -> str:
    """Generates a summary with Markdown lists for variables and constants."""
    summary = []

    variables = documentation.get("variables", [])
    if variables:
        summary.append("## Variables")
        for var in variables:
            name = var.get("name", "unknown")
            desc = truncate_description(var.get("description", ""))
            summary.append(f"- **{name}**: {desc}")

    constants = documentation.get("constants", [])
    if constants:
        summary.append("## Constants")
        for const in constants:
            name = const.get("name", "unknown")
            desc = truncate_description(const.get("description", ""))
            summary.append(f"- **{name}**: {desc}")

    return "\n".join(summary)


def generate_documentation_prompt(file_name, code_structure, project_info, style_guidelines, language, function_schema):
    """Generates a prompt for documentation generation."""

    docstring_format = function_schema["functions"][0]["parameters"]["properties"]["docstring_format"]["enum"][0]

    prompt = f"""
    You are a code documentation generator. Your task is to generate documentation for the following {language} file: '{file_name}'.
    
    Project Info:
    {project_info}

    Style Guidelines:
    {style_guidelines}

    Use the {docstring_format} style for docstrings.

    The documentation should strictly follow this schema:
    {{
        "docstring_format": "{docstring_format}",
        "summary": "A detailed summary of the file.",
        "changes_made": ["List of changes made"],
        "functions": [
            {{
                "name": "Function name",
                "docstring": "Detailed description in {docstring_format} style",
                "args": ["List of argument names"],
                "async": true/false,
                "complexity": integer
            }}
        ],
        "classes": [
            {{
                "name": "Class name",
                "docstring": "Detailed description in {docstring_format} style",
                "methods": [
                    {{
                        "name": "Method name",
                        "docstring": "Detailed description in {docstring_format} style",
                        "args": ["List of argument names"],
                        "async": true/false,
                        "type": "instance/class/static",
                        "complexity": integer
                    }}
                ]
            }}
        ],
        "halstead": {{
            "volume": number,
            "difficulty": number,
            "effort": number
        }},
        "maintainability_index": number,
        "variables": [
            {{
                "name": "Variable name",
                "type": "Inferred data type",
                "description": "Description of the variable",
                "file": "File name",
                "line": integer,
                "link": "Link to definition",
                "example": "Example usage",
                "references": "References to the variable"
            }}
        ],
        "constants": [
            {{
                "name": "Constant name",
                "type": "Inferred data type",
                "description": "Description of the constant",
                "file": "File name",
                "line": integer,
                "link": "Link to definition",
                "example": "Example usage",
                "references": "References to the constant"
            }}
        ]
    }}

    Ensure that all required fields are included and properly formatted according to the schema.

    Given the following code structure:
    {json.dumps(code_structure, indent=2)}

    Generate detailed documentation that strictly follows the provided schema. Include the following:
    1. A comprehensive summary of the file's purpose and functionality.
    2. A list of recent changes or modifications made to the file.
    3. Detailed documentation for all functions, including their arguments, return types, and whether they are asynchronous.
    4. Comprehensive documentation for all classes and their methods, including inheritance information if applicable.
    5. Information about all variables and constants, including their types, descriptions, and usage examples.
    6. Accurate Halstead metrics (volume, difficulty, and effort) for the entire file.
    7. The maintainability index of the file.

    Ensure that all docstrings follow the {docstring_format} format and provide clear, concise, and informative descriptions.
    """
    return textwrap.dedent(prompt).strip()



async def write_documentation_report(
    documentation: Optional[Dict[str, Any]],
    language: str,
    file_path: str,
    repo_root: str,
    output_dir: str,
    project_id: str,  # Added project_id
) -> Optional[Dict[str, Any]]:
    """Writes documentation to JSON and Markdown files."""

    if not documentation:
        logger.warning(f"No documentation to write for '{file_path}'")
        return None

    try:
        # Construct the project-specific output path
        project_output_dir = Path(output_dir) / project_id
        project_output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        relative_path = os.path.relpath(file_path, repo_root)
        safe_filename = sanitize_filename(os.path.basename(file_path))
        base_path = project_output_dir / safe_filename  # Use project_output_dir

        # Frontend-compatible JSON
        json_path = base_path.with_suffix(".json")
        async with aiofiles.open(json_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(documentation, indent=2))  # Write the full documentation

        # Markdown for reference (optional - check if needed)
        if documentation.get("generate_markdown", True):  # Add a flag to control Markdown generation
            markdown_content = await generate_markdown_content(documentation, language, file_path, relative_path)
            md_path = base_path.with_suffix(".md")
            async with aiofiles.open(md_path, "w", encoding="utf-8") as f:
                await f.write(markdown_content)

        logger.info(f"Documentation written to {json_path}")
        return documentation  # Return the documentation

    except Exception as e:
        logger.error(f"Error writing documentation report: {e}", exc_info=True)
        return None



async def generate_markdown_content(
    documentation: Dict[str, Any],
    language: str,
    file_path: str,
    relative_path: str
) -> str:
    """Generates enhanced markdown content with collapsible sections and better formatting."""

    badges = generate_all_badges(documentation.get("metrics", {}))
    summary = generate_summary(documentation)
    functions = format_functions(documentation.get("functions", []))
    classes = format_classes(documentation.get("classes", []))


    content = f"""
# Documentation for {os.path.basename(file_path)}

{badges}

{summary}

## Functions
{functions}

## Classes
{classes}
    """.strip()

    toc = generate_table_of_contents(content) # Generate TOC last
    return f"# Table of Contents\n\n{toc}\n\n{content}"


def get_metric_status(value: float, thresholds: Dict[str, int]) -> str:
    """Returns a status indicator based on metric value and thresholds."""
    if value <= thresholds["low"]:
        return "Low"
    elif value <= thresholds["medium"]:
        return "Medium"
    else:
        return "High"


def sanitize_filename(filename: str) -> str:
    """Sanitizes filename by removing invalid characters."""
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)
