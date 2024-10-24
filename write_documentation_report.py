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

    complexity = metrics.get("cyclomatic")
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
    return (description[:max_length] + '...') if len(description) > max_length else description


def sanitize_text(text: str) -> str:
    """Sanitizes text for Markdown."""
    markdown_special_chars = ['\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '#', '+', '-', '.', '!', '|']
    for char in markdown_special_chars:
        text = text.replace(char, f"\\{char}")
    return text.replace('\n', ' ').strip()


def generate_table_of_contents(content: str) -> str:
    """Generates a table of contents from Markdown headers."""
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
    """Formats method information into a Markdown table."""
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
    """Formats class information into a Markdown table."""
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
    """Formats function information into a Markdown table."""

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



def generate_summary(documentation: Dict[str, Any]) -> str:
    """Generates a summary with Markdown lists for variables and constants."""
    variables = documentation.get("variables", [])
    constants = documentation.get("constants", [])

    summary_parts = []
    if variables:
        summary_parts.append("**Variables:**")
        summary_parts.extend([f"- {sanitize_text(var.get('name', 'N/A'))}: {sanitize_text(var.get('description', ''))}" for var in variables])

    if constants:
        summary_parts.append("**Constants:**")
        summary_parts.extend([f"- {sanitize_text(const.get('name', 'N/A'))}: {sanitize_text(const.get('description', ''))}" for const in constants])

    return "### **Summary**\n\n" + "\n".join(summary_parts) if summary_parts else "### **Summary**\n\nNo variables or constants found."



def generate_documentation_prompt(file_name, code_structure, project_info, style_guidelines, language, function_schema):
    """
    Generates a prompt for documentation generation that is compatible with the provided schema.

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
    output_dir: str
) -> Optional[Dict[str, Any]]:
    """
    Generates documentation in both frontend-compatible JSON and Markdown formats.

    Args:
        documentation: The documentation data.
        language: The programming language.
        file_path: Path to the source file.
        repo_root: Root directory of the repository.
        output_dir: Output directory for documentation.

    Returns:
        Optional[Dict[str, Any]]: The formatted documentation data, or None if generation fails.
    """
    if not documentation:
        logger.warning(f"No documentation to write for '{file_path}'")
        return None

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        relative_path = os.path.relpath(file_path, repo_root)

        # Generate frontend-compatible JSON structure
        frontend_docs = {
            "summary": documentation.get("summary", ""),
            "classes": [
                {
                    "name": cls["name"],
                    "docstring": cls["docstring"],
                    "methods": [
                        {
                            "name": method["name"],
                            "docstring": method["docstring"],
                            "args": method.get("args", []),
                            "async": method.get("async", False),
                            "complexity": method.get("complexity", 0),
                            "type": method.get("type", "instance")
                        }
                        for method in cls.get("methods", [])
                    ]
                }
                for cls in documentation.get("classes", [])
            ],
            "functions": [
                {
                    "name": func["name"],
                    "docstring": func["docstring"],
                    "args": func.get("args", []),
                    "async": func.get("async", False),
                    "complexity": func.get("complexity", 0)
                }
                for func in documentation.get("functions", [])
            ],
            "metrics": {
                "maintainability_index": documentation.get("maintainability_index", 0),
                "complexity": documentation.get("complexity", 0),
                "halstead": documentation.get("halstead", {
                    "volume": 0,
                    "difficulty": 0,
                    "effort": 0
                })
            }
        }

        # Also generate Markdown content for reference
        markdown_content = await generate_markdown_content(documentation, language, file_path, relative_path)

        # Write both formats
        safe_filename = sanitize_filename(os.path.basename(file_path))
        base_path = Path(output_dir) / safe_filename

        # Write JSON for frontend
        json_path = base_path.with_suffix('.json')
        async with aiofiles.open(json_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(frontend_docs, indent=2))

        # Write Markdown for reference
        md_path = base_path.with_suffix('.md')
        async with aiofiles.open(md_path, 'w', encoding='utf-8') as f:
            await f.write(markdown_content)

        logger.info(f"Documentation written to {json_path} and {md_path}")
        return frontend_docs

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
    
    # Generate table of contents
    toc = [
        "# Table of Contents\n",
        "1. [Overview](#overview)",
        "2. [Code Structure](#code-structure)",
        "3. [Dependencies](#dependencies)",
        "4. [Metrics](#metrics)",
        "\n---\n"
    ]

    # Generate Overview section
    overview = [
        "# Overview\n",
        f"**File:** `{os.path.basename(file_path)}`  ",
        f"**Language:** {language}  ",
        f"**Path:** `{relative_path}`  \n",
        "## Summary\n",
        f"{documentation.get('summary', 'No summary available.')}\n",
        "## Recent Changes\n",
        "\n".join(f"- {change}" for change in documentation.get('changes_made', ['No recent changes.'])),
        "\n"
    ]

    # Generate Code Structure section
    code_structure = ["# Code Structure\n"]
    
    if documentation.get("classes"):
        code_structure.append("## Classes\n")
        code_structure.append(format_classes(documentation["classes"]))

    if documentation.get("functions"):
        code_structure.append("## Functions\n")
        code_structure.append(format_functions(documentation["functions"]))

    # Generate Dependencies section
    dependencies = [
        "# Dependencies\n",
        "```mermaid",
        "graph TD;",
    ]
    
    # Create dependency graph
    dep_map = {}
    for dep in documentation.get('variables', []) + documentation.get('constants', []):
        if dep.get('type') == 'import':
            dep_name = dep['name']
            dep_refs = dep.get('references', [])
            dep_map[dep_name] = dep_refs
            dependencies.append(f"    {dep_name}[{dep_name}];")
    
    for dep, refs in dep_map.items():
        for ref in refs:
            if ref in dep_map:
                dependencies.append(f"    {dep} --> {ref};")
    
    dependencies.extend(["```\n"])

    # Generate Metrics section
    metrics = [
        "# Metrics\n",
        "## Code Quality\n",
        generate_all_badges(
            complexity=documentation.get("complexity"),
            halstead=documentation.get("halstead"),
            mi=documentation.get("maintainability_index")
        ),
        "\n"
    ]

    if "halstead" in documentation:
        halstead = documentation["halstead"]
        metrics.extend([
            "## Halstead Metrics\n",
            "| Metric | Value |",
            "|--------|--------|",
            f"| Volume | {halstead.get('volume', 0):.1f} |",
            f"| Difficulty | {halstead.get('difficulty', 0):.1f} |",
            f"| Effort | {halstead.get('effort', 0):.1f} |",
        ])

    # Combine all sections
    content = "\n".join([
        *toc,
        *overview,
        *code_structure,
        *dependencies,
        *metrics
    ])

    return content


def get_metric_status(value: float) -> str:
    """Returns a status indicator based on metric value."""
    if value >= 80:
        return "✅ Good"
    elif value >= 60:
        return "⚠️ Warning"
    return "❌ Needs Improvement"


def sanitize_filename(filename: str) -> str:
    """Sanitizes filename by removing invalid characters."""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)
