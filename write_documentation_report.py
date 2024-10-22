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

def extract_module_docstring(file_path: str) -> str:
    """
    Extracts the module docstring from a Python file.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        str: The module docstring if found, otherwise an empty string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        module = ast.parse(source)
        return ast.get_docstring(module) or ''
    except Exception as e:
        logger.error(f"Error extracting module docstring from '{file_path}': {e}", exc_info=True)
        return ''
    
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
    # Replace invalid characters with underscores
    invalid_chars = r'<>:"/\\|?*'
    return ''.join('_' if c in invalid_chars else c for c in filename)

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
    documentation: Optional[dict],
    language: str,
    file_path: str,
    repo_root: str,
    output_dir: str
) -> Optional[str]:
    try:
        if not documentation:
            logger.warning(f"No documentation to write for '{file_path}'")
            return None

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create documentation directory '{output_dir}': {e}")
            return None

        relative_path = os.path.relpath(file_path, repo_root)
        safe_file_name = sanitize_filename(os.path.basename(relative_path))
        doc_file_path = os.path.join(output_dir, safe_file_name + '.md')

        # Read the source code
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            logger.error(f"Failed to read source file '{file_path}': {e}")
            source_code = "Failed to read source code"

        # Start building the Markdown content
        sections = []

        # File header and badges
        sections.append(f"# {relative_path}")
        
        badges = [
            generate_badge("Complexity", documentation.get('complexity', 0), DEFAULT_COMPLEXITY_THRESHOLDS),
            generate_badge("Halstead Volume", documentation.get('halstead', {}).get('volume', 0), DEFAULT_HALSTEAD_THRESHOLDS['volume']),
            generate_badge("Maintainability Index", documentation.get('maintainability_index', 0), DEFAULT_MAINTAINABILITY_THRESHOLDS)
        ]
        sections.append(" ".join(badges))

        # Source Code (at the top, right after badges)
        sections.append("## Source Code")
        sections.append(f"```{language}\n{source_code}\n```")

        # Summary
        sections.append("## Summary")
        sections.append(documentation.get('summary', 'No summary available.'))

        # Recent Changes
        changes = documentation.get('changes_made', [])
        if changes:
            sections.append("## Recent Changes")
            sections.append("\n".join(f"- {change}" for change in changes))

        # Functions
        functions = documentation.get('functions', [])
        if functions:
            sections.append("## Functions")
            table = ["| Name | Async | Complexity | Arguments | Description |",
                    "|------|-------|------------|-----------|-------------|"]
            for func in functions:
                name = func.get('name', 'N/A')
                is_async = 'Yes' if func.get('async', False) else 'No'
                complexity = func.get('complexity', 'N/A')
                args = ', '.join(func.get('args', []))
                description = func.get('docstring', 'No description').split('\n')[0]
                table.append(f"| {name} | {is_async} | {complexity} | {args} | {description} |")
            sections.append("\n".join(table))

        # Classes
        classes = documentation.get('classes', [])
        if classes:
            sections.append("## Classes")
            for cls in classes:
                class_name = cls.get('name', 'N/A')
                class_description = cls.get('docstring', 'No description').split('\n')[0]
                sections.append(f"### {class_name}")
                sections.append(class_description)
                
                methods = cls.get('methods', [])
                if methods:
                    table = ["| Method | Async | Type | Complexity | Arguments | Description |",
                            "|--------|-------|------|------------|-----------|-------------|"]
                    for method in methods:
                        name = method.get('name', 'N/A')
                        is_async = 'Yes' if method.get('async', False) else 'No'
                        method_type = method.get('type', 'N/A')
                        complexity = method.get('complexity', 'N/A')
                        args = ', '.join(method.get('args', []))
                        description = method.get('docstring', 'No description').split('\n')[0]
                        table.append(f"| {name} | {is_async} | {method_type} | {complexity} | {args} | {description} |")
                    sections.append("\n".join(table))

        # Variables
        variables = documentation.get('variables', [])
        if variables:
            sections.append("## Variables")
            table = ["| Name | Type | Description | File | Line | Example | References |",
                    "|------|------|-------------|------|------|---------|------------|"]
            for var in variables:
                name = var.get('name', 'N/A')
                var_type = var.get('type', 'N/A')
                description = var.get('description', 'No description')
                file = var.get('file', 'N/A')
                line = var.get('line', 'N/A')
                example = var.get('example', 'N/A')
                references = var.get('references', 'N/A')
                table.append(f"| {name} | {var_type} | {description} | {file} | {line} | {example} | {references} |")
            sections.append("\n".join(table))

        # Constants
        constants = documentation.get('constants', [])
        if constants:
            sections.append("## Constants")
            table = ["| Name | Type | Description | File | Line | Example | References |",
                    "|------|------|-------------|------|------|---------|------------|"]
            for const in constants:
                name = const.get('name', 'N/A')
                const_type = const.get('type', 'N/A')
                description = const.get('description', 'No description')
                file = const.get('file', 'N/A')
                line = const.get('line', 'N/A')
                example = const.get('example', 'N/A')
                references = const.get('references', 'N/A')
                table.append(f"| {name} | {const_type} | {description} | {file} | {line} | {example} | {references} |")
            sections.append("\n".join(table))

        # Metrics
        halstead = documentation.get('halstead', {})
        if halstead:
            sections.append("## Halstead Metrics")
            metrics = [
                f"- Volume: {halstead.get('volume', 'N/A')}",
                f"- Difficulty: {halstead.get('difficulty', 'N/A')}",
                f"- Effort: {halstead.get('effort', 'N/A')}"
            ]
            sections.append("\n".join(metrics))

        maintainability_index = documentation.get('maintainability_index')
        if maintainability_index is not None:
            sections.append("## Maintainability Index")
            sections.append(str(maintainability_index))

        # Join all sections with double newlines for better readability
        content = "\n\n".join(sections)

        # Generate and add Table of Contents at the start
        toc = generate_table_of_contents(content)
        content = f"# Table of Contents\n\n{toc}\n\n{content}"

        # Write the file
        try:
            async with aiofiles.open(doc_file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            logger.info(f"Documentation written to '{doc_file_path}' successfully.")
            return content
        except Exception as e:
            logger.error(f"Error writing to file '{doc_file_path}': {e}")
            return None

    except Exception as e:
        logger.error(f"Error writing documentation report for '{file_path}': {e}", exc_info=True)
        return None