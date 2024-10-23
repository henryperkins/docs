import aiofiles
import re
import json
import os
import textwrap
import logging
import sys
from typing import Optional, Dict, Any, List, cast
import ast  # Import ast module

# Import default thresholds from utils
from utils import (
    DEFAULT_COMPLEXITY_THRESHOLDS,
    DEFAULT_HALSTEAD_THRESHOLDS,
    DEFAULT_MAINTAINABILITY_THRESHOLDS,
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
    """Generates a prompt for documentation generation."""
    # ... (no changes)


async def write_documentation_report(
    documentation: Optional[dict],
    language: str,
    file_path: str,
    repo_root: str,
    output_dir: str
) -> Optional[str]:
    """Generates and writes the documentation report."""

    if not documentation:
        logger.warning(f"No documentation to write for '{file_path}'")
        return None

    try:
        os.makedirs(output_dir, exist_ok=True)
        relative_path = os.path.relpath(file_path, repo_root)
        file_name = os.path.basename(file_path)  # Get file name

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
            f"**File:** `{file_name}`  ", # Use file_name variable
            f"**Language:** {language}  ",
            f"**Path:** `{relative_path}`  \n",
            generate_summary(documentation), # Call generate_summary with documentation
            "## Recent Changes\n",
            "\n".join(f"- {change}" for change in documentation.get('changes_made', ['No recent changes.'])),
            "\n"
        ]

        # Generate Code Structure section
        code_structure_parts = ["# Code Structure\n"]

        if documentation.get("classes"):
            code_structure_parts.append("## Classes\n")
            code_structure_parts.append(format_classes(documentation["classes"]))

        if documentation.get("functions"):
            code_structure_parts.append("## Functions\n")
            code_structure_parts.append(format_functions(documentation["functions"]))

        code_structure_md = "\n".join(code_structure_parts)


        # Generate Dependencies section
        dependencies = [
            "# Dependencies\n",
            "```mermaid",
            "graph TD;",
        ]
        
        # Create dependency graph (Consider replacing with a more robust library)
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
        metrics_parts = [
            "# Metrics\n",
            "## Code Quality\n",
            generate_all_badges(documentation.get("metrics", {})),  # Pass the metrics dictionary
        ]

        if "halstead" in documentation:
            halstead = documentation["halstead"]
            metrics_parts.extend([ # Use metrics_parts
                "## Halstead Metrics\n",
                "| Metric | Value |",
                "|--------|--------|",
                f"| Volume | {halstead.get('volume', 0):.1f} |",
                f"| Difficulty | {halstead.get('difficulty', 0):.1f} |",
                f"| Effort | {halstead.get('effort', 0):.1f} |",
                # ... Add other Halstead metrics here ...
            ])

        if "raw" in documentation:
            raw = documentation["raw"]
            metrics_parts.extend([ # Use metrics_parts
                "## Raw Metrics\n",
                "| Metric | Value |",
                "|--------|--------|",
                f"| Lines of Code (LOC) | {raw.get('loc', 0)} |",
                f"| Logical Lines of Code (LLOC) | {raw.get('lloc', 0)} |",
                f"| Source Lines of Code (SLOC) | {raw.get('sloc', 0)} |",
                f"| Comments | {raw.get('comments', 0)} |",
                f"| Multiline Strings | {raw.get('multi', 0)} |",
                f"| Blank Lines | {raw.get('blank', 0)} |",
            ])

        if "quality" in documentation:
            quality = documentation["quality"]
            metrics_parts.extend([ # Use metrics_parts
                "## Code Quality Metrics\n",
                "| Metric | Value |",
                "|--------|--------|",
                f"| Average Method Length | {quality.get('avg_method_length', 0):.1f} |",
                f"| Average Argument Count | {quality.get('avg_argument_count', 0):.1f} |",
                f"| Max Nesting Level | {quality.get('max_nesting_level', 0)} |",
                f"| Average Nesting Level | {quality.get('avg_nesting_level', 0):.1f} |",
                # ... Add other code quality metrics here ...
            ])

        # Combine all sections
        content = "\n".join([
            *toc,
            *overview,
            code_structure_md,
            *dependencies,
            *metrics_parts # Correct variable name here
        ])

        # Write the documentation file
        safe_filename = sanitize_filename(os.path.basename(file_path))
        output_path = os.path.join(output_dir, f"{safe_filename}.md")
        
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(content)

        logger.info(f"Documentation written to {output_path}")
        return content

    except Exception as e:
        logger.error(f"Error writing documentation report: {e}", exc_info=True)
        return None


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
