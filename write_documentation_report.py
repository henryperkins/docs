"""
write_documentation_report.py

This module provides functions for generating documentation reports in both JSON
and Markdown formats. It is thread-safe for concurrent write operations, handles
file I/O exceptions gracefully, and uses f-strings for string formatting.
"""

import aiofiles
import re
import json
import os
import logging
import sys
from typing import Optional, Dict, Any, List, Union, cast
from pathlib import Path
import asyncio

from utils import (
    DEFAULT_COMPLEXITY_THRESHOLDS,
    DEFAULT_HALSTEAD_THRESHOLDS,
    DEFAULT_MAINTAINABILITY_THRESHOLDS,
)
from token_utils import TokenManager
from code_chunk import CodeChunk
from context_manager import HierarchicalContextManager

logger = logging.getLogger(__name__)

# Create a lock for file writing
write_lock = asyncio.Lock()


def generate_badge(metric_name: str, value: Union[int, float], thresholds: Dict[str, int], logo: str = None) -> str:
    """Generates a Markdown badge for a given metric."""
    low, medium, high = thresholds["low"], thresholds["medium"], thresholds["high"]
    color = "green" if value <= low else "yellow" if value <= medium else "red"
    badge_label = metric_name.replace("_", " ").title()
    logo_part = f"&logo={logo}" if logo else ""
    return f"![{badge_label}](https://img.shields.io/badge/{badge_label}-{value:.2f}-{color}?style=flat-square{logo_part})"


def generate_all_badges(metrics: Dict[str, Any]) -> str:
    """Generates badges for all metrics."""
    badges = []

    if complexity := metrics.get("complexity"):
        badges.append(generate_badge("Complexity", complexity, DEFAULT_COMPLEXITY_THRESHOLDS, logo="codeClimate"))

    if halstead := metrics.get("halstead"):
        badges.append(generate_badge("Halstead Volume", halstead["volume"], DEFAULT_HALSTEAD_THRESHOLDS["volume"], logo="stackOverflow"))
        badges.append(generate_badge("Halstead Difficulty", halstead["difficulty"], DEFAULT_HALSTEAD_THRESHOLDS["difficulty"], logo="codewars"))
        badges.append(generate_badge("Halstead Effort", halstead["effort"], DEFAULT_HALSTEAD_THRESHOLDS["effort"], logo="atlassian"))

    if mi := metrics.get("maintainability_index"):
        badges.append(generate_badge("Maintainability Index", mi, DEFAULT_MAINTAINABILITY_THRESHOLDS, logo="codeclimate"))

    return " ".join(badges)


def format_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Formats data into a Markdown table."""
    headers = [sanitize_text(str(header)) for header in headers]
    table = f"| {' | '.join(headers)} |\n"
    table += f"| {' | '.join(['---'] * len(headers))} |\n"
    for row in rows:
        sanitized_row = [sanitize_text(str(cell)) for cell in row]
        table += f"| {' | '.join(sanitized_row)} |\n"
    return table


def truncate_description(description: str, max_length: int = 100) -> str:
    """Truncates a description to a maximum length."""
    return f"{description[:max_length]}..." if len(description) > max_length else description


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


def calculate_prompt_tokens(base_info: str, context: str, chunk_content: str, schema: str) -> int:
    """Calculates total tokens needed for the prompt using TokenManager."""
    total = 0
    for text in [base_info, context, chunk_content, schema]:
        token_result = TokenManager.count_tokens(text)
        total += token_result.token_count
    return total


# Helper functions (these would also be included in the file)
def truncate_description(description: str, max_length: int = 100) -> str:
    """Truncates a description to a maximum length."""
    return f"{description[:max_length]}..." if len(description) > max_length else description


def format_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Formats data into a Markdown table."""
    headers = [sanitize_text(str(header)) for header in headers]
    table = f"| {' | '.join(headers)} |\n"
    table += f"| {' | '.join(['---'] * len(headers))} |\n"
    for row in rows:
        sanitized_row = [sanitize_text(str(cell)) for cell in row]
        table += f"| {' | '.join(sanitized_row)} |\n"
    return table


def sanitize_text(text: str) -> str:
    """Sanitizes text for Markdown."""
    escaped_text = re.escape(text)
    return escaped_text.replace('\n', ' ').replace('\r', '').strip()


def generate_documentation_prompt(
    chunk: CodeChunk,
    context_manager: HierarchicalContextManager,
    project_info: str,
    style_guidelines: str,
    function_schema: Dict[str, Any],
    max_total_tokens: int = 4096,
    max_completion_tokens: int = 1024
) -> List[Dict[str, str]]:
    """Enhanced prompt generation for comprehensive documentation with optimized token management."""

    base_info = f"""
Project Info:
{project_info}

Style Guidelines:
{style_guidelines}
"""

    detailed_instructions = (
        "Please generate comprehensive documentation with the following sections:\n"
        "- **Summary**: A detailed summary of the file or module.\n"
        "- **Changelog**: Include a list of recent changes if available.\n"
        "- **Functions**: For each function, provide the name, docstring, arguments, "
        "whether it's asynchronous, complexity, and Halstead metrics (volume, difficulty, and effort).\n"
        "- **Classes**: For each class, provide the name, docstring, and details of each method, "
        "including arguments, whether it's asynchronous, return type, complexity, and Halstead metrics.\n"
        "- **Variables and Constants**: Document each variable and constant with its name, type, and description.\n"
        "- **Metrics**: Include maintainability index and any other relevant code quality metrics.\n"
    )

    schema_str = json.dumps(function_schema, indent=2)

    fixed_tokens = calculate_prompt_tokens(
        base_info=base_info,
        context="",
        chunk_content=chunk.chunk_content,
        schema=schema_str
    )

    available_context_tokens = max_total_tokens - fixed_tokens - max_completion_tokens
    if available_context_tokens <= 0:
        logger.warning(f"No tokens available for context in chunk {chunk.chunk_id}")
        available_context_tokens = 0

    context_chunks = []
    if chunk.function_name:
        context_chunks = context_manager.get_context_for_function(
            module_path=chunk.file_path,
            function_name=chunk.function_name,
            language=chunk.language,
            max_tokens=available_context_tokens
        )
    elif chunk.class_name:
        context_chunks = context_manager.get_context_for_class(
            module_path=chunk.file_path,
            class_name=chunk.class_name,
            language=chunk.language,
            max_tokens=available_context_tokens
        )
    else:
        context_chunks = context_manager.get_context_for_module(
            module_path=chunk.file_path,
            language=chunk.language,
            max_tokens=available_context_tokens
        )

    context = "Related code and documentation:\n\n"
    current_tokens = 0
    for ctx_chunk in context_chunks:
        if ctx_chunk.chunk_id == chunk.chunk_id:
            continue

        context_addition = f"""
# From {ctx_chunk.get_context_string()}:
{ctx_chunk.chunk_content}
"""
        doc = context_manager.get_documentation_for_chunk(ctx_chunk.chunk_id)
        if doc:
            context_addition += f"""
Existing documentation:
{json.dumps(doc, indent=2)}
"""

        addition_tokens = len(TokenManager.get_encoder().encode(context_addition))
        if current_tokens + addition_tokens > available_context_tokens:
            break

        context += context_addition
        current_tokens += addition_tokens

    prompt_messages = [
        {"role": "system", "content": base_info},
        {"role": "user", "content": detailed_instructions},
        {"role": "assistant", "content": context},
        {"role": "user", "content": chunk.chunk_content},
        {"role": "system", "content": f"Schema:\n{schema_str}"}
    ]

    return prompt_messages


async def write_documentation_report(
    documentation: Optional[Dict[str, Any]],
    language: str,
    file_path: str,
    repo_root: str,
    output_dir: str,
    project_id: str,
) -> Optional[Dict[str, Any]]:
    """Writes documentation to JSON and Markdown files with thread safety and exception handling."""

    if not documentation:
        logger.warning(f"No documentation to write for '{file_path}'")
        return None

    try:
        async with write_lock:
            project_output_dir = Path(output_dir) / project_id
            project_output_dir.mkdir(parents=True, exist_ok=True)

            relative_path = os.path.relpath(file_path, repo_root)
            safe_filename = sanitize_filename(os.path.basename(file_path))
            base_path = project_output_dir / safe_filename

            json_path = base_path.with_suffix(".json")
            try:
                async with aiofiles.open(json_path, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(documentation, indent=2))
            except (FileNotFoundError, PermissionError, OSError) as e:
                logger.error(f"Error writing JSON to {json_path}: {e}", exc_info=True)
                return None

            if documentation.get("generate_markdown", True):
                markdown_content = await generate_markdown_content(documentation, language, file_path, relative_path)
                md_path = base_path.with_suffix(".md")
                try:
                    async with aiofiles.open(md_path, "w", encoding="utf-8") as f:
                        await f.write(markdown_content)
                except (FileNotFoundError, PermissionError, OSError) as e:
                    logger.error(f"Error writing Markdown to {md_path}: {e}", exc_info=True)
                    return None

        logger.info(f"Documentation written to {json_path}")
        return documentation

    except Exception as e:
        logger.error(f"Error writing documentation report: {e}", exc_info=True)
        return None


async def generate_markdown_content(
    documentation: Dict[str, Any],
    language: str,  # Though not used directly in this function, it might be useful for future extensions
    file_path: str,
    relative_path: str  # Though not used directly in this function, it's part of the original signature
) -> str:
    """Generates enhanced markdown content with metrics analysis."""

    badges = generate_all_badges(documentation.get("metrics", {}))
    summary = generate_summary(documentation)
    functions = format_functions(documentation.get("functions", []))
    classes = format_classes(documentation.get("classes", []))

    metrics_summary = documentation.get("metrics_summary", {})
    problematic_files = documentation.get("problematic_files", [])

    metrics_content = f"""
## Metrics Analysis

### Overall Metrics
- Maintainability Index: {metrics_summary.get('maintainability_index', 0):.2f}
- Average Complexity: {metrics_summary.get('average_complexity', 0):.2f}
- Total Files Analyzed: {metrics_summary.get('processed_files', 0)}
- Success Rate: {metrics_summary.get('success_rate', 0):.1f}%

### Warnings and Issues
- Error Count: {metrics_summary.get('error_count', 0)}
- Warning Count: {metrics_summary.get('warning_count', 0)}

### Problematic Files
"""

    if problematic_files:
        metrics_content += "\n".join(
            f"- {file['file_path']}: {', '.join(issue['type'] for issue in file['issues'])}"
            for file in problematic_files
        )
    else:
        metrics_content += "No problematic files found."

    content = f"""
# Documentation for {os.path.basename(file_path)}

{badges}

{summary}

{metrics_content}

## Functions
{functions}

## Classes
{classes}
    """.strip()

    toc = generate_table_of_contents(content)
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
