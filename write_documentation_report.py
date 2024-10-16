import aiofiles
import re
import json
import os
from typing import Optional, Dict, Any
from utils import logger

def get_threshold(metric: str, key: str, default: int) -> int:
    try:
        return int(os.getenv(f"{metric.upper()}_{key.upper()}_THRESHOLD", default))
    except ValueError:
        logger.error(f"Invalid environment variable for {metric.upper()}_{key.upper()}_THRESHOLD")
        return default
        
def format_table(headers: list, rows: list) -> str:
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        table += "| " + " | ".join(row) + " |\n"
    return table
    
def generate_all_badges(
    complexity: Optional[int] = None,
    halstead: Optional[dict] = None,
    mi: Optional[float] = None
) -> str:
    badges = []

    if complexity is not None:
        low_threshold = get_threshold('complexity', 'low', 10)
        medium_threshold = get_threshold('complexity', 'medium', 20)
        color = "green" if complexity < low_threshold else "yellow" if complexity < medium_threshold else "red"
        complexity_badge = f'![Complexity](https://img.shields.io/badge/Complexity-{complexity}-{color}.svg?style=flat)'
        badges.append(complexity_badge)

    if halstead:
        volume = halstead.get('volume', 0)
        difficulty = halstead.get('difficulty', 0)
        effort = halstead.get('effort', 0)

        volume_low = get_threshold('halstead_volume', 'low', 100)
        volume_medium = get_threshold('halstead_volume', 'medium', 500)
        volume_color = "green" if volume < volume_low else "yellow" if volume < volume_medium else "red"

        difficulty_low = get_threshold('halstead_difficulty', 'low', 10)
        difficulty_medium = get_threshold('halstead_difficulty', 'medium', 20)
        difficulty_color = "green" if difficulty < difficulty_low else "yellow" if difficulty < difficulty_medium else "red"

        effort_low = get_threshold('halstead_effort', 'low', 500)
        effort_medium = get_threshold('halstead_effort', 'medium', 1000)
        effort_color = "green" if effort < effort_low else "yellow" if effort < effort_medium else "red"

        volume_badge = f'![Volume](https://img.shields.io/badge/Volume-{volume}-{volume_color}.svg?style=flat)'
        difficulty_badge = f'![Difficulty](https://img.shields.io/badge/Difficulty-{difficulty}-{difficulty_color}.svg?style=flat)'
        effort_badge = f'![Effort](https://img.shields.io/badge/Effort-{effort}-{effort_color}.svg?style=flat)'

        badges.extend([volume_badge, difficulty_badge, effort_badge])

    if mi is not None:
        high_threshold = get_threshold('maintainability_index', 'high', 80)
        medium_threshold = get_threshold('maintainability_index', 'medium', 50)
        color = "green" if mi > high_threshold else "yellow" if mi > medium_threshold else "red"
        mi_badge = f'![Maintainability](https://img.shields.io/badge/Maintainability-{mi:.2f}-{color}.svg?style=flat)'
        badges.append(mi_badge)

    return ' '.join(badges).strip()

def truncate_description(description: str, max_length: int = 100) -> str:
    return (description[:max_length] + '...') if len(description) > max_length else description

def sanitize_text(text: str) -> str:
    markdown_special_chars = ['*', '_', '`', '~', '<', '>', '#']
    for char in markdown_special_chars:
        text = text.replace(char, f"\\{char}")
    return text.replace('|', '\\|').replace('\n', ' ').strip()

def generate_table_of_contents(content: str) -> str:
    toc = []
    for line in content.splitlines():
        if line.startswith("#"):
            level = line.count("#")
            title = line.lstrip("#").strip()
            anchor = re.sub(r'[^a-zA-Z0-9\s]', '', title).replace(' ', '-').lower()
            anchor = anchor.replace('--', '-').strip('-')
            toc.append(f"{'  ' * (level - 1)}- [{title}](#{anchor})")
    return "\n".join(toc)

def format_halstead_metrics(halstead: Dict[str, Any]) -> str:
    if not halstead:
        return ''
    volume = halstead.get('volume', 0)
    difficulty = halstead.get('difficulty', 0)
    effort = halstead.get('effort', 0)
    metrics = f'![Halstead Volume](https://img.shields.io/badge/Halstead%20Volume-{volume}-blue)\n'
    metrics += f'![Halstead Difficulty](https://img.shields.io/badge/Halstead%20Difficulty-{difficulty}-blue)\n'
    metrics += f'![Halstead Effort](https://img.shields.io/badge/Halstead%20Effort-{effort}-blue)\n'
    return metrics

def format_maintainability_index(mi_score: float) -> str:
    if mi_score is None:
        return ''
    return f'![Maintainability Index](https://img.shields.io/badge/Maintainability%20Index-{mi_score:.2f}-brightgreen)\n'

def format_functions(functions: list) -> str:
    content = ''
    for func in functions:
        name = func.get('name', '')
        docstring = func.get('docstring', '')
        args = func.get('args', [])
        is_async = func.get('async', False)
        async_str = 'async ' if is_async else ''
        arg_list = ', '.join(args)
        content += f'#### Function: `{async_str}{name}({arg_list})`\n\n'
        content += f'{docstring}\n\n'
    return content

def format_methods(methods: list) -> str:
    content = ''
    for method in methods:
        name = method.get('name', '')
        docstring = method.get('docstring', '')
        args = method.get('args', [])
        is_async = method.get('async', False)
        method_type = method.get('type', 'instance')
        async_str = 'async ' if is_async else ''
        arg_list = ', '.join(args)
        content += f'- **Method**: `{async_str}{name}({arg_list})` ({method_type} method)\n\n'
        content += f'  {docstring}\n\n'
    return content

def format_classes(classes: list) -> str:
    content = ''
    for cls in classes:
        name = cls.get('name', '')
        docstring = cls.get('docstring', '')
        methods = cls.get('methods', [])
        content += f"### Class: `{name}`\n\n"
        content += f"{docstring}\n\n"
        if methods:
            content += f"#### Methods:\n\n"
            content += format_methods(methods)
    return content

def format_variables(variables: list) -> str:
    if not variables:
        return ''
    content = "### Variables\n\n"
    for var in variables:
        content += f'- `{var}`\n'
    content += "\n"
    return content

def format_constants(constants: list) -> str:
    if not constants:
        return ''
    content = "### Constants\n\n"
    for const in constants:
        content += f'- `{const}`\n'
    content += "\n"
    return content

def generate_documentation_prompt(
    file_name: str,
    code_structure: Dict[str, Any],
    project_info: str,
    style_guidelines: str,
    language: str,
    function_schema: Dict[str, Any]
) -> str:
    prompt = f"""
You are a code documentation generator.

Project Info:
{project_info}

Style Guidelines:
{style_guidelines}

Given the following code structure of the {language} file '{file_name}', generate detailed documentation according to the specified schema.

Code Structure:
{json.dumps(code_structure, indent=2)}

Schema:
{json.dumps(function_schema["functions"][0]["parameters"], indent=2)}

Ensure that the output is a JSON object that follows the schema exactly, including all required fields.

Example Output:
{{
  "summary": "Brief summary of the file.",
  "changes_made": ["List of changes made to the file."],
  "functions": [
    {{
      "name": "function_name",
      "docstring": "Detailed description of the function.",
      "args": ["arg1", "arg2"],
      "async": false
    }}
  ],
  "classes": [
    {{
      "name": "ClassName",
      "docstring": "Detailed description of the class.",
      "methods": [
        {{
          "name": "method_name",
          "docstring": "Detailed description of the method.",
          "args": ["arg1"],
          "async": false,
          "type": "instance"
        }}
      ]
    }}
  ]
}}

Ensure all strings are properly escaped and the JSON is valid.

Output:"""
    return prompt

async def write_documentation_report(
    documentation: Optional[dict],
    language: str,
    file_path: str,
    repo_root: str,
    new_content: str,
    output_dir: str
) -> str:
    try:
        if not documentation:
            logger.warning(f"No documentation to write for '{file_path}'")
            return ''
    
        relative_path = os.path.relpath(file_path, repo_root)
        file_header = f'# File: {relative_path}\n\n'
        documentation_content = file_header

        # Add Halstead metrics and Maintainability Index
        halstead_content = format_halstead_metrics(documentation.get('halstead', {}))
        mi_content = format_maintainability_index(documentation.get('maintainability_index'))
        documentation_content += halstead_content + mi_content + "\n"

        # Add Summary
        summary = documentation.get('summary', '')
        if summary:
            documentation_content += f"## Summary\n\n{summary}\n\n"

        # Add Changes Made
        changes_made = documentation.get('changes_made', [])
        if changes_made:
            documentation_content += f"## Changes Made\n\n"
            for change in changes_made:
                documentation_content += f"- {change}\n"
            documentation_content += "\n"

        # Add Classes
        classes = documentation.get('classes', [])
        if classes:
            documentation_content += "## Classes\n\n"
            documentation_content += format_classes(classes)

        # Add Functions
        functions = documentation.get('functions', [])
        if functions:
            documentation_content += "## Functions\n\n"
            documentation_content += format_functions(functions)

        # Add Variables
        variables = documentation.get('variables', [])
        if variables:
            documentation_content += format_variables(variables)

        # Add Constants
        constants = documentation.get('constants', [])
        if constants:
            documentation_content += format_constants(constants)

        # Add Source Code
        with open(file_path, 'r') as file:
            source_code = file.read()
        documentation_content += f"## Source Code\n\n```{language}\n{source_code}\n```\n"

        # Generate Table of Contents
        toc = generate_table_of_contents(documentation_content)
        documentation_content = "# Table of Contents\n\n" + toc + "\n\n" + documentation_content

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Convert relative path to a safe file name
        safe_file_name = relative_path.replace(os.sep, '_')
        doc_file_path = os.path.join(output_dir, f"{safe_file_name}.md")

        async with aiofiles.open(doc_file_path, 'w', encoding='utf-8') as f:
            await f.write(documentation_content)
        logger.info(f"Documentation written to '{doc_file_path}' successfully.")
        return documentation_content

    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e} for file {file_path}")
        return ''
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return ''
    except Exception as e:
        logger.error(f"Unexpected error: {e} for file {file_path}", exc_info=True)
        return ''
