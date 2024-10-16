import aiofiles
import re
import json
import os
from typing import Optional
from utils import logger


def get_threshold(metric: str, key: str, default: int) -> int:
    try:
        return int(os.getenv(f"{metric.upper()}_{key.upper()}_THRESHOLD", default))
    except ValueError:
        logger.error(f"Invalid environment variable for {metric.upper()}_{key.upper()}_THRESHOLD")
        return default

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
        complexity_badge = f'![Complexity: {complexity}](https://img.shields.io/badge/Complexity-{complexity}-{color}?style=flat-square)'
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

        volume_badge = f'![Halstead Volume: {volume}](https://img.shields.io/badge/Volume-{volume}-{volume_color}?style=flat-square)'
        difficulty_badge = f'![Halstead Difficulty: {difficulty}](https://img.shields.io/badge/Difficulty-{difficulty}-{difficulty_color}?style=flat-square)'
        effort_badge = f'![Halstead Effort: {effort}](https://img.shields.io/badge/Effort-{effort}-{effort_color}?style=flat-square)'

        badges.extend([volume_badge, difficulty_badge, effort_badge])

    if mi is not None:
        high_threshold = get_threshold('maintainability_index', 'high', 80)
        medium_threshold = get_threshold('maintainability_index', 'medium', 50)
        color = "green" if mi > high_threshold else "yellow" if mi > medium_threshold else "red"
        mi_badge = f'![Maintainability Index: {mi}](https://img.shields.io/badge/Maintainability-{mi}-{color}?style=flat-square)'
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
            toc.append(f"{'  ' * (level - 1)}- [{title}](#{anchor})")
    return "\n".join(toc)

async def write_documentation_report(
    documentation: Optional[dict],
    language: str,
    file_path: str,
    repo_root: str,
    new_content: str,
    output_dir: str  # Added output_dir parameter
) -> str:
    try:
        relative_path = os.path.relpath(file_path, repo_root)
        file_header = f'# File: {relative_path}\n\n'
        documentation_content = file_header

        summary = documentation.get('summary', '') if documentation else ''
        summary = sanitize_text(summary)
        if summary:
            summary_section = f'## Summary\n\n{summary}\n'
            documentation_content += summary_section

        changes = documentation.get('changes_made', []) if documentation else []
        changes = [sanitize_text(change) for change in changes if change.strip()]
        if changes:
            changes_formatted = '\n'.join((f'- {change}' for change in changes))
            changes_section = f'## Changes Made\n\n{changes_formatted}\n'
            documentation_content += changes_section

        halstead = documentation.get('halstead') if documentation else {}
        mi = documentation.get('maintainability_index') if documentation else None
        complexity = max(
            [
                *(func.get('complexity', 0) for func in documentation.get('functions', [])),
                *(method.get('complexity', 0) for cls in documentation.get('classes', []) for method in cls.get('methods', []))
            ],
            default=0
        )
        overall_badges = generate_all_badges(complexity, halstead, mi)
        if overall_badges:
            documentation_content += f"{overall_badges}\n\n"

        functions = documentation.get('functions', []) if documentation else []
        if functions:
            functions_section = '## Functions\n\n'
            functions_section += '| Function | Arguments | Description | Async | Complexity |\n'
            functions_section += '|----------|-----------|-------------|-------|------------|\n'
            for func in functions:
                func_name = func.get('name', 'N/A')
                func_args = ', '.join(func.get('args', []))
                func_doc = sanitize_text(func.get('docstring', ''))
                first_line_doc = truncate_description(func_doc)
                func_async = 'Yes' if func.get('async', False) else 'No'
                func_complexity = func.get('complexity', 0)
                complexity_badge = generate_all_badges(func_complexity, {}, 0) if func_complexity > 0 else ''
                functions_section += f'| `{func_name}` | `{func_args}` | {first_line_doc} | {func_async} | {complexity_badge} |\n'
            documentation_content += functions_section + '\n'

        classes = documentation.get('classes', []) if documentation else []
        if classes:
            classes_section = '## Classes\n\n'
            for cls in classes:
                cls_name = cls.get('name', 'N/A')
                cls_doc = sanitize_text(cls.get('docstring', 'No description provided.'))
                classes_section += f'### Class: `{cls_name}`\n\n{cls_doc}\n\n'

                methods = cls.get('methods', [])
                if methods:
                    classes_section += '| Method | Arguments | Description | Async | Type | Complexity |\n'
                    classes_section += '|--------|-----------|-------------|-------|------|------------|\n'
                    for method in methods:
                        method_name = method.get('name', 'N/A')
                        method_args = ', '.join(method.get('args', []))
                        method_doc = sanitize_text(method.get('docstring', ''))
                        first_line_method_doc = truncate_description(method_doc)
                        method_async = 'Yes' if method.get('async', False) else 'No'
                        method_type = method.get('type', 'N/A')
                        method_complexity = method.get('complexity', 0)
                        complexity_badge = generate_all_badges(method_complexity, {}, 0) if method_complexity > 0 else ''
                        classes_section += (
                            f'| `{method_name}` | `{method_args}` | {first_line_method_doc} | '
                            f'{method_async} | {method_type} | {complexity_badge} |\n'
                        )
                    classes_section += '\n'
            documentation_content += classes_section

        code_content = new_content.strip()
        code_block = f'```{language}\n{code_content}\n```\n\n---\n'
        documentation_content += code_block

        toc = generate_table_of_contents(documentation_content)
        documentation_content = toc + "\n\n" + documentation_content

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save documentation to a separate .md file
        doc_file_path = os.path.join(output_dir, f"{os.path.basename(file_path)}.md")

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
        logger.error(f"Unexpected error: {e} for file {file_path}")
        return ''

def generate_documentation_prompt(
    file_name: str,
    code_structure: dict,
    project_info: Optional[str],
    style_guidelines: Optional[str],
    language: str,
) -> str:
    prompt = (
        "You are an expert software engineer tasked with generating comprehensive documentation for the following code structure."
    )
    if project_info:
        prompt += f"\n\n**Project Information:**\n{project_info}"
    if style_guidelines:
        prompt += f"\n\n**Style Guidelines:**\n{style_guidelines}"
    prompt += f"\n\n**File Name:** {file_name}"
    prompt += f"\n**Language:** {language}"
    prompt += f"\n\n**Code Structure:**\n{json.dumps(code_structure, indent=2)}"
    prompt += """

**Instructions:**

Using the code structure provided, generate detailed documentation in JSON format that matches the following schema:

{
  "summary": "A detailed and comprehensive summary of the file, covering its purpose, functionality, and any important details.",
  "functions": [
    {
      "name": "Function name",
      "docstring": "Detailed description of the function, including its purpose and any important details.",
      "args": ["List of argument names"],
      "async": true,
      "complexity": 10
    }
    // Repeat for each function
  ],
  "classes": [
    {
      "name": "Class name",
      "docstring": "Detailed description of the class.",
      "methods": [
        {
          "name": "Method name",
          "docstring": "Detailed description of the method.",
          "args": ["List of argument names"],
          "async": false,
          "type": "Method type (e.g., 'instance', 'class', 'static')",
          "complexity": 5
        }
        // Repeat for each method
      ]
    }
    // Repeat for each class
  ],
  "halstead_metrics": {
    "volume": 0.0,
    "difficulty": 0.0,
    "effort": 0.0
  },
  "maintainability_index": 0.0
}
"""
    return prompt