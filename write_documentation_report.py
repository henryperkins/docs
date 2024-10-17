import aiofiles
import re
import json
import os
import textwrap
import logging
import sys
from typing import Optional, Dict, Any, List
from utils import logger  # Ensure logger is properly configured elsewhere

# Configure logging if not already configured
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def get_threshold(metric: str, key: str, default: int) -> int:
    try:
        return int(os.getenv(f"{metric.upper()}_{key.upper()}_THRESHOLD", default))
    except ValueError:
        logger.error(f"Invalid environment variable for {metric.upper()}_{key.upper()}_THRESHOLD")
        return default


def format_table(headers: List[str], rows: List[List[str]]) -> str:
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

        volume_badge = f'![Volume](https://img.shields.io/badge/Halstead%20Volume-{volume}-{volume_color}.svg?style=flat)'
        difficulty_badge = f'![Difficulty](https://img.shields.io/badge/Halstead%20Difficulty-{difficulty}-{difficulty_color}.svg?style=flat)'
        effort_badge = f'![Effort](https://img.shields.io/badge/Halstead%20Effort-{effort}-{effort_color}.svg?style=flat)'

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
            # Replace non-alphanumerics except spaces and dashes
            anchor = re.sub(r'[^a-zA-Z0-9\s-]', '', title)
            anchor = re.sub(r'\s+', '-', anchor).lower()
            anchor = re.sub(r'-+', '-', anchor).strip('-')
            toc.append(f"{'  ' * (level - 1)}- [{title}](#{anchor})")
    return "\n".join(toc)


def format_halstead_metrics(halstead: Dict[str, Any]) -> str:
    if not halstead:
        return ''
    volume = halstead.get('volume', 0)
    difficulty = halstead.get('difficulty', 0)
    effort = halstead.get('effort', 0)

    # Define thresholds
    volume_low, volume_medium = 100, 500
    difficulty_low, difficulty_medium = 10, 20
    effort_low, effort_medium = 500, 1000

    volume_color = "green" if volume < volume_low else "yellow" if volume < volume_medium else "red"
    difficulty_color = "green" if difficulty < difficulty_low else "yellow" if difficulty < difficulty_medium else "red"
    effort_color = "green" if effort < effort_low else "yellow" if effort < effort_medium else "red"

    metrics = f'![Halstead Volume](https://img.shields.io/badge/Halstead%20Volume-{volume}-{volume_color}.svg?style=flat)\n'
    metrics += f'![Halstead Difficulty](https://img.shields.io/badge/Halstead%20Difficulty-{difficulty}-{difficulty_color}.svg?style=flat)\n'
    metrics += f'![Halstead Effort](https://img.shields.io/badge/Halstead%20Effort-{effort}-{effort_color}.svg?style=flat)\n'
    return metrics


def format_maintainability_index(mi_score: float) -> str:
    if mi_score is None:
        return ''
    return f'![Maintainability Index](https://img.shields.io/badge/Maintainability%20Index-{mi_score:.2f}-brightgreen.svg?style=flat)\n'


def format_functions(functions: List[Dict[str, Any]]) -> str:
    headers = ["Function Name", "Complexity", "Async", "Docstring"]
    rows = [
        [
            func.get("name", "N/A"),
            str(func.get("complexity", 0)),
            str(func.get("async", False)),
            sanitize_text(func.get("docstring", ""))
        ]
        for func in functions
    ]
    return format_table(headers, rows)


def format_methods(methods: List[Dict[str, Any]]) -> str:
    headers = ["Method Name", "Type", "Async", "Docstring"]
    rows = [
        [
            method.get("name", "N/A"),
            method.get("type", "instance"),
            str(method.get("async", False)),
            sanitize_text(method.get("docstring", ""))
        ]
        for method in methods
    ]
    return format_table(headers, rows)


def format_classes(classes: List[Dict[str, Any]]) -> str:
    headers = ["Class Name", "Docstring", "Methods"]
    rows = [
        [
            cls.get("name", "N/A"),
            sanitize_text(cls.get("docstring", "")),
            format_methods(cls.get("methods", []))
        ]
        for cls in classes
    ]
    return format_table(headers, rows)


def format_variables_and_constants(variables: List[Dict[str, Any]], constants: List[Dict[str, Any]]) -> str:
    """
    Formats variables and constants into markdown tables with detailed information.

    Args:
        variables (List[Dict[str, Any]]): A list of variable dictionaries.
        constants (List[Dict[str, Any]]): A list of constant dictionaries.

    Returns:
        str: Markdown-formatted tables for variables and constants.
    """
    headers = ["Name", "Type", "Data Type", "Description", "Defined At", "Usage Example", "References"]
    rows = []

    logger.debug(f"Formatting {len(variables)} variables and {len(constants)} constants.")

    # Define required keys
    required_keys = ["name", "type", "description", "file", "line", "link", "example", "references"]

    # Process Variables
    for var in variables:
        missing_keys = [key for key in required_keys if key not in var]
        if missing_keys:
            logger.warning(f"Variable '{var.get('name', 'N/A')}' is missing keys: {missing_keys}")
        row = [
            var.get("name", "N/A"),
            "Variable",
            f"`{var.get('type', 'Unknown')}`",
            sanitize_text(var.get("description", "No description provided.")),
            f"[{var.get('file', 'N/A')}:{var.get('line', 'N/A')}]({var.get('link', '#')})",
            sanitize_text(var.get("example", "No example provided.")),
            sanitize_text(var.get("references", "N/A"))
        ]
        rows.append(row)

    # Process Constants
    for const in constants:
        missing_keys = [key for key in required_keys if key not in const]
        if missing_keys:
            logger.warning(f"Constant '{const.get('name', 'N/A')}' is missing keys: {missing_keys}")
        row = [
            const.get("name", "N/A"),
            "Constant",
            f"`{const.get('type', 'Unknown')}`",
            sanitize_text(const.get("description", "No description provided.")),
            f"[{const.get('file', 'N/A')}:{const.get('line', 'N/A')}]({const.get('link', '#')})",
            sanitize_text(const.get("example", "No example provided.")),
            sanitize_text(const.get("references", "N/A"))
        ]
        rows.append(row)

    table = format_table(headers, rows)
    logger.debug("Completed formatting variables and constants tables.")
    return table


def generate_summary(variables: List[Dict[str, Any]], constants: List[Dict[str, Any]]) -> str:
    """Generate a summary section for variables and constants."""
    total_vars = len(variables)
    total_consts = len(constants)
    summary = f"### **Summary**\n\n- **Total Variables:** {total_vars}\n- **Total Constants:** {total_consts}\n"
    return summary


def sanitize_filename(filename: str) -> str:
    """Sanitize filenames by replacing invalid characters."""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def generate_documentation_prompt(
    file_name: str,
    code_structure: Dict[str, Any],
    project_info: str,
    style_guidelines: str,
    language: str,
    function_schema: Dict[str, Any]
) -> str:
    functions = function_schema.get("functions", [])
    if functions and "parameters" in functions[0]:
        schema = json.dumps(function_schema["functions"][0]["parameters"], indent=2)
    else:
        logger.error("Function schema is missing or empty.")
        schema = "{}"  # Fallback or handle accordingly

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
    {schema}
    
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
    return textwrap.dedent(prompt).strip()


async def write_documentation_report(
    documentation: Optional[dict],
    language: str,
    file_path: str,
    repo_root: str,
    output_dir: str
) -> str:
    """
    Writes the documentation report to a markdown file with enhanced variable and constant information.

    Args:
        documentation (Optional[dict]): The documentation data.
        language (str): Programming language of the source code.
        file_path (str): Path to the source code file.
        repo_root (str): Root directory of the repository.
        output_dir (str): Directory where the documentation will be saved.

    Returns:
        str: The generated documentation content.
    """
    try:
        if not documentation:
            logger.warning(f"No documentation to write for '{file_path}'")
            return ''

        relative_path = os.path.relpath(file_path, repo_root)
        safe_file_name = sanitize_filename(relative_path.replace(os.sep, '_'))
        doc_file_path = os.path.join(output_dir, f"{safe_file_name}.md")

        file_header = f'# File: {relative_path}\n\n'
        documentation_content = file_header

        # Add Halstead metrics and Maintainability Index
        halstead_content = format_halstead_metrics(documentation.get('halstead', {}))
        mi_content = format_maintainability_index(documentation.get('maintainability_index'))
        documentation_content += halstead_content + mi_content + "\n"

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

        # Add Classes
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

        # Add Variables and Constants
        variables = documentation.get('variables', [])
        constants = documentation.get('constants', [])
        if variables or constants:
            documentation_content += "## Variables and Constants\n\n"
            documentation_content += format_variables_and_constants(variables, constants)
            documentation_content += "\n"

        # Add Source Code
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            source_code = await file.read()
        documentation_content += f"## Source Code\n\n```{language}\n{source_code}\n```\n"

        # Generate Table of Contents
        toc = generate_table_of_contents(documentation_content)
        documentation_content = "# Table of Contents\n\n" + toc + "\n\n" + documentation_content

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Write to markdown file
        async with aiofiles.open(doc_file_path, 'w', encoding='utf-8') as f:
            await f.write(documentation_content)
        logger.info(f"Documentation written to '{doc_file_path}' successfully.")
        return documentation_content

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return ''
    except subprocess.SubprocessError as e:
        logger.error(f"Subprocess error during flake8 execution: {e}")
        return ''
    except Exception as e:
        logger.error(f"Unexpected error: {e} for file {file_path}", exc_info=True)
        return ''
