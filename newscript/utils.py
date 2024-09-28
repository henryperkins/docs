import os
import sys
import json
import logging
import aiohttp
import asyncio
from dotenv import load_dotenv
from typing import Set, List, Optional

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Constants
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_EXCLUDED_DIRS = {'.git', '__pycache__', 'node_modules'}
DEFAULT_EXCLUDED_FILES = {'.DS_Store'}
DEFAULT_SKIP_TYPES = {'.json', '.md', '.txt', '.csv'}
LANGUAGE_MAPPING = {
    '.py': 'python',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.html': 'html',
    '.htm': 'html',
    '.css': 'css',
}

function_schema = {
    "name": "generate_documentation",
    "description": "Generates documentation for code structures.",
    "parameters": {
        "type": "object",
        "properties": {
            "functions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "docstring": {"type": "string"},
                        "args": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string"}
                                },
                                "required": ["name", "type"]
                            }
                        },
                        "returns": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"}
                            },
                            "required": ["type"]
                        },
                        "decorators": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["name", "docstring", "args", "returns", "decorators"]
                }
            },
            "classes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "docstring": {"type": "string"},
                        "bases": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "decorators": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "methods": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "docstring": {"type": "string"},
                                    "args": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "type": {"type": "string"}
                                            },
                                            "required": ["name", "type"]
                                        }
                                    },
                                    "returns": {
                                        "type": "object",
                                        "properties": {
                                            "type": {"type": "string"}
                                        },
                                        "required": ["type"]
                                    },
                                    "decorators": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["name", "docstring", "args", "returns", "decorators"]
                            }
                        }
                    },
                    "required": ["name", "docstring", "bases", "decorators", "methods"]
                }
            }
        },
        "required": ["functions", "classes"]
    }
}

def get_language(ext: str) -> str:
    """Determines the programming language based on file extension."""
    return LANGUAGE_MAPPING.get(ext.lower(), 'plaintext')

def is_binary(file_path: str) -> bool:
    """Checks if a file is binary."""
    try:
        with open(file_path, 'rb') as file:
            return b'\0' in file.read(1024)
    except Exception as e:
        logger.error(f"Error checking binary file '{file_path}': {e}")
        return True

def load_config(config_path: str, excluded_dirs: Set[str], excluded_files: Set[str], skip_types: Set[str]) -> None:
    """Loads configuration from a JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
            for key, default_set in [
                ('excluded_dirs', excluded_dirs),
                ('excluded_files', excluded_files),
                ('skip_types', skip_types),
            ]:
                items = config.get(key, [])
                if isinstance(items, list):
                    default_set.update(items)
                else:
                    logger.error(f"'{key}' must be a list in '{config_path}'.")
            logger.info(f"Loaded config from '{config_path}'.")
    except FileNotFoundError:
        logger.warning(f"Config file '{config_path}' not found. Using defaults.")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing config file '{config_path}': {e}")
    except Exception as e:
        logger.error(f"Error loading config file '{config_path}': {e}")

def get_all_file_paths(repo_path: str, excluded_dirs: Set[str], excluded_files: Set[str]) -> List[str]:
    """Gets all file paths in a repository, excluding specified directories and files."""
    file_paths = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('.')]
        file_paths.extend([os.path.join(root, file) for file in files if file not in excluded_files and not file.startswith('.')])
    logger.info(f"Collected {len(file_paths)} files from '{repo_path}'.")
    return file_paths

def is_valid_extension(ext: str, skip_types: Set[str]) -> bool:
    """Checks if a file extension is valid (not in the skip list)."""
    return ext.lower() not in skip_types

def extract_json_from_response(response: str) -> Optional[str]:
    """Extracts JSON content from the model's response."""
    # Try to find JSON enclosed in triple backticks with 'json' language identifier
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # Try to find any JSON enclosed in triple backticks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # As a last resort, attempt to use the entire response
    if response.strip().startswith('{') and response.strip().endswith('}'):
        return response.strip()

    return None
async def fetch_documentation(session: aiohttp.ClientSession, prompt: str, semaphore: asyncio.Semaphore, model_name: str, retry: int = 3) -> Optional[str]:
    """Fetches generated documentation from the OpenAI API."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.2
    }
    for attempt in range(1, retry + 1):
        try:
            async with semaphore:
                async with session.post(
                    OPENAI_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=120
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'choices' in data and len(data['choices']) > 0 and 'message' in data['choices'][0]:
                            documentation = data['choices'][0]['message']['content'].strip()
                            logger.info("Generated documentation.")
                            return documentation
                        else:
                            logger.error("Unexpected API response structure.")
                            return None
                    elif response.status in {429, 500, 502, 503, 504}:
                        error_text = await response.text()
                        logger.warning(
                            f"API rate limit or server error (status {response.status}). "
                            f"Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds. "
                            f"Response: {error_text}"
                        )
                        await asyncio.sleep(2 ** attempt)
                    else:
                        error_text = await response.text()
                        logger.error(f"API request failed with status {response.status}: {error_text}")
                        return None
        except asyncio.TimeoutError:
            logger.error(
                f"Request timed out during attempt {attempt}/{retry}. "
                f"Retrying in {2 ** attempt} seconds."
            )
            await asyncio.sleep(2 ** attempt)
        except aiohttp.ClientError as e:
            logger.error(
                f"Client error during API request: {e}. "
                f"Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds."
            )
            await asyncio.sleep(2 ** attempt)
    logger.error("Failed to generate documentation after multiple attempts.")
    return None

def generate_documentation_prompt(
    code_structure: dict,
    project_info: str = None,
    style_guidelines: str = None
) -> str:
    """Generates a prompt for the OpenAI API to create documentation."""
    language = code_structure.get('language', 'code')
    json_structure = json.dumps(code_structure, indent=2)

    prompt_parts = [
        f"You are an expert {language} developer and technical writer.",
    ]

    if project_info:
        prompt_parts.append(
            f"The code belongs to a project that {project_info}."
        )

    if style_guidelines:
        prompt_parts.append(
            f"Please follow these documentation style guidelines: {style_guidelines}"
        )

    prompt_parts.append(
        f"""
Given the following {language} code structure in JSON format, generate detailed docstrings or comments for each function, method, class, element, or rule. Include descriptions of all parameters, return types, and any relevant details. Preserve and enhance existing documentation where applicable.

**Important Instructions**:
- Provide the updated code structure with the 'docstring' fields filled in.
- Output **only** the JSON, without any additional text or explanations.
- Enclose the entire JSON output within triple backticks like this: ```json ... ```
- Ensure the JSON is valid and properly formatted.

Code Structure:
{json_structure}
"""
    )

    prompt = '\n'.join(prompt_parts)
    return prompt

