# utils.py

import os
import sys
import json
import logging
import aiohttp
import asyncio
import re
from dotenv import load_dotenv
from typing import Set, List, Optional, Dict

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_EXCLUDED_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', '.idea'} # Added .venv and .idea
DEFAULT_EXCLUDED_FILES = {'.DS_Store'}
DEFAULT_SKIP_TYPES = {'.json', '.md', '.txt', '.csv', '.lock'} # Added .lock files

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

    # First, try to extract JSON using the function calling format
    try:
        response_json = json.loads(response)
        if "function_call" in response_json and "arguments" in response_json["function_call"]:
            return response_json["function_call"]["arguments"]
    except json.JSONDecodeError:
        pass  # Fallback to other extraction methods


    # Try to find JSON enclosed in triple backticks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # As a last resort, attempt to use the entire response if it's valid JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None


function_schema = {  # Updated and complete schema
    "name": "generate_documentation",
    "description": "Generates documentation, summaries, and lists of changes for code structures.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "A concise summary of the changes."},
            "changes": {
                "type": "array",
                "description": "A list of specific changes.",
                "items": {"type": "string"},
            },
            "functions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "docstring": {"type": "string"},
                        "args": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "type": {"type": "string"}}, "required": ["name", "type"]}},
                        "returns": {"type": "object", "properties": {"type": {"type": "string"}}, "required": ["type"]},
                        "decorators": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "docstring", "args", "returns", "decorators"],
                },
            },
            "classes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "docstring": {"type": "string"},
                        "bases": {"type": "array", "items": {"type": "string"}},
                        "decorators": {"type": "array", "items": {"type": "string"}},
                        "methods": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "docstring": {"type": "string"},
                                    "args": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "type": {"type": "string"}}, "required": ["name", "type"]}},
                                    "returns": {"type": "object", "properties": {"type": {"type": "string"}}, "required": ["type"]},
                                    "decorators": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["name", "docstring", "args", "returns", "decorators"],
                            },
                        },
                    },
                    "required": ["name", "docstring", "bases", "decorators", "methods"],
                },
            },
            "elements": { # HTML structure
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "attributes": {"type": "object"},
                        "text": {"type": "string"},
                        "docstring": {"type": "string"},
                    },
                    "required": ["tag", "attributes", "text", "docstring"],
                },
            },
            "rules": {  # CSS structure
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string"},
                        "declarations": {"type": "string"},
                        "docstring": {"type": "string"},
                    },
                    "required": ["selector", "declarations", "docstring"],
                },
            },
        },
        "required": ["summary", "changes", "functions", "classes", "elements", "rules"], # Updated required fields
    },
}


async def fetch_documentation(session: aiohttp.ClientSession, prompt: str, semaphore: asyncio.Semaphore, model_name: str, function_schema: dict, retry: int = 3) -> Optional[dict]:
    """Fetches generated documentation, summaries, and change lists from the OpenAI API using function calling.

    Args:
        session: The aiohttp client session.
        prompt: The prompt to send to the API.
        semaphore: Semaphore to control concurrency.
        model_name: The name of the OpenAI model to use.
        function_schema: The JSON schema for the expected function call output.
        retry: The number of retry attempts.

    Returns:
        A dictionary containing the generated documentation, or None if the request fails.
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set. Please set it in your environment or .env file.")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "functions": [function_schema],
        "function_call": {"name": "generate_documentation"},
        "max_tokens": 3000,  # Increased token limit for additional data
        "temperature": 0.2
    }

    for attempt in range(1, retry + 1):
        try:
            async with semaphore:
                async with session.post(
                    OPENAI_API_URL, headers=headers, json=payload, timeout=120
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        try:
                            # Attempt to extract JSON using function calling format first
                            message = data["choices"][0]["message"]
                            if "function_call" in message:
                                arguments = message["function_call"]["arguments"]
                                documentation = json.loads(arguments)
                                logger.info("Generated documentation using function call.")
                            else:  # Fallback to direct extraction if not a function call
                                content = message["content"]
                                documentation = json.loads(extract_json_from_response(content))
                                logger.info("Generated documentation from raw content.")

                            return documentation

                        except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
                            logger.error(f"Error parsing JSON response: {e}")
                            logger.error(f"Response content: {data}")  # Log the full response for debugging
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
                        logger.error(
                            f"API request failed with status {response.status}: {error_text}"
                        )
                        return None

        except asyncio.TimeoutError:
            logger.error(
                f"Request timed out during attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds."
            )
            await asyncio.sleep(2 ** attempt)
        except aiohttp.ClientError as e:
            logger.error(
                f"Client error during API request: {e}. Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds."
            )
            await asyncio.sleep(2 ** attempt)

    logger.error("Failed to generate documentation after multiple attempts.")
    return None
    
async def fetch_summary(
    session: aiohttp.ClientSession, 
    prompt: str, 
    semaphore: asyncio.Semaphore, 
    model_name: str, 
    retry: int = 3
) -> Optional[str]:
    """
    Fetches a summary from the OpenAI API.

    Args:
        session (aiohttp.ClientSession): The session to use for making the API request.
        prompt (str): The prompt to send to the API.
        semaphore (asyncio.Semaphore): A semaphore to limit the number of concurrent API requests.
        model_name (str): The model to use for the OpenAI request (e.g., 'gpt-4').
        retry (int, optional): Number of retry attempts for failed requests. Defaults to 3.

    Returns:
        Optional[str]: The summary text if successful, otherwise None.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,  # Adjust depending on the length of the summary
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

                        # Ensure the response contains 'choices' and it's well-formed
                        choices = data.get('choices', [])
                        if choices and 'message' in choices[0]:
                            summary = choices[0]['message']['content'].strip()
                            return summary
                        else:
                            logger.error(f"Unexpected API response structure: {data}")
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
                        logger.error(f"Unhandled API request failure with status {response.status}: {error_text}")
                        return None

        except asyncio.TimeoutError:
            logger.error(f"Request timed out during attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds.")
            await asyncio.sleep(2 ** attempt)

        except aiohttp.ClientError as e:
            logger.error(f"Client error during API request: {e}. Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds.")
            await asyncio.sleep(2 ** attempt)
    
    logger.error("Failed to generate summary after multiple attempts.")
    return None


def generate_documentation_prompt(
    code_structure: dict,
    project_info: str = None,
    style_guidelines: str = None
) -> str:
    """Generates a prompt for GPT-4o-mini to create documentation, summaries, and change lists."""

    language = code_structure.get("language", "code")
    json_structure = json.dumps(code_structure, indent=2)

    prompt_parts = [
        f"You are an expert {language} developer and technical writer."
    ]

    if project_info:
        prompt_parts.append(f"The code belongs to a project that {project_info}.")

    if style_guidelines:
        prompt_parts.append(f"Please follow these documentation style guidelines: {style_guidelines}")

    # Core instruction for documentation generation
    prompt_parts.append(
        f"""
Given the following {language} code structure in JSON format, perform the following tasks:

1. Generate detailed docstrings or comments for each function, method, and class. Include descriptions of parameters, return types, and relevant details. Enhance existing documentation where applicable.
2. Provide a concise summary of the changes or additions you made to the code documentation.
3. List the specific changes made to the code (e.g., added docstrings, updated class descriptions).

Instructions:
- Output the results by invoking the `generate_documentation` function.
- The output must conform to the provided JSON schema.
- Respond only with the function call output, without additional text.
"""
    )

    # Special instruction for CSS files
    if language == 'css':
        prompt_parts.append("For CSS files, place docstrings *above* the selector they apply to.")

    # Append the schema for guidance
    prompt_parts.append(f"Schema:\n{json.dumps(function_schema, indent=2)}")

    # Combine the prompt parts
    prompt = '\n'.join(prompt_parts)

    return prompt
