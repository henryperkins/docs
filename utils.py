# utils.py

import os
import sys
import json
import fnmatch
import black
import logging
import aiohttp
import asyncio
import re
from dotenv import load_dotenv
from typing import Set, List, Optional, Dict, Tuple  # Added Tuple

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_EXCLUDED_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', '.idea'}  # Added .venv and .idea
DEFAULT_EXCLUDED_FILES = {'.DS_Store'}
DEFAULT_SKIP_TYPES = {'.json', '.md', '.txt', '.csv', '.lock'}  # Added .lock files

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

def load_config(config_path: str, excluded_dirs: Set[str], excluded_files: Set[str], skip_types: Set[str]) -> Tuple[str, str]:
    """
    Loads additional configurations from a config.json file.
    
    Parameters:
        config_path (str): Path to the config.json file.
        excluded_dirs (Set[str]): Set to add excluded directories.
        excluded_files (Set[str]): Set to add excluded files.
        skip_types (Set[str]): Set to add skipped file extensions.
    
    Returns:
        Tuple[str, str]: Project information and style guidelines.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        project_info = config.get('project_info', '')
        style_guidelines = config.get('style_guidelines', '')
        excluded_dirs.update(config.get('excluded_dirs', []))
        excluded_files.update(config.get('excluded_files', []))
        skip_types.update(config.get('skip_types', []))
        logger.debug(f"Loaded configuration from '{config_path}'.")
        return project_info, style_guidelines
    except Exception as e:
        logger.error(f"Error loading config file '{config_path}': {e}")
        return '', ''

def get_all_file_paths(repo_path: str, excluded_dirs: Set[str], excluded_files: Set[str]) -> List[str]:
    """
    Recursively retrieves all file paths in the repository, excluding specified directories and files.
    
    Parameters:
        repo_path (str): Path to the repository.
        excluded_dirs (Set[str]): Directories to exclude.
        excluded_files (Set[str]): Files to exclude.
    
    Returns:
        List[str]: List of file paths.
    """
    file_paths = []
    for root, dirs, files in os.walk(repo_path):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for file in files:
            if any(fnmatch.fnmatch(file, pattern) for pattern in excluded_files):
                continue
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    logger.debug(f"Retrieved {len(file_paths)} files from '{repo_path}'.")
    return file_paths

def format_with_black(code: str) -> str:
    """
    Formats the given Python code using Black.
    
    Parameters:
        code (str): The Python code to format.
    
    Returns:
        str: The formatted Python code.
    """
    try:
        formatted_code = black.format_str(code, mode=black.FileMode())
        logger.debug("Successfully formatted code with Black.")
        return formatted_code
    except black.NothingChanged:
        logger.debug("No changes made by Black; code is already formatted.")
        return code
    except Exception as e:
        logger.error(f"Error formatting code with Black: {e}")
        return code  # Return the original code if formatting fails

def check_with_flake8(file_path: str) -> bool:
    """
    Checks Python code compliance using flake8.
    
    Parameters:
        file_path (str): Path to the Python file to check.
    
    Returns:
        bool: True if the code passes flake8 checks, False otherwise.
    """
    try:
        result = subprocess.run(['flake8', file_path], capture_output=True, text=True)
        if result.returncode == 0:
            logger.debug(f"flake8 passed for '{file_path}'.")
            return True
        else:
            logger.error(f"flake8 issues in '{file_path}':\n{result.stdout}")
            return False
    except FileNotFoundError:
        logger.error("flake8 is not installed. Please install it via 'pip install flake8'.")
        return False
    except Exception as e:
        logger.error(f"Error running flake8 on '{file_path}': {e}")
        return False

def is_valid_extension(ext: str, skip_types: Set[str]) -> bool:
    """Checks if a file extension is valid (not in the skip list)."""
    return ext.lower() not in skip_types

def extract_json_from_response(response: str) -> Optional[dict]:
    """Extracts JSON content from the model's response.

    Attempts multiple methods to extract JSON:
    1. Function calling format.
    2. JSON enclosed in triple backticks.
    3. Entire response as JSON.

    Args:
        response (str): The raw response string from the model.

    Returns:
        Optional[dict]: The extracted JSON as a dictionary, or None if extraction fails.
    """
    # First, try to extract JSON using the function calling format
    try:
        response_json = json.loads(response)
        if "function_call" in response_json and "arguments" in response_json["function_call"]:
            return json.loads(response_json["function_call"]["arguments"])
    except json.JSONDecodeError:
        pass  # Fallback to other extraction methods

    # Try to find JSON enclosed in triple backticks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # As a last resort, attempt to use the entire response if it's valid JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None

# Updated function_schema with 'description' instead of 'docstring'
# utils.py

# utils.py

function_schema = {
    "name": "generate_documentation",
    "description": "Generates comprehensive documentation based on the provided code structure.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A detailed summary of the codebase, explaining its purpose and functionality."
            },
            "changes": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "A comprehensive list of changes or updates made to the code, including new features, bug fixes, and optimizations."
            },
            "functions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the function."
                        },
                        "description": {
                            "type": "string",
                            "description": "A detailed description of what the function does."
                        },
                        "parameters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The name of the parameter."
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "The data type of the parameter."
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "A brief description of the parameter."
                                    }
                                },
                                "required": ["name", "type"]
                            },
                            "description": "A list of parameters that the function accepts."
                        },
                        "returns": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "description": "The return data type of the function."
                                },
                                "description": {
                                    "type": "string",
                                    "description": "A brief description of what the function returns."
                                }
                            },
                            "required": ["type"]
                        }
                    },
                    "required": ["name", "description"]
                },
                "description": "A detailed list of functions within the codebase, including their descriptions, parameters, and return types."
            },
            "classes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the class."
                        },
                        "description": {
                            "type": "string",
                            "description": "A detailed description of what the class represents."
                        },
                        "methods": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The name of the method."
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "A detailed description of what the method does."
                                    },
                                    "parameters": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {
                                                    "type": "string",
                                                    "description": "The name of the parameter."
                                                },
                                                "type": {
                                                    "type": "string",
                                                    "description": "The data type of the parameter."
                                                },
                                                "description": {
                                                    "type": "string",
                                                    "description": "A brief description of the parameter."
                                                }
                                            },
                                            "required": ["name", "type"]
                                        },
                                        "description": "A list of parameters that the method accepts."
                                    },
                                    "returns": {
                                        "type": "object",
                                        "properties": {
                                            "type": {
                                                "type": "string",
                                                "description": "The return data type of the method."
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "A brief description of what the method returns."
                                            }
                                        },
                                        "required": ["type"]
                                    }
                                },
                                "required": ["name", "description"]
                            },
                            "description": "A detailed list of methods within the class, including their descriptions, parameters, and return types."
                        }
                    },
                    "required": ["name", "description"]
                },
                "description": "A detailed list of classes within the codebase, including their descriptions and methods."
            }
        },
        "required": ["summary", "changes"]
    }
}


async def fetch_documentation(session: aiohttp.ClientSession, prompt: str, semaphore: asyncio.Semaphore, model_name: str, function_schema: dict) -> Optional[dict]:
    """
    Fetches documentation from OpenAI's API based on the provided prompt.
    
    Parameters:
        session (aiohttp.ClientSession): The aiohttp client session.
        prompt (str): The prompt to send to the API.
        semaphore (asyncio.Semaphore): Semaphore to control concurrency.
        model_name (str): The OpenAI model to use.
        function_schema (dict): The JSON schema for the function call.
    
    Returns:
        Optional[dict]: The generated documentation, or None if failed.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates code documentation."},
            {"role": "user", "content": prompt}
        ],
        "functions": [function_schema],
        "function_call": "auto"  # Let the model decide which function to call
    }
    
    async with semaphore:
        try:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as resp:
                response_text = await resp.text()
                if resp.status != 200:
                    logger.error(f"OpenAI API request failed with status {resp.status}: {response_text}")
                    return None
                response = await resp.json()
                logger.debug(f"Full API Response: {json.dumps(response, indent=2)}")
                choice = response.get("choices", [])[0]
                message = choice.get('message', {})
                
                # Check for function_call
                if 'function_call' in message:
                    arguments = message['function_call'].get('arguments', '{}')
                    try:
                        documentation = json.loads(arguments)
                        logger.debug("Received documentation via function_call.")
                        return documentation
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON from function_call arguments: {e}")
                        logger.error(f"Arguments Content: {arguments}")
                        return None
                else:
                    # Fallback: Extract documentation directly from content
                    content = message.get('content', '')
                    if content:
                        logger.debug("No function_call detected. Attempting to extract documentation from content.")
                        # Implement parsing logic based on expected content format
                        try:
                            # Example Parsing Logic:
                            # Assume the content has sections like "Summary:" and "Changes:"
                            summary = ""
                            changes = []
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if line.startswith("Summary:"):
                                    summary = line.replace("Summary:", "").strip()
                                elif line.startswith("Changes:"):
                                    changes = [l.replace("-", "").strip() for l in lines[i+1:] if l.startswith("-")]
                            documentation = {
                                "summary": summary,
                                "changes": changes
                            }
                            logger.debug("Extracted documentation from content.")
                            return documentation
                        except Exception as e:
                            logger.error(f"Error parsing documentation content: {e}")
                            logger.error(f"Content Received: {content}")
                            return None
                    else:
                        logger.error("No content found in the API response.")
                        return None
        except Exception as e:
            logger.error(f"Error fetching documentation from OpenAI API: {e}")
            return None

async def fetch_summary(
    session: aiohttp.ClientSession,
    prompt: str,
    semaphore: asyncio.Semaphore,
    model_name: str,
    retry: int = 3,
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
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set. Please set it in your environment or .env file.")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "system", "content": "You are an AI assistant that summarizes code."},
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.2,
    }

    for attempt in range(1, retry + 1):
        try:
            async with semaphore:
                async with session.post(
                    OPENAI_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=120,
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
                        logger.error(
                            f"Unhandled API request failure with status {response.status}: {error_text}"
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

    logger.error("Failed to generate summary after multiple attempts.")
    return None

def generate_documentation_prompt(code_structure: dict, project_info: Optional[str], style_guidelines: Optional[str], language: str) -> str:
    """
    Generates a prompt for the OpenAI API based on the code structure.
    
    Parameters:
        code_structure (dict): The extracted structure of the code.
        project_info (Optional[str]): Information about the project.
        style_guidelines (Optional[str]): Documentation style guidelines.
        language (str): The programming language of the code.
    
    Returns:
        str: The generated prompt.
    """
    prompt = "You are an experienced software developer tasked with generating comprehensive documentation for a codebase."
    if project_info:
        prompt += f"\n\n**Project Information:** {project_info}"
    if style_guidelines:
        prompt += f"\n\n**Style Guidelines:** {style_guidelines}"
    prompt += f"\n\n**Language:** {language.capitalize()}"
    prompt += f"\n\n**Code Structure:**\n```json\n{json.dumps(code_structure, indent=2)}\n```"
    prompt += "\n\n**Instructions:** Based on the above code structure, generate the following documentation sections:\n1. **Summary:** A detailed summary of the codebase.\n2. **Changes Made:** A comprehensive list of changes or updates made to the code.\n\n**Please ensure that the documentation is clear, detailed, and adheres to the provided style guidelines.**"
    return prompt
