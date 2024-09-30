# utils.py

import os
import sys
import json
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
    Loads configuration from a JSON file and updates the provided sets.

    Args:
        config_path (str): Path to the configuration JSON file.
        excluded_dirs (Set[str]): Set of directories to exclude.
        excluded_files (Set[str]): Set of files to exclude.
        skip_types (Set[str]): Set of file extensions to skip.

    Returns:
        Tuple[str, str]: A tuple containing project information and style guidelines.
    """
    project_info = ''
    style_guidelines = ''
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update excluded directories, files, and skip types
        excluded_dirs.update(set(config.get('excluded_dirs', [])))
        excluded_files.update(set(config.get('excluded_files', [])))
        skip_types.update(set(config.get('skip_types', [])))

        # Load project info and style guidelines
        project_info = config.get('project_info', '')
        style_guidelines = config.get('style_guidelines', '')

    except Exception as e:
        logger.error(f"Error loading configuration from '{config_path}': {e}")
    finally:
        return project_info, style_guidelines

def get_all_file_paths(repo_path: str, excluded_dirs: Set[str], excluded_files: Set[str]) -> List[str]:
    """Gets all file paths in a repository, excluding specified directories and files."""
    file_paths = []
    for root, dirs, files in os.walk(repo_path):
        # Exclude specified directories and those starting with a dot
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('.')]
        # Add files that are not excluded and do not start with a dot
        file_paths.extend([os.path.join(root, file) for file in files if file not in excluded_files and not file.startswith('.')])
    logger.info(f"Collected {len(file_paths)} files from '{repo_path}'.")
    return file_paths

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

function_schema = {
    "name": "generate_documentation",
    "description": "Generates detailed documentation, summaries, and lists of changes for code structures.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A concise summary of the code's purpose and functionality."
            },
            "changes": {
                "type": "array",
                "description": "A list of specific changes made in the code.",
                "items": {
                    "type": "string"
                }
            },
            "functions": {
                "type": "array",
                "description": "Details about functions in the code.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Function name."
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed explanation of what the function does."
                        },
                        "parameters": {
                            "type": "array",
                            "description": "List of parameters the function accepts.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Parameter name."
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "Parameter data type."
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Description of the parameter."
                                    }
                                },
                                "required": ["name", "type"]
                            }
                        },
                        "returns": {
                            "type": "object",
                            "description": "Information about the return value.",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "description": "Return data type."
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Description of the return value."
                                }
                            },
                            "required": ["type"]
                        },
                        "decorators": {
                            "type": "array",
                            "description": "List of decorators applied to the function.",
                            "items": {
                                "type": "string"
                            }
                        },
                        "examples": {
                            "type": "array",
                            "description": "Usage examples of the function.",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["name", "description", "parameters", "returns"]
                }
            },
            "classes": {
                "type": "array",
                "description": "Details about classes in the code.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Class name."
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description of the class."
                        },
                        "bases": {
                            "type": "array",
                            "description": "Base classes the class inherits from.",
                            "items": {
                                "type": "string"
                            }
                        },
                        "decorators": {
                            "type": "array",
                            "description": "List of decorators applied to the class.",
                            "items": {
                                "type": "string"
                            }
                        },
                        "methods": {
                            "type": "array",
                            "description": "List of methods within the class.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Method name."
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Detailed explanation of the method."
                                    },
                                    "parameters": {
                                        "type": "array",
                                        "description": "List of parameters the method accepts.",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {
                                                    "type": "string",
                                                    "description": "Parameter name."
                                                },
                                                "type": {
                                                    "type": "string",
                                                    "description": "Parameter data type."
                                                },
                                                "description": {
                                                    "type": "string",
                                                    "description": "Description of the parameter."
                                                }
                                            },
                                            "required": ["name", "type"]
                                        }
                                    },
                                    "returns": {
                                        "type": "object",
                                        "description": "Information about the return value.",
                                        "properties": {
                                            "type": {
                                                "type": "string",
                                                "description": "Return data type."
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "Description of the return value."
                                            }
                                        },
                                        "required": ["type"]
                                    },
                                    "decorators": {
                                        "type": "array",
                                        "description": "List of decorators applied to the method.",
                                        "items": {
                                            "type": "string"
                                        }
                                    },
                                    "examples": {
                                        "type": "array",
                                        "description": "Usage examples of the method.",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                },
                                "required": ["name", "description", "parameters", "returns"]
                            }
                        }
                    },
                    "required": ["name", "description", "bases", "decorators", "methods"]
                }
            },
            "api_endpoints": {
                "type": "array",
                "description": "Details about API endpoints (for backend files).",
                "items": {
                    "type": "object",
                    "properties": {
                        "route": {
                            "type": "string",
                            "description": "API route path."
                        },
                        "method": {
                            "type": "string",
                            "description": "HTTP method (GET, POST, etc.)."
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of what the endpoint does."
                        },
                        "authentication": {
                            "type": "string",
                            "description": "Authentication requirements."
                        },
                        "parameters": {
                            "type": "array",
                            "description": "Parameters accepted by the API endpoint.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Parameter name."
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "Parameter data type."
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Description of the parameter."
                                    },
                                    "required": {
                                        "type": "boolean",
                                        "description": "Whether the parameter is required."
                                    }
                                },
                                "required": ["name", "type", "required"]
                            }
                        },
                        "responses": {
                            "type": "array",
                            "description": "Possible responses from the API endpoint.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "status_code": {
                                        "type": "integer",
                                        "description": "HTTP status code."
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Description of the response."
                                    },
                                    "body": {
                                        "type": "string",
                                        "description": "Response body content."
                                    }
                                },
                                "required": ["status_code", "description"]
                            }
                        },
                        "examples": {
                            "type": "array",
                            "description": "Example requests to the API endpoint.",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["route", "method", "description"]
                }
            },
            "elements": {
                "type": "array",
                "description": "Details about HTML elements.",
                "items": {
                    "type": "object",
                    "properties": {
                        "tag": {
                            "type": "string",
                            "description": "HTML tag name."
                        },
                        "attributes": {
                            "type": "object",
                            "description": "Attributes of the HTML element.",
                            "additionalProperties": {
                                "type": "string",
                                "description": "Value of the attribute."
                            }
                        },
                        "text": {
                            "type": "string",
                            "description": "Inner text of the element."
                        },
                        "description": {
                            "type": "string",
                            "description": "Purpose of the element."
                        }
                    },
                    "required": ["tag"]
                }
            },
            "rules": {
                "type": "array",
                "description": "Details about CSS rules.",
                "items": {
                    "type": "object",
                    "properties": {
                        "selector": {
                            "type": "string",
                            "description": "CSS selector."
                        },
                        "declarations": {
                            "type": "string",
                            "description": "CSS declarations (properties and values)."
                        },
                        "description": {
                            "type": "string",
                            "description": "Purpose or effect of the rule."
                        }
                    },
                    "required": ["selector"]
                }
            }
        },
        "required": ["summary", "changes"]
    }
}


async def fetch_documentation(
    session: aiohttp.ClientSession,
    prompt: str,
    semaphore: asyncio.Semaphore,
    model_name: str,
    function_schema: dict,
    retry: int = 3,
) -> Optional[dict]:
    """Fetches generated documentation from the OpenAI API using function calling.

    Args:
        session (aiohttp.ClientSession): The aiohttp client session.
        prompt (str): The prompt to send to the API.
        semaphore (asyncio.Semaphore): Semaphore to control concurrency.
        model_name (str): The name of the OpenAI model to use.
        function_schema (dict): The JSON schema for the expected function call output.
        retry (int, optional): The number of retry attempts. Defaults to 3.

    Returns:
        Optional[dict]: A dictionary containing the generated documentation, or None if the request fails.
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set. Please set it in your environment or .env file.")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Prepare the messages for the API request
    messages = [
        {"role": "system", "content": "You are an AI assistant that generates code documentation."},
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "functions": [function_schema],
        "function_call": {"name": "generate_documentation"},
        "temperature": 0.2,
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
                            # Attempt to extract JSON using function calling format
                            message = data["choices"][0]["message"]
                            if "function_call" in message:
                                function_call = message["function_call"]
                                arguments_str = function_call["arguments"]
                                # Ensure that 'arguments_str' is a valid JSON string
                                arguments = json.loads(arguments_str)
                                documentation = arguments
                                logger.info("Generated documentation using function call.")
                            else:
                                logger.error("Function call not found in the response.")
                                return None

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

def generate_documentation_prompt(
    code_structure: dict,
    project_info: str = None,
    style_guidelines: str = None
) -> str:
    """
    Generates a prompt for the AI model to create comprehensive documentation, summaries, and change lists.

    Parameters:
        code_structure (dict): The structured representation of the code.
        project_info (str, optional): Information about the project the code belongs to.
        style_guidelines (str, optional): Specific documentation style guidelines to follow.

    Returns:
        str: The generated prompt to be sent to the AI model.
    """

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
Based on the provided {language} code structure in JSON format, generate comprehensive documentation following these guidelines:

1. **Detailed Summaries**:
   - Provide a clear and concise summary of the file's purpose and functionality.
   - Explain the role of the code within the larger application context.

2. **Changes Made**:
   - List specific changes implemented in the code.
   - Categorize changes if applicable (e.g., Added, Updated, Fixed).

3. **Function and Method Documentation**:
   - For each function and method, include:
     - **Name**: The function or method name.
     - **Description**: A detailed explanation of what it does.
     - **Parameters**:
       - List all parameters with their names, types, and detailed descriptions.
     - **Returns**:
       - Specify the return type and provide a description of the return value.
     - **Usage Examples**:
       - Provide code snippets demonstrating how to use the function or method.

4. **Class Documentation**:
   - For each class, include:
     - **Name**: The class name.
     - **Description**: A detailed explanation of the class's purpose.
     - **Inheritance**: List base classes if any.
     - **Methods**: Document each method as per the function guidelines.

5. **API Endpoint Documentation** (for backend files):
   - For each API endpoint, include:
     - **Route**: The API route path.
     - **Method**: HTTP method (GET, POST, etc.).
     - **Description**: What the endpoint does.
     - **Authentication**: Authentication requirements.
     - **Parameters**:
       - List all expected request parameters with details.
     - **Responses**:
       - Describe possible responses, status codes, and response bodies.
     - **Usage Examples**:
       - Provide example requests to the endpoint.

6. **HTML Element Documentation**:
   - For each HTML element, include:
     - **Tag**: The HTML tag name.
     - **Attributes**: List attributes with descriptions.
     - **Text**: Any inner text.
     - **Description**: Purpose of the element.

7. **CSS Rule Documentation**:
   - For each CSS rule, include:
     - **Selector**: The CSS selector.
     - **Declarations**: The properties and values.
     - **Description**: Purpose or effect of the rule.

8. **Organizational Structure**:
   - Organize the documentation with clear headings and subheadings.
   - Use markdown formatting for readability (e.g., `#`, `##`, `###` for headings).

9. **Code Blocks**:
   - Include code snippets where appropriate, using the correct language identifiers for syntax highlighting.
   - Ensure code examples are complete and functional.

10. **Consistency and Clarity**:
    - Maintain a consistent format throughout the documentation.
    - Use clear and precise language suitable for developers.

Instructions:
- Output the results by invoking the `generate_documentation` function.
- The output must conform to the provided JSON schema.
- Respond only with the function call output, without additional text.

Schema:
{json.dumps(function_schema, indent=2)}

Code Structure:
{json_structure}
"""
    )

    # Special instruction for CSS files
    if language == 'css':
        prompt_parts.append("For CSS files, place comments *above* the selector they apply to.")

    # Combine the prompt parts
    prompt = '\n'.join(prompt_parts)

    return prompt
