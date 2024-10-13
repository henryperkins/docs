# language_functions.py

import json
import logging
from typing import Dict, Any
from .base_handler import PythonHandler  # Ensure correct import path

logger = logging.getLogger(__name__)

def get_handler(language: str, function_schema: Dict[str, Any]):
    """
    Returns the appropriate handler based on the programming language.

    Args:
        language (str): The programming language of the source code.
        function_schema (Dict[str, Any]): The function schema loaded from JSON.

    Returns:
        BaseHandler or None: An instance of the appropriate handler or None if unsupported.
    """
    language = language.lower()
    if language == 'python':
        return PythonHandler(function_schema)
    # Add other language handlers here (e.g., JavaHandler) as needed
    else:
        logger.warning(f"No handler available for language: {language}")
        return None

def insert_docstrings(
    original_code: str, documentation: Dict[str, Any], language: str, schema_path: str = 'schemas/function_schema.json'
) -> str:
    """
    Inserts docstrings/comments into code based on the language.

    Args:
        original_code (str): The original source code.
        documentation (Dict[str, Any]): Documentation details obtained from AI.
        language (str): Programming language of the source code.
        schema_path (str): Path to the function schema JSON file.

    Returns:
        str: The source code with inserted documentation.
    """
    logger.debug(f"Processing docstrings for language: {language}")

    # Load the function schema from the specified path
    function_schema = None
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            function_schema = json.load(f)
        logger.debug(f"Loaded function schema from '{schema_path}'.")
    except FileNotFoundError:
        logger.error(f"Function schema file not found at '{schema_path}'.")
        return original_code
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from function schema file: {e}")
        return original_code
    except OSError as e:
        logger.error(f"OS error occurred while loading function schema: {e}")
        return original_code
    except Exception as e:
        logger.error(f"Unexpected error loading function schema: {e}", exc_info=True)
        return original_code

    handler = get_handler(language, function_schema)
    if not handler:
        logger.warning(f"Unsupported language '{language}'. Skipping docstring insertion.")
        return original_code

    if documentation is None:
        logger.error("Documentation is None. Skipping docstring insertion.")
        return original_code

    # Insert docstrings using the handler
    try:
        updated_code = handler.insert_docstrings(original_code, documentation)
        logger.debug("Docstring insertion completed successfully.")
        return updated_code
    except Exception as e:
        logger.error(f"Error inserting docstrings: {e}", exc_info=True)
        return original_code
