"""
language_functions.py

This module provides utility functions for handling different programming languages within the documentation generation pipeline.
It includes functions to retrieve the appropriate language handler and to insert docstrings/comments into source code based on AI-generated documentation.

Functions:
    - get_handler(language, function_schema): Retrieves the appropriate handler for a given programming language.
    - insert_docstrings(original_code, documentation, language, schema_path): Inserts docstrings/comments into the source code using the specified language handler.
"""

import json
import logging
from typing import Dict, Any, Optional

from .base_handler import BaseHandler
from .python_handler import PythonHandler
from .java_handler import JavaHandler
from .js_ts_handler import JSTsHandler
from .go_handler import GoHandler
from .cpp_handler import CppHandler
from .html_handler import HTMLHandler
from .css_handler import CSSHandler
from utils import load_function_schema  # Import for dynamic schema loading

logger = logging.getLogger(__name__)


def get_handler(language: str, function_schema: Dict[str, Any]) -> Optional[BaseHandler]:
    """
    Factory function to retrieve the appropriate language handler.

    This function matches the provided programming language with its corresponding handler class.
    If the language is supported, it returns an instance of the handler initialized with the given function schema.
    If the language is unsupported, it logs a warning and returns None.

    Args:
        language (str): The programming language of the source code (e.g., "python", "java", "javascript").
        function_schema (Dict[str, Any]): The schema defining functions and their documentation structure.

    Returns:
        Optional[BaseHandler]: An instance of the corresponding language handler or None if unsupported.
    """
    if function_schema is None:
        logger.error("Function schema is None. Cannot retrieve handler.")
        return None

    language = language.lower()
    if language == "python":
        return PythonHandler(function_schema)
    elif language == "java":
        return JavaHandler(function_schema)
    elif language in ["javascript", "js", "typescript", "ts"]:
        return JSTsHandler(function_schema)
    elif language == "go":
        return GoHandler(function_schema)
    elif language in ["cpp", "c++", "cxx"]:
        return CppHandler(function_schema)
    elif language in ["html", "htm"]:
        return HTMLHandler(function_schema)
    elif language == "css":
        return CSSHandler(function_schema)
    else:
        logger.warning(f"No handler available for language: {language}")
        return None


def insert_docstrings(
    original_code: str, 
    documentation: Dict[str, Any], 
    language: str, 
    schema_path: str  # schema_path is now required
) -> str:
    """
    Inserts docstrings/comments into code based on the specified programming language.

    This function dynamically loads the function schema from a JSON file, retrieves the appropriate
    language handler, and uses it to insert documentation comments into the original source code.
    If any errors occur during schema loading or docstring insertion, the original code is returned.

    Args:
        original_code (str): The original source code to be documented.
        documentation (Dict[str, Any]): Documentation details obtained from AI, typically including descriptions of functions, classes, and methods.
        language (str): The programming language of the source code (e.g., "python", "java", "javascript").
        schema_path (str): Path to the function schema JSON file, which defines the structure and expected documentation format.

    Returns:
        str: The source code with inserted documentation comments, or the original code if errors occur.
    """
    logger.debug(f"Processing docstrings for language: {language}")

    try:
        # Load the function schema from the provided schema path
        function_schema = load_function_schema(schema_path)
    except (ValueError, FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.error(f"Error loading function schema: {e}")
        return original_code  # Return original code on schema loading error
    except Exception as e:  # Catch-all for other unexpected errors
        logger.error(f"An unexpected error occurred during schema loading: {e}", exc_info=True)
        return original_code

    # Retrieve the appropriate handler for the specified language
    handler = get_handler(language, function_schema)
    if not handler:
        logger.warning(f"Unsupported language '{language}'. Skipping docstring insertion.")
        return original_code

    if documentation is None:
        logger.error("Documentation is None. Skipping docstring insertion.")
        return original_code

    try:
        # Use the handler to insert docstrings/comments into the original code
        updated_code = handler.insert_docstrings(original_code, documentation)
        logger.debug("Docstring insertion completed successfully.")
        return updated_code
    except Exception as e:
        logger.error(f"Error inserting docstrings: {e}", exc_info=True)
        return original_code  # Return original code on docstring insertion error
