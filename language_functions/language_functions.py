"""
language_functions.py

Utility for inserting docstrings into source code using language handlers.
"""

import json
import logging
from typing import Dict, Any, Optional

from utils import load_function_schema

logger = logging.getLogger(__name__)


def insert_docstrings(
    original_code: str,
    documentation: Dict[str, Any],
    language: str,
    schema_path: str,
) -> str:
    """Inserts docstrings/comments into code using the appropriate language handler."""
    logger.debug(f"Processing docstrings for language: {language}")

    try:
        function_schema = load_function_schema(schema_path)
    except (ValueError, FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.error(f"Error loading function schema: {e}")
        return original_code
    except Exception as e:
        logger.error(f"Unexpected error during schema loading: {e}", exc_info=True)
        return original_code

    # Import get_handler from the package __init__
    from language_functions import get_handler

    handler = get_handler(language, function_schema)
    if not handler:
        logger.warning(f"Unsupported language '{language}'. Skipping docstring insertion.")
        return original_code

    if documentation is None:
        logger.error("Documentation is None. Skipping docstring insertion.")
        return original_code

    try:
        updated_code = handler.insert_docstrings(original_code, documentation)
        logger.debug("Docstring insertion completed successfully.")
        return updated_code
    except Exception as e:
        logger.error(f"Error inserting docstrings: {e}", exc_info=True)
        return original_code
