# language_functions.py

from typing import Optional, Dict, Any
from language_functions import get_handler
import logging

logger = logging.getLogger(__name__)

def insert_docstrings(
    original_code: str, documentation: Dict[str, Any], language: str
) -> str:
    """
    Inserts docstrings/comments into code based on the language.
    
    Parameters:
        original_code (str): The original source code.
        documentation (Dict[str, Any]): Documentation details obtained from AI.
        language (str): Programming language of the source code.
    
    Returns:
        str: The source code with inserted documentation.
    """
    logger.debug(f"Processing docstrings for language: {language}")
    handler = get_handler(language)
    if not handler:
        logger.warning(f"Unsupported language '{language}'. Skipping docstring insertion.")
        return original_code
    try:
        if language.lower() in ["javascript", "typescript"]:
            # For JS/TS, pass language to handler
            return handler.insert_docstrings(original_code, documentation, language.lower())
        else:
            return handler.insert_docstrings(original_code, documentation)
    except Exception as e:
        logger.error(f"Error inserting docstrings for language '{language}': {e}", exc_info=True)
        return original_code
