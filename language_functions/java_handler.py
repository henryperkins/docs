# language_functions/java_handler.py

import logging
from typing import Optional, Dict, Any
from parser.java_parser import extract_structure, insert_docstrings

logger = logging.getLogger(__name__)

def insert_javadoc_docstrings(
    original_code: str, documentation: Dict[str, Any], language: str
) -> str:
    """
    Inserts Javadoc comments into Java code based on the provided documentation.

    Parameters:
        original_code (str): The original Java source code.
        documentation (Dict[str, Any]): A dictionary containing documentation to insert.
        language (str): The programming language, either 'java'.

    Returns:
        str: The modified Java source code with inserted Javadoc comments.
    """
    logger.debug("Starting insert_javadoc_docstrings")
    try:
        structure = extract_structure(original_code)
        if not structure:
            logger.error("Failed to extract structure from Java code.")
            return original_code
        
        modified_code = insert_docstrings(original_code, documentation)
        logger.debug("Completed inserting Javadoc docstrings")
        return modified_code
    except Exception as e:
        logger.error(f"Exception in insert_javadoc_docstrings: {e}", exc_info=True)
        return original_code
