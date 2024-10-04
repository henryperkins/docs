# language_functions/java_handler.py

import logging
from typing import Optional, Dict, Any
from language_functions.base_handler import BaseHandler
from parser.java_parser import extract_structure, insert_docstrings

logger = logging.getLogger(__name__)

class JavaHandler(BaseHandler):
    """Handler for Java language."""

    def extract_structure(self, code: str) -> Dict[str, Any]:
        """Parses Java code to extract classes and methods."""
        logger.debug("Extracting Java code structure.")
        structure = extract_structure(code)
        if structure is None:
            logger.error("Failed to extract structure from Java code.")
            return {}
        return structure

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """Inserts Javadoc comments into Java code based on the provided documentation."""
        logger.debug("Inserting Javadoc docstrings into Java code.")
        modified_code = insert_docstrings(code, documentation)
        return modified_code

    def validate_code(self, code: str) -> bool:
        """Validates the modified Java code for syntax correctness."""
        # Basic validation by attempting to parse
        from parser.java_parser import extract_structure
        try:
            structure = extract_structure(code)
            if structure:
                logger.debug("Java code validation successful.")
                return True
            else:
                logger.error("Java code validation failed: Structure extraction returned empty.")
                return False
        except Exception as e:
            logger.error(f"Java code validation failed: {e}")
            return False
