# language_functions/html_handler.py

import logging
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup, Comment
from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class HTMLHandler(BaseHandler):
    """Handler for HTML language."""

    def extract_structure(self, code: str) -> Dict[str, Any]:
        """Extracts the structure of HTML code."""
        logger.debug("Extracting HTML structure.")
        try:
            soup = BeautifulSoup(code, "lxml")
            structure = {"tags": []}
            for tag in soup.find_all(True):
                structure["tags"].append({"name": tag.name, "attributes": tag.attrs})
                logger.debug(f"Extracted tag: {tag.name}")
            return structure
        except Exception as e:
            logger.error(f"Error extracting HTML structure: {e}")
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """Inserts comments into HTML code based on provided documentation."""
        logger.debug("Inserting HTML comments.")
        try:
            soup = BeautifulSoup(code, "lxml")
            summary = documentation.get("summary", "").strip()
            changes = documentation.get("changes_made", [])
            if not summary and not changes:
                logger.warning("No summary or changes provided in documentation. Skipping comment insertion.")
                return code
            new_comment_parts = []
            if summary:
                new_comment_parts.append(f"Summary: {summary}")
            if changes:
                changes_formatted = "; ".join(changes)
                new_comment_parts.append(f"Changes: {changes_formatted}")
            new_comment = Comment(" " + " | ".join(new_comment_parts) + " ")
            if soup.body:
                soup.body.insert(0, new_comment)
                logger.debug("Inserted comment at the beginning of the body.")
            else:
                soup.insert(0, new_comment)
                logger.debug("Inserted comment at the beginning of the document.")
            modified_code = soup.prettify()
            logger.debug("Completed inserting HTML comments.")
            return modified_code
        except Exception as e:
            logger.error(f"Error inserting HTML comments: {e}")
            return code

    def validate_code(self, code: str) -> bool:
        """Validates the modified HTML code for correctness."""
        # Basic validation using BeautifulSoup's parser
        try:
            BeautifulSoup(code, "lxml")
            logger.debug("HTML code validation successful.")
            return True
        except Exception as e:
            logger.error(f"HTML code validation failed: {e}")
            return False
