# language_functions/html_handler.py

import logging
import subprocess
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup, Comment
from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class HTMLHandler(BaseHandler):
    """Handler for HTML language."""
    def __init__(self, function_schema: Dict[str, Any]):
        """Initializes the HTMLHandler with a function schema."""
        self.function_schema = function_schema
        
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

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates HTML code using the tidy utility.

        Args:
            code (str): The HTML code to validate.
            file_path (Optional[str]): The path to the HTML file being validated.

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        logger.debug('Starting HTML code validation.')
        if not file_path:
            logger.warning('File path not provided for HTML validation. Skipping tidy validation.')
            return True  # Assuming no validation without a file

        try:
            # Write code to the specified file path
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)

            # Attempt to validate the HTML file using tidy
            process = subprocess.run(
                ['tidy', '-e', file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if process.returncode != 0:
                logger.error(f'Tidy validation failed for {file_path}:\n{process.stderr}')
                return False
            else:
                logger.debug('Tidy validation successful.')
            return True
        except FileNotFoundError:
            logger.error("Tidy utility not found. Please install it using your package manager (e.g., 'sudo apt-get install tidy').")
            return False
        except Exception as e:
            logger.error(f'Unexpected error during HTML code validation: {e}')
            return False