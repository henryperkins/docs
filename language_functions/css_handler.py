# language_functions/css_handler.py

import logging
from typing import Dict, Any
import tinycss2
from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class CSSHandler(BaseHandler):
    """Handler for CSS language."""

    def __init__(self, function_schema):
        """
        Initialize CSSHandler with a function schema.

        Args:
            function_schema (dict): The schema defining functions.
        """
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str) -> Dict[str, Any]:
        """
        Extracts structure from CSS code.

        Args:
            code (str): The CSS source code to parse.
            file_path (str): The path to the CSS file.

        Returns:
            dict: A dictionary containing the structure of the CSS code.
        """
        logger.debug("Extracting CSS structure.")
        try:
            rules = tinycss2.parse_rule_list(code, skip_whitespace=True, skip_comments=True)
            structure = {"rules": []}
            for rule in rules:
                if rule.type == "qualified-rule":
                    selectors = "".join([token.serialize() for token in rule.prelude]).strip()
                    declarations = []
                    for decl in tinycss2.parse_declaration_list(rule.content):
                        if decl.type == "declaration":
                            declarations.append({
                                "property": decl.lower_name,
                                "value": "".join([token.serialize() for token in decl.value]).strip(),
                            })
                    structure["rules"].append({
                        "selectors": selectors,
                        "declarations": declarations
                    })
                    logger.debug("Extracted rule: %s", selectors)
            return structure
        except tinycss2.ParseError as e:
            logger.error("Parse error while extracting CSS structure: %s", e)
            return {}
        except Exception as e:
            logger.error("Unexpected error: %s", e)
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """Inserts comments into CSS code based on provided documentation."""
        logger.debug("Inserting CSS docstrings.")
        try:
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
            new_comment = "/* " + " | ".join(new_comment_parts) + " */\n"
            modified_code = new_comment + code
            logger.debug("Completed inserting CSS docstrings.")
            return modified_code
        except Exception as e:
            logger.error(f"Error inserting CSS docstrings: {e}")
            return code

    def validate_code(self, code: str) -> bool:
        """Validates the modified CSS code for correctness."""
        # Basic validation using tinycss2
        try:
            tinycss2.parse_stylesheet(code, skip_whitespace=True, skip_comments=True)
            logger.debug("CSS code validation successful.")
            return True
        except Exception as e:
            logger.error(f"CSS code validation failed: {e}")
            return False
