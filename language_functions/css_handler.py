# language_functions/css_handler.py

import logging
import subprocess  # Ensure subprocess is imported
from typing import Dict, Any, Optional
import tinycss2

logger = logging.getLogger(__name__)


class CSSHandler:
    """Handler for CSS language."""

    def __init__(self, function_schema: Dict[str, Any]):
        """Initializes the CSSHandler with a function schema."""
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
                            declarations.append(
                                {
                                    "property": decl.lower_name,
                                    "value": "".join([token.serialize() for token in decl.value]).strip(),
                                }
                            )
                    structure["rules"].append({"selectors": selectors, "declarations": declarations})
                    logger.debug("Extracted rule: %s", selectors)
            return structure
        except tinycss2.ParseError as e:
            logger.error("Parse error while extracting CSS structure: %s", e)
            return {}
        except Exception as e:
            logger.error("Unexpected error while extracting CSS structure: %s", e)
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
            logger.error("Error inserting CSS docstrings: %s", e)
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates CSS code using Stylelint.

        Args:
            code (str): The CSS code to validate.
            file_path (Optional[str]): The path to the CSS file being validated.

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        logger.debug("Starting CSS code validation for file: %s", file_path)
        if not file_path:
            logger.warning("File path not provided for Stylelint validation. Skipping validation.")
            return True  # Assuming no validation without a file

        try:
            # Write code to the specified file path
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Attempt to validate the CSS file using Stylelint
            process = subprocess.run(
                ["stylelint", file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if process.returncode != 0:
                logger.error("Stylelint validation failed for %s:\n%s", file_path, process.stdout)
                return False
            else:
                logger.debug("Stylelint validation successful for %s.", file_path)
            return True
        except FileNotFoundError:
            logger.error("Stylelint not found. Please install it using 'npm install -g stylelint'.")
            return False
        except Exception as e:
            logger.error("Unexpected error during Stylelint validation for %s: %s", file_path, e)
            return False
