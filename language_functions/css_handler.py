"""
css_handler.py

This module provides the `CSSHandler` class, which is responsible for handling CSS code files.
It includes methods for extracting the code structure, inserting comments, and validating CSS code.
The handler uses external JavaScript scripts for parsing and modifying the code.

The `CSSHandler` class extends the `BaseHandler` abstract class.
"""

import os
import logging
import subprocess
import json
from typing import Dict, Any, Optional

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class CSSHandler(BaseHandler):
    """Handler for the CSS programming language."""

    def __init__(self, function_schema: Dict[str, Any]):
        """
        Initializes the `CSSHandler` with a function schema.

        Args:
            function_schema (Dict[str, Any]): The schema defining functions for documentation generation.
        """
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """
        Extracts the structure of the CSS code, analyzing selectors, properties, and rules.

        This method runs an external JavaScript parser script that processes the CSS code and outputs
        a JSON structure representing the code elements.

        Args:
            code (str): The source code to analyze.
            file_path (str, optional): The file path for code reference. Defaults to None.

        Returns:
            Dict[str, Any]: A detailed structure of the CSS components.
        """
        try:
            # Path to the CSS parser script
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "css_parser.js")
            # Prepare input data for the parser
            input_data = {"code": code, "language": "css"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running CSS parser script: {script_path}")

            # Run the parser script using Node.js
            result = subprocess.run(
                ["node", script_path],
                input=input_json,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse the output JSON structure
            structure = json.loads(result.stdout)
            logger.debug(f"Extracted CSS code structure successfully from file: {file_path}")
            return structure

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running css_parser.js for file {file_path}: {e.stderr}")
            return {}

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing output from css_parser.js for file {file_path}: {e}")
            return {}

        except Exception as e:
            logger.error(f"Unexpected error extracting CSS structure from file {file_path}: {e}")
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts comments into CSS code based on the provided documentation.

        This method runs an external JavaScript inserter script that processes the code and documentation
        to insert comments.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The CSS code with inserted documentation.
        """
        logger.debug("Inserting comments into CSS code.")
        try:
            # Path to the CSS inserter script
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "css_inserter.js")
            # Prepare input data for the inserter
            input_data = {"code": code, "documentation": documentation, "language": "css"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running CSS inserter script: {script_path}")

            # Run the inserter script using Node.js
            result = subprocess.run(
                ["node", script_path],
                input=input_json,
                capture_output=True,
                text=True,
                check=True
            )

            modified_code = result.stdout
            logger.debug("Completed inserting comments into CSS code.")
            return modified_code

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running css_inserter.js: {e.stderr}")
            return code

        except Exception as e:
            logger.error(f"Unexpected error inserting comments into CSS code: {e}")
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates CSS code for correctness using 'stylelint'.

        Args:
            code (str): The CSS code to validate.
            file_path (Optional[str]): The path to the CSS source file.

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        logger.debug("Starting CSS code validation.")
        try:
            # Use stylelint to validate CSS code
            process = subprocess.run(
                ["stylelint", "--stdin"],
                input=code,
                capture_output=True,
                text=True
            )

            # Check the result of the validation
            if process.returncode != 0:
                logger.error(f"stylelint validation failed:\n{process.stderr}")
                return False
            else:
                logger.debug("stylelint validation passed.")
            return True

        except FileNotFoundError:
            logger.error(
                "stylelint is not installed or not found in PATH. Please install it using 'npm install -g stylelint'."
            )
            return False

        except Exception as e:
            logger.error(f"Unexpected error during CSS code validation: {e}")
            return False