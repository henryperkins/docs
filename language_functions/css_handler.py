# language_functions/css_handler.py

import os
import logging
import subprocess
import json
from typing import Dict, Any, Optional

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class CSSHandler(BaseHandler):
    """Handler for CSS language."""

    def __init__(self, function_schema: Dict[str, Any]):
        """
        Initializes the CSSHandler with a function schema.
    
        Args:
            function_schema (Dict[str, Any]): The schema used for function operations.
        """
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """Extracts the structure of the CSS code, analyzing selectors, properties, and rules.

        Args:
            code (str): The source code to analyze.
            file_path (str): The file path for code reference.

        Returns:
            Dict[str, Any]: A detailed structure of the CSS components.
        """
        try:
            script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'css_parser.js')
            input_data = {
                "code": code,
                "language": "css"
            }
            input_json = json.dumps(input_data)
            logger.debug(f"Running CSS parser script: {script_path}")
            result = subprocess.run(
                ["node", script_path],
                input=input_json,
                capture_output=True,
                text=True,
                check=True
            )
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
        """Inserts comments into CSS code based on the provided documentation.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The source code with inserted documentation.
        """
        logger.debug("Inserting comments into CSS code.")
        try:
            script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'css_inserter.js')
            input_data = {
                "code": code,
                "documentation": documentation,
                "language": "css"
            }
            input_json = json.dumps(input_data)
            logger.debug(f"Running CSS inserter script: {script_path}")
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
        logger.debug('Starting CSS code validation.')
        try:
            # Use stylelint to validate CSS code
            process = subprocess.run(
                ["stylelint", "--stdin"],
                input=code,
                capture_output=True,
                text=True
            )
            if process.returncode != 0:
                logger.error(f"stylelint validation failed:\n{process.stderr}")
                return False
            else:
                logger.debug("stylelint validation passed.")
            return True
        except FileNotFoundError:
            logger.error("stylelint is not installed or not found in PATH. Please install it using 'npm install -g stylelint'.")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during CSS code validation: {e}")
            return False
