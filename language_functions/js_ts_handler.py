# language_functions/js_ts_handler.py
import os
import logging
import subprocess
import json
from typing import Dict, Any, Optional
from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class JSTsHandler(BaseHandler):
    """Handler for JavaScript and TypeScript languages."""

    def __init__(self, function_schema: Dict[str, Any]):
        """Initializes the JSTsHandler with a function schema."""
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        try:
            # You might need to adjust the script path
            script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'acorn_parser.js')
            input_data = {
                "code": code,
                "language": "javascript"  # or determine based on file_path
            }
            input_json = json.dumps(input_data)
            result = subprocess.run(
                ["node", script_path],
                input=input_json,
                capture_output=True,
                text=True,
                check=True
            )
            structure = json.loads(result.stdout)
            logger.debug("Extracted JS/TS code structure successfully.")
            return structure
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running acorn_parser.js: {e.stderr}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing output from acorn_parser.js: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error extracting JS/TS structure: {e}")
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """Inserts JSDoc comments into JS/TS code based on the provided documentation."""
        logger.debug("Inserting JSDoc docstrings into JS/TS code.")
        try:
            script_path = "scripts/acorn_inserter.js"
            input_data = {
                "code": code,
                "documentation": documentation,
                "language": "javascript"  # or "typescript" based on actual language
            }
            input_json = json.dumps(input_data)
            result = subprocess.run(
                ["node", script_path],
                input=input_json,
                capture_output=True,
                text=True,
                check=True
            )
            modified_code = result.stdout
            logger.debug("Completed inserting JSDoc docstrings.")
            return modified_code
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running acorn_inserter.js: {e.stderr}")
            return code
        except Exception as e:
            logger.error(f"Unexpected error inserting JSDoc docstrings: {e}")
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates JavaScript/TypeScript code for syntax correctness and style compliance.

        Args:
            code (str): The JS/TS code to validate.
            file_path (Optional[str]): The path to the file being validated.

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        logger.debug('Starting JavaScript/TypeScript code validation.')
        if not file_path:
            logger.warning('File path not provided for ESLint validation. Skipping ESLint.')
            return True  # Assuming no linting without a file

        try:
            # Write code to a temporary file for ESLint
            temp_file = f"{file_path}.temp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code)

            process = subprocess.run(
                ['eslint', temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Remove temporary file
            subprocess.run(['rm', temp_file])

            if process.returncode != 0:
                logger.error(f'ESLint validation failed for {file_path}:\n{process.stdout}')
                return False
            else:
                logger.debug('ESLint validation successful.')
            return True
        except FileNotFoundError:
            logger.error("ESLint is not installed. Please install it using 'npm install eslint'.")
            return False
        except Exception as e:
            logger.error(f'Unexpected error during ESLint validation: {e}')
            return False