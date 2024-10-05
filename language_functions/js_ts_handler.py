# language_functions/js_ts_handler.py
import json
import subprocess
import logging
from typing import Optional, Dict, Any
from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class JSTsHandler(BaseHandler):
    def __init__(self, function_schema):
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str) -> Dict[str, Any]:
        try:
            # You might need to adjust the script path
            script_path = "scripts/acorn_parser.js"
            result = subprocess.run(
                ["node", script_path],
                input=code,
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

    def validate_code(self, code: str) -> bool:
        """Validates the modified JS/TS code for syntax correctness."""
        # Simple validation using ESLint or similar tool
        try:
            result = subprocess.run(
                ["eslint", "--no-eslintrc", "--env", "es2021", "--parser-options", "ecmaVersion=12"],
                input=code,
                capture_output=True,
                text=True,
                check=False  # Do not raise exception
            )
            if result.returncode == 0:
                logger.debug("JS/TS code validation successful.")
                return True
            else:
                logger.error(f"JS/TS code validation failed:\n{result.stdout}")
                return False
        except FileNotFoundError:
            logger.error("ESLint is not installed. Please install it using 'npm install eslint'.")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during JS/TS code validation: {e}")
            return False
