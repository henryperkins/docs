# language_functions/go_handler.py

import os
import logging
import subprocess
import json
from typing import Dict, Any, Optional

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class GoHandler(BaseHandler):
    """Handler for Go language."""

    def __init__(self, function_schema: Dict[str, Any]):
        """
        Initializes the GoHandler with a function schema.
    
        Args:
            function_schema (Dict[str, Any]): The schema used for function operations.
        """
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """Extracts the structure of the Go code, analyzing functions, types, and variables.

        Args:
            code (str): The source code to analyze.
            file_path (str): The file path for code reference.

        Returns:
            Dict[str, Any]: A detailed structure of the code components.
        """
        try:
            script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'go_parser.go')
            input_data = {
                "code": code,
                "language": "go"
            }
            input_json = json.dumps(input_data)
            logger.debug(f"Running Go parser script: {script_path}")
            result = subprocess.run(
                ["go", "run", script_path],
                input=input_json,
                capture_output=True,
                text=True,
                check=True
            )
            structure = json.loads(result.stdout)
            logger.debug(f"Extracted Go code structure successfully from file: {file_path}")
            return structure
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running go_parser.go for file {file_path}: {e.stderr}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing output from go_parser.go for file {file_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error extracting Go structure from file {file_path}: {e}")
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """Inserts comments into Go code based on the provided documentation.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The source code with inserted documentation.
        """
        logger.debug("Inserting comments into Go code.")
        try:
            script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'go_inserter.go')
            input_data = {
                "code": code,
                "documentation": documentation,
                "language": "go"
            }
            input_json = json.dumps(input_data)
            logger.debug(f"Running Go inserter script: {script_path}")
            result = subprocess.run(
                ["go", "run", script_path],
                input=input_json,
                capture_output=True,
                text=True,
                check=True
            )
            modified_code = result.stdout
            logger.debug("Completed inserting comments into Go code.")
            return modified_code
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running go_inserter.go: {e.stderr}")
            return code
        except Exception as e:
            logger.error(f"Unexpected error inserting comments into Go code: {e}")
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates Go code for syntax correctness using 'go fmt' and 'go vet'.

        Args:
            code (str): The Go code to validate.
            file_path (Optional[str]): The path to the Go source file.

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        logger.debug('Starting Go code validation.')
        if not file_path:
            logger.warning('File path not provided for Go validation. Skipping validation.')
            return True  # Assuming no validation without a file

        try:
            # Write code to a temporary file for validation
            temp_file = f"{file_path}.temp.go"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.debug(f"Wrote temporary Go file for validation: {temp_file}")

            # Run 'go fmt' to format the code and check syntax
            fmt_process = subprocess.run(
                ["go", "fmt", temp_file],
                capture_output=True,
                text=True
            )
            if fmt_process.returncode != 0:
                logger.error(f"go fmt validation failed for {file_path}:\n{fmt_process.stderr}")
                return False
            else:
                logger.debug("go fmt validation passed.")

            # Run 'go vet' to check for potential errors
            vet_process = subprocess.run(
                ["go", "vet", temp_file],
                capture_output=True,
                text=True
            )
            if vet_process.returncode != 0:
                logger.error(f"go vet validation failed for {file_path}:\n{vet_process.stderr}")
                return False
            else:
                logger.debug("go vet validation passed.")

            # Remove the temporary file
            os.remove(temp_file)
            logger.debug(f"Removed temporary Go file: {temp_file}")

            return True
        except FileNotFoundError:
            logger.error("Go is not installed or not found in PATH. Please install Go.")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Go code validation: {e}")
            return False
