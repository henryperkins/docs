"""
go_handler.py

This module provides the `GoHandler` class, which is responsible for handling Go language code files.
It includes methods for extracting the code structure, inserting comments, and validating Go code.
The handler uses external Go scripts for parsing and modifying the code.

The `GoHandler` class extends the `BaseHandler` abstract class.
"""

import os
import logging
import subprocess
import json
from typing import Dict, Any, Optional

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class GoHandler(BaseHandler):
    """Handler for the Go programming language."""

    def __init__(self, function_schema: Dict[str, Any]):
        """
        Initializes the `GoHandler` with a function schema.

        Args:
            function_schema (Dict[str, Any]): The schema defining functions for documentation generation.
        """
        self.function_schema = function_schema

    async def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """
        Extracts the structure of the Go code, analyzing functions, types, and variables.

        This method runs an external Go parser script that processes the code and outputs
        a JSON structure representing the code elements.

        Args:
            code (str): The source code to analyze.
            file_path (str, optional): The file path for code reference. Defaults to None.

        Returns:
            Dict[str, Any]: A detailed structure of the code components.
        """
        try:
            # Path to the Go parser script
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "go_parser.go")
            # Prepare input data for the parser
            input_data = {"code": code, "language": "go"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running Go parser script: {script_path}")

            # Run the parser script
            result = subprocess.run(
                ["go", "run", script_path], input=input_json, capture_output=True, text=True, check=True
            )

            # Parse the output JSON structure
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
        """
        Inserts comments into Go code based on the provided documentation.

        This method runs an external Go inserter script that processes the code and documentation
        to insert comments.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The Go code with inserted documentation.
        """
        logger.debug("Inserting comments into Go code.")
        try:
            # Path to the Go inserter script
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "go_inserter.go")
            # Prepare input data for the inserter
            input_data = {"code": code, "documentation": documentation, "language": "go"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running Go inserter script: {script_path}")

            # Run the inserter script
            result = subprocess.run(
                ["go", "run", script_path], input=input_json, capture_output=True, text=True, check=True
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
        logger.debug("Starting Go code validation.")
        if not file_path:
            logger.warning("File path not provided for Go validation. Skipping validation.")
            return True  # Assuming no validation without a file

        try:
            # Write code to a temporary file for validation
            temp_file = f"{file_path}.temp.go"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(code)
            logger.debug(f"Wrote temporary Go file for validation: {temp_file}")

            # Run 'go fmt' to format the code and check syntax
            fmt_process = subprocess.run(["go", "fmt", temp_file], capture_output=True, text=True)

            # Check the result of 'go fmt'
            if fmt_process.returncode != 0:
                logger.error(f"go fmt validation failed for {file_path}:\n{fmt_process.stderr}")
                return False
            else:
                logger.debug("go fmt validation passed.")

            # Run 'go vet' to check for potential issues
            vet_process = subprocess.run(["go", "vet", temp_file], capture_output=True, text=True)

            # Check the result of 'go vet'
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
