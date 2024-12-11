"""
java_handler.py

This module provides the `JavaHandler` class, which is responsible for handling Java code files.
It includes methods for extracting the code structure, inserting Javadoc comments, and validating Java code.
The handler uses external JavaScript scripts for parsing and modifying the code.

The `JavaHandler` class extends the `BaseHandler` abstract class.
"""

import os
import logging
import subprocess
import json
from typing import Dict, Any, Optional

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class JavaHandler(BaseHandler):
    """Handler for the Java language."""

    def __init__(self, function_schema: Dict[str, Any]):
        """
        Initializes the JavaHandler with a function schema.

        Args:
            function_schema (Dict[str, Any]): The schema used for function operations.
        """
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """
        Extracts the structure of the Java code, analyzing classes, methods, and fields.

        This method runs an external JavaScript parser script that processes the Java code and outputs
        a JSON structure representing the code elements.

        Args:
            code (str): The source code to analyze.
            file_path (str, optional): The file path for code reference.

        Returns:
            Dict[str, Any]: A detailed structure of the code components.
        """
        try:
            script_path = os.path.join(os.path.dirname(
                __file__), "..", "scripts", "java_parser.js")
            input_data = {"code": code, "language": "java"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running Java parser script: {script_path}")

            result = subprocess.run(
                ["node", script_path], input=input_json, capture_output=True, text=True, check=True)

            structure = json.loads(result.stdout)
            logger.debug(
                f"Extracted Java code structure successfully from file: {file_path}")
            return structure

        except subprocess.CalledProcessError as e:
            logger.error(
                f"Error running java_parser.js for file {file_path}: {e.stderr}")
            return {}

        except json.JSONDecodeError as e:
            logger.error(
                f"Error parsing output from java_parser.js for file {file_path}: {e}")
            return {}

        except Exception as e:
            logger.error(
                f"Unexpected error extracting Java structure from file {file_path}: {e}")
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts Javadoc comments into Java code based on the provided documentation.

        This method runs an external JavaScript inserter script that processes the code and documentation
        to insert comments.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The source code with inserted documentation.
        """
        logger.debug("Inserting Javadoc docstrings into Java code.")
        try:
            script_path = os.path.join(os.path.dirname(
                __file__), "..", "scripts", "java_inserter.js")
            input_data = {"code": code,
                          "documentation": documentation, "language": "java"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running Java inserter script: {script_path}")

            result = subprocess.run(
                ["node", script_path], input=input_json, capture_output=True, text=True, check=True)

            modified_code = result.stdout
            logger.debug(
                "Completed inserting Javadoc docstrings into Java code.")
            return modified_code

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running java_inserter.js: {e.stderr}")
            return code

        except Exception as e:
            logger.error(f"Unexpected error inserting Javadoc docstrings: {e}")
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates Java code for syntax correctness using javac.

        Args:
            code (str): The Java code to validate.
            file_path (Optional[str]): The path to the Java source file.

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        logger.debug("Starting Java code validation.")
        if not file_path:
            logger.warning(
                "File path not provided for javac validation. Skipping javac.")
            return True  # Assuming no validation without a file

        try:
            # Write code to a temporary file for validation
            temp_file = f"{file_path}.temp.java"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(code)
            logger.debug(
                f"Wrote temporary Java file for validation: {temp_file}")

            # Compile the temporary Java file
            process = subprocess.run(
                ["javac", temp_file], capture_output=True, text=True)

            # Remove the temporary file and class file if compilation was successful
            if process.returncode != 0:
                logger.error(
                    f"javac validation failed for {file_path}:\n{process.stderr}")
                return False
            else:
                logger.debug("javac validation passed.")
                os.remove(temp_file)
                # Remove the generated class file
                class_file = temp_file.replace(".java", ".class")
                if os.path.exists(class_file):
                    os.remove(class_file)
            return True

        except FileNotFoundError:
            logger.error(
                "javac is not installed or not found in PATH. Please install the JDK.")
            return False

        except Exception as e:
            logger.error(f"Unexpected error during Java code validation: {e}")
            return False
