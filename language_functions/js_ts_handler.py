"""
js_ts_handler.py

This module provides the JSTsHandler class, which is responsible for handling JavaScript and TypeScript code files.
It includes methods for extracting the code structure, inserting JSDoc comments, and validating JS/TS code.
The handler uses Node.js scripts to interface with JavaScript tools like Babel for parsing and modifying the code.

The JSTsHandler class extends the BaseHandler abstract class.
"""

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
        """
        Initializes the JSTsHandler with a function schema.

        Args:
            function_schema (Dict[str, Any]): The schema defining functions for documentation generation.
        """
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """
        Extracts the structure of the JavaScript/TypeScript code, analyzing classes, functions, methods, variables, and constants.

        This method runs an external Node.js script that uses Babel to parse the code and outputs
        a JSON structure representing the code elements.

        Args:
            code (str): The source code to analyze.
            file_path (str, optional): The file path for code reference.

        Returns:
            Dict[str, Any]: A detailed structure of the code components.
        """
        try:
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "js_ts_parser.js")
            _, ext = os.path.splitext(file_path or "")
            language = "typescript" if ext.lower() in [".ts", ".tsx"] else "javascript"

            input_data = {"code": code, "language": language}
            input_json = json.dumps(input_data)
            logger.debug(f"Running JS/TS parser script: {script_path}")

            result = subprocess.run(
                ["node", script_path],
                input=input_json,
                capture_output=True,
                text=True,
                check=True,
            )

            structure = json.loads(result.stdout)
            logger.debug(f"Extracted JS/TS code structure successfully from file: {file_path}")
            return structure

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running js_ts_parser.js for file {file_path}: {e.stderr}")
            return {}

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing output from js_ts_parser.js for file {file_path}: {e}")
            return {}

        except Exception as e:
            logger.error(f"Unexpected error extracting JS/TS structure from file {file_path}: {e}")
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts JSDoc comments into JS/TS code based on the provided documentation.

        This method runs an external Node.js script that processes the code and documentation
        to insert JSDoc comments.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The source code with inserted documentation.
        """
        logger.debug("Inserting JSDoc comments into JS/TS code.")
        try:
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "js_ts_inserter.js")
            language = documentation.get("language", "javascript")

            input_data = {
                "code": code,
                "documentation": documentation,
                "language": language,
            }
            input_json = json.dumps(input_data)
            logger.debug(f"Running JS/TS inserter script: {script_path}")

            result = subprocess.run(
                ["node", script_path],
                input=input_json,
                capture_output=True,
                text=True,
                check=True,
            )

            modified_code = result.stdout
            logger.debug("Completed inserting JSDoc comments into JS/TS code.")
            return modified_code

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running js_ts_inserter.js: {e.stderr}")
            return code

        except Exception as e:
            logger.error(f"Unexpected error inserting JSDoc comments: {e}")
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates JavaScript/TypeScript code for syntax correctness and style compliance.

        For JavaScript, uses ESLint to check for syntax errors.
        For TypeScript, uses the TypeScript compiler 'tsc' to perform type checking and syntax validation.

        Args:
            code (str): The JS/TS code to validate.
            file_path (Optional[str]): The path to the file being validated.

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        logger.debug(f"Starting JS/TS code validation for file: {file_path}")

        try:
            _, ext = os.path.splitext(file_path or "")
            is_typescript = ext.lower() in [".ts", ".tsx"]
            temp_file_ext = ".temp.ts" if is_typescript else ".temp.js"

            temp_file = f"{(file_path or 'temp')}{temp_file_ext}"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(code)

            if is_typescript:
                process = subprocess.run(
                    ["tsc", "--noEmit", temp_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            else:
                process = subprocess.run(
                    ["eslint", temp_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

            os.remove(temp_file)

            if process.returncode != 0:
                logger.error(f"Code validation failed for {file_path}:\n{process.stdout}\n{process.stderr}")
                return False
            else:
                logger.debug(f"Code validation successful for {file_path}.")
            return True

        except FileNotFoundError as e:
            logger.error(f"Validation tool not found: {e}. Please ensure ESLint and TypeScript are installed.")
            return False

        except Exception as e:
            logger.error(f"Unexpected error during code validation for {file_path}: {e}")
            return False

    def calculate_metrics(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """
        Calculates code complexity metrics for JS/TS code.

        Uses external tools to compute cyclomatic complexity, Halstead metrics, and maintainability index.

        Args:
            code (str): The JS/TS code to analyze.
            file_path (str, optional): The file path for reference.

        Returns:
            Dict[str, Any]: A dictionary containing the metrics.
        """
        logger.debug(f"Calculating metrics for file: {file_path}")
        try:
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "js_ts_metrics.js")
            _, ext = os.path.splitext(file_path or "")
            language = "typescript" if ext.lower() in [".ts", ".tsx"] else "javascript"

            input_data = {"code": code, "language": language}
            input_json = json.dumps(input_data)
            logger.debug(f"Running metrics calculation script: {script_path}")

            result = subprocess.run(
                ["node", script_path],
                input=input_json,
                capture_output=True,
                text=True,
                check=True,
            )

            metrics = json.loads(result.stdout)
            logger.debug(f"Calculated metrics for file: {file_path}")
            return metrics

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running js_ts_metrics.js for file {file_path}: {e.stderr}")
            return {}

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing metrics output for file {file_path}: {e}")
            return {}

        except Exception as e:
            logger.error(f"Unexpected error calculating metrics for file {file_path}: {e}")
            return {}