# Module: html_handler

## Overview
**File:** `docs/language_functions/html_handler.py`
**Description:** 

## AI-Generated Documentation


**Summary:** No summary provided.


**Description:** No description provided.



## Classes

| Class | Inherits From | Complexity Score* |
|-------|---------------|-------------------|
| `HTMLHandler` | `BaseHandler` | 0 |

### Class Methods

| Class | Method | Parameters | Returns | Complexity Score* |
|-------|--------|------------|---------|-------------------|
| `HTMLHandler` | `extract_structure` | `(self: Any, code: str, file_path: str, metrics: Optional[Dict[(str, Any)]] = None)` | `Dict[(str, Any)]` | 0 |
| `HTMLHandler` | `insert_docstrings` | `(self: Any, code: str, documentation: Dict[(str, Any)])` | `str` | 0 |
| `HTMLHandler` | `validate_code` | `(self: Any, code: str, file_path: Optional[str] = None)` | `bool` | 0 |

## Source Code
```python
"""
html_handler.py

This module provides the `HTMLHandler` class, which is responsible for handling HTML code files.
It includes methods for extracting the code structure, inserting comments, and validating HTML code.
The handler uses external JavaScript scripts for parsing and modifying the code.

The `HTMLHandler` class extends the `BaseHandler` abstract class.
"""

import os
import logging
import subprocess
import json
from typing import Dict, Any, Optional

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class HTMLHandler(BaseHandler):
    """Handler for the HTML language."""

    def __init__(self, function_schema: Dict[str, Any]):
        """
        Initializes the HTMLHandler with a function schema.

        Args:
            function_schema (Dict[str, Any]): The schema used for function operations.
        """
        self.function_schema = function_schema

    async def extract_structure(self, code: str, file_path: str, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extracts the structure of the HTML code, analyzing tags, attributes, and nesting.

        This method runs an external JavaScript parser script that processes the HTML code and outputs
        a JSON structure representing the code elements.

        Args:
            code (str): The source code to analyze.
            file_path (str, optional): The file path for code reference.

        Returns:
            Dict[str, Any]: A detailed structure of the HTML components.
        """
        try:
            script_path = os.path.join(os.path.dirname(
                __file__), "..", "scripts", "html_parser.js")
            input_data = {"code": code, "language": "html"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running HTML parser script: {script_path}")

            result = subprocess.run(
                ["node", script_path], input=input_json, capture_output=True, text=True, check=True)

            structure = json.loads(result.stdout)
            logger.debug(
                f"Extracted HTML code structure successfully from file: {file_path}")
            return structure

        except subprocess.CalledProcessError as e:
            logger.error(
                f"Error running html_parser.js for file {file_path}: {e.stderr}")
            return {}

        except json.JSONDecodeError as e:
            logger.error(
                f"Error parsing output from html_parser.js for file {file_path}: {e}")
            return {}

        except Exception as e:
            logger.error(
                f"Unexpected error extracting HTML structure from file {file_path}: {e}")
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts comments into HTML code based on the provided documentation.

        This method runs an external JavaScript inserter script that processes the code and documentation
        to insert comments.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The source code with inserted documentation.
        """
        logger.debug("Inserting comments into HTML code.")
        try:
            script_path = os.path.join(os.path.dirname(
                __file__), "..", "scripts", "html_inserter.js")
            input_data = {"code": code,
                          "documentation": documentation, "language": "html"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running HTML inserter script: {script_path}")

            result = subprocess.run(
                ["node", script_path], input=input_json, capture_output=True, text=True, check=True)

            modified_code = result.stdout
            logger.debug("Completed inserting comments into HTML code.")
            return modified_code

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running html_inserter.js: {e.stderr}")
            return code

        except Exception as e:
            logger.error(
                f"Unexpected error inserting comments into HTML code: {e}")
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates HTML code for correctness using an HTML validator like 'tidy'.

        Args:
            code (str): The HTML code to validate.
            file_path (Optional[str]): The path to the HTML source file.

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        logger.debug("Starting HTML code validation.")
        try:
            # Using 'tidy' for HTML validation
            process = subprocess.run(
                ["tidy", "-errors", "-quiet", "-utf8"], input=code, capture_output=True, text=True)

            if process.returncode > 0:
                logger.error(f"HTML validation failed:\n{process.stderr}")
                return False
            else:
                logger.debug("HTML validation passed.")
            return True

        except FileNotFoundError:
            logger.error(
                "tidy is not installed or not found in PATH. Please install it for HTML validation.")
            return False

        except Exception as e:
            logger.error(f"Unexpected error during HTML code validation: {e}")
            return False

```