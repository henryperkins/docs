"""
js_ts_handler.py

Handles JavaScript and TypeScript code analysis, documentation generation, and modification.
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
        Initialize the JavaScript/TypeScript handler.

        Args:
            function_schema (Dict[str, Any]): Schema defining documentation structure.
        """
        self.function_schema = function_schema
        self.script_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """
        Extracts the structure of the JavaScript/TypeScript code.

        Args:
            code (str): Source code to analyze.
            file_path (str, optional): Path to the source file.

        Returns:
            Dict[str, Any]: Extracted code structure or empty structure on failure.
        """
        try:
            # Skip if file should be excluded
            if self._should_skip_file(file_path):
                return self._get_empty_structure(f"Skipped: {file_path}")

            # Determine language based on file extension
            language = self._get_language(file_path)

            # Prepare input for parser
            input_data = {
                "code": code,
                "language": language,
                "filePath": file_path or "unknown",
                "options": {
                    "errorRecovery": True,
                    "plugins": self._get_parser_plugins(language)
                }
            }

            # Run parser script
            structure = self._run_script(
                script_name="js_ts_parser.js",
                input_data=input_data,
                error_message=f"Error parsing {file_path}"
            )

            # Run metrics analysis
            metrics = self._run_script(
                script_name="js_ts_metrics.js",
                input_data=input_data,
                error_message=f"Error calculating metrics for {file_path}"
            )

            # Merge metrics into structure
            if metrics and isinstance(metrics, dict):
                structure.update(metrics)

            return structure

        except Exception as e:
            logger.error(f"Unexpected error analyzing {file_path}: {e}", exc_info=True)
            return self._get_empty_structure(f"Error: {str(e)}")

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts documentation into JavaScript/TypeScript code.

        Args:
            code (str): Original source code.
            documentation (Dict[str, Any]): Documentation to insert.

        Returns:
            str: Modified source code with documentation.
        """
        try:
            input_data = {
                "code": code,
                "documentation": documentation,
                "options": {
                    "docStyle": "JSDoc",  # or 'TSDoc' for TypeScript
                    "insertSpacing": True,
                    "preserveExisting": True
                }
            }

            result = self._run_script(
                script_name="js_ts_inserter.js",
                input_data=input_data,
                error_message="Error inserting documentation"
            )

            if result and isinstance(result, str):
                return result

            return code  # Return original code if insertion fails

        except Exception as e:
            logger.error(f"Error inserting documentation: {e}", exc_info=True)
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates JavaScript/TypeScript code syntax.

        Args:
            code (str): Code to validate.
            file_path (Optional[str]): Path to the source file.

        Returns:
            bool: True if code is valid, False otherwise.
        """
        try:
            input_data = {
                "code": code,
                "language": self._get_language(file_path),
                "filePath": file_path or "unknown"
            }

            result = self._run_script(
                script_name="js_ts_validator.js",
                input_data=input_data,
                error_message=f"Error validating {file_path}"
            )

            return bool(result and result.get("isValid", False))

        except Exception as e:
            logger.error(f"Error during validation: {e}", exc_info=True)
            return False

    def _run_script(
        self,
        script_name: str,
        input_data: Dict[str, Any],
        error_message: str,
        timeout: int = 30
    ) -> Any:
        """
        Runs a Node.js script with the given input data.

        Args:
            script_name (str): Name of the script to run.
            input_data (Dict[str, Any]): Input data for the script.
            error_message (str): Error message prefix for logging.
            timeout (int): Timeout in seconds.

        Returns:
            Any: Script output or None on failure.
        """
        script_path = os.path.join(self.script_dir, script_name)
        
        try:
            # Ensure proper string encoding
            input_json = json.dumps(input_data, ensure_ascii=False)

            process = subprocess.run(
                ["node", script_path],
                input=input_json,
                encoding='utf-8',
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout
            )

            if process.returncode == 0:
                try:
                    return json.loads(process.stdout)
                except json.JSONDecodeError:
                    if script_name == "js_ts_inserter.js":
                        return process.stdout  # Return raw output for inserter
                    logger.error(f"{error_message}: Invalid JSON output")
                    return None
            else:
                logger.error(f"{error_message}: {process.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"{error_message}: Script timeout after {timeout}s")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"{error_message}: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"{error_message}: {str(e)}")
            return None

    def _get_language(self, file_path: Optional[str]) -> str:
        """Determines the language based on file extension."""
        if not file_path:
            return "javascript"
        ext = os.path.splitext(file_path)[1].lower()
        return "typescript" if ext in [".ts", ".tsx"] else "javascript"

    def _should_skip_file(self, file_path: Optional[str]) -> bool:
        """Determines if a file should be skipped."""
        if not file_path:
            return False
            
        skip_patterns = [
            "node_modules",
            ".d.ts",
            ".test.",
            ".spec.",
            ".min.",
            "dist/",
            "build/"
        ]
        return any(pattern in file_path for pattern in skip_patterns)

    def _get_parser_plugins(self, language: str) -> list:
        """Gets the appropriate parser plugins based on language."""
        plugins = [
            "jsx",
            "decorators-legacy",
            ["decorators", { "decoratorsBeforeExport": True }],
            "classProperties",
            "classPrivateProperties",
            "classPrivateMethods",
            "exportDefaultFrom",
            "exportNamespaceFrom",
            "dynamicImport"
        ]
        
        if language == "typescript":
            plugins.append("typescript")
        
        return plugins

    def _get_empty_structure(self, reason: str = "") -> Dict[str, Any]:
        """Returns an empty structure with optional reason."""
        return {
            "classes": [],
            "functions": [],
            "variables": [],
            "constants": [],
            "summary": f"Skipped: {reason}" if reason else "Empty structure",
            "changes_made": [],
            "halstead": {
                "volume": 0,
                "difficulty": 0,
                "effort": 0
            },
            "maintainability_index": 0
        }