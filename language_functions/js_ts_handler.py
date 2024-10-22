"""
Enhanced JavaScript/TypeScript Handler with comprehensive support for modern features.

This handler provides robust parsing, analysis, and documentation generation for JavaScript and TypeScript code,
including support for modern language features, React components, and detailed metrics calculation.
"""

import os
import logging
import subprocess
import json
import tempfile
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

from language_functions.base_handler import BaseHandler
from metrics import calculate_all_metrics

logger = logging.getLogger(__name__)

class JSDocStyle(Enum):
    """Enumeration of supported documentation styles."""
    JSDOC = "jsdoc"
    TSDOC = "tsdoc"

@dataclass
class MetricsResult:
    """Container for code metrics results."""
    complexity: int
    maintainability: float
    halstead: Dict[str, float]
    function_metrics: Dict[str, Dict[str, Any]]

class JSTsHandler(BaseHandler):
    """Handler for JavaScript and TypeScript languages with enhanced capabilities."""

    def __init__(self, function_schema: Dict[str, Any]):
        """Initialize the handler with configuration."""
        self.function_schema = function_schema
        self.script_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """
        Extracts detailed code structure with enhanced error handling and TypeScript support.

        Args:
            code (str): Source code to analyze
            file_path (str, optional): Path to the source file

        Returns:
            Dict[str, Any]: Comprehensive code structure and metrics
        """
        try:
            # Determine language and parser options
            is_typescript = self._is_typescript_file(file_path)
            parser_options = self._get_parser_options(is_typescript)
            
            # Prepare input for parser
            input_data = {
                "code": code,
                "language": "typescript" if is_typescript else "javascript",
                "filePath": file_path or "unknown",
                "options": parser_options
            }

            # Get metrics first
            metrics = self._calculate_metrics(code, is_typescript)

            # Run parser script
            structure = self._run_parser_script(input_data)
            if not structure:
                return self._get_empty_structure("Parser error")

            # Enhance structure with metrics
            structure.update({
                "halstead": metrics.halstead,
                "complexity": metrics.complexity,
                "maintainability_index": metrics.maintainability,
                "function_metrics": metrics.function_metrics
            })

            # Add React-specific analysis if needed
            if self._is_react_file(file_path):
                react_info = self._analyze_react_components(code, is_typescript)
                structure["react_components"] = react_info

            return structure

        except Exception as e:
            logger.error(f"Error extracting structure: {str(e)}", exc_info=True)
            return self._get_empty_structure(f"Error: {str(e)}")

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts JSDoc/TSDoc comments with improved formatting and type information.

        Args:
            code (str): Original source code
            documentation (Dict[str, Any]): Documentation to insert

        Returns:
            str: Modified source code with documentation
        """
        try:
            is_typescript = self._is_typescript_file(documentation.get("file_path"))
            doc_style = JSDocStyle.TSDOC if is_typescript else JSDocStyle.JSDOC

            input_data = {
                "code": code,
                "documentation": documentation,
                "language": "typescript" if is_typescript else "javascript",
                "options": {
                    "style": doc_style.value,
                    "includeTypes": is_typescript,
                    "preserveExisting": True
                }
            }

            return self._run_inserter_script(input_data) or code

        except Exception as e:
            logger.error(f"Error inserting documentation: {str(e)}", exc_info=True)
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates code using ESLint with TypeScript support.

        Args:
            code (str): Code to validate
            file_path (Optional[str]): Path to source file

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if not file_path:
                logger.warning("File path not provided for validation")
                return True

            is_typescript = self._is_typescript_file(file_path)
            config_path = self._get_eslint_config(is_typescript)

            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.ts' if is_typescript else '.js',
                encoding='utf-8',
                delete=False
            ) as tmp:
                tmp.write(code)
                temp_path = tmp.name

            try:
                result = subprocess.run(
                    ["eslint", "--config", config_path, temp_path],
                    capture_output=True,
                    text=True
                )
                return result.returncode == 0
            finally:
                os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Validation error: {str(e)}", exc_info=True)
            return False

    def _calculate_metrics(self, code: str, is_typescript: bool) -> MetricsResult:
        """Calculates comprehensive code metrics."""
        try:
            # Use typhonjs-escomplex for detailed metrics
            input_data = {
                "code": code,
                "options": {
                    "typescript": is_typescript,
                    "sourceType": "module",
                    "loc": True,
                    "cyclomatic": True,
                    "halstead": True,
                    "maintainability": True
                }
            }

            result = self._run_script(
                script_name="js_ts_metrics.js",
                input_data=input_data,
                error_message="Error calculating metrics"
            )

            if not result:
                return MetricsResult(0, 0.0, {}, {})

            return MetricsResult(
                complexity=result.get("complexity", 0),
                maintainability=result.get("maintainability", 0.0),
                halstead=result.get("halstead", {}),
                function_metrics=result.get("functions", {})
            )

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
            return MetricsResult(0, 0.0, {}, {})

    def _analyze_react_components(self, code: str, is_typescript: bool) -> Dict[str, Any]:
        """Analyzes React components and their properties."""
        try:
            input_data = {
                "code": code,
                "options": {
                    "typescript": is_typescript,
                    "plugins": ["jsx", "react"]
                }
            }

            return self._run_script(
                script_name="react_analyzer.js",
                input_data=input_data,
                error_message="Error analyzing React components"
            ) or {}

        except Exception as e:
            logger.error(f"Error analyzing React components: {str(e)}", exc_info=True)
            return {}

    def _get_parser_options(self, is_typescript: bool) -> Dict[str, Any]:
        """Gets appropriate parser options based on file type."""
        options = {
            "sourceType": "module",
            "plugins": [
                "jsx",
                "decorators-legacy",
                ["decorators", { "decoratorsBeforeExport": True }],
                "classProperties",
                "classPrivateProperties",
                "classPrivateMethods",
                "exportDefaultFrom",
                "exportNamespaceFrom",
                "dynamicImport",
                "nullishCoalescing",
                "optionalChaining",
            ]
        }
        
        if is_typescript:
            options["plugins"].extend([
                "typescript",
                "decorators-legacy",
                "classProperties"
            ])

        return options

    @staticmethod
    def _is_typescript_file(file_path: Optional[str]) -> bool:
        """Determines if a file is TypeScript based on extension."""
        if not file_path:
            return False
        return file_path.lower().endswith(('.ts', '.tsx'))

    @staticmethod
    def _is_react_file(file_path: Optional[str]) -> bool:
        """Determines if a file contains React components."""
        if not file_path:
            return False
        return file_path.lower().endswith(('.jsx', '.tsx'))

    def _get_eslint_config(self, is_typescript: bool) -> str:
        """Gets appropriate ESLint configuration file path."""
        config_name = '.eslintrc.typescript.json' if is_typescript else '.eslintrc.json'
        return os.path.join(self.script_dir, config_name)

    def _get_empty_structure(self, reason: str = "") -> Dict[str, Any]:
        """Returns an empty structure with optional reason."""
        return {
            "classes": [],
            "functions": [],
            "variables": [],
            "constants": [],
            "imports": [],
            "exports": [],
            "react_components": [],
            "summary": f"Empty structure: {reason}" if reason else "Empty structure",
            "halstead": {
                "volume": 0,
                "difficulty": 0,
                "effort": 0
            },
            "complexity": 0,
            "maintainability_index": 0
        }

    def _run_parser_script(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Runs the parser script with error handling."""
        return self._run_script(
            script_name="js_ts_parser.js",
            input_data=input_data,
            error_message="Error running parser"
        )

    def _run_inserter_script(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Runs the documentation inserter script with error handling."""
        return self._run_script(
            script_name="js_ts_inserter.js",
            input_data=input_data,
            error_message="Error running inserter"
        )

    def _run_script(
        self,
        script_name: str,
        input_data: Dict[str, Any],
        error_message: str,
        timeout: int = 30
    ) -> Any:
        """
        Runs a Node.js script with robust error handling.

        Args:
            script_name (str): Name of the script to run
            input_data (Dict[str, Any]): Input data for the script
            error_message (str): Error message prefix
            timeout (int): Timeout in seconds

        Returns:
            Any: Script output or None on failure
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