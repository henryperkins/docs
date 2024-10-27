# js_ts_handler.py

import os
import logging
import subprocess
import json
import tempfile
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from jsonschema import validate, ValidationError

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class JSDocStyle(Enum):
    JSDOC = "jsdoc"
    TSDOC = "tsdoc"

@dataclass
class MetricsResult:
    complexity: int
    maintainability: float
    halstead: Dict[str, float]
    function_metrics: Dict[str, Dict[str, Any]]

class JSTsHandler(BaseHandler):

    def __init__(self, function_schema: Dict[str, Any]):
        self.function_schema = function_schema
        self.script_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")

    async def extract_structure(self, code: str, file_path: str, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extracts the structure of the JavaScript/TypeScript code.

        Checklist:
        - [x] Parsing: Uses external js_ts_parser.js script.
        - [x] Data Structure: Conforms to function_schema.json.
        - [x] Schema Validation: Implemented using jsonschema.validate.
        - [x] Metrics Calculation: Uses external js_ts_metrics.js script.
        - [x] Language-Specific Features: Extracts React components.
        """
        logger.info(f"Extracting structure for file: {file_path}")
        try:
            is_typescript = self._is_typescript_file(file_path)
            parser_options = self._get_parser_options(is_typescript)
            input_data = {
                "code": code,
                "language": "typescript" if is_typescript else "javascript",
                "filePath": file_path or "unknown",
                "options": parser_options
            }

            # Get metrics
            metrics_result = self._calculate_metrics(code, is_typescript)
            if metrics_result is None:
                return self._get_empty_structure("Metrics calculation failed")

            # Run parser script
            parsed_data = self._run_parser_script(input_data)
            if parsed_data is None:
                return self._get_empty_structure("Parsing failed")

            # Map parsed data to function_schema.json structure
            structured_data = {
                "docstring_format": "JSDoc" if not is_typescript else "TSDoc",
                "summary": parsed_data.get("summary", ""),
                "changes_made": [],  # Placeholder for changelog
                "functions": self._map_functions(parsed_data.get("functions", []), metrics_result.function_metrics),
                "classes": self._map_classes(parsed_data.get("classes", []), metrics_result.function_metrics),
                "variables": parsed_data.get("variables", []),
                "constants": parsed_data.get("constants", []),
                "imports": parsed_data.get("imports", []),
                "metrics": {
                    "complexity": metrics_result.complexity,
                    "halstead": metrics_result.halstead,
                    "maintainability_index": metrics_result.maintainability,
                }
            }

            # React analysis
            if self._is_react_file(file_path):
                react_info = self._analyze_react_components(code, is_typescript)
                if react_info is not None:
                    structured_data["react_components"] = react_info

            # Schema validation
            try:
                validate(instance=structured_data, schema=self.function_schema)
            except ValidationError as e:
                logger.warning(f"Schema validation failed: {e}")

            return structured_data

        except Exception as e:
            logger.error(f"Error extracting structure: {str(e)}", exc_info=True)
            return self._get_empty_structure(f"Error: {str(e)}")

    def _map_functions(self, functions: List[Dict], function_metrics: Dict) -> List[Dict]:
        """Maps function data to the schema."""
        mapped_functions = []
        for func in functions:
            func_name = func.get("name", "")
            metrics = function_metrics.get(func_name, {})
            mapped_functions.append({
                "name": func_name,
                "docstring": func.get("docstring", ""),
                "args": func.get("params", []),
                "async": func.get("async", False),
                "returns": {"type": func.get("returnType", ""), "description": ""},  # Map return type
                "complexity": metrics.get("complexity", 0),
                "halstead": metrics.get("halstead", {})
            })
        return mapped_functions

    def _map_classes(self, classes: List[Dict], function_metrics: Dict) -> List[Dict]:
        """Maps class data to the schema."""
        mapped_classes = []
        for cls in classes:
            mapped_methods = self._map_functions(cls.get("methods", []), function_metrics)
            mapped_classes.append({
                "name": cls.get("name", ""),
                "docstring": cls.get("docstring", ""),
                "methods": mapped_methods
            })
        return mapped_classes

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts JSDoc/TSDoc comments into JavaScript/TypeScript code.

        Checklist:
        - [x] Docstring Generation: Generates JSDoc/TSDoc style comments.
        - [x] Docstring Formats: Handles JSDoc and TSDoc based on file type.
        - [x] Insertion Method: Uses external js_ts_inserter.js script.
        - [x] Error Handling: Includes error handling and logging.
        - [x] Preservation of Existing Docstrings: Controlled by script options.
        """
        logger.info("Inserting docstrings...")
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
                    "preserveExisting": True  # Or False, depending on your requirement
                }
            }

            updated_code = self._run_inserter_script(input_data)
            return updated_code if updated_code is not None else code

        except Exception as e:
            logger.error(f"Error inserting documentation: {str(e)}", exc_info=True)
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates JavaScript/TypeScript code using ESLint.

        Checklist:
        - [x] Validation Tool: Uses ESLint.
        - [x] Error Handling: Handles validation errors.
        - [x] Temporary Files: Uses and cleans up temporary files.
        """
        logger.info("Validating code...")
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
                if result.returncode == 0:
                    logger.debug("ESLint validation passed.")
                else:
                    logger.error(f"ESLint validation failed: {result.stdout}\n{result.stderr}")
                return result.returncode == 0
            finally:
                try:
                    os.unlink(temp_path)
                except OSError as e:
                    logger.error(f"Error deleting temporary file {temp_path}: {e}")

        except Exception as e:
            logger.error(f"Validation error: {str(e)}", exc_info=True)
            return False

    def _calculate_metrics(self, code: str, is_typescript: bool) -> Optional[MetricsResult]:
        """Calculates code metrics using the js_ts_metrics.js script."""
        try:
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
                error_message="Metrics calculation failed"
            )
            logger.debug(f"Metrics calculation result: {result}")

            if result is None:
                logger.error("Metrics calculation returned None.")
                return None

            if not isinstance(result, dict):
                logger.error(f"Metrics result is not a dictionary: {type(result)}")
                return None

            required_keys = ["complexity", "maintainability", "halstead", "functions"]
            if not all(key in result for key in required_keys):
                missing_keys = [key for key in required_keys if key not in result]
                logger.error(f"Metrics result is missing keys: {missing_keys}")
                return None

            if not isinstance(result["halstead"], dict):
                logger.error("Halstead metrics should be a dictionary.")
                return None

            if not isinstance(result["functions"], dict):
                logger.error("Function metrics should be a dictionary.")
                return None

            return MetricsResult(
                complexity=result.get("complexity", 0),
                maintainability=result.get("maintainability", 0.0),
                halstead=result.get("halstead", {}),
                function_metrics=result.get("functions", {})
            )

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
            return None

    def _analyze_react_components(self, code: str, is_typescript: bool) -> Optional[Dict[str, Any]]:
        """Analyzes React components using the react_analyzer.js script."""
        try:
            input_data = {
                "code": code,
                "options": {
                    "typescript": is_typescript,
                    "plugins": ["jsx", "react"]
                }
            }
            result = self._run_script(
                script_name="react_analyzer.js",
                input_data=input_data,
                error_message="React analysis failed"
            )
            logger.debug(f"React analysis result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing React components: {str(e)}", exc_info=True)
            return None

    def _get_parser_options(self, is_typescript: bool) -> Dict[str, Any]:
        """Returns parser options for the js_ts_parser.js script."""
        options = {
            "sourceType": "module",
            "plugins": [
                "jsx",
                "decorators-legacy",
                ["decorators", {"decoratorsBeforeExport": True}],
                "classProperties",
                "classPrivateProperties",
                "classPrivateMethods",
                "exportDefaultFrom",
                "exportNamespaceFrom",
                "dynamicImport",
                "nullishCoalescingOperator",
                "optionalChaining",
            ]
        }

        if is_typescript:
            options["plugins"].extend([
                "typescript"
            ])

        return options

    def _is_typescript_file(self, file_path: Optional[str]) -> bool:
        """Checks if a file is a TypeScript file."""
        if not file_path:
            return False
        return file_path.lower().endswith(('.ts', '.tsx'))

    def _is_react_file(self, file_path: Optional[str]) -> bool:
        """Checks if a file is a React file (JSX or TSX)."""
        if not file_path:
            return False
        return file_path.lower().endswith(('.jsx', '.tsx'))

    def _get_eslint_config(self, is_typescript: bool) -> str:
        """Returns the path to the appropriate ESLint config file."""
        config_name = '.eslintrc.typescript.json' if is_typescript else '.eslintrc.json'
        return os.path.join(self.script_dir, config_name)

    def _get_empty_structure(self, reason: str = "") -> Dict[str, Any]:
        """Returns an empty structure dictionary with a reason."""
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
            "maintainability_index": 0,
            "function_metrics": {}
        }

    def _run_parser_script(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Runs the js_ts_parser.js script and returns the parsed data."""
        return self._run_script(
            script_name="js_ts_parser.js",
            input_data=input_data,
            error_message="Parsing failed"
        )

    def _run_inserter_script(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Runs the js_ts_inserter.js script and returns the updated code."""
        result = self._run_script(
            script_name="js_ts_inserter.js",
            input_data=input_data,
            error_message="Error running inserter"
        )
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return result.get("code")
        else:
            logger.error("Inserter script did not return code string.")
            return None

    def _run_script(self, script_name: str, input_data: Dict[str, Any], error_message: str) -> Any:
        """
        Runs a Node.js script with improved error handling and encoding management.
        """
        try:
            script_path = os.path.join(self.script_dir, script_name)
            if not os.path.isfile(script_path):
                logger.error(f"Script not found: {script_path}")
                return None

            logger.debug(f"Running script: {script_path} with input data: {input_data}")

            # Convert input data to JSON string with proper encoding handling
            try:
                input_json = json.dumps(input_data, ensure_ascii=False)
                input_bytes = input_json.encode('utf-8', errors='surrogateescape')
            except UnicodeEncodeError as e:
                logger.error(f"Unicode encoding error in input data: {e}", exc_info=True)
                return None

            process = subprocess.run(
                ['node', script_path],
                input=input_json,  # Pass the JSON string
                capture_output=True,
                text=True, 
                check=True,
                timeout=60
            )

            if process.returncode != 0:
                logger.error(f"{error_message}: {process.stderr}")
                return None

            output = process.stdout.strip()
            logger.debug(f"Script Output ({script_name}): {output}")

            try:
                return json.loads(output)
            except json.JSONDecodeError as e:
                if script_name == "js_ts_inserter.js":
                    # If inserter script returns plain code, not JSON
                    return output
                logger.error(f"{error_message}: Invalid JSON output. Error: {e}")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"{error_message}: Process error: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            logger.error(f"{error_message}: Script timed out after 60 seconds")
            return None
        except Exception as e:
            logger.error(f"{error_message}: Unexpected error: {e}", exc_info=True)
            return None
