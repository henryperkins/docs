# Module: extraction_manager

## Overview
**File:** `simple2/extract/extraction_manager.py`
**Description:** import ast
from typing import Dict, Any, Optional, List, Tuple
from core.logger import log_info, log_error, log_debug
from extract.classes import ClassExtractor
from extract.functions import FunctionExtractor


class ExtractionManager:
    """
    Enhanced extraction manager with support for exception classes and robust error handling.
    """

    def extract_metadata(self, source_code: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract metadata from source code with special handling for exception classes.

        Args:
            source_code (str): The source code to analyze.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Extracted metadata for classes and functions.
        """
        try:
            log_debug("Starting metadata extraction")

            # Parse the source code into an AST
            tree = ast.parse(source_code)
            log_debug("AST parsing complete")

            # Use ClassExtractor and FunctionExtractor
            class_extractor = ClassExtractor(source_code)
            function_extractor = FunctionExtractor(source_code)

            classes = class_extractor.extract_classes()
            functions = function_extractor.extract_functions()

            log_info(
                f"Extraction complete. Found {len(classes)} classes and {len(functions)} functions")
            return {
                'classes': classes,
                'functions': functions
            }

        except SyntaxError as e:
            log_error(f"Syntax error in source code: {e}")
            return {'classes': [], 'functions': []}
        except Exception as e:
            log_error(f"Failed to extract metadata: {str(e)}")
            return {'classes': [], 'functions': []}

    def detect_exceptions(self, function_node: ast.FunctionDef) -> List[str]:
        """Public method to detect exceptions in a function node.

        Args:
            function_node (ast.FunctionDef): The function node to analyze.

        Returns:
            List[str]: A list of detected exceptions.
        """
        return self._detect_exceptions(function_node)

    def _detect_exceptions(self, function_node: ast.FunctionDef) -> List[str]:
        """Protected method to detect exceptions in a function node.

        Args:
            function_node (ast.FunctionDef): The function node to analyze.

        Returns:
            List[str]: A list of detected exceptions.
        """
        detected_exceptions = []
        for node in ast.walk(function_node):
            if isinstance(node, ast.Raise):
                if isinstance(node.exc, ast.Name):
                    detected_exceptions.append(node.exc.id)
                elif isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                    detected_exceptions.append(node.exc.func.id)
        return detected_exceptions


## AI-Generated Documentation


**Summary:** No summary provided.


**Description:** No description provided.



## Classes

| Class | Inherits From | Complexity Score* |
|-------|---------------|-------------------|
| `ExtractionManager` | `` | 0 |

### Class Methods

| Class | Method | Parameters | Returns | Complexity Score* |
|-------|--------|------------|---------|-------------------|
| `ExtractionManager` | `extract_metadata` | `(self: Any, source_code: str)` | `Dict[(str, List[Dict[(str, Any)]])]` | 0 |
| `ExtractionManager` | `detect_exceptions` | `(self: Any, function_node: ast.FunctionDef)` | `List[str]` | 0 |

## Source Code
```python
import ast
from typing import Dict, Any, Optional, List, Tuple
from core.logger import log_info, log_error, log_debug
from extract.classes import ClassExtractor
from extract.functions import FunctionExtractor


class ExtractionManager:
    """
    Enhanced extraction manager with support for exception classes and robust error handling.
    """

    def extract_metadata(self, source_code: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract metadata from source code with special handling for exception classes.

        Args:
            source_code (str): The source code to analyze.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Extracted metadata for classes and functions.
        """
        try:
            log_debug("Starting metadata extraction")

            # Parse the source code into an AST
            tree = ast.parse(source_code)
            log_debug("AST parsing complete")

            # Use ClassExtractor and FunctionExtractor
            class_extractor = ClassExtractor(source_code)
            function_extractor = FunctionExtractor(source_code)

            classes = class_extractor.extract_classes()
            functions = function_extractor.extract_functions()

            log_info(
                f"Extraction complete. Found {len(classes)} classes and {len(functions)} functions")
            return {
                'classes': classes,
                'functions': functions
            }

        except SyntaxError as e:
            log_error(f"Syntax error in source code: {e}")
            return {'classes': [], 'functions': []}
        except Exception as e:
            log_error(f"Failed to extract metadata: {str(e)}")
            return {'classes': [], 'functions': []}

    def detect_exceptions(self, function_node: ast.FunctionDef) -> List[str]:
        """Public method to detect exceptions in a function node.

        Args:
            function_node (ast.FunctionDef): The function node to analyze.

        Returns:
            List[str]: A list of detected exceptions.
        """
        return self._detect_exceptions(function_node)

    def _detect_exceptions(self, function_node: ast.FunctionDef) -> List[str]:
        """Protected method to detect exceptions in a function node.

        Args:
            function_node (ast.FunctionDef): The function node to analyze.

        Returns:
            List[str]: A list of detected exceptions.
        """
        detected_exceptions = []
        for node in ast.walk(function_node):
            if isinstance(node, ast.Raise):
                if isinstance(node.exc, ast.Name):
                    detected_exceptions.append(node.exc.id)
                elif isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                    detected_exceptions.append(node.exc.func.id)
        return detected_exceptions

```