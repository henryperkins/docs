# Module: functions

## Overview
**File:** `simple2/extract/functions.py`
**Description:** """
Function Extraction Module

This module provides functionality to extract function definitions and their metadata
from Python source code. It uses the Abstract Syntax Tree (AST) to analyze source code
and extract relevant information such as parameters, return types, and docstrings.

Version: 1.0.1
Author: Development Team
"""

import ast
from typing import List, Dict, Any, Optional, Tuple
from core.logger import log_info, log_error, log_debug
from extract.base import BaseExtractor
from core.utils import handle_exceptions  # Import the decorator from utils


class FunctionExtractor(BaseExtractor):
    """
    Extract function definitions and their metadata from Python source code.

    Inherits from BaseExtractor and provides methods to extract detailed information
    about functions, including parameters, return types, and docstrings.
    """

    @handle_exceptions(log_error)
    def extract_functions(
        self, source_code: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract all function definitions and their metadata from the source code.

        Parameters:
        source_code (str): The source code to analyze.

        Returns:
        list: A list of dictionaries containing function metadata.
        """
        if source_code:
            log_debug("Initializing FunctionExtractor with new source code.")
            self.__init__(source_code)

        log_debug("Starting extraction of function definitions.")
        functions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                function_info = self.extract_details(node)
                if function_info:
                    functions.append(function_info)
                    log_info(
                        f"Extracted function '{node.name}' with metadata.")

        log_debug(f"Total functions extracted: {len(functions)}")
        return functions

    @handle_exceptions(log_error)
    def extract_details(self, node: ast.AST) -> Dict[str, Any]:
        """
        Extract details from a function definition node.

        Args:
            node (ast.AST): The AST node to extract details from.

        Returns:
            dict: A dictionary containing function details.
        """
        if not isinstance(node, ast.FunctionDef):
            raise ValueError(
                f"Expected FunctionDef node, got {type(node).__name__}")

        log_debug(f"Extracting details for function: {node.name}")

        details = self._extract_common_details(node)
        details.update(
            {
                "args": self.extract_parameters(node),
                "return_type": self.extract_return_type(node),
                "decorators": self._extract_decorators(node),
                "exceptions": self._detect_exceptions(node),
                "body_summary": self.get_body_summary(node),
            }
        )
        log_debug(f"Successfully extracted details for function {node.name}")
        return details

    def extract_parameters(self, node: ast.FunctionDef) -> List[Tuple[str, str]]:
        """
        Extract parameters with type annotations and default values.

        Args:
            node (ast.FunctionDef): The function node to extract parameters from.

        Returns:
            list: A list of tuples containing parameter names and types.
        """
        parameters = []
        for arg in node.args.args:
            param_name = arg.arg
            param_type = self._get_type_annotation(arg)
            parameters.append((param_name, param_type))
        return parameters

    def extract_return_type(self, node: ast.FunctionDef) -> str:
        """
        Extract return type annotation from a function.

        Args:
            node (ast.FunctionDef): The function node to extract return type from.

        Returns:
            str: The return type annotation.
        """
        return ast.unparse(node.returns) if node.returns else "Any"

    def get_body_summary(self, node: ast.FunctionDef) -> str:
        """
        Generate a summary of the function body.

        Args:
            node (ast.FunctionDef): The function node to summarize.

        Returns:
            str: A summary of the function body.
        """
        return " ".join(ast.unparse(stmt) for stmt in node.body[:3]) + "..."


## AI-Generated Documentation


**Summary:** No summary provided.


**Description:** No description provided.



## Classes

| Class | Inherits From | Complexity Score* |
|-------|---------------|-------------------|
| `FunctionExtractor` | `BaseExtractor` | 0 |

### Class Methods

| Class | Method | Parameters | Returns | Complexity Score* |
|-------|--------|------------|---------|-------------------|
| `FunctionExtractor` | `extract_functions` | `(self: Any, source_code: Optional[str] = None)` | `List[Dict[(str, Any)]]` | 0 |
| `FunctionExtractor` | `extract_details` | `(self: Any, node: ast.AST)` | `Dict[(str, Any)]` | 0 |
| `FunctionExtractor` | `extract_parameters` | `(self: Any, node: ast.FunctionDef)` | `List[Tuple[(str, str)]]` | 0 |
| `FunctionExtractor` | `extract_return_type` | `(self: Any, node: ast.FunctionDef)` | `str` | 0 |
| `FunctionExtractor` | `get_body_summary` | `(self: Any, node: ast.FunctionDef)` | `str` | 0 |

## Source Code
```python
"""
Function Extraction Module

This module provides functionality to extract function definitions and their metadata
from Python source code. It uses the Abstract Syntax Tree (AST) to analyze source code
and extract relevant information such as parameters, return types, and docstrings.

Version: 1.0.1
Author: Development Team
"""

import ast
from typing import List, Dict, Any, Optional, Tuple
from core.logger import log_info, log_error, log_debug
from extract.base import BaseExtractor
from core.utils import handle_exceptions  # Import the decorator from utils


class FunctionExtractor(BaseExtractor):
    """
    Extract function definitions and their metadata from Python source code.

    Inherits from BaseExtractor and provides methods to extract detailed information
    about functions, including parameters, return types, and docstrings.
    """

    @handle_exceptions(log_error)
    def extract_functions(
        self, source_code: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract all function definitions and their metadata from the source code.

        Parameters:
        source_code (str): The source code to analyze.

        Returns:
        list: A list of dictionaries containing function metadata.
        """
        if source_code:
            log_debug("Initializing FunctionExtractor with new source code.")
            self.__init__(source_code)

        log_debug("Starting extraction of function definitions.")
        functions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                function_info = self.extract_details(node)
                if function_info:
                    functions.append(function_info)
                    log_info(
                        f"Extracted function '{node.name}' with metadata.")

        log_debug(f"Total functions extracted: {len(functions)}")
        return functions

    @handle_exceptions(log_error)
    def extract_details(self, node: ast.AST) -> Dict[str, Any]:
        """
        Extract details from a function definition node.

        Args:
            node (ast.AST): The AST node to extract details from.

        Returns:
            dict: A dictionary containing function details.
        """
        if not isinstance(node, ast.FunctionDef):
            raise ValueError(
                f"Expected FunctionDef node, got {type(node).__name__}")

        log_debug(f"Extracting details for function: {node.name}")

        details = self._extract_common_details(node)
        details.update(
            {
                "args": self.extract_parameters(node),
                "return_type": self.extract_return_type(node),
                "decorators": self._extract_decorators(node),
                "exceptions": self._detect_exceptions(node),
                "body_summary": self.get_body_summary(node),
            }
        )
        log_debug(f"Successfully extracted details for function {node.name}")
        return details

    def extract_parameters(self, node: ast.FunctionDef) -> List[Tuple[str, str]]:
        """
        Extract parameters with type annotations and default values.

        Args:
            node (ast.FunctionDef): The function node to extract parameters from.

        Returns:
            list: A list of tuples containing parameter names and types.
        """
        parameters = []
        for arg in node.args.args:
            param_name = arg.arg
            param_type = self._get_type_annotation(arg)
            parameters.append((param_name, param_type))
        return parameters

    def extract_return_type(self, node: ast.FunctionDef) -> str:
        """
        Extract return type annotation from a function.

        Args:
            node (ast.FunctionDef): The function node to extract return type from.

        Returns:
            str: The return type annotation.
        """
        return ast.unparse(node.returns) if node.returns else "Any"

    def get_body_summary(self, node: ast.FunctionDef) -> str:
        """
        Generate a summary of the function body.

        Args:
            node (ast.FunctionDef): The function node to summarize.

        Returns:
            str: A summary of the function body.
        """
        return " ".join(ast.unparse(stmt) for stmt in node.body[:3]) + "..."

```