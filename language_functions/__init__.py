"""
language_functions Package

This package provides language-specific handlers for extracting code structures,
inserting documentation comments (docstrings), and validating code across various
programming languages. It includes handlers for languages such as Python, Java,
JavaScript/TypeScript, Go, C++, HTML, and CSS.

Modules:
    - python_handler.py: Handler for Python code.
    - java_handler.py: Handler for Java code.
    - js_ts_handler.py: Handler for JavaScript and TypeScript code.
    - go_handler.py: Handler for Go code.
    - cpp_handler.py: Handler for C++ code.
    - html_handler.py: Handler for HTML code.
    - css_handler.py: Handler for CSS code.
    - base_handler.py: Abstract base class defining the interface for all handlers.

Functions:
    - get_handler(language, function_schema): Factory function to retrieve the appropriate language handler.

Example:
    ```python
    from language_functions import get_handler
    from utils import load_function_schema

    function_schema = load_function_schema('path/to/schema.json')
    handler = get_handler('python', function_schema)
    if handler:
        updated_code = handler.insert_docstrings(original_code, documentation)
    ```
"""

import logging
from typing import Dict, Any, Optional

from .python_handler import PythonHandler
from .java_handler import JavaHandler
from .js_ts_handler import JSTsHandler
from .go_handler import GoHandler
from .cpp_handler import CppHandler
from .html_handler import HTMLHandler
from .css_handler import CSSHandler
from .base_handler import BaseHandler

logger = logging.getLogger(__name__)


def get_handler(language: str, function_schema: Dict[str, Any]) -> Optional[BaseHandler]:
    """
    Factory function to retrieve the appropriate language handler.

    This function matches the provided programming language with its corresponding handler class.
    If the language is supported, it returns an instance of the handler initialized with the given
    function schema. If the language is unsupported, it logs a warning and returns None.

    Args:
        language (str): The programming language of the source code (e.g., "python", "java", "javascript").
        function_schema (Dict[str, Any]): The schema defining functions and their documentation structure.

    Returns:
        Optional[BaseHandler]: An instance of a subclass of `BaseHandler` tailored to the specified language
                               or `None` if the language is unsupported.
    """
    if function_schema is None:
        logger.error("Function schema is None. Cannot retrieve handler.")
        return None

    # Normalize the language string to lowercase to ensure case-insensitive matching
    language = language.lower()

    # Mapping of language identifiers to their corresponding handler classes
    if language == "python":
        return PythonHandler(function_schema)
    elif language == "java":
        return JavaHandler(function_schema)
    elif language in ["javascript", "js", "typescript", "ts"]:
        return JSTsHandler(function_schema)
    elif language == "go":
        return GoHandler(function_schema)
    elif language in ["cpp", "c++", "cxx"]:
        return CppHandler(function_schema)
    elif language in ["html", "htm"]:
        return HTMLHandler(function_schema)
    elif language == "css":
        return CSSHandler(function_schema)
    else:
        logger.warning(f"No handler available for language: {language}")
        return None
