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
from .language_functions import insert_docstrings  # Import the function
from metrics import MetricsAnalyzer  # Import MetricsAnalyzer

logger = logging.getLogger(__name__)

__all__ = ["get_handler", "insert_docstrings"]

def get_handler(language: str, function_schema: Dict[str, Any], metrics_analyzer: MetricsAnalyzer) -> Optional[BaseHandler]:
    """
    Factory function to retrieve the appropriate language handler.

    Args:
        language (str): The programming language of the source code.
        function_schema (Dict[str, Any]): The schema defining functions.
        metrics_analyzer (MetricsAnalyzer): The metrics analyzer object.

    Returns:
        Optional[BaseHandler]: An instance of the corresponding language handler or None if unsupported.
    """
    if function_schema is None:
        logger.error("Function schema is None. Cannot retrieve handler.")
        return None

    language = language.lower()

    # Map of supported languages to their handlers
    handlers = {
        "python": PythonHandler,
        "javascript": JSTsHandler,
        "js": JSTsHandler,
        "typescript": JSTsHandler,
        "ts": JSTsHandler,
        # ... (Add other language handlers here when you're ready)
    }

    handler_class = handlers.get(language)
    if handler_class:
        return handler_class(function_schema, metrics_analyzer)
    else:
        logger.debug(f"No handler available for language: {language}")
        return None  # Return None instead of raising an exc
    