# language_functions/__init__.py

from typing import Dict, Any, Optional  # Import Dict, Any, and Optional
from language_functions.python_handler import PythonHandler
from language_functions.java_handler import JavaHandler
from language_functions.js_ts_handler import JSTsHandler
from language_functions.go_handler import GoHandler
from language_functions.cpp_handler import CppHandler
from language_functions.html_handler import HTMLHandler
from language_functions.css_handler import CSSHandler
from language_functions.base_handler import BaseHandler  # Ensure BaseHandler is imported

def get_handler(language: str, function_schema: Dict[str, Any]) -> Optional[BaseHandler]:
    """
    Factory function to retrieve the appropriate language handler.

    Args:
        language (str): The programming language.
        function_schema (Dict[str, Any]): The schema defining functions.

    Returns:
        Optional[BaseHandler]: The corresponding language handler or None if unsupported.
    """
    language = language.lower()
    if language == 'python':
        return PythonHandler(function_schema)
    elif language == 'java':
        return JavaHandler(function_schema)
    elif language in ['javascript', 'js', 'typescript', 'ts']:
        return JSTsHandler(function_schema)
    elif language == 'go':
        return GoHandler(function_schema)
    elif language in ['cpp', 'c++', 'cxx']:
        return CppHandler(function_schema)
    elif language in ['html', 'htm']:
        return HTMLHandler(function_schema)
    elif language == 'css':
        return CSSHandler(function_schema)
    else:
        return None
