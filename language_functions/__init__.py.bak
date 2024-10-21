# language_functions/__init__.py

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

    Args:
        language (str): The programming language.
        function_schema (Dict[str, Any]): The schema defining functions.

    Returns:
        Optional[BaseHandler]: The corresponding language handler or None if unsupported.
    """
    if function_schema is None:
        logger.error("Function schema is None. Cannot retrieve handler.")
        return None

    language = language.lower()
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
