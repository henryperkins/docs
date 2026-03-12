"""
language_functions Package

Provides language-specific handlers for extracting code structures,
inserting documentation, and validating code.
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

__all__ = ["get_handler", "BaseHandler"]

_HANDLER_MAP = {
    "python": PythonHandler,
    "java": JavaHandler,
    "javascript": JSTsHandler,
    "js": JSTsHandler,
    "typescript": JSTsHandler,
    "ts": JSTsHandler,
    "go": GoHandler,
    "cpp": CppHandler,
    "c++": CppHandler,
    "cxx": CppHandler,
    "html": HTMLHandler,
    "htm": HTMLHandler,
    "css": CSSHandler,
}


def get_handler(
    language: str,
    function_schema: Dict[str, Any],
    metrics_analyzer=None,
) -> Optional[BaseHandler]:
    """Factory function to retrieve the appropriate language handler."""
    if function_schema is None:
        logger.error("Function schema is None. Cannot retrieve handler.")
        return None

    handler_class = _HANDLER_MAP.get(language.lower())
    if handler_class:
        return handler_class(function_schema, metrics_analyzer)

    logger.debug(f"No handler available for language: {language}")
    return None
