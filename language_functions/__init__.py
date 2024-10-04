# language_functions/__init__.py

from typing import Optional, Dict, Any
from language_functions.python_handler import PythonHandler
from language_functions.java_handler import JavaHandler
from language_functions.js_ts_handler import JSTsHandler
from language_functions.html_handler import HTMLHandler
from language_functions.css_handler import CSSHandler
import logging

logger = logging.getLogger(__name__)

def get_handler(language: str) -> Optional[BaseHandler]:
    """
    Returns the appropriate handler based on the language.
    
    Parameters:
        language (str): The programming language.
    
    Returns:
        Optional[BaseHandler]: An instance of the language handler or None if unsupported.
    """
    handlers = {
        "python": PythonHandler(),
        "java": JavaHandler(),
        "javascript": JSTsHandler(),
        "typescript": JSTsHandler(),
        "html": HTMLHandler(),
        "css": CSSHandler(),
    }
    handler = handlers.get(language.lower())
    if not handler:
        logger.warning(f"No handler found for language '{language}'.")
    return handler
