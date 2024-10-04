# language_functions/__init__.py

from typing import Optional
from language_functions.python_handler import PythonHandler
from language_functions.java_handler import JavaHandler
from language_functions.js_ts_handler import JSTsHandler
from language_functions.html_handler import HTMLHandler
from language_functions.css_handler import CSSHandler
from language_functions.go_handler import GoHandler  # Add this line
from language_functions.base_handler import BaseHandler

def get_handler(language: str) -> Optional[BaseHandler]:
    """
    Factory function to retrieve the appropriate language handler.
    
    Args:
        language (str): The programming language.
    
    Returns:
        Optional[BaseHandler]: The corresponding language handler or None if unsupported.
    """
    language = language.lower()
    if language == "python":
        return PythonHandler()
    elif language == "java":
        return JavaHandler()
    elif language in ["javascript", "typescript"]:
        return JSTsHandler()
    elif language in ["html", "htm"]:
        return HTMLHandler()
    elif language == "css":
        return CSSHandler()
    else:
        return None
