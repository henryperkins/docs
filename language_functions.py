# language_functions.py

import os
import sys
import json
import ast
import subprocess
import logging
import astor
import tempfile
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup, Comment
import tinycss2
import javalang  # Ensure javalang is installed
from parser.java_parser import extract_structure, insert_docstrings
from language_functions.python_handler import insert_python_docstrings
from language_functions.java_handler import insert_javadoc_docstrings
# Import other handlers as they are implemented (e.g., js_ts_handler, html_handler, css_handler)

logger = logging.getLogger(__name__)


def insert_docstrings(
    original_code: str, documentation: Dict[str, Any], language: str
) -> str:
    """
    Inserts docstrings/comments into code based on the language.

    Parameters:
        original_code (str): The original source code.
        documentation (Dict[str, Any]): Documentation details obtained from AI.
        language (str): Programming language of the source code.

    Returns:
        str: The source code with inserted documentation.
    """
    logger.debug(f"Processing docstrings for language: {language}")
    try:
        if language == "python":
            modified_code = insert_python_docstrings(original_code, documentation)
            # Additional Python-specific formatting or validation can be done here
        elif language == "java":
            modified_code = insert_javadoc_docstrings(original_code, documentation, language)
            # Additional Java-specific formatting or validation can be done here
        elif language in ["javascript", "typescript"]:
            # Placeholder for JS/TS handler
            from language_functions.js_ts_handler import insert_js_ts_docstrings
            modified_code = insert_js_ts_docstrings(original_code, documentation, language)
        elif language == "html":
            # Placeholder for HTML handler
            from language_functions.html_handler import insert_html_comments
            modified_code = insert_html_comments(original_code, documentation)
        elif language == "css":
            # Placeholder for CSS handler
            from language_functions.css_handler import insert_css_docstrings
            modified_code = insert_css_docstrings(original_code, documentation)
        else:
            logger.warning(
                f"Unsupported language '{language}'. Skipping documentation insertion."
            )
            modified_code = original_code
        return modified_code
    except Exception as e:
        logger.error(
            f"Error processing docstrings for language '{language}': {e}", exc_info=True
        )
        return original_code
