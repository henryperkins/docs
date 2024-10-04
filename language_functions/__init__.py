# language_functions/__init__.py

from .python_handler import insert_python_docstrings
from .java_handler import insert_javadoc_docstrings
from .js_ts_handler import insert_jsdoc_comments
from .html_handler import insert_html_comments
from .css_handler import insert_css_comments
import logging

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
        elif language == "java":
            modified_code = insert_javadoc_docstrings(original_code, documentation, language)
        elif language in ["javascript", "typescript"]:
            modified_code = insert_jsdoc_comments(original_code, documentation, language)
        elif language in ["html", "htm"]:
            modified_code = insert_html_comments(original_code, documentation)
        elif language == "css":
            modified_code = insert_css_comments(original_code, documentation)
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
