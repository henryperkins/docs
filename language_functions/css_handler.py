# language_functions/css_handler.py

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def insert_css_comments(original_code: str, documentation: Dict[str, Any]) -> str:
    """
    Inserts comments into CSS code based on provided documentation to enhance understanding.

    Parameters:
        original_code (str): The original CSS source code.
        documentation (Dict[str, Any]): A dictionary containing documentation details for comments.

    Returns:
        str: The modified CSS source code with inserted comments.
    """
    logger.debug("Starting insert_css_comments")
    try:
        summary = documentation.get("summary", "").strip()
        changes = documentation.get("changes_made", [])
        if not summary and not changes:
            logger.warning(
                "No summary or changes provided in documentation. Skipping comment insertion."
            )
            return original_code
        new_comment_parts = []
        if summary:
            new_comment_parts.append(f"Summary: {summary}")
        if changes:
            changes_formatted = "; ".join(changes)
            new_comment_parts.append(f"Changes: {changes_formatted}")
        new_comment_text = "/* " + " | ".join(new_comment_parts) + " */\n"
        modified_code = new_comment_text + original_code
        logger.debug("Inserted new CSS comment at the beginning of the file.")
        logger.debug("Completed inserting CSS comments")
        return modified_code
    except Exception as e:
        logger.error(f"Error inserting CSS comments: {e}", exc_info=True)
        return original_code
