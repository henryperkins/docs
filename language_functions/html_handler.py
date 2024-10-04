# language_functions/html_handler.py

import logging
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)

def insert_html_comments(original_code: str, documentation: Dict[str, Any]) -> str:
    """
    Inserts comments into HTML code based on provided documentation to improve clarity.

    Parameters:
        original_code (str): The original HTML source code.
        documentation (Dict[str, Any]): A dictionary containing documentation details for comments.

    Returns:
        str: The modified HTML source code with inserted comments.
    """
    logger.debug("Starting insert_html_comments")
    try:
        soup = BeautifulSoup(original_code, "html.parser")
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
        new_comment_text = " | ".join(new_comment_parts)
        comment = Comment(f" {new_comment_text} ")
        if soup.body:
            soup.body.insert(0, comment)
            logger.debug("Inserted comment at the beginning of the body.")
        else:
            soup.insert(0, comment)
            logger.debug("Inserted comment at the beginning of the document.")
        modified_code = str(soup)
        logger.debug("Completed inserting HTML comments")
        return modified_code
    except Exception as e:
        logger.error(f"Error inserting HTML comments: {e}", exc_info=True)
        return original_code
