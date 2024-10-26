# shared_functions.py

"""
shared_functions.py

Provides shared functions for both Gemini and Azure model integrations,
such as token encoding, prompt token calculation, logging setup, and 
data transformation utilities.
"""

import json
import logging
from typing import List, Dict, Any
from token_utils import TokenManager

logger = logging.getLogger(__name__)

# Define default thresholds and configurations
DEFAULT_COMPLEXITY_THRESHOLDS = {"low": 5, "medium": 10, "high": 15}
DEFAULT_HALSTEAD_THRESHOLDS = {
    "volume": {"low": 500, "medium": 1000, "high": 2000},
    "difficulty": {"low": 5, "medium": 10, "high": 20},
    "effort": {"low": 500, "medium": 1000, "high": 2000},
}
DEFAULT_MAINTAINABILITY_THRESHOLDS = {"low": 65, "medium": 85, "high": 100}

def calculate_prompt_tokens(base_info: str, context: str, chunk_content: str, schema: str) -> int:
    """
    Calculates total tokens for the prompt content using TokenManager.

    Args:
        base_info: Project and style information
        context: Related code/documentation
        chunk_content: Content of the chunk being documented
        schema: JSON schema

    Returns:
        Total token count
    """
    total = 0
    for text in [base_info, context, chunk_content, schema]:
        token_result = TokenManager.count_tokens(text)
        total += token_result.token_count
    return total

def format_prompt(base_info: str, context: str, chunk_content: str, schema: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generates prompt messages to be used in the documentation generation request.

    Args:
        base_info: Information about the project
        context: Contextual information from other parts of the code
        chunk_content: Code chunk content
        schema: JSON schema for structure

    Returns:
        List of messages as prompt input
    """
    schema_str = json.dumps(schema, indent=2)
    return [
        {"role": "system", "content": base_info},
        {"role": "user", "content": context},
        {"role": "assistant", "content": chunk_content},
        {"role": "schema", "content": schema_str}
    ]

def log_error(message: str, exc: Exception):
    """Logs errors with traceback information."""
    logger.error(f"{message}: {str(exc)}", exc_info=True)