# language_functions/base_handler.py

import abc
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BaseHandler(abc.ABC):
    """Abstract base class for language-specific handlers."""

    @abc.abstractmethod
    def extract_structure(self, code: str) -> Dict[str, Any]:
        """Extracts the structure of the code (classes, functions, etc.)."""
        pass

    @abc.abstractmethod
    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """Inserts docstrings/comments into the code based on the documentation."""
        pass

    @abc.abstractmethod
    def validate_code(self, code: str) -> bool:
        """Validates the modified code for syntax correctness."""
        pass
