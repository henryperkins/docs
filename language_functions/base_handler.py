# language_functions/base_handler.py

import abc
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaseHandler(abc.ABC):
    """Abstract base class for language-specific handlers."""

    @abc.abstractmethod
    def extract_structure(self, code: str, file_path: str) -> Dict[str, Any]:
        """
        Extracts the structure of the code (classes, functions, etc.).

        Args:
            code (str): The source code to analyze.
            file_path (str): Path to the source file.

        Returns:
            Dict[str, Any]: A dictionary representing the code structure.
        """
        pass

    @abc.abstractmethod
    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts docstrings/comments into the code based on the documentation.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The source code with inserted documentation.
        """
        pass

    @abc.abstractmethod
    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates the modified code for syntax correctness.

        Args:
            code (str): The modified source code.
            file_path (Optional[str]): Path to the source file (optional).

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        pass
