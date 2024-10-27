"""
base_handler.py

This module defines the abstract base class `BaseHandler` for language-specific handlers.
Each handler is responsible for extracting code structure, inserting docstrings/comments,
and validating code for a specific programming language.

Classes:
    - BaseHandler: Abstract base class defining the interface for all language handlers.
"""

from __future__ import annotations  # For forward references in type hints
import abc
import logging
from typing import Dict, Any, Optional, List

from metrics import MetricsAnalyzer  # Import for type hinting

logger = logging.getLogger(__name__)

class BaseLanguageHandler(ABC):
    """
    Abstract base class for language-specific code handlers.

    Provides a common interface and shared functionality for handling different programming languages
    in the documentation generation process.  Subclasses must implement the `extract_structure` and
    `validate_code` methods.
    """

	 def __init__(self, function_schema: Dict[str, Any], metrics_analyzer: MetricsAnalyzer):
        """
        Initializes the BaseLanguageHandler.

        Args:
            function_schema (Dict[str, Any]): The schema defining functions and their documentation structure.
            metrics_analyzer: The metrics analyzer object for collecting code metrics.
        """
        self.function_schema = function_schema
        self.metrics_analyzer = metrics_analyzer

    @abc.abstractmethod
    async def extract_structure(self, code: str, file_path: str) -> Dict[str, Any]:
        """
        Extracts the structure of the code (classes, functions, etc.).

        Subclasses must implement this method to parse the source code and identify key components
        such as classes, functions, methods, variables, and other relevant elements.

        Args:
            code (str): The source code to analyze.
            file_path (str): Path to the source file.

        Returns:
            Dict[str, Any]: A dictionary representing the code structure, including details
                            like classes, functions, variables, and their attributes.  Should conform
                            to the provided `function_schema`.
        """
        raise NotImplementedError
        
    def insert_docstrings(self, code: str, documentation: Dict[str, Any], docstring_format: str = "default") -> str:
        """
        Inserts docstrings into the code based on the documentation.

        Provides a default implementation that logs the docstring insertion attempt. Subclasses can override
        this method to implement language-specific docstring insertion logic.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.
            docstring_format (str, optional): The format of the docstrings to be inserted. Defaults to "default".

        Returns:
            str: The source code, potentially with inserted documentation. The default implementation returns
                 the original code unchanged.
        """
        logger.info(f"Inserting docstrings (format: {docstring_format})...")
        return code  # Return the original code if no specific logic is implemented

    @abc.abstractmethod
	 def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates the modified code for syntax correctness.

        Subclasses must implement this method to ensure that the code remains syntactically correct after
        inserting docstrings/comments. It may involve compiling the code or running
        language-specific linters/validators.

        Args:
            code (str): The modified source code.
            file_path (Optional[str]): Path to the source file (optional).

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        raise NotImplementedError
	 @abc.abstractmethod # Make this abstract if all handlers need to implement it
    def _calculate_complexity(self, code: str) -> Optional[float]:
        """
        Calculates code complexity.

        This method provides a default implementation that returns None. Subclasses can override
        this method to provide language-specific complexity calculations.

        Args:
            code (str): The source code to analyze.

        Returns:
            Optional[float]: The calculated complexity, or None if not implemented.
        """
        raise NotImplementedError