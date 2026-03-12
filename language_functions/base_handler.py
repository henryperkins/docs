"""
base_handler.py

Abstract base class for language-specific handlers.
"""

from __future__ import annotations

import abc
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseHandler(abc.ABC):
    """
    Abstract base class for language-specific code handlers.

    Provides a common interface for extracting structure, inserting docstrings,
    and validating code.
    """

    def __init__(self, function_schema: Dict[str, Any], metrics_analyzer=None):
        self.function_schema = function_schema
        self.metrics_analyzer = metrics_analyzer

    @abc.abstractmethod
    async def extract_structure(self, code: str, file_path: str) -> Dict[str, Any]:
        raise NotImplementedError

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        logger.info("Inserting docstrings (base — no-op)...")
        return code

    @abc.abstractmethod
    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        raise NotImplementedError

    def _calculate_complexity(self, code: str) -> Optional[float]:
        return None
