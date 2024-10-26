# results.py

"""
results.py

Contains data classes for storing processing results, such as FileProcessingResult.
These classes are used across multiple modules to standardize result storage and access.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class FileProcessingResult:
    """
    Stores the result of processing a file.

    Attributes:
        file_path: Path to the processed file
        success: Whether processing succeeded
        error: Error message if processing failed
        documentation: Generated documentation if successful
        metrics: Metrics about the processing
        chunk_count: Number of chunks processed
        successful_chunks: Number of successfully processed chunks
        timestamp: When the processing completed
    """
    file_path: str
    success: bool
    error: Optional[str] = None
    documentation: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    chunk_count: int = 0
    successful_chunks: int = 0
    timestamp: datetime = field(default_factory=datetime.now)