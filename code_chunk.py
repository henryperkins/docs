"""
code_chunk.py

This module defines the CodeChunk dataclass for storing metadata about code chunks
in the documentation generation system.
"""

import uuid
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class CodeChunk:
    """
    Immutable dataclass representing a chunk of code with associated metadata.
    
    Each chunk contains information about its location in the source file,
    its content, and structural information (e.g., whether it belongs to
    a function or class).

    Attributes:
        file_path (str): Path to the source file
        start_line (int): Starting line number (inclusive)
        end_line (int): Ending line number (inclusive)
        function_name (Optional[str]): Name of the function if chunk is a function
        class_name (Optional[str]): Name of the class if chunk is part of a class
        chunk_id (str): Unique identifier for the chunk
        chunk_content (str): The actual code string
        tokens (List[str]): List of token strings from the code
        token_count (int): Number of tokens in the chunk
        language (str): Programming language of the chunk (e.g., "python")
    """
    file_path: str
    start_line: int
    end_line: int
    function_name: Optional[str]
    class_name: Optional[str]
    chunk_content: str
    tokens: List[str]
    token_count: int
    language: str
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_async: bool = False
    chunk_type: str = field(init=False)

    def __post_init__(self):
        """Validate chunk data and set chunk type"""
        if self.start_line > self.end_line:
            raise ValueError(f"start_line ({self.start_line}) must be <= end_line ({self.end_line})")
        if not self.tokens:
            raise ValueError("tokens list cannot be empty")
        if self.token_count != len(self.tokens):
            raise ValueError(f"token_count ({self.token_count}) does not match length of tokens ({len(self.tokens)})")

        # Set chunk type
        object.__setattr__(self, 'chunk_type', self._determine_chunk_type())

    def _determine_chunk_type(self) -> str:
        """Determines the type of this chunk based on its properties"""
        if self.class_name and self.function_name:
            return 'method'
        elif self.class_name:
            return 'class'
        elif self.function_name:
            return 'function'
        return 'module'

    def can_merge_with(self, other: 'CodeChunk') -> bool:
        """Determines if this chunk can be merged with another"""
        return (
            self.file_path == other.file_path
            and self.class_name == other.class_name
            and self.function_name == other.function_name
            and self.end_line + 1 == other.start_line
        )

    def merge(self, other: 'CodeChunk') -> 'CodeChunk':
        """Merges this chunk with another, if possible"""
        if not self.can_merge_with(other):
            raise ValueError("Chunks cannot be merged")
        
        return CodeChunk(
            file_path=self.file_path,
            start_line=self.start_line,
            end_line=other.end_line,
            function_name=self.function_name,
            class_name=self.class_name,
            chunk_content=f"{self.chunk_content}\n{other.chunk_content}",
            tokens=self.tokens + other.tokens,
            token_count=self.token_count + other.token_count,
            language=self.language,
            is_async=self.is_async
        )
    
    def get_context_string(self) -> str:
        """
        Returns a concise string representation of the chunk's context.
        
        The string includes file path, line numbers, and optionally the
        function and/or class name if they exist.
        
        Returns:
            str: A formatted context string like:
                "File: path/to/file.py, Lines: 10-20, Function: my_function"
                or "File: path/to/file.py, Lines: 30-40, Class: MyClass"
                or "File: path/to/file.py, Lines: 50-60"
        """
        context_parts = [
            f"File: {self.file_path}",
            f"Lines: {self.start_line}-{self.end_line}"
        ]

        if self.class_name is not None:
            context_parts.append(f"Class: {self.class_name}")
        if self.function_name is not None:
            context_parts.append(f"Function: {self.function_name}")

        return ", ".join(context_parts)

    def __repr__(self) -> str:
        """
        Returns a detailed string representation of the CodeChunk.
        
        Includes all metadata but truncates the chunk_content and tokens
        to prevent overly long representations.
        
        Returns:
            str: A string representation of the CodeChunk
        """
        content_preview = (
            f"{self.chunk_content[:50]}..." 
            if len(self.chunk_content) > 50 
            else self.chunk_content
        )
        tokens_preview = (
            f"{str(self.tokens[:3])[:-1]}, ...]" 
            if len(self.tokens) > 3 
            else str(self.tokens)
        )
        
        return (
            f"CodeChunk(file_path='{self.file_path}', "
            f"start_line={self.start_line}, end_line={self.end_line}, "
            f"function_name={self.function_name!r}, "
            f"class_name={self.class_name!r}, "
            f"chunk_id='{self.chunk_id}', "
            f"chunk_content='{content_preview}', "
            f"tokens={tokens_preview}, "
            f"token_count={self.token_count}, "
            f"language='{self.language}')"
        )
