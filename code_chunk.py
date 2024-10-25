"""
code_chunk.py

Defines the CodeChunk dataclass for representing segments of code with associated 
metadata and analysis capabilities. Provides core functionality for code 
organization and documentation generation.
"""

import uuid
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Set
from pathlib import Path

class ChunkType(Enum):
    """Enumeration of possible chunk types."""
    MODULE = "module"
    CLASS = "class"
    METHOD = "method"
    FUNCTION = "function"
    NESTED_FUNCTION = "nested_function"
    CLASS_METHOD = "class_method"
    STATIC_METHOD = "static_method"
    PROPERTY = "property"
    ASYNC_FUNCTION = "async_function"
    ASYNC_METHOD = "async_method"
    DECORATOR = "decorator"

@dataclass(frozen=True)
class ChunkMetadata:
    """
    Stores metadata about a code chunk.
    
    Attributes:
        start_line: Starting line number in source
        end_line: Ending line number in source
        chunk_type: Type of code chunk
        complexity: Cyclomatic complexity if calculated
        token_count: Number of tokens in chunk
        dependencies: Set of chunk IDs this chunk depends on
        used_by: Set of chunk IDs that depend on this chunk
    """
    start_line: int
    end_line: int
    chunk_type: ChunkType
    complexity: Optional[float] = None
    token_count: int = 0
    dependencies: Set[str] = field(default_factory=set)
    used_by: Set[str] = field(default_factory=set)

@dataclass(frozen=True)
class CodeChunk:
    """
    Immutable representation of a code chunk with metadata.
    
    Each chunk represents a logical segment of code (function, class, etc.)
    with associated metadata about its structure, content, and relationships.
    
    Attributes:
        file_path: Path to source file
        start_line: Starting line number
        end_line: Ending line number
        function_name: Name if chunk is a function
        class_name: Name if chunk is part of a class
        chunk_content: Actual code content
        tokens: List of token strings
        token_count: Number of tokens
        language: Programming language
        chunk_id: Unique identifier
        is_async: Whether chunk is async
        decorator_list: List of decorators
        docstring: Original docstring if any
        parent_chunk_id: ID of parent chunk
        metadata: Additional metadata
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
    decorator_list: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validates chunk data and sets immutable metadata."""
        if self.start_line > self.end_line:
            raise ValueError(
                f"start_line ({self.start_line}) must be <= end_line ({self.end_line})"
            )
        if not self.tokens:
            raise ValueError("tokens list cannot be empty")
        if self.token_count != len(self.tokens):
            raise ValueError(
                f"token_count ({self.token_count}) does not match "
                f"length of tokens ({len(self.tokens)})"
            )
        
        # Set chunk type in metadata
        super().__setattr__('metadata', {
            **self.metadata,
            'chunk_type': self._determine_chunk_type(),
            'hash': self._calculate_hash(),
            'size': len(self.chunk_content)
        })

    def _determine_chunk_type(self) -> ChunkType:
        """Determines the type of this chunk based on its properties."""
        if self.class_name and self.function_name:
            if 'staticmethod' in self.decorator_list:
                return ChunkType.STATIC_METHOD
            elif 'classmethod' in self.decorator_list:
                return ChunkType.CLASS_METHOD
            elif 'property' in self.decorator_list:
                return ChunkType.PROPERTY
            elif self.is_async:
                return ChunkType.ASYNC_METHOD
            return ChunkType.METHOD
        elif self.class_name:
            return ChunkType.CLASS
        elif self.function_name:
            if self.is_async:
                return ChunkType.ASYNC_FUNCTION
            return (ChunkType.NESTED_FUNCTION 
                   if self.parent_chunk_id 
                   else ChunkType.FUNCTION)
        elif self.decorator_list:
            return ChunkType.DECORATOR
        return ChunkType.MODULE

    def _calculate_hash(self) -> str:
        """Calculates a hash of the chunk content."""
        return hashlib.sha256(
            self.chunk_content.encode('utf-8')
        ).hexdigest()

    def get_context_string(self) -> str:
        """
        Returns a concise string representation of the chunk's context.
        
        Returns:
            str: Formatted string with file path, lines, and chunk info
        """
        parts = [
            f"File: {self.file_path}",
            f"Lines: {self.start_line}-{self.end_line}"
        ]
        
        if self.class_name:
            parts.append(f"Class: {self.class_name}")
        if self.function_name:
            prefix = "Async " if self.is_async else ""
            parts.append(f"{prefix}Function: {self.function_name}")
        if self.decorator_list:
            parts.append(f"Decorators: {', '.join(self.decorator_list)}")
        
        return ", ".join(parts)

    def get_hierarchy_path(self) -> str:
        """
        Returns the full hierarchy path of the chunk.
        
        Returns:
            str: Path in form "module.class.method" or "module.function"
        """
        parts = [Path(self.file_path).stem]
        if self.class_name:
            parts.append(self.class_name)
        if self.function_name:
            parts.append(self.function_name)
        return ".".join(parts)

    def can_merge_with(self, other: 'CodeChunk') -> bool:
        """
        Determines if this chunk can be merged with another.
        
        Args:
            other: Another chunk to potentially merge with
            
        Returns:
            bool: True if chunks can be merged
        """
        return (
            self.file_path == other.file_path and
            self.class_name == other.class_name and
            self.function_name == other.function_name and
            self.end_line + 1 == other.start_line and
            self.language == other.language and
            self.parent_chunk_id == other.parent_chunk_id
        )

    @staticmethod
    def merge(chunk1: 'CodeChunk', chunk2: 'CodeChunk') -> 'CodeChunk':
        """
        Creates a new chunk by merging two chunks.
        
        Args:
            chunk1: First chunk to merge
            chunk2: Second chunk to merge
            
        Returns:
            CodeChunk: New merged chunk
            
        Raises:
            ValueError: If chunks cannot be merged
        """
        if not chunk1.can_merge_with(chunk2):
            raise ValueError("Chunks cannot be merged")
        
        return CodeChunk(
            file_path=chunk1.file_path,
            start_line=chunk1.start_line,
            end_line=chunk2.end_line,
            function_name=chunk1.function_name,
            class_name=chunk1.class_name,
            chunk_content=f"{chunk1.chunk_content}\n{chunk2.chunk_content}",
            tokens=chunk1.tokens + chunk2.tokens,
            token_count=chunk1.token_count + chunk2.token_count,
            language=chunk1.language,
            is_async=chunk1.is_async,
            decorator_list=chunk1.decorator_list,
            docstring=chunk1.docstring,
            parent_chunk_id=chunk1.parent_chunk_id,
            metadata={
                **chunk1.metadata,
                **chunk2.metadata,
                'merged_from': [chunk1.chunk_id, chunk2.chunk_id]
            }
        )

    def split(
        self, 
        split_point: int,
        overlap_tokens: int = 0
    ) -> List['CodeChunk']:
        """
        Splits chunk at specified line number.
        
        Args:
            split_point: Line number to split at
            overlap_tokens: Number of tokens to overlap
            
        Returns:
            List[CodeChunk]: List of split chunks
            
        Raises:
            ValueError: If split point is invalid
        """
        if (split_point <= self.start_line or 
            split_point >= self.end_line):
            raise ValueError("Invalid split point")
        
        lines = self.chunk_content.splitlines()
        split_idx = split_point - self.start_line
        
        # Create first chunk
        chunk1_lines = lines[:split_idx]
        chunk1_content = '\n'.join(chunk1_lines)
        chunk1_tokens = self.tokens[:split_idx]
        
        # Create second chunk with overlap
        if overlap_tokens > 0:
            overlap_start = max(0, len(chunk1_tokens) - overlap_tokens)
            overlap_tokens_list = chunk1_tokens[overlap_start:]
        else:
            overlap_tokens_list = []
            
        chunk2_lines = lines[split_idx:]
        chunk2_content = '\n'.join(chunk2_lines)
        chunk2_tokens = overlap_tokens_list + self.tokens[split_idx:]
        
        chunks = []
        for i, (content, tok) in enumerate(
            [(chunk1_content, chunk1_tokens),
             (chunk2_content, chunk2_tokens)],
            1
        ):
            chunks.append(CodeChunk(
                file_path=self.file_path,
                start_line=self.start_line + (0 if i == 1 else split_idx),
                end_line=split_point if i == 1 else self.end_line,
                function_name=f"{self.function_name}_part{i}" if self.function_name else None,
                class_name=f"{self.class_name}_part{i}" if self.class_name else None,
                chunk_content=content,
                tokens=tok,
                token_count=len(tok),
                language=self.language,
                is_async=self.is_async,
                decorator_list=self.decorator_list if i == 1 else [],
                docstring=self.docstring if i == 1 else None,
                parent_chunk_id=self.parent_chunk_id,
                metadata={
                    **self.metadata,
                    'split_from': self.chunk_id,
                    'split_part': i
                }
            ))
        
        return chunks

    def get_metrics(self) -> Dict[str, Any]:
        """
        Gets all metrics associated with this chunk.
        
        Returns:
            Dict[str, Any]: Combined metrics from metadata
        """
        return {
            'complexity': self.metadata.get('complexity'),
            'token_count': self.token_count,
            'size': self.metadata.get('size'),
            'start_line': self.start_line,
            'end_line': self.end_line,
            'type': self.metadata.get('chunk_type').value,
            'has_docstring': self.docstring is not None,
            'is_async': self.is_async,
            'decorator_count': len(self.decorator_list),
            'has_parent': self.parent_chunk_id is not None
        }

    def __repr__(self) -> str:
        """Returns a detailed string representation of the chunk."""
        content_preview = (
            f"{self.chunk_content[:50]}..." 
            if len(self.chunk_content) > 50 
            else self.chunk_content
        ).replace('\n', '\\n')
        
        return (
            f"CodeChunk(file='{self.file_path}', "
            f"lines={self.start_line}-{self.end_line}, "
            f"type={self.metadata['chunk_type'].value}, "
            f"content='{content_preview}', "
            f"tokens={len(self.tokens)})"
        )