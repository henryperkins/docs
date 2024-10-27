"""
code_chunk.py

Defines the CodeChunk dataclass for representing segments of code with associated
metadata and analysis capabilities. Provides core functionality for code
organization and documentation generation, including AST-based merging and splitting.
"""

from __future__ import annotations
import uuid
import hashlib
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import List, Optional, Dict, Any, Set
from pathlib import Path
import ast
from token_utils import TokenManager, TokenizationError, TokenizationResult
import itertools
import logging
from radon.complexity import cc_visit

# Configure logger
logger = logging.getLogger(__name__)



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


@dataclass(frozen=True)  # ChunkMetadata is immutable
class ChunkMetadata:
    """Stores metadata about a code chunk, including complexity."""
    start_line: int
    end_line: int
    chunk_type: ChunkType
    token_count: int = 0
    dependencies: Set[int] = field(default_factory=set)
    used_by: Set[int] = field(default_factory=set)
    complexity: Optional[float] = None  # Cached complexity

    def __post_init__(self):
        # Complexity is handled in CodeChunk.__post_init__
        pass  # No need for __post_init__ here


dataclass(frozen=True)
class CodeChunk:
    """Immutable representation of a code chunk with metadata."""

    file_path: str
    start_line: int
    end_line: int
    function_name: Optional[str]
    class_name: Optional[str]
    chunk_content: str
    language: str  # Keep language attribute
    is_async: bool = False
    decorator_list: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    parent_chunk_id: Optional[int] = None
    chunk_id: int = field(init=False)
    _chunk_counter = itertools.count()
    _tokens: Optional[List[int]] = field(default=None, init=False, repr=False)
    metadata: ChunkMetadata = field(init=False) # Only define metadata and _tokens ONCE


    @property
    def tokens(self) -> Optional[List[int]]:
        """Returns cached tokens."""
        return self._tokens

    @property
    def token_count(self) -> int:
        """Returns the number of tokens."""
        return len(self._tokens) if self._tokens else 0

    def __post_init__(self):  # Correct implementation
        """Initializes CodeChunk with tokens and complexity."""
        object.__setattr__(self, "chunk_id", next(self._chunk_counter))

        try:
            token_result: TokenizationResult = TokenManager.count_tokens(self.chunk_content, include_special_tokens=True)
            if token_result.error:
                raise ValueError(f"Token counting failed: {token_result.error}")

            object.__setattr__(self, "_tokens", token_result.tokens)

            complexity = self._calculate_complexity(self.chunk_content)

            metadata = ChunkMetadata(
                start_line=self.start_line,
                end_line=self.end_line,
                chunk_type=self._determine_chunk_type(),
                token_count=self.token_count,  # Use the property
                complexity=complexity
            )
            object.__setattr__(self, "metadata", metadata)

        except TokenizationError as e:
            logger.error(f"Tokenization error in chunk: {str(e)}")
            object.__setattr__(self, "token_count", 0)
            object.__setattr__(self, "_tokens", [])
            # Create metadata with default values in case of error
            metadata = ChunkMetadata(
                start_line=self.start_line,
                end_line=self.end_line,
                chunk_type=self._determine_chunk_type(),
                token_count=0,
                complexity=None
            )
            object.__setattr__(self, "metadata", metadata)

	 def _calculate_complexity(self, code: str) -> Optional[float]:
        """Calculates complexity using radon."""
        logger.debug(f"Calculating complexity for chunk (lines {self.start_line}-{self.end_line})")
        try:
            complexity_blocks = cc_visit(code)
            calculated_complexity = sum(block.complexity for block in complexity_blocks)
            logger.debug(f"Calculated complexity: {calculated_complexity}")
            return calculated_complexity
        except Exception as e:
            logger.error(f"Error calculating complexity: {e}")
            return None

    def add_dependency(self, other: CodeChunk) -> CodeChunk:
        """
        Adds a dependency relationship between chunks immutably.

        Args:
            other: The CodeChunk that this chunk depends on.

        Returns:
            A new CodeChunk object with the updated dependencies.
        """
        if self.chunk_id == other.chunk_id:
            logger.warning("Attempting to add self-dependency, skipping.")
            return self

        if other.chunk_id in self.metadata.dependencies:
            logger.debug(f"Dependency {other.chunk_id} already exists for chunk {self.chunk_id}, skipping.")
            return self

        new_dependencies = self.metadata.dependencies.union({other.chunk_id})
        new_self_metadata = replace(self.metadata, dependencies=new_dependencies)
        new_self = replace(self, metadata=new_self_metadata)

        new_other_used_by = other.metadata.used_by.union({self.chunk_id})
        new_other_metadata = replace(other.metadata, used_by=new_other_used_by)
        new_other = replace(other, metadata=new_other_metadata)

        return new_self

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
            return ChunkType.NESTED_FUNCTION if self.parent_chunk_id else ChunkType.FUNCTION
        elif self.decorator_list:  # Check for decorators before defaulting to MODULE
            return ChunkType.DECORATOR
        return ChunkType.MODULE  # Default to MODULE if no other type is identified

    def _calculate_hash(self) -> str:
        """Calculates a SHA256 hash of the chunk content."""
        return hashlib.sha256(self.chunk_content.encode('utf-8')).hexdigest()

    def get_context_string(self) -> str:
        """Returns a concise string representation of the chunk's context."""
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
        """Returns the full hierarchy path of the chunk."""
        parts = [Path(self.file_path).stem]  # Use stem for module name
        if self.class_name:
            parts.append(self.class_name)
        if self.function_name:
            parts.append(self.function_name)
        return ".".join(parts)

    def can_merge_with(self, other: 'CodeChunk') -> bool:
        """
        Determines if this chunk can be merged with another using AST analysis.

        Args:
            other: Another chunk to potentially merge with.

        Returns:
            bool: True if chunks can be merged.
        """
        if not (
            self.file_path == other.file_path and
            self.language == other.language and
            self.end_line + 1 == other.start_line
        ):
            return False

        # Use AST to check if merging maintains valid syntax
        combined_content = self.chunk_content + '\n' + other.chunk_content
        try:
            ast.parse(combined_content)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def merge(chunk1: 'CodeChunk', chunk2: 'CodeChunk') -> 'CodeChunk':
        """
        Creates a new chunk by merging two chunks using AST analysis.

        Args:
            chunk1: First chunk to merge.
            chunk2: Second chunk to merge.

        Returns:
            CodeChunk: New merged chunk.

        Raises:
            ValueError: If chunks cannot be merged.
        """
        if not chunk1.can_merge_with(chunk2):
            raise ValueError("Chunks cannot be merged, AST validation failed.")

        combined_content = chunk1.chunk_content + '\n' + chunk2.chunk_content
        tokens = TokenManager.count_tokens(combined_content)

        # Aggregate dependencies and used_by
        combined_dependencies = chunk1.metadata.dependencies.union(chunk2.metadata.dependencies)
        combined_used_by = chunk1.metadata.used_by.union(chunk2.metadata.used_by)

        # Determine new chunk type
        chunk_type = chunk1.metadata.chunk_type  # Simplistic approach; can be enhanced

        # Calculate complexity
        complexity = chunk1.metadata.complexity
        if chunk2.metadata.complexity is not None:
            complexity += chunk2.metadata.complexity

        new_metadata = ChunkMetadata(
            start_line=chunk1.start_line,
            end_line=chunk2.end_line,
            chunk_type=chunk_type,
            token_count=chunk1.token_count + chunk2.token_count,
            dependencies=combined_dependencies,
            used_by=combined_used_by,
            complexity=complexity
        )

        return CodeChunk(
            file_path=chunk1.file_path,
            start_line=chunk1.start_line,
            end_line=chunk2.end_line,
            function_name=chunk1.function_name or chunk2.function_name,
            class_name=chunk1.class_name or chunk2.class_name,
            chunk_content=combined_content,
            token_count=tokens.token_count,
            language=chunk1.language,
            is_async=chunk1.is_async or chunk2.is_async,
            decorator_list=list(set(chunk1.decorator_list + chunk2.decorator_list)),
            docstring=chunk1.docstring or chunk2.docstring,
            parent_chunk_id=chunk1.parent_chunk_id,
            metadata=new_metadata
        )

    def get_possible_split_points(self) -> List[int]:
        """
        Returns a list of line numbers where the chunk can be split without breaking syntax.

        Returns:
            List[int]: List of valid split line numbers.
        """
        try:
            tree = ast.parse(self.chunk_content)
        except SyntaxError:
            # If the chunk has invalid syntax, it can't be split safely
            return []

        possible_split_points = []
        for node in ast.walk(tree):  # Use ast.walk to traverse all nodes
            if isinstance(node, (ast.stmt, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check for statements, class definitions, and function definitions
                if hasattr(node, 'lineno'):
                    split_line = self.start_line + node.lineno - 1
                    if split_line < self.end_line:  # Avoid splitting on the last line
                        possible_split_points.append(split_line)

        return sorted(set(possible_split_points))  # Remove duplicates and sort

    def split(
        self,
        split_point: int
    ) -> List['CodeChunk']:
        """
        Splits chunk at specified line number using AST analysis.

        Args:
            split_point: Line number to split at.

        Returns:
            List[CodeChunk]: List of split chunks.

        Raises:
            ValueError: If split point is invalid or violates boundary conditions.
        """
        valid_split_points = self.get_possible_split_points()

        # Boundary Checks
        if split_point == self.start_line:
            raise ValueError(f"Cannot split at the very beginning of the chunk (line {split_point}).")
        if split_point == self.end_line:
            raise ValueError(f"Cannot split at the very end of the chunk (line {split_point}).")

        if split_point not in valid_split_points:
            raise ValueError(
                f"Invalid split point at line {split_point}. "
                f"Valid split points are: {valid_split_points}"
            )

        lines = self.chunk_content.splitlines(keepends=True)
        split_idx = split_point - self.start_line

        # Ensure split_idx is within the bounds of the lines list
        if split_idx <= 0 or split_idx >= len(lines):
            raise ValueError(
                f"Split index {split_idx} derived from split point {split_point} is out of bounds."
            )

        # Create first chunk
        chunk1_content = ''.join(lines[:split_idx])
        tokens1 = TokenManager.count_tokens(chunk1_content)

        chunk1 = CodeChunk(
            file_path=self.file_path,
            start_line=self.start_line,
            end_line=split_point - 1,
            function_name=self.function_name,
            class_name=self.class_name,
            chunk_content=chunk1_content,
            token_count=tokens1.token_count,
            language=self.language,
            is_async=self.is_async,
            decorator_list=self.decorator_list,
            docstring=self.docstring,
            parent_chunk_id=self.parent_chunk_id,
            metadata=replace(self.metadata, end_line=split_point - 1)
        )

        # Create second chunk
        chunk2_content = ''.join(lines[split_idx:])
        tokens2 = TokenManager.count_tokens(chunk2_content)

        chunk2 = CodeChunk(
            file_path=self.file_path,
            start_line=split_point,
            end_line=self.end_line,
            function_name=self.function_name,
            class_name=self.class_name,
            chunk_content=chunk2_content,
            token_count=tokens2.token_count,
            language=self.language,
            is_async=self.is_async,
            decorator_list=self.decorator_list,
            docstring=self.docstring,
            parent_chunk_id=self.parent_chunk_id,
            metadata=replace(self.metadata, start_line=split_point)
        )

        return [chunk1, chunk2]

    def get_metrics(self) -> Dict[str, Any]:
        return {
            'complexity': self.metadata.complexity,
            'token_count': self.token_count,
            'start_line': self.metadata.start_line,
            'end_line': self.metadata.end_line,
            'type': self.metadata.chunk_type.value,
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
            f"type={self.metadata.chunk_type.value}, "
            f"content='{content_preview}', "
            f"tokens={self.token_count})"
        )
