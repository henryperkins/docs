"""
code_chunk.py

Defines the CodeChunk dataclass for representing segments of code with associated
metadata and analysis capabilities. Provides core functionality for code
organization and documentation generation, including AST-based merging and splitting.
"""

import uuid
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Set
from pathlib import Path
import ast  # Import ast for AST-based analysis
from token_utils import TokenManager

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
        start_line: Starting line number in source.
        end_line: Ending line number in source.
        chunk_type: Type of code chunk.
        complexity: Cyclomatic complexity if calculated.
        token_count: Number of tokens in chunk.
        dependencies: Set of chunk IDs this chunk depends on.
        used_by: Set of chunk IDs that depend on this chunk.
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
        file_path: Path to source file.
        start_line: Starting line number.
        end_line: Ending line number.
        function_name: Name if chunk is a function.
        class_name: Name if chunk is part of a class.
        chunk_content: Actual code content.
        token_count: Number of tokens. Calculated on initialization.
        language: Programming language.
        chunk_id: Unique identifier.
        is_async: Whether chunk is async.
        decorator_list: List of decorators.
        docstring: Original docstring if any.
        parent_chunk_id: ID of parent chunk.
        metadata: Additional metadata.
    """
    file_path: str
    start_line: int
    end_line: int
    function_name: Optional[str]
    class_name: Optional[str]
    chunk_content: str
    token_count: int
    language: str
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_async: bool = False
    decorator_list: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    metadata: ChunkMetadata = field(init=False)

    @property
    def tokens(self) -> List[str]:
        """Tokenizes chunk content on demand using TokenManager."""
        return TokenManager.count_tokens(self.chunk_content).tokens
        
    def __post_init__(self) -> None:
        try:
            token_result = TokenManager.count_tokens(
                self.chunk_content,
                include_special_tokens=True
            )
            if token_result.error:
                raise ValueError(f"Token counting failed: {token_result.error}")

            object.__setattr__(self, "token_count", token_result.token_count)
            object.__setattr__(self, "_tokens", token_result.tokens)

            # Initialize metadata
            metadata = ChunkMetadata(
                start_line=self.start_line,
                end_line=self.end_line,
                chunk_type=self._determine_chunk_type(),
                token_count=token_result.token_count
            )
            object.__setattr__(self, "metadata", metadata)

        except TokenizationError as e:
            logger.error(f"Tokenization error in chunk: {str(e)}")
            object.__setattr__(self, "token_count", 0)
            object.__setattr__(self, "_tokens", [])

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
        
        return CodeChunk(
            file_path=chunk1.file_path,
            start_line=chunk1.start_line,
            end_line=chunk2.end_line,
            function_name=chunk1.function_name or chunk2.function_name,
            class_name=chunk1.class_name or chunk2.class_name,
            chunk_content=combined_content,
            tokens=tokens.tokens,
            token_count=tokens.token_count,
            language=chunk1.language,
            is_async=chunk1.is_async or chunk2.is_async,
            decorator_list=list(set(chunk1.decorator_list + chunk2.decorator_list)),
            docstring=chunk1.docstring or chunk2.docstring,
            parent_chunk_id=chunk1.parent_chunk_id,
            metadata={
                **chunk1.metadata,
                **chunk2.metadata,
                'merged_from': [chunk1.chunk_id, chunk2.chunk_id]
            }
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
            ValueError: If split point is invalid.
        """
        valid_split_points = self.get_possible_split_points()
        if split_point not in valid_split_points:
            raise ValueError(f"Invalid split point at line {split_point}. Valid split points are: {valid_split_points}")
        
        lines = self.chunk_content.splitlines(keepends=True)
        split_idx = split_point - self.start_line
        
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
            tokens=tokens1.tokens,
            token_count=tokens1.token_count,
            language=self.language,
            is_async=self.is_async,
            decorator_list=self.decorator_list,
            docstring=self.docstring,
            parent_chunk_id=self.parent_chunk_id,
            metadata={
                **self.metadata,
                'split_from': self.chunk_id,
                'split_part': 1
            }
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
            tokens=tokens2.tokens,
            token_count=tokens2.token_count,
            language=self.language,
            is_async=self.is_async,
            decorator_list=self.decorator_list,
            docstring=self.docstring,
            parent_chunk_id=self.parent_chunk_id,
            metadata={
                **self.metadata,
                'split_from': self.chunk_id,
                'split_part': 2
            }
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
            f"type={self.metadata['chunk_type'].value}, "
            f"content='{content_preview}', "
            f"tokens={self.token_count})"
        )
