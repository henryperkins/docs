"""
context_manager.py

Hierarchical context management system for code documentation generation.
This module provides a robust implementation for managing code chunks and their
associated documentation in a hierarchical project structure.
"""

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Iterator
import aiofiles
import shutil

from code_chunk import CodeChunk, ChunkType

logger = logging.getLogger(__name__)

class ChunkNotFoundError(Exception):
    """Raised when a requested chunk is not found in the context manager."""
    pass

@dataclass
class ChunkLocation:
    """
    Represents the location of a chunk in the project hierarchy.
    
    Attributes:
        project_path: Root path of the project containing the chunk
        module_path: Path to the module containing the chunk
        class_name: Name of the containing class, if any
        function_name: Name of the containing function, if any
    """
    project_path: str
    module_path: str
    class_name: Optional[str]
    function_name: Optional[str]

    def get_hierarchy_path(self) -> str:
        """Returns the full path in the hierarchy."""
        parts = [self.project_path, self.module_path]
        if self.class_name:
            parts.append(self.class_name)
        if self.function_name:
            parts.append(self.function_name)
        return '/'.join(parts)

class HierarchicalContextManager:
    """
    Manages code chunks and documentation in a hierarchical structure.
    
    The hierarchy follows: Project -> Module -> Class -> Function -> Chunk.
    Provides efficient storage, retrieval, and management of code chunks and
    their associated documentation while maintaining relationships and context.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the context manager.

        Args:
            cache_dir: Optional directory for caching documentation
        """
        # Type hints for the nested structure
        ChunkDict = Dict[str, List[CodeChunk]]
        DocDict = Dict[str, Dict[str, Any]]
        
        # Initialize hierarchical storage
        self._chunks: Dict[str, Dict[str, Dict[str, Dict[str, List[CodeChunk]]]]] = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(list)
                )
            )
        )
        
        # Initialize documentation and tracking
        self._docs: Dict[str, Any] = {}
        self._chunk_locations: Dict[str, ChunkLocation] = {}
        self._chunk_ids: Set[str] = set()
        
        # Initialize cache if specified
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = asyncio.Lock()

    def _get_location(self, chunk: CodeChunk) -> ChunkLocation:
        """
        Determines the hierarchical location for a chunk.
        
        Args:
            chunk: The code chunk to locate
            
        Returns:
            ChunkLocation: The chunk's location in the hierarchy
        """
        project_path = str(Path(chunk.file_path).parent)
        module_path = str(Path(chunk.file_path).stem)
        return ChunkLocation(
            project_path=project_path,
            module_path=module_path,
            class_name=chunk.class_name,
            function_name=chunk.function_name
        )

    async def add_code_chunk(self, chunk: CodeChunk) -> None:
        """
        Adds a code chunk to the hierarchy.
        
        Args:
            chunk: The code chunk to add
            
        Raises:
            ValueError: If chunk is invalid or already exists
        """
        async with self._lock:
            if chunk.chunk_id in self._chunk_ids:
                raise ValueError(f"Chunk with ID {chunk.chunk_id} already exists")
                
            location = self._get_location(chunk)
            
            # Store chunk in hierarchy
            self._chunks[location.project_path][location.module_path][
                location.class_name or ''
            ][location.function_name or ''].append(chunk)
            
            # Update tracking
            self._chunk_ids.add(chunk.chunk_id)
            self._chunk_locations[chunk.chunk_id] = location
            
            logger.debug(f"Added chunk {chunk.chunk_id} at {location.get_hierarchy_path()}")

    async def add_doc_chunk(
        self,
        chunk_id: str,
        documentation: Dict[str, Any]
    ) -> None:
        """
        Adds documentation for a specific chunk.
        
        Args:
            chunk_id: ID of the chunk to document
            documentation: Documentation dictionary
            
        Raises:
            ChunkNotFoundError: If chunk_id doesn't exist
        """
        async with self._lock:
            if chunk_id not in self._chunk_ids:
                raise ChunkNotFoundError(f"No chunk found with ID {chunk_id}")
                
            self._docs[chunk_id] = documentation.copy()
            
            # Cache documentation if enabled
            if self._cache_dir:
                await self._cache_documentation(chunk_id, documentation)
            
            logger.debug(f"Added documentation for chunk {chunk_id}")

    async def _cache_documentation(
        self,
        chunk_id: str,
        documentation: Dict[str, Any]
    ) -> None:
        """
        Caches documentation to disk.
        
        Args:
            chunk_id: ID of the chunk
            documentation: Documentation to cache
        """
        if not self._cache_dir:
            return
            
        cache_path = self._cache_dir / f"{chunk_id}.json"
        try:
            async with aiofiles.open(cache_path, 'w') as f:
                await f.write(json.dumps(documentation))
        except Exception as e:
            logger.error(f"Failed to cache documentation: {e}")

    def _get_chunks_with_limit(
        self,
        chunks: List[CodeChunk],
        max_tokens: int,
        language: Optional[str] = None
    ) -> List[CodeChunk]:
        """
        Returns chunks up to the token limit.
        
        Args:
            chunks: List of chunks to filter
            max_tokens: Maximum total tokens
            language: Optional language filter
            
        Returns:
            List[CodeChunk]: Filtered chunks within token limit
        """
        filtered_chunks = []
        total_tokens = 0
        
        for chunk in chunks:
            if language and chunk.language != language:
                continue
                
            if total_tokens + chunk.token_count > max_tokens:
                break
                
            filtered_chunks.append(chunk)
            total_tokens += chunk.token_count
            
        return filtered_chunks

    async def get_context_for_function(
        self,
        module_path: str,
        function_name: str,
        language: str,
        max_tokens: int = 4096
    ) -> List[CodeChunk]:
        """
        Gets context chunks for a function.
        
        Args:
            module_path: Path to the module
            function_name: Name of the function
            language: Programming language filter
            max_tokens: Maximum total tokens
            
        Returns:
            List[CodeChunk]: Related chunks within token limit
        """
        async with self._lock:
            project_path = str(Path(module_path).parent)
            module_name = Path(module_path).stem
            
            all_chunks: List[CodeChunk] = []
            module_dict = self._chunks[project_path][module_name]
            
            # Get function chunks (highest priority)
            for class_chunks in module_dict.values():
                for func_chunks in class_chunks.values():
                    all_chunks.extend(
                        chunk for chunk in func_chunks
                        if chunk.function_name == function_name
                    )
            
            # Get class chunks if function is a method
            for class_name, class_chunks in module_dict.items():
                if any(chunk.class_name == class_name and chunk.function_name == function_name 
                      for chunk in all_chunks):
                    all_chunks.extend(
                        chunk for chunks in class_chunks.values()
                        for chunk in chunks
                        if chunk.class_name == class_name
                    )
            
            # Get module chunks (lowest priority)
            all_chunks.extend(
                chunk for chunks in module_dict[''].values()
                for chunk in chunks
            )
            
            return self._get_chunks_with_limit(all_chunks, max_tokens, language)

    async def get_context_for_class(
        self,
        module_path: str,
        class_name: str,
        language: str,
        max_tokens: int = 4096
    ) -> List[CodeChunk]:
        """
        Gets context chunks for a class.
        
        Args:
            module_path: Path to the module
            class_name: Name of the class
            language: Programming language filter
            max_tokens: Maximum total tokens
            
        Returns:
            List[CodeChunk]: Related chunks within token limit
        """
        async with self._lock:
            project_path = str(Path(module_path).parent)
            module_name = Path(module_path).stem
            
            all_chunks: List[CodeChunk] = []
            
            # Get class chunks (highest priority)
            class_dict = self._chunks[project_path][module_name][class_name]
            for chunks in class_dict.values():
                all_chunks.extend(chunks)
            
            # Get module chunks (lower priority)
            module_chunks = self._chunks[project_path][module_name]['']['']
            all_chunks.extend(module_chunks)
            
            return self._get_chunks_with_limit(all_chunks, max_tokens, language)

    async def get_context_for_module(
        self,
        module_path: str,
        language: str,
        max_tokens: int = 4096
    ) -> List[CodeChunk]:
        """
        Gets context chunks for a module.
        
        Args:
            module_path: Path to the module
            language: Programming language filter
            max_tokens: Maximum total tokens
            
        Returns:
            List[CodeChunk]: Module chunks within token limit
        """
        async with self._lock:
            project_path = str(Path(module_path).parent)
            module_name = Path(module_path).stem
            
            all_chunks = [
                chunk
                for class_dict in self._chunks[project_path][module_name].values()
                for func_dict in class_dict.values()
                for chunk in func_dict
            ]
            
            return self._get_chunks_with_limit(all_chunks, max_tokens, language)

    async def get_documentation_for_chunk(
        self,
        chunk_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Gets documentation for a specific chunk.
        
        Args:
            chunk_id: ID of the chunk
            
        Returns:
            Optional[Dict[str, Any]]: The chunk's documentation or None if not found
        """
        async with self._lock:
            # Check memory first
            if chunk_id in self._docs:
                return self._docs[chunk_id]
            
            # Check cache if enabled
            if self._cache_dir:
                cache_path = self._cache_dir / f"{chunk_id}.json"
                try:
                    if cache_path.exists():
                        async with aiofiles.open(cache_path, 'r') as f:
                            return json.loads(await f.read())
                except Exception as e:
                    logger.error(f"Failed to read cached documentation: {e}")
            
            return None

    async def update_code_chunk(self, chunk: CodeChunk) -> None:
        """
        Updates an existing chunk.
        
        Args:
            chunk: The updated chunk
            
        Raises:
            ChunkNotFoundError: If chunk doesn't exist
        """
        async with self._lock:
            if chunk.chunk_id not in self._chunk_ids:
                raise ChunkNotFoundError(f"No chunk found with ID {chunk.chunk_id}")
            
            # Remove old chunk
            await self.remove_code_chunk(chunk.chunk_id)
            
            # Add updated chunk
            await self.add_code_chunk(chunk)
            
            logger.debug(f"Updated chunk {chunk.chunk_id}")

    async def remove_code_chunk(self, chunk_id: str) -> None:
        """
        Removes a chunk and its documentation.
        
        Args:
            chunk_id: ID of the chunk to remove
            
        Raises:
            ChunkNotFoundError: If chunk doesn't exist
        """
        async with self._lock:
            if chunk_id not in self._chunk_ids:
                raise ChunkNotFoundError(f"No chunk found with ID {chunk_id}")
            
            location = self._chunk_locations[chunk_id]
            chunks = self._chunks[location.project_path][location.module_path][
                location.class_name or ''
            ][location.function_name or '']
            
            # Remove chunk
            chunks[:] = [chunk for chunk in chunks if chunk.chunk_id != chunk_id]
            
            # Remove documentation
            self._docs.pop(chunk_id, None)
            
            # Remove from cache if enabled
            if self._cache_dir:
                cache_path = self._cache_dir / f"{chunk_id}.json"
                try:
                    if cache_path.exists():
                        cache_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove cached documentation: {e}")
            
            # Update tracking
            self._chunk_ids.remove(chunk_id)
            del self._chunk_locations[chunk_id]
            
            logger.debug(f"Removed chunk {chunk_id}")

    async def clear_context(self) -> None:
        """Clears all chunks and documentation."""
        async with self._lock:
            self._chunks.clear()
            self._docs.clear()
            self._chunk_ids.clear()
            self._chunk_locations.clear()
            
            # Clear cache if enabled
            if self._cache_dir:
                try:
                    shutil.rmtree(self._cache_dir)
                    self._cache_dir.mkdir()
                except Exception as e:
                    logger.error(f"Failed to clear cache directory: {e}")
            
            logger.debug("Cleared all context")

    @contextmanager
    def batch_updates(self) -> Iterator[None]:
        """
        Context manager for batching multiple updates.
        
        Use this when making multiple related changes to avoid
        intermediate inconsistencies.
        
        Example:
            ```python
            async with context_manager.batch_updates():
                await context_manager.remove_code_chunk(old_id)
                await context_manager.add_code_chunk(new_chunk)
            ```
        """
        try:
            yield
        finally:
            # Could add consistency checks here
            pass
