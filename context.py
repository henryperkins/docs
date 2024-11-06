import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import aiofiles
import shutil
import hashlib
import networkx as nx
from sentence_transformers import SentenceTransformer
from collections import deque

# Assuming these are imported from another module
from code_chunk import CodeChunk, ChunkMetadata, ChunkLocation
from token_utils import TokenManager, TokenizationError
from dependency_analyzer import DependencyAnalyzer  # Import the unified DependencyAnalyzer

logger = logging.getLogger(__name__)

# Custom Exceptions
class ChunkNotFoundError(Exception):
    """Raised when a requested chunk cannot be found."""
    pass

class InvalidChunkError(Exception):
    """Raised when a chunk is invalid or corrupted."""
    pass

class CacheError(Exception):
    """Raised when cache operations fail."""
    pass

# Node Class for Hierarchical Structure
class Node:
    """Node in the context tree."""

    def __init__(
        self,
        name: str,
        chunk: Optional[CodeChunk] = None,
        location: Optional[ChunkLocation] = None,
        metadata: Optional[ChunkMetadata] = None
    ):
        self.name = name
        self.chunk = chunk
        self.location = location
        self.metadata = metadata or ChunkMetadata()
        self.children: Dict[str, 'Node'] = {}
        self.parent: Optional['Node'] = None

    def add_child(self, child: 'Node') -> None:
        """Adds a child node."""
        self.children[child.name] = child
        child.parent = self

    def remove_child(self, name: str) -> None:
        """Removes a child node."""
        if name in self.children:
            self.children[name].parent = None
            del self.children[name]

    def get_ancestors(self) -> List['Node']:
        """Gets all ancestor nodes."""
        ancestors = []
        current = self.parent
        while current:
            ancestors.append(current)
            current = current.parent
        return ancestors

    def get_descendants(self) -> List['Node']:
        """Gets all descendant nodes."""
        descendants = []
        queue = deque(self.children.values())
        while queue:
            node = queue.popleft()
            descendants.append(node)
            queue.extend(node.children.values())
        return descendants

# HierarchicalContextManager Class
class HierarchicalContextManager:
    """Manages code chunks and documentation with a tree structure, caching, dependency graphs, and semantic similarity."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_cache_size: int = 1000,
        token_model: str = 'cl100k_base',
        embedding_model: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initializes the context manager.

        Args:
            cache_dir: Directory for persistent cache
            max_cache_size: Maximum number of items in memory cache
            token_model: Tokenizer model to use
            embedding_model: Sentence transformer model for embeddings
        """
        self._root = Node("root")
        self._docs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._max_cache_size = max_cache_size
        self._token_model = token_model
        self._metrics: Dict[str, Any] = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_chunks': 0,
            'total_tokens': 0
        }

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Dependency Graph for Advanced Relationship Management
        self._dependency_graph = nx.DiGraph()
        self._embedding_model = SentenceTransformer(embedding_model)

    async def add_code_chunk(self, chunk: CodeChunk) -> None:
        """
        Adds a code chunk to the tree with validation, dependency tracking, and metrics.

        Args:
            chunk: CodeChunk to add

        Raises:
            InvalidChunkError: If chunk is invalid
            TokenizationError: If token counting fails
        """
        try:
            # Validate chunk
            if not chunk.chunk_content.strip():
                raise InvalidChunkError("Empty chunk content")

            # Count tokens
            token_result = TokenManager.count_tokens(
                chunk.chunk_content,
                model=self._token_model
            )

            location = ChunkLocation(
                project_path=str(Path(chunk.file_path).parent),
                module_path=Path(chunk.file_path).stem,
                class_name=chunk.class_name,
                function_name=chunk.function_name,
                start_line=chunk.start_line,
                end_line=chunk.end_line
            )

            metadata = ChunkMetadata(
                token_count=token_result.token_count,
                dependencies=set()
            )
            metadata.update_hash(chunk.chunk_content)

            async with self._lock:
                # Check for overlapping chunks
                if self._has_overlap(location):
                    logger.warning(f"Overlapping chunk detected at {location.get_hierarchy_path()}")

                # Add to tree
                path = location.get_hierarchy_path().split(".")
                current = self._root

                for part in path:
                    if part not in current.children:
                        current.children[part] = Node(part)
                    current = current.children[part]

                current.chunk = chunk
                current.location = location
                current.metadata = metadata

                # Update metrics
                self._metrics['total_chunks'] += 1
                self._metrics['total_tokens'] += token_result.token_count

                logger.debug(f"Added chunk {chunk.chunk_id} to context")

                # Add to dependency graph
                self._add_to_dependency_graph(chunk)

        except (InvalidChunkError, TokenizationError) as e:
            logger.error(f"Validation error adding chunk: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding chunk: {str(e)}")
            raise

    def _has_overlap(self, location: ChunkLocation) -> bool:
        """Checks if a location overlaps with existing chunks."""
        for node in self._get_module_nodes(location.module_path):
            if node.location and node.location.overlaps_with(location):
                return True
        return False

    def _get_module_nodes(self, module_path: str) -> List[Node]:
        """Gets all nodes in a module."""
        nodes = []
        if module_path in self._root.children:
            module_node = self._root.children[module_path]
            nodes.extend([module_node] + module_node.get_descendants())
        return nodes

    def _add_to_dependency_graph(self, chunk: CodeChunk) -> None:
        """
        Adds a chunk to the dependency graph based on semantic similarity.

        Args:
            chunk: CodeChunk to add
        """
        self._dependency_graph.add_node(chunk.chunk_id, chunk=chunk)

        analyzer = DependencyAnalyzer()
        dependencies = analyzer.analyze(chunk.chunk_content)
        for dep in dependencies:
            if dep in self._dependency_graph:
                self._dependency_graph.add_edge(chunk.chunk_id, dep)

    async def get_context_for_function(
        self,
        module_path: str,
        function_name: str,
        language: str,
        max_tokens: int = 4096
    ) -> List[CodeChunk]:
        """
        Gets context chunks for a function with token limit awareness.

        Args:
            module_path: Path to the module
            function_name: Name of the function
            language: Programming language
            max_tokens: Maximum total tokens

        Returns:
            List[CodeChunk]: Relevant context chunks
        """
        async with self._lock:
            module_name = Path(module_path).stem
            path = [module_name, function_name]
            node = self._find_node(path)

            if node and node.chunk:
                context_chunks = self._get_context_from_graph(
                    chunk_id=node.chunk.chunk_id,
                    language=language,
                    max_tokens=max_tokens
                )

                return context_chunks
            return []

    async def get_context_for_class(
        self,
        module_path: str,
        class_name: str,
        language: str,
        max_tokens: int = 4096
    ) -> List[CodeChunk]:
        """Gets context chunks for a class."""
        async with self._lock:
            module_name = Path(module_path).stem
            path = [module_name, class_name]
            node = self._find_node(path)

            if node:
                return self._get_context_from_graph(
                    chunk_id=node.chunk.chunk_id if node.chunk else "",
                    language=language,
                    max_tokens=max_tokens
                )
            return []

    async def get_context_for_module(
        self,
        module_path: str,
        language: str,
        max_tokens: int = 4096
    ) -> List[CodeChunk]:
        """Gets context chunks for a module."""
        async with self._lock:
            module_name = Path(module_path).stem
            path = [module_name]
            node = self._find_node(path)

            if node:
                return self._get_context_from_graph(
                    chunk_id=node.chunk.chunk_id if node.chunk else "",
                    language=language,
                    max_tokens=max_tokens
                )
            return []

    def _get_context_from_graph(
        self,
        chunk_id: str,
        language: str,
        max_tokens: int
    ) -> List[CodeChunk]:
        """
        Performs intelligent context gathering using dependency graph traversal with token limiting.

        Args:
            chunk_id: Starting chunk ID
            language: Programming language
            max_tokens: Maximum total tokens

        Returns:
            List[CodeChunk]: Context chunks within token limit
        """
        context_chunks = []
        total_tokens = 0
        visited = set()
        queue = deque([(chunk_id, 0)])  # (chunk_id, depth)

        while queue and total_tokens < max_tokens:
            current_id, depth = queue.popleft()

            if current_id in visited:
                continue

            visited.add(current_id)

            current_chunk = self._dependency_graph.nodes[current_id]['chunk']

            if current_chunk.language == language:
                if total_tokens + current_chunk.token_count <= max_tokens:
                    context_chunks.append(current_chunk)
                    total_tokens += current_chunk.token_count

                    # Enqueue related chunks based on dependencies
                    for neighbor in self._dependency_graph.neighbors(current_id):
                        if neighbor not in visited:
                            queue.append((neighbor, depth + 1))

            # Optionally, you can limit the depth to avoid too deep traversal
            if depth >= 5:  # Example depth limit
                continue

        return context_chunks

    async def get_documentation_for_chunk(
        self,
        chunk_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves documentation with caching and validation.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Optional[Dict[str, Any]]: Documentation if found

        Raises:
            ChunkNotFoundError: If chunk not found
            CacheError: If cache operations fail
        """
        async with self._lock:
            # Check memory cache
            if chunk_id in self._docs:
                self._metrics['cache_hits'] += 1
                return self._docs[chunk_id]

            # Check disk cache
            if self._cache_dir:
                cache_path = self._cache_dir / f"{chunk_id}.json"
                if cache_path.exists():
                    try:
                        async with aiofiles.open(
                            cache_path, 'r', encoding="utf-8"
                        ) as f:
                            cached = json.loads(await f.read())

                        # Validate cache
                        node = self._find_node_by_id(chunk_id)
                        if node and node.metadata:
                            cached_hash = cached.get('metadata', {}).get('hash')
                            if cached_hash == node.metadata.hash:
                                doc = cached['documentation']
                                self._docs[chunk_id] = doc
                                self._metrics['cache_hits'] += 1
                                return doc

                    except Exception as e:
                        logger.error(f"Cache read failed: {e}")

            self._metrics['cache_misses'] += 1
            return None

    async def update_code_chunk(self, chunk: CodeChunk) -> None:
        """
        Updates an existing chunk with change detection.

        Args:
            chunk: Updated chunk

        Raises:
            ChunkNotFoundError: If chunk not found
            InvalidChunkError: If chunk is invalid
        """
        async with self._lock:
            path = chunk.get_hierarchy_path().split(".")
            node = self._find_node(path)

            if not node:
                raise ChunkNotFoundError(
                    f"No chunk found for path: {'.'.join(path)}"
                )

            # Calculate new metadata
            token_result = TokenManager.count_tokens(
                chunk.chunk_content,
                model=self._token_model
            )

            new_metadata = ChunkMetadata(
                token_count=token_result.token_count,
                dependencies=node.metadata.dependencies if node.metadata else set()
            )
            new_metadata.update_hash(chunk.chunk_content)

            # Check if content actually changed
            if (node.metadata and
                node.metadata.hash == new_metadata.hash):
                logger.debug(f"Chunk {chunk.chunk_id} unchanged, skipping update")
                return

            # Update node
            node.chunk = chunk
            node.metadata = new_metadata

            # Update dependency graph
            self._dependency_graph.remove_node(chunk.chunk_id)
            self._add_to_dependency_graph(chunk)

            # Invalidate cached documentation
            self._docs.pop(chunk.chunk_id, None)
            if self._cache_dir:
                cache_path = self._cache_dir / f"{chunk.chunk_id}.json"
                try:
                    if cache_path.exists():
                        cache_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove cached documentation: {e}")

            logger.debug(f"Updated chunk {chunk.chunk_id}")

    async def remove_code_chunk(self, chunk_id: str) -> None:
        """
        Removes a chunk and its documentation.

        Args:
            chunk_id: Chunk to remove

        Raises:
            ChunkNotFoundError: If chunk not found
        """
        async with self._lock:
            node = self._find_node_by_id(chunk_id)
            if not node:
                raise ChunkNotFoundError(f"No chunk found with ID {chunk_id}")

            # Remove from tree
            if node.parent:
                node.parent.remove_child(node.name)

            # Remove from dependency graph
            if self._dependency_graph.has_node(chunk_id):
                self._dependency_graph.remove_node(chunk_id)

            # Remove documentation
            self._docs.pop(chunk_id, None)

            # Remove from cache
            if self._cache_dir:
                cache_path = self._cache_dir / f"{chunk_id}.json"
                try:
                    if cache_path.exists():
                        cache_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove cached documentation: {e}")

            # Update metrics
            if node.metadata:
                self._metrics['total_chunks'] -= 1
                self._metrics['total_tokens'] -= node.metadata.token_count

            logger.debug(f"Removed chunk {chunk_id}")

    async def clear_context(self) -> None:
        """Clears all chunks, documentation, and cache."""
        async with self._lock:
            self._root = Node("root")
            self._docs.clear()
            self._metrics = {
                'cache_hits': 0,
                'cache_misses': 0,
                'total_chunks': 0,
                'total_tokens': 0
            }

            if self._cache_dir:
                try:
                    shutil.rmtree(self._cache_dir)
                    self._cache_dir.mkdir()
                    logger.debug("Cache directory cleared")
                except OSError as e:
                    logger.error(f"Error clearing cache directory: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Returns current metrics."""
        return {
            **self._metrics,
            'cache_hit_ratio': (
                self._metrics['cache_hits'] /
                (self._metrics['cache_hits'] + self._metrics['cache_misses'])
                if (self._metrics['cache_hits'] + self._metrics['cache_misses']) > 0
                else 0
            ),
            'avg_tokens_per_chunk': (
                self._metrics['total_tokens'] / self._metrics['total_chunks']
                if self._metrics['total_chunks'] > 0
                else 0
            )
        }

    def _find_node_by_id(self, chunk_id: str) -> Optional[Node]:
        """Finds a node by chunk ID."""
        queue = deque([self._root])
        while queue:
            current = queue.popleft()
            if current.chunk and current.chunk.chunk_id == chunk_id:
                return current
            queue.extend(current.children.values())
        return None

    def _find_node(self, path: List[str]) -> Optional[Node]:
        """Finds a node by path."""
        current = self._root
        for part in path:
            if part not in current.children:
                return None
            current = current.children[part]
        return current

    async def optimize_cache(self) -> None:
        """Optimizes cache by removing least recently used items."""
        if len(self._docs) > self._max_cache_size:
            sorted_docs = sorted(
                self._docs.items(),
                key=lambda x: x[1].get('metadata', {}).get('last_modified', datetime.min)
            )
            to_remove = sorted_docs[:-self._max_cache_size]
            for chunk_id, _ in to_remove:
                self._docs.pop(chunk_id)
                logger.debug(f"Optimized cache by removing chunk {chunk_id}")
