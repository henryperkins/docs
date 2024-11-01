"""
combined_context_manager.py

Combines context retrieval and management functionalities.
Manages code context and relationships between code chunks with enhanced caching,
metrics tracking, hierarchical organization, dependency graphs, and semantic similarity.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import aiofiles
import math
from collections import deque, defaultdict
import shutil
import hashlib
from functools import lru_cache
import ast
import networkx as nx
from sentence_transformers import SentenceTransformer
from heapq import heappush, heappop

from code_chunk import CodeChunk, ChunkType
from token_utils import TokenManager, TokenizerModel, TokenizationError
from metrics import MetricsResult
from metrics_utils import calculate_code_metrics_with_metadata, CodeMetadata

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

# Data Classes
@dataclass
class ChunkLocation:
    """Represents the location of a chunk in the project hierarchy."""
    project_path: str
    module_path: str
    class_name: Optional[str] = None
    function_name: Optional[str] = None
    start_line: int = 0
    end_line: int = 0

    def get_hierarchy_path(self) -> str:
        """Returns the full hierarchical path of the chunk."""
        parts = [self.module_path]
        if self.class_name:
            parts.append(self.class_name)
        if self.function_name:
            parts.append(self.function_name)
        return ".".join(parts)

    def overlaps_with(self, other: 'ChunkLocation') -> bool:
        """Checks if this location overlaps with another."""
        if self.module_path != other.module_path:
            return False
        return (self.start_line <= other.end_line and
                self.end_line >= other.start_line)

@dataclass
class ChunkMetadata:
    """Metadata for a code chunk."""
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    hash: str = ""
    dependencies: Set[str] = field(default_factory=set)
    metrics: Optional[MetricsResult] = None

    def update_hash(self, content: str) -> None:
        """Updates the content hash."""
        self.hash = hashlib.sha256(content.encode()).hexdigest()
        self.last_modified = datetime.now()

@dataclass
class RelevanceScore:
    """Represents the relevance score of a code chunk."""
    chunk_id: str
    score: float
    distance: int
    last_accessed: datetime
    dependency_depth: int
    usage_count: int = 0
    tokens: int = 0

@dataclass
class ChunkPriority:
    """Represents the priority of a code chunk for context retrieval."""
    proximity_score: float
    dependency_score: float
    temporal_score: float
    usage_score: float
    final_score: float = field(init=False)

    def __post_init__(self):
        weights = {
            'proximity': 0.4,
            'dependency': 0.3,
            'temporal': 0.2,
            'usage': 0.1
        }
        self.final_score = (
            self.proximity_score * weights['proximity'] +
            self.dependency_score * weights['dependency'] +
            self.temporal_score * weights['temporal'] +
            self.usage_score * weights['usage']
        )

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
        token_model: TokenizerModel = TokenizerModel.GPT4,
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

    # Core Methods

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

        related_chunks = self._find_related_chunks(chunk)
        for related_chunk_id in related_chunks:
            self._dependency_graph.add_edge(chunk.chunk_id, related_chunk_id)

    def _find_related_chunks(self, chunk: CodeChunk, threshold: float = 0.7) -> List[str]:
        """
        Finds related chunks based on semantic similarity.

        Args:
            chunk: CodeChunk to compare
            threshold: Similarity threshold

        Returns:
            List of related chunk IDs
        """
        chunk_embedding = self._embedding_model.encode(chunk.chunk_content)
        related_chunks = []

        for node_id in self._dependency_graph.nodes:
            other_chunk = self._dependency_graph.nodes[node_id]['chunk']
            other_embedding = self._embedding_model.encode(other_chunk.chunk_content)
            similarity = self._cosine_similarity(chunk_embedding, other_embedding)

            if similarity > threshold:
                related_chunks.append(node_id)

        return related_chunks

    @staticmethod
    def _cosine_similarity(vec1: Any, vec2: Any) -> float:
        """Calculates cosine similarity between two vectors."""
        return float((vec1 @ vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

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

    def _get_context_from_tree(
        self,
        node: Node,
        language: str,
        max_tokens: int
    ) -> List[CodeChunk]:
        """
        Performs intelligent context gathering using BFS with token limiting.

        Args:
            node: Starting node
            language: Programming language
            max_tokens: Maximum total tokens

        Returns:
            List[CodeChunk]: Context chunks within token limit
        """
        context_chunks = []
        total_tokens = 0
        visited = set()
        queue = deque([(node, 0)])  # (node, depth)

        while queue and total_tokens < max_tokens:
            current, depth = queue.popleft()

            if current.chunk_id in visited:
                continue

            visited.add(current.chunk_id)

            if (current.chunk and
                current.chunk.language == language and
                current.metadata):

                # Check if adding this chunk would exceed token limit
                if (total_tokens + current.metadata.token_count <= max_tokens):
                    context_chunks.append(current.chunk)
                    total_tokens += current.metadata.token_count

                    # Add related chunks based on dependencies
                    for dep_id in current.metadata.dependencies:
                        dep_node = self._find_node_by_id(dep_id)
                        if dep_node and dep_node.chunk_id not in visited:
                            queue.append((dep_node, depth + 1))

            # Add siblings and children with priority based on depth
            siblings = [
                (n, depth) for n in current.parent.children.values()
                if n.chunk_id not in visited
            ] if current.parent else []

            children = [
                (n, depth + 1) for n in current.children.values()
                if n.chunk_id not in visited
            ]

            # Prioritize closer relationships
            queue.extend(sorted(siblings + children, key=lambda x: x[1]))

        return context_chunks

    # Caching Methods

    async def _cache_documentation(
        self,
        chunk_id: str,
        documentation: Dict[str, Any]
    ) -> None:
        """Caches documentation to disk with error handling."""
        if not self._cache_dir:
            return

        cache_path = self._cache_dir / f"{chunk_id}.json"
        try:
            async with aiofiles.open(cache_path, 'w', encoding="utf-8") as f:
                await f.write(json.dumps({
                    'documentation': documentation,
                    'timestamp': datetime.now().isoformat(),
                    'metadata': {
                        'token_count': self._get_doc_token_count(documentation),
                        'hash': self._calculate_doc_hash(documentation)
                    }
                }, indent=2))
            logger.debug(f"Cached documentation for chunk {chunk_id}")

        except Exception as e:
            logger.error(f"Failed to cache documentation: {e}")
            raise CacheError(f"Cache write failed: {str(e)}")

    def _get_doc_token_count(self, documentation: Dict[str, Any]) -> int:
        """Calculates token count for documentation."""
        try:
            return TokenManager.count_tokens(
                json.dumps(documentation),
                model=self._token_model
            ).token_count
        except TokenizationError:
            return 0

    def _calculate_doc_hash(self, documentation: Dict[str, Any]) -> str:
        """Calculates hash for documentation content."""
        return hashlib.sha256(
            json.dumps(documentation, sort_keys=True).encode()
        ).hexdigest()

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

    async def get_related_chunks(
        self,
        chunk_id: str,
        max_distance: int = 2
    ) -> List[CodeChunk]:
        """
        Gets related chunks based on dependencies and proximity.

        Args:
            chunk_id: Starting chunk
            max_distance: Maximum relationship distance

        Returns:
            List[CodeChunk]: Related chunks
        """
        node = self._find_node_by_id(chunk_id)
        if not node:
            return []

        related = []
        visited = set()
        queue = deque([(node, 0)])  # (node, distance)

        while queue:
            current, distance = queue.popleft()

            if distance > max_distance:
                continue

            if current.chunk_id in visited:
                continue

            visited.add(current.chunk_id)

            if current.chunk and current != node:
                related.append(current.chunk)

            # Add dependencies from the dependency graph
            if self._dependency_graph.has_node(current.chunk_id):
                for dep_id in self._dependency_graph.predecessors(current.chunk_id):
                    dep_node = self._find_node_by_id(dep_id)
                    if dep_node and dep_node.chunk_id not in visited:
                        queue.append((dep_node, distance + 1))

            # Add siblings
            if current.parent:
                for sibling in current.parent.children.values():
                    if sibling != current and sibling.chunk_id not in visited:
                        queue.append((sibling, distance + 1))

            # Add children
            for child in current.children.values():
                if child.chunk_id not in visited:
                    queue.append((child, distance + 1))

        return related

    # Helper Methods

    async def _process_dependencies(self, chunk: CodeChunk) -> None:
        """Processes and updates dependencies for a chunk."""
        # Placeholder for any additional dependency processing logic
        pass

# ContextRetriever Class
class ContextRetriever:
    """Enhanced context retrieval with smart prioritization."""

    def __init__(self, token_manager: TokenManager):
        self.token_manager = token_manager
        self._usage_history: Dict[str, int] = defaultdict(int)
        self._last_access: Dict[str, datetime] = {}
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._importance_cache: Dict[str, float] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self.CACHE_DURATION = timedelta(hours=1)

    @lru_cache(maxsize=128)
    async def get_relevant_chunks(
        self,
        target_chunk: CodeChunk,
        all_chunks: List[CodeChunk],
        max_tokens: int,
        max_distance: int = 3
    ) -> List[CodeChunk]:
        """Gets relevant chunks with semantic and dependency awareness."""
        self._update_usage_stats(target_chunk.chunk_id)
        
        scored_chunks = []
        base_path = self._get_base_path(target_chunk)
        
        for chunk in all_chunks:
            if chunk.chunk_id == target_chunk.chunk_id:
                continue
                
            distance = self._calculate_distance(
                base_path,
                self._get_base_path(chunk)
            )
            
            if distance > max_distance:
                continue
                
            score = await self._calculate_relevance_score(
                chunk,
                target_chunk,
                distance
            )
            
            scored_chunks.append((chunk, score))
        
        return self._select_chunks_within_limit(
            scored_chunks,
            max_tokens
        )

    async def _calculate_relevance_score(
        self,
        chunk: CodeChunk,
        target_chunk: CodeChunk,
        distance: int
    ) -> float:
        """Calculates comprehensive relevance score."""
        cache_key = f"{chunk.chunk_id}:{target_chunk.chunk_id}"
        
        if self._is_cached_score_valid(cache_key):
            return self._importance_cache[cache_key]
            
        scores = {
            'proximity': self._calculate_proximity_score(distance),
            'dependency': await self._calculate_dependency_score(
                chunk.chunk_id,
                target_chunk.chunk_id
            ),
            'usage': self._calculate_usage_score(chunk.chunk_id),
            'semantic': await self._calculate_semantic_similarity(chunk, target_chunk)
        }
        
        weights = {
            'proximity': 0.3,
            'dependency': 0.25,
            'temporal': 0.15,
            'usage': 0.15,
            'semantic': 0.15
        }
        
        final_score = sum(
            score * weights[factor]
            for factor, score in scores.items()
        )
        
        # Cache the calculated score
        self._importance_cache[cache_key] = final_score
        self._cache_expiry[cache_key] = datetime.now() + self.CACHE_DURATION
        
        return final_score

    def _select_chunks_within_limit(
        self,
        scored_chunks: List[Tuple[CodeChunk, float]],
        max_tokens: int
    ) -> List[CodeChunk]:
        """Selects optimal combination of chunks within token limit."""
        selected_chunks = []
        current_tokens = 0
        
        # Sort chunks by score/token ratio for optimal selection
        chunks_by_efficiency = sorted(
            scored_chunks,
            key=lambda x: x[1] / x[0].token_count,
            reverse=True
        )
        
        for chunk, score in chunks_by_efficiency:
            if current_tokens + chunk.token_count <= max_tokens:
                selected_chunks.append(chunk)
                current_tokens += chunk.token_count
            else:
                break
                
        return selected_chunks

    @lru_cache(maxsize=1024)
    def _calculate_distance(self, path1: str, path2: str) -> int:
        """Calculates hierarchical distance between two paths."""
        parts1 = path1.split(".")
        parts2 = path2.split(".")
        
        # Calculate LCS length
        m, n = len(parts1), len(parts2)
        lcs = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if parts1[i-1] == parts2[j-1]:
                    lcs[i][j] = lcs[i-1][j-1] + 1
                else:
                    lcs[i][j] = max(lcs[i-1][j], lcs[i][j-1])
        
        common_path_length = lcs[m][n]
        return (m - common_path_length) + (n - common_path_length)

    async def _calculate_dependency_score(
        self,
        source_id: str,
        target_id: str
    ) -> float:
        """Calculates dependency-based relevance score."""
        if source_id not in self._dependency_graph:
            return 0.0
            
        # Perform BFS to find shortest path
        visited = {source_id}
        queue = [(source_id, 0)]
        min_distance = float('inf')
        
        while queue:
            current_id, distance = queue.pop(0)
            
            if current_id == target_id:
                min_distance = min(min_distance, distance)
                continue
                
            for dep_id in self._dependency_graph[current_id]:
                if dep_id not in visited:
                    visited.add(dep_id)
                    queue.append((dep_id, distance + 1))
        
        if min_distance == float('inf'):
            return 0.0
            
        # Convert distance to score (closer = higher score)
        return 1.0 / (1.0 + min_distance)

    async def _calculate_semantic_similarity(
        self,
        chunk1: CodeChunk,
        chunk2: CodeChunk
    ) -> float:
        """Calculates semantic similarity between chunks."""
        try:
            # Get AST-based features
            features1 = await self._extract_semantic_features(chunk1)
            features2 = await self._extract_semantic_features(chunk2)
            
            # Calculate Jaccard similarity
            intersection = len(features1.intersection(features2))
            union = len(features1.union(features2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0

    async def _extract_semantic_features(self, chunk: CodeChunk) -> Set[str]:
        """Extracts semantic features from a chunk."""
        features = set()
        
        try:
            tree = ast.parse(chunk.content)
            
            class FeatureExtractor(ast.NodeVisitor):
                def visit_Name(self, node: ast.Name):
                    features.add(f"name:{node.id}")
                    
                def visit_Call(self, node: ast.Call):
                    if isinstance(node.func, ast.Name):
                        features.add(f"call:{node.func.id}")
                        
                def visit_FunctionDef(self, node: ast.FunctionDef):
                    features.add(f"func:{node.name}")
                    for arg in node.args.args:
                        features.add(f"arg:{arg.arg}")
                        
                def visit_ClassDef(self, node: ast.ClassDef):
                    features.add(f"class:{node.name}")
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            features.add(f"inherits:{base.id}")
            
            extractor = FeatureExtractor()
            extractor.visit(tree)
            
        except Exception as e:
            logger.error(f"Error extracting semantic features: {e}")
            
        return features

    def _calculate_temporal_score(self, chunk_id: str) -> float:
        """Calculates time-based relevance score."""
        if chunk_id not in self._last_access:
            return 0.0
            
        time_diff = datetime.now() - self._last_access[chunk_id]
        hours_diff = time_diff.total_seconds() / 3600
        
        # Exponential decay with 24-hour half-life
        return math.exp(-hours_diff / 24)

    def _calculate_usage_score(self, chunk_id: str) -> float:
        """Calculates usage-based relevance score."""
        if chunk_id not in self._usage_history:
            return 0.0
            
        usage_count = self._usage_history[chunk_id]
        # Logarithmic scaling to prevent over-dominance of frequently used chunks
        return math.log1p(usage_count) / 10.0

    def _calculate_proximity_score(self, distance: int) -> float:
        """Calculates proximity-based relevance score."""
        if distance <= 0:
            return 1.0
        # Exponential decay based on distance
        return math.exp(-distance / 2)

    def _update_usage_stats(self, chunk_id: str) -> None:
        """Updates usage statistics for a chunk."""
        self._usage_history[chunk_id] += 1
        self._last_access[chunk_id] = datetime.now()

    def _is_cached_score_valid(self, cache_key: str) -> bool:
        """Checks if cached score is still valid."""
        if cache_key not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[cache_key]

    @staticmethod
    def _get_base_path(chunk: CodeChunk) -> str:
        """Gets the base path for a chunk's location."""
        parts = []
        if chunk.file_path:
            parts.append(Path(chunk.file_path).stem)
        if chunk.class_name:
            parts.append(chunk.class_name)
        if chunk.function_name:
            parts.append(chunk.function_name)
        return ".".join(parts)

    async def update_dependency_graph(self, chunks: List[CodeChunk]) -> None:
        """Updates the dependency graph with new chunks."""
        try:
            # Clear existing dependencies for these chunks
            for chunk in chunks:
                self._dependency_graph[chunk.chunk_id] = set()

            # Rebuild dependencies
            for chunk in chunks:
                await self._analyze_chunk_dependencies(chunk)
                
        except Exception as e:
            logger.error(f"Error updating dependency graph: {e}")

    async def _analyze_chunk_dependencies(self, chunk: CodeChunk) -> None:
        """Analyzes and updates dependencies for a single chunk."""
        try:
            tree = ast.parse(chunk.content)
            
            class DependencyVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.dependencies = set()
                    self.imported_names = set()
                    
                def visit_Import(self, node: ast.Import):
                    for alias in node.names:
                        self.imported_names.add(alias.name)
                        
                def visit_ImportFrom(self, node: ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        full_name = f"{module}.{alias.name}"
                        self.imported_names.add(full_name)
                        
                def visit_Name(self, node: ast.Name):
                    if isinstance(node.ctx, ast.Load) and node.id in self.imported_names:
                        self.dependencies.add(node.id)
                        
                def visit_Call(self, node: ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.imported_names:
                            self.dependencies.add(node.func.id)
                    self.generic_visit(node)

            visitor = DependencyVisitor()
            visitor.visit(tree)
            
            # Update dependency graph
            self._dependency_graph[chunk.chunk_id].update(visitor.dependencies)
            
        except Exception as e:
            logger.error(f"Error analyzing chunk dependencies: {e}")

    async def get_dependency_chain(
        self,
        chunk: CodeChunk,
        max_depth: int = 3
    ) -> List[CodeChunk]:
        """Gets the dependency chain for a chunk up to max_depth."""
        visited = set()
        chain = []
        
        async def traverse_dependencies(current_chunk: CodeChunk, depth: int):
            if depth > max_depth or current_chunk.chunk_id in visited:
                return
                
            visited.add(current_chunk.chunk_id)
            chain.append(current_chunk)
            
            # Get direct dependencies
            for dep_id in self._dependency_graph[current_chunk.chunk_id]:
                dep_chunk = self._find_chunk_by_id(dep_id)
                if dep_chunk:
                    await traverse_dependencies(dep_chunk, depth + 1)

        await traverse_dependencies(chunk, 0)
        return chain

    def _find_chunk_by_id(self, chunk_id: str) -> Optional[CodeChunk]:
        """Finds a chunk by its ID in the current context."""
        # This method should be implemented based on how chunks are stored
        # in your application
        pass

    async def get_related_contexts(
        self,
        chunk: CodeChunk,
        all_chunks: List[CodeChunk],
        max_contexts: int = 5
    ) -> List[CodeChunk]:
        """Gets related contexts based on semantic similarity."""
        if not chunk.content.strip():
            return []

        try:
            # Extract semantic features for target chunk
            target_features = await self._extract_semantic_features(chunk)
            
            # Calculate similarities
            similarities = []
            for other_chunk in all_chunks:
                if other_chunk.chunk_id == chunk.chunk_id:
                    continue
                    
                other_features = await self._extract_semantic_features(other_chunk)
                similarity = len(target_features.intersection(other_features)) / \
                           len(target_features.union(other_features)) if target_features else 0
                           
                similarities.append((other_chunk, similarity))
            
            # Sort by similarity and return top contexts
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [chunk for chunk, _ in similarities[:max_contexts]]
            
        except Exception as e:
            logger.error(f"Error getting related contexts: {e}")
            return []

    def clear_caches(self) -> None:
        """Clears all internal caches."""
        self._importance_cache.clear()
        self._cache_expiry.clear()
        self._calculate_distance.cache_clear()
        logger.debug("Cleared all context retriever caches")

    async def optimize_context_window(
        self,
        chunks: List[CodeChunk],
        max_tokens: int
    ) -> List[CodeChunk]:
        """Optimizes the context window for the given chunks."""
        try:
            # Calculate importance scores for all chunks
            scored_chunks = []
            for chunk in chunks:
                importance = await self._calculate_chunk_importance(chunk)
                token_count = chunk.token_count
                scored_chunks.append((chunk, importance, token_count))

            # Sort by importance/token ratio
            scored_chunks.sort(key=lambda x: x[1]/x[2], reverse=True)
            
            # Select chunks while respecting token limit
            selected_chunks = []
            current_tokens = 0
            
            for chunk, importance, tokens in scored_chunks:
                if current_tokens + tokens <= max_tokens:
                    selected_chunks.append(chunk)
                    current_tokens += tokens
                else:
                    break
                    
            return selected_chunks
            
        except Exception as e:
            logger.error(f"Error optimizing context window: {e}")
            return chunks

    async def _calculate_chunk_importance(self, chunk: CodeChunk) -> float:
        """Calculates overall importance score for a chunk."""
        try:
            # Combine multiple factors for importance
            factors = {
                'complexity': chunk.metadata.complexity or 0,
                'dependencies': len(self._dependency_graph[chunk.chunk_id]),
                'usage': self._usage_history.get(chunk.chunk_id, 0),
                'recency': self._calculate_temporal_score(chunk.chunk_id)
            }
            
            weights = {
                'complexity': 0.3,
                'dependencies': 0.3,
                'usage': 0.2,
                'recency': 0.2
            }
            
            return sum(
                score * weights[factor]
                for factor, score in factors.items()
            )
            
        except Exception as e:
            logger.error(f"Error calculating chunk importance: {e}")
            return 0.0
