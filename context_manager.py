import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import timeit  # Import for benchmarking
from collections import deque
import shutil
import aiofiles
from dataclasses import dataclass

from code_chunk import CodeChunk

logger = logging.getLogger(__name__)


class ChunkNotFoundError(Exception):
    """Custom exception."""
    pass


class Node:
    """Node in the context tree."""

    def __init__(self, name: str, chunk: Optional[CodeChunk] = None, location: Optional[ChunkLocation] = None):
        self.name = name
        self.chunk = chunk
        self.location = location
        self.children: Dict[str, "Node"] = {}

@dataclass
class ChunkLocation:
    """Represents the location of a chunk in the project hierarchy."""
    project_path: str
    module_path: str
    class_name: Optional[str]
    function_name: Optional[str]

    def get_hierarchy_path(self) -> str:
        """Returns a string representation of the chunk's hierarchical path."""
        parts = [self.module_path]
        if self.class_name:
            parts.append(self.class_name)
        if self.function_name:
            parts.append(self.function_name)
        return ".".join(parts)
        

class HierarchicalContextManager:
    """Manages code chunks and documentation with a tree structure and caching."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initializes the context manager."""
        self._root = Node("root")
        self._docs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            

    def _get_location(self, chunk: CodeChunk) -> ChunkLocation:
        """Determines the hierarchical location for a chunk."""
        project_path = str(Path(chunk.file_path).parent)
        module_path = Path(chunk.file_path).stem
        return ChunkLocation(
            project_path=project_path,
            module_path=module_path,
            class_name=chunk.class_name,
            function_name=chunk.function_name,
        )

    async def add_code_chunk(self, chunk: CodeChunk) -> None:
        """Adds a code chunk to the tree."""
        async with self._lock:
            path = chunk.get_hierarchy_path().split(".")
            current = self._root
            for part in path:
                current = current.children.setdefault(part, Node(part))
            current.chunk = chunk
            current.location = self._get_location(chunk)  # Store location
            logger.debug(f"Added chunk {chunk.chunk_id}")

    async def update_code_chunk(self, chunk: CodeChunk) -> None:
        """Updates an existing chunk in the tree."""
        async with self._lock:
            path = chunk.get_hierarchy_path().split(".")
            node = self._find_node(path)
            if node:
                node.chunk = chunk
                node.location = self._get_location(chunk)  # Update location
                logger.debug(f"Updated chunk {chunk.chunk_id}")
            else:
                raise ChunkNotFoundError(f"No chunk found for path: {'.'.join(path)}")

    async def remove_code_chunk(self, chunk_id: str) -> None:
        """Removes a chunk and its documentation."""
        async with self._lock:
            if chunk_id not in self._chunks:
                raise ChunkNotFoundError(f"No chunk found with ID {chunk_id}")

            del self._chunks[chunk_id]
            self._docs.pop(chunk_id, None)  # Remove associated documentation
            del self._chunk_locations[chunk_id]

            if self._cache_dir:  # Remove from cache if enabled
                cache_path = self._cache_dir / f"{chunk_id}.json"
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                    except Exception as e:
                        logger.error(f"Failed to remove cached documentation: {e}")

            logger.debug(f"Removed chunk {chunk_id}")

    def _get_all_chunk_ids(self) -> Set[str]:
        """Helper function to get all chunk IDs from the tree."""
        chunk_ids = set()
        queue = deque([self._root])
        while queue:
            current = queue.popleft()
            if current.chunk:
                chunk_ids.add(current.chunk.chunk_id)
            for child in current.children.values():
                queue.append(child)
        return chunk_ids


	def _find_node(self, path: List[str]) -> Optional[Node]:
        """Finds a node in the tree by its path."""
        current = self._root
        for part in path:
            if part not in current.children:
                return None
            current = current.children[part]
        return current

    async def get_context_for_function(
        self, module_path: str, function_name: str, language: str, max_tokens: int = 4096
    ) -> List[CodeChunk]:
        """Gets context chunks for a function using the tree."""
        async with self._lock:
            module_name = Path(module_path).stem
            path = [module_name, function_name]
            node = self._find_node(path)
            if node and node.chunk:
                return self._get_context_from_tree(node, language, max_tokens)
            return []

    async def get_context_for_class(
        self, module_path: str, class_name: str, language: str, max_tokens: int = 4096
    ) -> List[CodeChunk]:
        """Gets context chunks for a class using the tree."""
        async with self._lock:
            module_name = Path(module_path).stem
            path = [module_name, class_name]  # Corrected path for class
            node = self._find_node(path)
            if node:  # Check if node is found
                return self._get_context_from_tree(node, language, max_tokens)
            return []

    async def get_context_for_module(
        self, module_path: str, language: str, max_tokens: int = 4096
    ) -> List[CodeChunk]:
        """Gets context chunks for a module using the tree."""
        async with self._lock:
            module_name = Path(module_path).stem
            path = [module_name]  # Corrected path for module
            node = self._find_node(path)
            if node:  # Check if node is found
                return self._get_context_from_tree(node, language, max_tokens)
            return []

    def _get_context_from_tree(self, node: Node, language: str, max_tokens: int) -> List[CodeChunk]:
        """Performs a breadth-first search to gather context chunks."""
        context_chunks = []
        total_tokens = 0
        queue = deque([node])

        while queue and total_tokens < max_tokens:
            current = queue.popleft()
            if current.chunk and current.chunk.language == language:
                if total_tokens + current.chunk.token_count <= max_tokens:
                    context_chunks.append(current.chunk)
                    total_tokens += current.chunk.token_count

            for child in current.children.values():
                queue.append(child)

        return context_chunks

	async def _cache_documentation(self, chunk_id: str, documentation: Dict[str, Any]) -> None:
        """Caches documentation to disk."""
        if not self._cache_dir:
            return

        cache_path = self._cache_dir / f"{chunk_id}.json"
        try:
            async with aiofiles.open(cache_path, 'w', encoding="utf-8") as f:
                await f.write(json.dumps(documentation, indent=2))  # Format JSON for readability
            logger.debug(f"Cached documentation for chunk {chunk_id}")
        except Exception as e:
            logger.error(f"Failed to cache documentation: {e}")
            

    async def get_documentation_for_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves documentation, checking cache first."""

        async with self._lock:
            # 1. Check in-memory cache
            if chunk_id in self._docs:
                logger.debug(f"Documentation found in memory for chunk {chunk_id}")
                return self._docs[chunk_id]

            # 2. Check disk cache
            if self._cache_dir:
                cache_path = self._cache_dir / f"{chunk_id}.json"
                if cache_path.exists():
                    try:
                        async with aiofiles.open(cache_path, 'r', encoding="utf-8") as f:
                            cached_doc = json.loads(await f.read())

                        # Cache Invalidation (Hash-Based)
                        chunk = self._chunks.get(chunk_id)
                        if chunk:
                            current_hash = chunk._calculate_hash()  # Assuming _calculate_hash is available
                            cached_hash = cached_doc.get("metadata", {}).get("hash")
                            if current_hash == cached_hash:
                                self._docs[chunk_id] = cached_doc  # Update in-memory cache
                                logger.debug(f"Documentation loaded from cache for chunk {chunk_id}")
                                return cached_doc
                            else:
                                logger.debug(f"Cache invalidated (hash mismatch) for chunk {chunk_id}")
                        else:
                            logger.warning(f"Chunk {chunk_id} not found in context, but documentation exists in cache.")

                    except Exception as e:
                        logger.error(f"Failed to load documentation from cache: {e}")

            # 3. Not in cache (or cache invalid), generate/fetch documentation
            logger.debug(f"Documentation not found in cache for chunk {chunk_id}")
            return None  # Or call the documentation generation function here if appropriate

    async def add_doc_chunk(self, chunk_id: str, documentation: Dict[str, Any]) -> None:
        """Adds or updates documentation for a specific chunk, updating the cache."""
        async with self._lock:
            if chunk_id not in self._chunks:
                raise ChunkNotFoundError(f"No chunk found with ID {chunk_id}")

            self._docs[chunk_id] = documentation
            if self._cache_dir:
                await self._cache_documentation(chunk_id, documentation)
            logger.debug(f"Added documentation for chunk {chunk_id}")

    async def clear_context(self) -> None:
        """Clears all chunks, documentation, and the cache."""
        async with self._lock:
            self._root = Node("root")  # Reset the tree
            self._docs.clear()

            if self._cache_dir:
                try:
                    # Remove the cache directory and recreate it
                    shutil.rmtree(self._cache_dir)
                    self._cache_dir.mkdir()  # Recreate an empty cache directory
                    logger.debug("Cache directory cleared")
                except OSError as e:
                    logger.error(f"Error clearing cache directory: {e}")



# Benchmarking
async def benchmark_context_retrieval(manager, module_path, function_name, language): # Make async
    """Benchmarks get_context_for_function asynchronously."""
    with manager.performance_monitor.measure("context_retrieval"):  # Assuming performance_monitor is available
        return await manager.get_context_for_function(module_path, function_name, language)


if __name__ == "__main__":
    import random
    import string
    from dataclasses import dataclass

    @dataclass  # Example ChunkLocation (adapt as needed)
    class ChunkLocation:
        project_path: str
        module_path: str
        class_name: Optional[str]
        function_name: Optional[str]

        def get_hierarchy_path(self) -> str:
            return ".".join([p for p in [self.module_path, self.class_name, self.function_name] if p])


    # Example usage and benchmarking
    cache_dir = "test_cache"
    manager_flat = HierarchicalContextManager()  # Original flat dictionary version (no cache)
    manager_tree = HierarchicalContextManager(cache_dir=cache_dir)  # Refactored tree version (with cache)

    # Create some sample CodeChunk objects with varying attributes
    chunks = []
    for i in range(1000):  # Example: 1000 chunks
        chunk = CodeChunk(
            file_path=f"module_{i % 10}.py",  # 10 different modules
            start_line=i * 10,
            end_line=i * 10 + 5,
            function_name=f"function_{i % 50}" if i % 2 == 0 else None,  # 50 different functions
            class_name=f"Class_{i % 20}" if i % 3 == 0 else None,  # 20 different classes
            chunk_content="".join(random.choices(string.ascii_letters, k=100)),  # Random content
            token_count=100,
            language="python",
        )
        chunks.append(chunk)

	async def populate_managers(chunks):
        """Populate both managers with chunks."""
        for chunk in chunks:
            await manager_flat.add_code_chunk(chunk)
            await manager_tree.add_code_chunk(chunk)

    asyncio.run(populate_managers(chunks))

    # Benchmark the original (flat dictionary) approach
    original_time = timeit.timeit(
        lambda: asyncio.run(benchmark_context_retrieval(manager_flat, "module_5.py", "function_25", "python")),
        number=100
    )

    # Benchmark the refactored (tree-based) approach
    refactored_time = timeit.timeit(
        lambda: asyncio.run(benchmark_context_retrieval(manager_tree, "module_5.py", "function_25", "python")),
        number=100
    )

    print(f"Original context retrieval time: {original_time:.4f} seconds")
    print(f"Refactored context retrieval time: {refactored_time:.4f} seconds")

    # Cleanup the test cache directory
    shutil.rmtree(cache_dir, ignore_errors=True)