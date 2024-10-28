"""
context_retrieval.py

Enhanced context retrieval system with smart chunk prioritization and relevance scoring.
Implements proximity-based ranking and dependency-aware context gathering.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime, timedelta
import math
from collections import defaultdict
from heapq import heappush, heappop
from functools import lru_cache
from pathlib import Path

from code_chunk import CodeChunk, ChunkType
from token_utils import TokenManager
from metrics import MetricsResult

logger = logging.getLogger(__name__)

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