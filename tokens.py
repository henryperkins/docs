# combined_token_management.py

"""
combined_token_management.py - Comprehensive module for tokenization and token management.

This module provides utilities for tokenizing text using various tokenizer models,
counting tokens, decoding tokens back to text, handling special tokens, validating
token limits, splitting text based on token limits, and batch token processing.
Additionally, it integrates advanced embedding calculations and similarity measurements
for enhanced text analysis, along with dynamic token limit management and adaptive
chunking strategies.
"""

import tiktoken
import asyncio
from typing import List, Optional, Union, Dict, Set
from dataclasses import dataclass, field
import logging
from enum import Enum, auto
import threading
import numpy as np
import ast
from collections import defaultdict

# Assuming these are imported from another module
from metrics_utils import EnhancedEmbeddingCalculator, CodeMetadata, EmbeddingManager, TokenizationError
from code_chunk import CodeChunk, ChunkType
from context_retrieval import ContextRetriever

logger = logging.getLogger(__name__)

class TokenizerModel(Enum):
    """Supported tokenizer models."""
    GPT4 = "cl100k_base"
    GPT3 = "p50k_base"
    CODEX = "p50k_edit"

@dataclass
class TokenizationResult:
    """Results from a tokenization operation."""
    tokens: List[int]
    token_count: int
    encoding_name: str
    special_tokens: Dict[str, int] = None
    error: Optional[str] = None

class TokenManager:
    """
    Manages tokenization operations with caching, thread-safety, and embedding functionalities.

    Features:
        - Token counting and decoding.
        - Handling of special tokens.
        - Validation against token limits.
        - Splitting text based on token limits.
        - Batch token processing.
        - Enhanced embedding calculations and similarity measurements.
    """

    _encoders = {}  # Cache for different encoders
    _default_model = TokenizerModel.GPT4
    _lock = threading.Lock()  # Lock for thread safety
    _embedding_manager: Optional[EmbeddingManager] = None  # Lazy initialization

    @classmethod
    def get_encoder(cls, model: TokenizerModel = None) -> tiktoken.Encoding:
        """Retrieves the appropriate encoder instance."""
        with cls._lock:  # Ensure thread-safe access
            try:
                model = model or cls._default_model
                if model not in cls._encoders:
                    logger.debug(f"Creating new encoder for model: {model.value}")
                    cls._encoders[model] = tiktoken.get_encoding(model.value)
                return cls._encoders[model]
            except Exception as e:
                logger.error(f"Failed to create encoder for model {model}: {e}")
                raise TokenizationError(f"Failed to create encoder: {str(e)}")

    @classmethod
    def count_tokens(cls, text: Union[str, List[str]], model: TokenizerModel = None, include_special_tokens: bool = False) -> TokenizationResult:
        """Counts tokens in the provided text with enhanced error handling."""
        logger.debug(f"Counting tokens for text: {text[:50]}...")  # Truncate text for logging
        try:
            if not text:
                logger.warning("Empty input provided for token counting.")
                return TokenizationResult([], 0, "", error="Empty input")

            encoder = cls.get_encoder(model)  # Get encoder (thread-safe)
            model = model or cls._default_model

            if isinstance(text, list):
                text = " ".join(text)

            tokens = encoder.encode(text)
            logger.debug(f"Encoded tokens: {tokens[:10]}...")  # Truncate tokens for logging

            special_tokens = None
            if include_special_tokens:
                special_tokens = cls._count_special_tokens(text, encoder)

            return TokenizationResult(
                tokens=tokens,
                token_count=len(tokens),
                encoding_name=model.value,
                special_tokens=special_tokens
            )

        except TokenizationError:
            raise  # Re-raise custom exceptions without modification
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            raise TokenizationError(f"Failed to count tokens: {str(e)}")

    @classmethod
    def decode_tokens(cls, tokens: List[int], model: TokenizerModel = None) -> str:
        """Decodes a list of tokens back to text."""
        logger.debug(f"Decoding tokens: {tokens[:10]}...")  # Truncate tokens for logging
        try:
            if not tokens:
                logger.warning("Empty token list provided for decoding.")
                return ""

            encoder = cls.get_encoder(model)
            decoded_text = encoder.decode(tokens)
            logger.debug(f"Decoded text: {decoded_text[:50]}...")  # Truncate text for logging
            return decoded_text

        except TokenizationError:
            raise
        except Exception as e:
            logger.error(f"Token decoding error: {e}")
            raise TokenizationError(f"Failed to decode tokens: {str(e)}")

    @classmethod
    def _count_special_tokens(cls, text: str, encoder: tiktoken.Encoding) -> Dict[str, int]:
        """Counts special tokens in the text."""
        special_tokens = {}
        try:
            # Example: count newlines and code blocks
            special_tokens["newlines"] = text.count("\n")
            special_tokens["code_blocks"] = text.count("```")
            logger.debug(f"Special tokens counted: {special_tokens}")
            return special_tokens
        except Exception as e:
            logger.warning(f"Error counting special tokens: {e}")
            return {}

    @classmethod
    def validate_token_limit(cls, text: str, max_tokens: int, model: TokenizerModel = None) -> bool:
        """Checks if the text exceeds the specified token limit."""
        try:
            result = cls.count_tokens(text, model)
            logger.debug(f"Token count {result.token_count} compared to max {max_tokens}")
            return result.token_count <= max_tokens
        except TokenizationError as e:
            logger.error(f"Validation failed: {e}")
            return False

    @classmethod
    def split_by_token_limit(cls, text: str, max_tokens: int, model: TokenizerModel = None) -> List[str]:
        """Splits the text into chunks, each not exceeding the specified token limit."""
        logger.debug(f"Splitting text with max tokens per chunk: {max_tokens}")
        try:
            encoder = cls.get_encoder(model)
            tokens = encoder.encode(text)
            chunks = []
            current_chunk = []
            current_count = 0

            for token in tokens:
                if current_count + 1 > max_tokens:
                    decoded_chunk = encoder.decode(current_chunk)
                    chunks.append(decoded_chunk)
                    logger.debug(f"Created chunk with {current_count} tokens.")
                    current_chunk = []
                    current_count = 0

                current_chunk.append(token)
                current_count += 1

            if current_chunk:
                decoded_chunk = encoder.decode(current_chunk)
                chunks.append(decoded_chunk)
                logger.debug(f"Created final chunk with {current_count} tokens.")

            logger.info(f"Total chunks created: {len(chunks)}")
            return chunks

        except TokenizationError:
            raise
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            raise TokenizationError(f"Failed to split text: {str(e)}")

    @classmethod
    def estimate_tokens_from_chars(cls, text: str) -> int:
        """Provides a rough estimation of token count based on character length."""
        estimate = len(text) // 4  # GPT models average ~4 characters per token
        logger.debug(f"Estimated tokens from characters: {estimate}")
        return estimate

    @classmethod
    def batch_count_tokens(cls, texts: List[str], model: TokenizerModel = None) -> List[TokenizationResult]:
        """Counts tokens for multiple texts efficiently."""
        logger.debug(f"Batch counting tokens for {len(texts)} texts.")
        results = []
        try:
            encoder = cls.get_encoder(model)

            for idx, text in enumerate(texts):
                try:
                    tokens = encoder.encode(text)
                    result = TokenizationResult(
                        tokens=tokens,
                        token_count=len(tokens),
                        encoding_name=model.value if model else cls._default_model.value
                    )
                    results.append(result)
                    logger.debug(f"Text {idx+1}: {result.token_count} tokens.")
                except Exception as e:
                    logger.error(f"Error in batch tokenization for text {idx+1}: {e}")
                    results.append(TokenizationResult(
                        tokens=[],
                        token_count=0,
                        encoding_name="",
                        error=str(e)
                    ))

            return results

        except TokenizationError as e:
            logger.error(f"Batch tokenization failed: {e}")
            # Return empty results with errors
            return [TokenizationResult(
                        tokens=[],
                        token_count=0,
                        encoding_name="",
                        error=str(e)
                    ) for _ in texts]

    @classmethod
    def clear_cache(cls):
        """Clears the encoder cache."""
        with cls._lock:
            cls._encoders.clear()
            logger.info("Encoder cache cleared.")

    # Embedding Integration
    @classmethod
    def get_enhanced_embedding(cls, code: str, metadata: CodeMetadata) -> np.ndarray:
        """Generates an enhanced embedding for the given code and metadata."""
        logger.debug(f"Generating enhanced embedding for code: {code[:50]}...")
        try:
            if cls._embedding_manager is None:
                cls._embedding_manager = EmbeddingManager()
                logger.debug("Initialized EmbeddingManager.")

            embedding = cls._embedding_manager.get_embedding(code, metadata)
            logger.debug(f"Generated embedding of shape: {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate enhanced embedding: {e}")
            raise TokenizationError(f"Failed to generate enhanced embedding: {str(e)}")

    @classmethod
    def calculate_similarity(cls, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculates the similarity between two embeddings."""
        logger.debug("Calculating similarity between two embeddings.")
        try:
            if cls._embedding_manager is None:
                cls._embedding_manager = EmbeddingManager()
                logger.debug("Initialized EmbeddingManager.")

            similarity = cls._embedding_manager.compare_embeddings(embedding1, embedding2)
            logger.debug(f"Calculated similarity: {similarity}")
            return similarity
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            raise TokenizationError(f"Failed to calculate similarity: {str(e)}")

    @classmethod
    def set_metadata_weights(cls, new_weights: Dict[str, float]) -> None:
        """Sets new weights for metadata features in embedding calculations."""
        logger.debug(f"Setting new metadata weights: {new_weights}")
        try:
            if cls._embedding_manager is None:
                cls._embedding_manager = EmbeddingManager()
                logger.debug("Initialized EmbeddingManager.")

            cls._embedding_manager.set_metadata_weights(new_weights)
            logger.info("Metadata weights updated successfully.")
        except Exception as e:
            logger.error(f"Failed to set metadata weights: {e}")
            raise TokenizationError(f"Failed to set metadata weights: {str(e)}")

@dataclass
class TokenBudget:
    """Represents token budget allocation for context retrieval."""
    total_limit: int
    reserved_tokens: int = field(default=100)  # Reserve for metadata
    allocation_strategy: 'TokenAllocationStrategy' = 'TokenAllocationStrategy.ADAPTIVE'
    
    @property
    def available_tokens(self) -> int:
        return max(0, self.total_limit - self.reserved_tokens)

class TokenAllocationStrategy(Enum):
    """Strategies for token allocation."""
    PROPORTIONAL = auto()  # Allocate tokens proportionally to importance
    PRIORITY_FIRST = auto()  # Prioritize most important chunks
    BALANCED = auto()  # Balance between importance and content preservation
    ADAPTIVE = auto()  # Adapt based on content type and relationships

class TokenLimitManager:
    """Enhanced token management with improved allocation strategies."""
    def __init__(self, token_manager: TokenManager):
        self.token_manager = token_manager
        self._chunk_tokens: Dict[str, int] = {}
        self._importance_cache: Dict[str, float] = {}
        self._ast_cache: Dict[str, ast.AST] = {}

    async def allocate_tokens(self, chunks: List[CodeChunk], budget: TokenBudget, context_retriever: ContextRetriever) -> Dict[str, int]:
        """Allocates tokens to chunks based on importance and strategy."""
        # Calculate importance scores
        importance_scores = await self._calculate_importance_scores(chunks, context_retriever)
        
        # Apply allocation strategy
        if budget.allocation_strategy == TokenAllocationStrategy.PROPORTIONAL:
            return await self._allocate_proportionally(chunks, importance_scores, budget)
        elif budget.allocation_strategy == TokenAllocationStrategy.PRIORITY_FIRST:
            return await self._allocate_priority_first(chunks, importance_scores, budget)
        elif budget.allocation_strategy == TokenAllocationStrategy.BALANCED:
            return await self._allocate_balanced(chunks, importance_scores, budget)
        else:  # ADAPTIVE
            return await self._allocate_adaptively(chunks, importance_scores, budget)

    async def optimize_chunk_tokens(self, chunk: CodeChunk, allocated_tokens: int) -> Optional[CodeChunk]:
        """Optimizes a chunk to fit within allocated tokens using progressive strategies."""
        optimization_levels = [
            self._optimize_remove_comments,
            self._optimize_simplify_whitespace,
            self._optimize_preserve_interfaces,
            self._optimize_preserve_dependencies,
            self._optimize_aggressive
        ]

        for level in optimization_levels:
            optimized = await level(chunk)
            if optimized:
                tokens = await self._count_tokens(optimized)
                if tokens <= allocated_tokens:
                    return optimized

        return await self._fallback_optimization(chunk, allocated_tokens)

    async def _allocate_proportionally(self, chunks: List[CodeChunk], importance_scores: Dict[str, float], budget: TokenBudget) -> Dict[str, int]:
        """Allocates tokens proportionally based on importance scores."""
        total_importance = sum(importance_scores.values())
        allocations = {}
        
        for chunk in chunks:
            if total_importance > 0:
                proportion = importance_scores[chunk.chunk_id] / total_importance
                allocation = int(budget.available_tokens * proportion)
            else:
                allocation = budget.available_tokens // len(chunks)
            
            allocations[chunk.chunk_id] = max(allocation, 0)
        
        return allocations

    async def _allocate_priority_first(self, chunks: List[CodeChunk], importance_scores: Dict[str, float], budget: TokenBudget) -> Dict[str, int]:
        """Allocates tokens prioritizing most important chunks."""
        sorted_chunks = sorted(chunks, key=lambda x: importance_scores[x.chunk_id], reverse=True)
        
        allocations = {}
        remaining_tokens = budget.available_tokens
        
        for chunk in sorted_chunks:
            current_tokens = await self._count_chunk_tokens(chunk)
            allocation = min(current_tokens, remaining_tokens)
            allocations[chunk.chunk_id] = allocation
            remaining_tokens -= allocation
            
            if remaining_tokens <= 0:
                break
        
        return allocations

    async def _allocate_balanced(self, chunks: List[CodeChunk], importance_scores: Dict[str, float], budget: TokenBudget) -> Dict[str, int]:
        """Balances between importance and content preservation."""
        base_allocation = budget.available_tokens // len(chunks)
        extra_tokens = budget.available_tokens % len(chunks)
        
        # Normalize importance scores
        max_importance = max(importance_scores.values(), default=1.0)
        normalized_scores = {k: v / max_importance for k, v in importance_scores.items()}
        
        allocations = {}
        for chunk in chunks:
            # Allocate base tokens plus extra based on importance
            extra = int(extra_tokens * normalized_scores[chunk.chunk_id])
            allocations[chunk.chunk_id] = base_allocation + extra
        
        return allocations

    async def _allocate_adaptively(self, chunks: List[CodeChunk], importance_scores: Dict[str, float], budget: TokenBudget) -> Dict[str, int]:
        """Adapts allocation based on content type and relationships."""
        # Calculate complexity-based weights
        complexity_weights = await self._calculate_complexity_weights(chunks)
        
        # Calculate relationship-based weights
        relationship_weights = await self._calculate_relationship_weights(chunks)
        
        # Combine weights with importance scores
        final_weights = {}
        for chunk in chunks:
            final_weights[chunk.chunk_id] = (
                importance_scores[chunk.chunk_id] * 0.4 +
                complexity_weights[chunk.chunk_id] * 0.3 +
                relationship_weights[chunk.chunk_id] * 0.3
            )
        
        # Normalize weights
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {k: v / total_weight for k, v in final_weights.items()}
        
        # Allocate tokens based on final weights
        allocations = {}
        for chunk in chunks:
            allocation = int(budget.available_tokens * final_weights[chunk.chunk_id])
            allocations[chunk.chunk_id] = max(allocation, 0)
        
        return allocations

    async def _optimize_remove_comments(self, chunk: CodeChunk) -> Optional[CodeChunk]:
        """Removes comments while preserving docstrings."""
        try:
            tree = self._get_ast(chunk.content)
            class CommentRemover(ast.NodeTransformer):
                def visit(self, node: ast.AST) -> ast.AST:
                    if hasattr(node, 'body'):
                        # Preserve docstrings
                        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and
                            len(node.body) > 0 and
                            isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Str)):
                            return node
                    return super().visit(node)

            transformer = CommentRemover()
            transformed = transformer.visit(tree)
            return self._create_optimized_chunk(chunk, transformed)
        except Exception as e:
            logger.error(f"Error removing comments: {e}")
            return None

    async def _optimize_simplify_whitespace(self, chunk: CodeChunk) -> Optional[CodeChunk]:
        """Simplifies whitespace while maintaining readability."""
        try:
            lines = chunk.content.split('\n')
            simplified = []
            prev_empty = False
            
            for line in lines:
                stripped = line.strip()
                if stripped:
                    simplified.append(line)
                    prev_empty = False
                elif not prev_empty:
                    simplified.append('')
                    prev_empty = True
                    
            return self._create_chunk_with_content(chunk, '\n'.join(simplified))
        except Exception as e:
            logger.error(f"Error simplifying whitespace: {e}")
            return None

    async def _optimize_preserve_interfaces(self, chunk: CodeChunk) -> Optional[CodeChunk]:
        """Optimizes while preserving interface definitions."""
        try:
            tree = self._get_ast(chunk.content)
            
            class InterfacePreserver(ast.NodeTransformer):
                def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                    # Preserve function signature and docstring
                    docstring = ast.get_docstring(node)
                    new_body = [ast.Pass()]
                    if docstring:
                        new_body.insert(0, ast.Expr(ast.Str(s=docstring)))
                    
                    return ast.FunctionDef(
                        name=node.name,
                        args=node.args,
                        body=new_body,
                        decorator_list=node.decorator_list,
                        returns=node.returns
                    )

                def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
                    # Preserve class structure and interface methods
                    new_body = []
                    docstring = ast.get_docstring(node)
                    if docstring:
                        new_body.append(ast.Expr(ast.Str(s=docstring)))
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            if self._is_interface_method(item):
                                new_body.append(self.visit_FunctionDef(item))
                    
                    return ast.ClassDef(
                        name=node.name,
                        bases=node.bases,
                        keywords=node.keywords,
                        body=new_body,
                        decorator_list=node.decorator_list
                    )

                def _is_interface_method(self, node: ast.FunctionDef) -> bool:
                    return (
                        hasattr(node, 'decorator_list') and
                        any(d.id == 'abstractmethod' for d in node.decorator_list
                            if isinstance(d, ast.Name))
                    )

            transformer = InterfacePreserver()
            transformed = transformer.visit(tree)
            return self._create_optimized_chunk(chunk, transformed)
        except Exception as e:
            logger.error(f"Error preserving interfaces: {e}")
            return None

    async def _optimize_preserve_dependencies(self, chunk: CodeChunk) -> Optional[CodeChunk]:
        """Optimizes while preserving dependency relationships."""
        try:
            tree = self._get_ast(chunk.content)
            
            class DependencyPreserver(ast.NodeTransformer):
                def __init__(self, dependencies: Set[str]):
                    self.dependencies = dependencies
                    self.used_names = set()

                def visit_Name(self, node: ast.AST) -> ast.AST:
                    if isinstance(node, ast.Name):
                        self.used_names.add(node.id)
                    return node

                def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                    # Check if function uses or provides dependencies
                    self.used_names.clear()
                    self.generic_visit(node)
                    
                    if self.used_names.intersection(self.dependencies):
                        return node
                    return self._create_minimal_function(node)

                def _create_minimal_function(self, node: ast.FunctionDef) -> ast.AST:
                    docstring = ast.get_docstring(node)
                    new_body = [ast.Pass()]
                    if docstring:
                        new_body.insert(0, ast.Expr(ast.Str(s=docstring)))
                    
                    return ast.FunctionDef(
                        name=node.name,
                        args=node.args,
                        body=new_body,
                        decorator_list=node.decorator_list,
                        returns=node.returns
                    )

            transformer = DependencyPreserver(chunk.metadata.dependencies)
            transformed = transformer.visit(tree)
            return self._create_optimized_chunk(chunk, transformed)
        except Exception as e:
            logger.error(f"Error preserving dependencies: {e}")
            return None

    async def _optimize_aggressive(self, chunk: CodeChunk) -> Optional[CodeChunk]:
        """Performs aggressive optimization while maintaining validity."""
        try:
            tree = self._get_ast(chunk.content)
            
            class AggressiveOptimizer(ast.NodeTransformer):
                def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                    # Keep only signature and minimal body
                    return ast.FunctionDef(
                        name=node.name,
                        args=node.args,
                        body=[ast.Pass()],
                        decorator_list=[],
                        returns=node.returns
                    )

                def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
                    # Keep only class definition and interface methods
                    return ast.ClassDef(
                        name=node.name,
                        bases=node.bases,
                        keywords=[],
                        body=[ast.Pass()],
                        decorator_list=[]
                    )

            transformer = AggressiveOptimizer()
            transformed = transformer.visit(tree)
            return self._create_optimized_chunk(chunk, transformed)
        except Exception as e:
            logger.error(f"Error in aggressive optimization: {e}")
            return None

    async def _fallback_optimization(self, chunk: CodeChunk, allocated_tokens: int) -> Optional[CodeChunk]:
        """Fallback optimization strategy when others fail."""
        try:
            # Simple truncation with basic structure preservation
            lines = chunk.content.split('\n')
            current_tokens = 0
            preserved_lines = []
            
            for line in lines:
                tokens = await self._count_tokens(line)
                if current_tokens + tokens > allocated_tokens:
                    break
                preserved_lines.append(line)
                current_tokens += tokens
            
            if preserved_lines:
                return self._create_chunk_with_content(chunk, '\n'.join(preserved_lines))
            return None
        except Exception as e:
            logger.error(f"Error in fallback optimization: {e}")
            return None

    def _get_ast(self, content: str) -> ast.AST:
        """Gets AST with caching."""
        content_hash = hash(content)
        if content_hash not in self._ast_cache:
            self._ast_cache[content_hash] = ast.parse(content)
        return self._ast_cache[content_hash]

    def _create_optimized_chunk(self, original_chunk: CodeChunk, transformed_tree: ast.AST) -> CodeChunk:
        """Creates a new chunk from transformed AST."""
        return self._create_chunk_with_content(original_chunk, ast.unparse(transformed_tree))

    def _create_chunk_with_content(self, original_chunk: CodeChunk, new_content: str) -> CodeChunk:
        """Creates a new chunk with modified content."""
        return CodeChunk(
            file_path=original_chunk.file_path,
            start_line=original_chunk.start_line,
            end_line=original_chunk.end_line,
            function_name=original_chunk.function_name,
            class_name=original_chunk.class_name,
            chunk_content=new_content,
            language=original_chunk.language,
            is_async=original_chunk.is_async,
            decorator_list=original_chunk.decorator_list,
            docstring=original_chunk.docstring,
            parent_chunk_id=original_chunk.parent_chunk_id
        )

    async def _count_tokens(self, content: str) -> int:
        """Counts tokens in content."""
        result = await self.token_manager.count_tokens(content)
        return result.token_count

    async def _count_chunk_tokens(self, chunk: CodeChunk) -> int:
        """Counts tokens in a chunk with caching."""
        if chunk.chunk_id not in self._chunk_tokens:
            result = await self.token_manager.count_tokens(chunk.chunk_content)
            self._chunk_tokens[chunk.chunk_id] = result.token_count
        return self._chunk_tokens[chunk.chunk_id]

    async def _calculate_complexity_weights(self, chunks: List[CodeChunk]) -> Dict[str, float]:
        """Calculates weights based on code complexity."""
        weights = {}
        max_complexity = 1.0  # Avoid division by zero
        
        # Find maximum complexity
        for chunk in chunks:
            if chunk.metadata.complexity is not None:
                max_complexity = max(max_complexity, chunk.metadata.complexity)
        
        # Calculate normalized weights
        for chunk in chunks:
            complexity = chunk.metadata.complexity or 0
            weights[chunk.chunk_id] = complexity / max_complexity
        
        return weights

    async def _calculate_relationship_weights(self, chunks: List[CodeChunk]) -> Dict[str, float]:
        """Calculates weights based on relationships between chunks."""
        weights = {}
        relationship_counts = defaultdict(int)
        
        # Count relationships
        for chunk in chunks:
            count = len(chunk.metadata.dependencies) + len(chunk.metadata.used_by)
            relationship_counts[chunk.chunk_id] = count
        
        # Normalize weights
        max_count = max(relationship_counts.values(), default=1)
        for chunk_id, count in relationship_counts.items():
            weights[chunk_id] = count / max_count
        
        return weights
