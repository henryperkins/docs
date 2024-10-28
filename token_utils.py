# token_utils.py

"""
token_utils.py - Enhanced tokenization utilities with embedding capabilities.

This module provides utilities for tokenizing text using various tokenizer models,
counting tokens, decoding tokens back to text, handling special tokens, validating
token limits, splitting text based on token limits, and batch token processing.
Additionally, it integrates advanced embedding calculations and similarity measurements
for enhanced text analysis.
"""

import tiktoken
from typing import List, Optional, Union, Dict
from dataclasses import dataclass
import logging
from enum import Enum
import threading
import numpy as np
from metrics_utils import EnhancedEmbeddingCalculator, CodeMetadata, EmbeddingManager, TokenizationError

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

class TokenizationError(Exception):
    """Custom exception for tokenization errors."""
    pass

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
        """
        Retrieves the appropriate encoder instance.

        Args:
            model (TokenizerModel, optional): TokenizerModel to use. Defaults to GPT4.

        Returns:
            tiktoken.Encoding: Encoder instance.

        Raises:
            TokenizationError: If encoder creation fails.
        """
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
    def count_tokens(
        cls,
        text: Union[str, List[str]],
        model: TokenizerModel = None,
        include_special_tokens: bool = False
    ) -> TokenizationResult:
        """
        Counts tokens in the provided text with enhanced error handling.

        Args:
            text (Union[str, List[str]]): Text to tokenize.
            model (TokenizerModel, optional): TokenizerModel to use. Defaults to GPT4.
            include_special_tokens (bool, optional): Whether to count special tokens. Defaults to False.

        Returns:
            TokenizationResult: Tokenization results.

        Raises:
            TokenizationError: If tokenization fails.
        """
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
    def decode_tokens(
        cls,
        tokens: List[int],
        model: TokenizerModel = None
    ) -> str:
        """
        Decodes a list of tokens back to text.

        Args:
            tokens (List[int]): List of token IDs.
            model (TokenizerModel, optional): TokenizerModel to use. Defaults to GPT4.

        Returns:
            str: Decoded text.

        Raises:
            TokenizationError: If decoding fails.
        """
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
    def _count_special_tokens(
        cls,
        text: str,
        encoder: tiktoken.Encoding
    ) -> Dict[str, int]:
        """
        Counts special tokens in the text.

        Args:
            text (str): Text to analyze.
            encoder (tiktoken.Encoding): Encoder instance.

        Returns:
            Dict[str, int]: Counts of special tokens.
        """
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
    def validate_token_limit(
        cls,
        text: str,
        max_tokens: int,
        model: TokenizerModel = None
    ) -> bool:
        """
        Checks if the text exceeds the specified token limit.

        Args:
            text (str): Text to check.
            max_tokens (int): Maximum allowed tokens.
            model (TokenizerModel, optional): TokenizerModel to use. Defaults to GPT4.

        Returns:
            bool: True if within limit, False otherwise.
        """
        try:
            result = cls.count_tokens(text, model)
            logger.debug(f"Token count {result.token_count} compared to max {max_tokens}")
            return result.token_count <= max_tokens
        except TokenizationError as e:
            logger.error(f"Validation failed: {e}")
            return False

    @classmethod
    def split_by_token_limit(
        cls,
        text: str,
        max_tokens: int,
        model: TokenizerModel = None
    ) -> List[str]:
        """
        Splits the text into chunks, each not exceeding the specified token limit.

        Args:
            text (str): Text to split.
            max_tokens (int): Maximum tokens per chunk.
            model (TokenizerModel, optional): TokenizerModel to use. Defaults to GPT4.

        Returns:
            List[str]: List of text chunks.

        Raises:
            TokenizationError: If splitting fails.
        """
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
        """
        Provides a rough estimation of token count based on character length.

        Useful for quick checks before full tokenization.

        Args:
            text (str): Text to estimate.

        Returns:
            int: Estimated token count.
        """
        estimate = len(text) // 4  # GPT models average ~4 characters per token
        logger.debug(f"Estimated tokens from characters: {estimate}")
        return estimate

    @classmethod
    def batch_count_tokens(
        cls,
        texts: List[str],
        model: TokenizerModel = None
    ) -> List[TokenizationResult]:
        """
        Counts tokens for multiple texts efficiently.

        Args:
            texts (List[str]): List of texts to tokenize.
            model (TokenizerModel, optional): TokenizerModel to use. Defaults to GPT4.

        Returns:
            List[TokenizationResult]: Results for each text.
        """
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
        """
        Clears the encoder cache.
        """
        with cls._lock:
            cls._encoders.clear()
            logger.info("Encoder cache cleared.")

    # Embedding Integration
    @classmethod
    def get_enhanced_embedding(cls, code: str, metadata: CodeMetadata) -> np.ndarray:
        """
        Generates an enhanced embedding for the given code and metadata.

        Args:
            code (str): Code snippet to embed.
            metadata (CodeMetadata): Metadata associated with the code.

        Returns:
            np.ndarray: Enhanced embedding vector.

        Raises:
            TokenizationError: If embedding generation fails.
        """
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
        """
        Calculates the similarity between two embeddings.

        Args:
            embedding1 (np.ndarray): First embedding vector.
            embedding2 (np.ndarray): Second embedding vector.

        Returns:
            float: Similarity score.

        Raises:
            TokenizationError: If similarity calculation fails.
        """
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
        """
        Sets new weights for metadata features in embedding calculations.

        Args:
            new_weights (Dict[str, float]): New weights for metadata features.

        Raises:
            TokenizationError: If setting weights fails.
        """
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