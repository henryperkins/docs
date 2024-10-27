# token_utils.py

"""
token_utils.py - Enhanced tokenization utilities
"""

import tiktoken
from typing import List, Optional, Union, Dict
from dataclasses import dataclass
import logging
from enum import Enum
import threading

logger = logging.getLogger(__name__)

class TokenizerModel(Enum):
    """Supported tokenizer models"""
    GPT4 = "cl100k_base"
    GPT3 = "p50k_base"
    CODEX = "p50k_edit"

@dataclass
class TokenizationResult:
    """Results from tokenization operation"""
    tokens: List[int]
    token_count: int
    encoding_name: str
    special_tokens: Dict[str, int] = None
    error: Optional[str] = None

class TokenizationError(Exception):
    """Custom exception for tokenization errors"""
    pass

class TokenManager:
    """Manages tokenization operations with caching and thread-safety."""

    _encoders = {}  # Cache for different encoders
    _default_model = TokenizerModel.GPT4
    _lock = threading.Lock()  # Lock for thread safety


    @classmethod
    def get_encoder(cls, model: TokenizerModel = None) -> tiktoken.Encoding:
        """
        Gets the appropriate encoder instance.
        
        Args:
            model: TokenizerModel to use
            
        Returns:
            tiktoken.Encoding: Encoder instance
            
        Raises:
            TokenizationError: If encoder creation fails
        """
        with cls._lock:  # Acquire lock before accessing shared resource
            try:
                model = model or cls._default_model
                if model not in cls._encoders:
                    cls._encoders[model] = tiktoken.get_encoding(model.value)
                return cls._encoders[model]
            except Exception as e:
                raise TokenizationError(f"Failed to create encoder: {str(e)}")

    @classmethod
    def count_tokens(
        cls,
        text: Union[str, List[str]],
        model: TokenizerModel = None,
        include_special_tokens: bool = False
    ) -> TokenizationResult:
        """
        Counts tokens in text with enhanced error handling.
        
        Args:
            text: Text to tokenize
            model: TokenizerModel to use
            include_special_tokens: Whether to count special tokens
            
        Returns:
            TokenizationResult: Tokenization results
            
        Raises:
            TokenizationError: If tokenization fails
        """
        logger.debug(f"Counting tokens for text: {text[:50]}...") # Truncate text for logging
        try:
            if not text:
                return TokenizationResult([], 0, "", error="Empty input")

            encoder = cls.get_encoder(model)  # Get encoder (thread-safe)
            model = model or cls._default_model

            if isinstance(text, list):
                text = " ".join(text)

            tokens = encoder.encode(text)

            special_tokens = None
            if include_special_tokens:
                special_tokens = cls._count_special_tokens(text, encoder)

            return TokenizationResult(
                tokens=tokens,
                token_count=len(tokens),
                encoding_name=model.value,
                special_tokens=special_tokens
            )

        except Exception as e:
            logger.error(f"Tokenization error: {str(e)}")
            raise TokenizationError(f"Failed to count tokens: {str(e)}")
            
    @classmethod
    def decode_tokens(
        cls,
        tokens: List[int],
        model: TokenizerModel = None
    ) -> str:
        """
        Decodes tokens back to text.
        
        Args:
            tokens: List of token IDs
            model: TokenizerModel to use
            
        Returns:
            str: Decoded text
            
        Raises:
            TokenizationError: If decoding fails
        """
        logger.debug(f"Decoding tokens: {tokens[:10]}...") # Truncate tokens for logging
        try:
            if not tokens:
                return ""

            encoder = cls.get_encoder(model)
            return encoder.decode(tokens)

        except Exception as e:
            logger.error(f"Token decoding error: {str(e)}")
            raise TokenizationError(f"Failed to decode tokens: {str(e)}")

    @classmethod
    def _count_special_tokens(
        cls,
        text: str,
        encoder: tiktoken.Encoding
    ) -> Dict[str, int]:
        """Counts special tokens in text"""
        special_tokens = {}
        try:
            # Add special token counting logic here
            # Example: count newlines, code blocks, etc.
            special_tokens["newlines"] = text.count("\n")
            special_tokens["code_blocks"] = text.count("```")
            return special_tokens
        except Exception as e:
            logger.warning(f"Error counting special tokens: {str(e)}")
            return {}

    @classmethod
    def validate_token_limit(
        cls,
        text: str,
        max_tokens: int,
        model: TokenizerModel = None
    ) -> bool:
        """
        Checks if text exceeds token limit.
        
        Args:
            text: Text to check
            max_tokens: Maximum allowed tokens
            model: TokenizerModel to use
            
        Returns:
            bool: True if within limit
        """
        try:
            result = cls.count_tokens(text, model)
            return result.token_count <= max_tokens
        except TokenizationError:
            return False

    @classmethod
    def split_by_token_limit(
        cls,
        text: str,
        max_tokens: int,
        model: TokenizerModel = None
    ) -> List[str]:
        """
        Splits text into chunks by token limit.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            model: TokenizerModel to use
            
        Returns:
            List[str]: Text chunks
        """
        try:
            encoder = cls.get_encoder(model)
            tokens = encoder.encode(text)
            chunks = []
            current_chunk = []
            current_count = 0

            for token in tokens:
                if current_count + 1 > max_tokens:
                    chunks.append(encoder.decode(current_chunk))
                    current_chunk = []
                    current_count = 0

                current_chunk.append(token)
                current_count += 1

            if current_chunk:
                chunks.append(encoder.decode(current_chunk))

            return chunks

        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            raise TokenizationError(f"Failed to split text: {str(e)}")

    @classmethod
    def estimate_tokens_from_chars(cls, text: str) -> int:
        """
        Rough estimation of tokens from character count.
        Useful for quick checks before full tokenization.
        
        Args:
            text: Text to estimate
            
        Returns:
            int: Estimated token count
        """
        # GPT models average ~4 characters per token
        return len(text) // 4

    @classmethod
    def batch_count_tokens(
        cls,
        texts: List[str],
        model: TokenizerModel = None
    ) -> List[TokenizationResult]:
        """
        Counts tokens for multiple texts efficiently.
        
        Args:
            texts: List of texts to tokenize
            model: TokenizerModel to use
            
        Returns:
            List[TokenizationResult]: Results for each text
        """
        results = []
        encoder = cls.get_encoder(model)
        
        for text in texts:
            try:
                tokens = encoder.encode(text)
                results.append(TokenizationResult(
                    tokens=tokens,
                    token_count=len(tokens),
                    encoding_name=model.value if model else cls._default_model.value
                ))
            except Exception as e:
                logger.error(f"Error in batch tokenization: {str(e)}")
                results.append(TokenizationResult(
                    tokens=[],
                    token_count=0,
                    encoding_name="",
                    error=str(e)
                ))
        
        return results

    @classmethod
    def clear_cache(cls):
        """Clears the encoder cache"""
        cls._encoders.clear()
