# token_utils.py

"""
token_utils.py

Provides tokenization functionalities using the tiktoken library.
Includes the TokenManager class for counting tokens in text strings,
retrieving the encoder, and decoding tokens back to text.
"""

import tiktoken
from typing import List, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TokenizationResult:
    """
    Stores the result of tokenizing text.

    Attributes:
        tokens: List of token IDs
        token_count: Number of tokens
        encoding_name: Name of the encoding used
    """
    tokens: List[int]
    token_count: int
    encoding_name: str

class TokenManager:
    _encoding_name = "cl100k_base"
    _encoder = None

    @classmethod
    def get_encoder(cls):
        """
        Gets the encoder for tokenization.

        Returns:
            Encoder: The encoder instance
        """
        if cls._encoder is None:
            cls._encoder = tiktoken.get_encoding(cls._encoding_name)
        return cls._encoder

    @classmethod
    def count_tokens(cls, text: str) -> TokenizationResult:
        """
        Counts the number of tokens in the given text.

        Args:
            text: The text to tokenize

        Returns:
            TokenizationResult: The result containing token count and tokens
        """
        encoder = cls.get_encoder()
        tokens = encoder.encode(text)
        return TokenizationResult(
            tokens=tokens,
            token_count=len(tokens),
            encoding_name=cls._encoding_name
        )

    @classmethod
    def decode_tokens(cls, tokens: List[int]) -> str:
        """
        Decodes tokens back to text.

        Args:
            tokens: List of token IDs

        Returns:
            str: Decoded text
        """
        encoder = cls.get_encoder()
        return encoder.decode(tokens)