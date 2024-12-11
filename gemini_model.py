# gemini_model.py

"""
gemini_model.py

Handles interaction with the Gemini model, including token counting and 
documentation generation via the Gemini API.
"""

import aiohttp
import logging
import json
from typing import List, Dict, Any
from utils import TokenManager  # Import TokenManager
from token_utils import TokenManager

logger = logging.getLogger(__name__)


class GeminiModel:
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint

    async def generate_documentation(self, prompt: List[Dict[str, str]], max_tokens: int = 1500) -> Dict[str, Any]:
        """Fetches documentation from the Gemini model API."""
        url = f"{self.endpoint}/generate-docs"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("Gemini documentation generated successfully.")
                    return data.get("documentation", {})
                else:
                    logger.error(
                        f"Error generating documentation from Gemini: {response.status}")
                    return {}

    def calculate_tokens(self, base_info: str, context: str, chunk_content: str, schema: str) -> int:
        """
        Calculates token count for Gemini model prompts using TokenManager.

        Args:
            base_info: Project and style information
            context: Related code/documentation
            chunk_content: Content of the chunk being documented
            schema: JSON schema

        Returns:
            Total token count
        """
        total = 0
        for text in [base_info, context, chunk_content, schema]:
            token_result = TokenManager.count_tokens(text)
            total += token_result.token_count
        return total

    def generate_prompt(self, base_info: str, context: str, chunk_content: str, schema: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generates prompt structure specifically for the Gemini model."""
        schema_str = json.dumps(schema, indent=2)
        prompt = [
            {"role": "system", "content": base_info},
            {"role": "user", "content": context},
            {"role": "assistant", "content": chunk_content},
            {"role": "schema", "content": schema_str}
        ]
        logger.debug("Generated prompt for Gemini model.")
        return prompt
