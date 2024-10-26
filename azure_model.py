# azure_model.py

"""
azure_model.py

Handles interaction with the Azure OpenAI API, including token counting,
documentation generation, and API request logic.
"""

import aiohttp
import logging
import json
from typing import List, Dict, Any
from utils import TokenManager  # Import TokenManager
from token_utils import TokenManager

logger = logging.getLogger(__name__)

class AzureModel:
    def __init__(self, api_key: str, endpoint: str, deployment_name: str, api_version: str):
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_version = api_version

    async def generate_documentation(self, prompt: List[Dict[str, str]], max_tokens: int = 1500) -> Dict[str, Any]:
        """Fetches documentation from the Azure OpenAI API."""
        url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "messages": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("Azure OpenAI documentation generated successfully.")
                    choice = data.get("choices", [{}])[0]
                    return choice.get("message", {}).get("content", {})
                else:
                    logger.error(f"Error generating documentation from Azure: {response.status}")
                    return {}

    def calculate_tokens(self, base_info: str, context: str, chunk_content: str, schema: str) -> int:
        """
        Calculates token count for Azure model prompts using TokenManager.

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
        """Generates prompt structure specifically for the Azure model."""
        schema_str = json.dumps(schema, indent=2)
        prompt = [
            {"role": "system", "content": base_info},
            {"role": "user", "content": context},
            {"role": "assistant", "content": chunk_content},
            {"role": "schema", "content": schema_str}
        ]
        logger.debug("Generated prompt for Azure model.")
        return prompt