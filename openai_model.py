# openai_model.py
import openai
import logging
import json
from typing import List, Dict, Any

# Set up logging for OpenAI model integration
logger = logging.getLogger(__name__)

class OpenAIModel:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def generate_prompt(self, base_info: str, context: str, chunk_content: str, schema: Dict[str, Any]) -> List[Dict[str, str]]:
        """Creates a structured prompt for OpenAI gpt-4o model."""
        schema_str = json.dumps(schema, indent=2)
        prompt = [
            {"role": "system", "content": base_info},
            {"role": "user", "content": context},
            {"role": "assistant", "content": chunk_content},
            {"role": "schema", "content": schema_str}
        ]
        logger.debug("Generated prompt for OpenAI model.")
        return prompt

    def calculate_tokens(self, prompt: List[Dict[str, str]]) -> int:
        """Calculates an approximate token count for OpenAI model prompts."""
        # Using a rough token calculation (OpenAI has ~4 chars per token as a baseline)
        return sum(len(item['content']) for item in prompt) // 4

    def generate_documentation(self, prompt: List[Dict[str, str]], max_tokens: int = 1500) -> str:
        """Fetches documentation generation from OpenAI gpt-4o model."""
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9
        )
        logger.info("OpenAI documentation generated successfully.")
        return response.choices[0].message['content']
