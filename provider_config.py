# provider_config.py

from pydantic import BaseModel, validator
from typing import Optional, Dict
import os
import yaml
import json
from pathlib import Path


class ProviderConfig(BaseModel):
    """Configuration model for AI providers."""
    name: str
    endpoint: str
    api_key: str
    deployment_name: Optional[str] = None
    api_version: Optional[str] = None
    model_name: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_enabled: bool = True
    timeout: float = 30.0
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_parallel_chunks: int = 3

    @validator("temperature")
    def validate_temperature(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v

    @validator("max_tokens")
    def validate_max_tokens(cls, v):
        if v < 1 or v > 8192:
            raise ValueError("max_tokens must be between 1 and 8192")
        return v


def load_provider_configs() -> Dict[str, ProviderConfig]:
    """
    Loads provider configurations from a configuration file or environment variables.

    Returns:
        Dict[str, ProviderConfig]: A dictionary of provider configurations.
    """
    config_file = os.getenv('PROVIDER_CONFIG_FILE')

    if config_file:
        config_path = Path(config_file)
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Provider config file not found: {config_file}")

        try:
            with open(config_path, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    config_data = json.load(f)
                else:
                    raise ValueError(
                        "Unsupported config file format. Must be .yaml or .json")

            # Convert loaded data to ProviderConfig instances
            configs = {name: ProviderConfig(**details)
                       for name, details in config_data.items()}
        except Exception as e:
            raise ValueError(f"Failed to load provider config file: {e}")

    else:
        # Load configurations from environment variables as a fallback
        configs = {
            "azure": ProviderConfig(
                name="azure",
                endpoint=os.getenv(
                    "AZURE_ENDPOINT", "https://your-azure-endpoint.com"),
                api_key=os.getenv("AZURE_API_KEY", "your-azure-api-key"),
                deployment_name=os.getenv(
                    "AZURE_DEPLOYMENT_NAME", "your-deployment-name"),
                api_version=os.getenv("AZURE_API_VERSION", "2023-01-01"),
                model_name=os.getenv("AZURE_MODEL_NAME", "gpt-3"),
                max_tokens=int(os.getenv("AZURE_MAX_TOKENS", "4096")),
                temperature=float(os.getenv("AZURE_TEMPERATURE", "0.7")),
                max_retries=int(os.getenv("AZURE_MAX_RETRIES", "3")),
                retry_delay=float(os.getenv("AZURE_RETRY_DELAY", "1.0")),
                cache_enabled=os.getenv(
                    "AZURE_CACHE_ENABLED", "True") == "True",
                timeout=float(os.getenv("AZURE_TIMEOUT", "30.0")),
                chunk_overlap=int(os.getenv("AZURE_CHUNK_OVERLAP", "200")),
                min_chunk_size=int(os.getenv("AZURE_MIN_CHUNK_SIZE", "100")),
                max_parallel_chunks=int(
                    os.getenv("AZURE_MAX_PARALLEL_CHUNKS", "3"))
            ),
            # Add other providers similarly
        }

    return configs
