# main.py

import os
import sys
import logging
import argparse
import asyncio
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from provider_config import load_provider_configs
from azure_model import AzureModel
from gemini_model import GeminiModel
from openai_model import OpenAIModel
from process_manager import DocumentationProcessManager
from utils import DEFAULT_EXCLUDED_FILES, DEFAULT_EXCLUDED_DIRS, DEFAULT_SKIP_TYPES, load_config, load_function_schema, get_all_file_paths, setup_logging

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and insert docstrings using Azure OpenAI, Gemini, or OpenAI models.")
    parser.add_argument("repo_path", help="Path to the code repository")
    parser.add_argument(
        "-c", "--config", help="Path to config.json", default="config.json")
    parser.add_argument(
        "--provider", help="Choose AI provider: 'azure', 'gemini', or 'openai'", default="azure")
    parser.add_argument(
        "--concurrency", help="Number of concurrent requests", type=int, default=5)
    parser.add_argument(
        "--skip-types", help="Comma-separated list of file extensions to skip", default="")
    parser.add_argument(
        "--project-info", help="Information about the project", default="")
    parser.add_argument("--style-guidelines",
                        help="Documentation style guidelines", default="")
    parser.add_argument(
        "--safe-mode", help="Run in safe mode (no files modified)", action="store_true")
    parser.add_argument("--log-level", help="Logging level", default="INFO")
    parser.add_argument("--schema", help="Path to function_schema.json",
                        default="schemas/function_schema.json")
    parser.add_argument(
        "--doc-output-dir", help="Directory to save documentation files", default="documentation")
    parser.add_argument(
        "--project-id", help="Unique identifier for the project", required=True)
    return parser.parse_args()


async def main():
    """Main function."""
    args = parse_arguments()
    load_dotenv()

    # Configure logging using utils.py
    log_file = "documentation_generation.log"
    if not setup_logging(log_file, log_level=args.log_level):
        print("Failed to set up logging. Exiting...")
        sys.exit(1)

    logger.info("Starting documentation generation process...")

    repo_path = args.repo_path
    config_path = args.config
    output_dir = args.doc_output_dir

    # Load provider configurations
    provider_configs = load_provider_configs(config_path)

    # Validate API configuration based on provider
    if args.provider not in provider_configs:
        logger.error(f"Unsupported provider: {args.provider}")
        sys.exit(1)

    provider_config = provider_configs[args.provider]

    try:
        # Initialize the appropriate model based on provider
        client = None
        if args.provider == "azure":
            client = AzureModel(
                api_key=provider_config.api_key,
                endpoint=provider_config.endpoint,
                deployment_name=provider_config.deployment_name,
                api_version=provider_config.api_version
            )
        elif args.provider == "gemini":
            client = GeminiModel(
                api_key=provider_config.api_key,
                endpoint=provider_config.endpoint
            )
        elif args.provider == "openai":
            client = OpenAIModel(api_key=provider_config.api_key)
        else:
            logger.error(f"Unsupported provider: {args.provider}")
            sys.exit(1)

        # Load configuration, schema, and file paths
        excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
        excluded_files = set(DEFAULT_EXCLUDED_FILES)
        skip_types_set = set(DEFAULT_SKIP_TYPES)
        if args.skip_types:
            skip_types_set.update(ext.strip()
                                  for ext in args.skip_types.split(","))

        project_info, style_guidelines = load_config(
            config_path, excluded_dirs, excluded_files, skip_types_set)
        project_info = args.project_info or project_info
        style_guidelines = args.style_guidelines or style_guidelines

        function_schema = load_function_schema(args.schema)
        file_paths = get_all_file_paths(
            repo_path, excluded_dirs, excluded_files, skip_types_set)

        # Initialize DocumentationProcessManager
        manager = DocumentationProcessManager(
            repo_root=repo_path,
            output_dir=output_dir,
            provider=args.provider,
            azure_config={
                "api_key": provider_config.api_key,
                "endpoint": provider_config.endpoint,
                "deployment_name": provider_config.deployment_name,
                "api_version": provider_config.api_version
            },
            gemini_config={
                "api_key": provider_config.api_key,
                "endpoint": provider_config.endpoint
            },
            openai_config={
                "api_key": provider_config.api_key
            },
            function_schema=function_schema,
            max_concurrency=args.concurrency
        )

        results = await manager.process_files(
            file_paths=file_paths,
            skip_types=skip_types_set,
            project_info=project_info,
            style_guidelines=style_guidelines,
            safe_mode=args.safe_mode
        )

        logger.info(f"Documentation generation completed. Results: {results}")

    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
