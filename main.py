# main.py
import os
import sys
import logging
import argparse
import asyncio
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from azure_model import AzureModel
from gemini_model import GeminiModel
from openai_model import OpenAIModel
from process_manager import DocumentationProcessManager
from utils import DEFAULT_EXCLUDED_FILES, DEFAULT_EXCLUDED_DIRS, DEFAULT_SKIP_TYPES, load_config, load_function_schema, get_all_file_paths
from logging_config import setup_logging

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate and insert docstrings using Azure OpenAI, Gemini, or OpenAI models.")
    parser.add_argument("repo_path", help="Path to the code repository")
    parser.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    parser.add_argument("--provider", help="Choose AI provider: 'azure', 'gemini', or 'openai'", default="azure")
    parser.add_argument("--concurrency", help="Number of concurrent requests", type=int, default=5)
    parser.add_argument("--skip-types", help="Comma-separated list of file extensions to skip", default="")
    parser.add_argument("--project-info", help="Information about the project", default="")
    parser.add_argument("--style-guidelines", help="Documentation style guidelines", default="")
    parser.add_argument("--safe-mode", help="Run in safe mode (no files modified)", action="store_true")
    parser.add_argument("--log-level", help="Logging level", default="INFO")
    parser.add_argument("--schema", help="Path to function_schema.json", default="schemas/function_schema.json")
    parser.add_argument("--doc-output-dir", help="Directory to save documentation files", default="documentation")
    parser.add_argument("--project-id", help="Unique identifier for the project", required=True)
    return parser.parse_args()

def validate_api_config(provider: str) -> Optional[str]:
    """Validates required API configuration based on the selected provider."""
    if provider == "azure":
        if not os.getenv("AZURE_OPENAI_API_KEY"):
            return "Missing AZURE_OPENAI_API_KEY"
        if not os.getenv("AZURE_OPENAI_ENDPOINT"):
            return "Missing AZURE_OPENAI_ENDPOINT"
        if not os.getenv("AZURE_OPENAI_DEPLOYMENT"):
            return "Missing AZURE_OPENAI_DEPLOYMENT"
    elif provider == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            return "Missing GEMINI_API_KEY"
        if not os.getenv("GEMINI_ENDPOINT"):
            return "Missing GEMINI_ENDPOINT"
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return "Missing OPENAI_API_KEY"
    else:
        return f"Unsupported provider: {provider}"
    return None

async def main():
    """Main function."""
    args = parse_arguments()
    load_dotenv()

    # Configure logging 
    log_file = "documentation_generation.log"
    if not setup_logging(log_file, log_level=args.log_level):
        print("Failed to set up logging. Exiting...")
        sys.exit(1)

    logger.info("Starting documentation generation process...")  # Example log message

    repo_path = args.repo_path
    config_path = args.config
    output_dir = args.doc_output_dir

    # Validate API configuration based on provider
    if error := validate_api_config(args.provider):
        logger.error(f"Configuration error: {error}")
        sys.exit(1)

    try:
        # Initialize the appropriate model based on provider
        client = None
        if args.provider == "azure":
            client = AzureModel(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_version=os.getenv("API_VERSION", "2023-05-15")
            )
        elif args.provider == "gemini":
            client = GeminiModel(
                api_key=os.getenv("GEMINI_API_KEY"),
                endpoint=os.getenv("GEMINI_ENDPOINT")
            )
        elif args.provider == "openai":
            client = OpenAIModel(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            logger.error(f"Unsupported provider: {args.provider}")
            sys.exit(1)

        # Load configuration, schema, and file paths
        excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
        excluded_files = set(DEFAULT_EXCLUDED_FILES)
        skip_types_set = set(DEFAULT_SKIP_TYPES)
        if args.skip_types:
            skip_types_set.update(ext.strip() for ext in args.skip_types.split(","))

        project_info, style_guidelines = load_config(config_path, excluded_dirs, excluded_files, skip_types_set)
        project_info = args.project_info or project_info
        style_guidelines = args.style_guidelines or style_guidelines

        function_schema = load_function_schema(args.schema)
        file_paths = get_all_file_paths(repo_path, excluded_dirs, excluded_files, skip_types_set)

        # Initialize DocumentationProcessManager
        manager = DocumentationProcessManager(
            repo_root=repo_path,
            output_dir=output_dir,
            provider=args.provider,
            azure_config={
                "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                "api_version": os.getenv("API_VERSION", "2024-05-01-preview")  # Moved inside azure_config
            },
            gemini_config={
                "api_key": os.getenv("GEMINI_API_KEY"),
                "endpoint": os.getenv("GEMINI_ENDPOINT")
            },
            openai_config={
                "api_key": os.getenv("OPENAI_API_KEY")
            },
            function_schema=function_schema,
            max_concurrency=args.concurrency
        )

        results = await manager.process_files(
            file_paths=file_paths,
            skip_types=skip_types_set,
            project_info=project_info,
            style_guidelines=style_guidelines,
            safe_mode=args.safe_mode  # safe_mode is still present
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
