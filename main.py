# main.py
import aiohttp
import os
import sys
import logging
import argparse
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Set, List

from utils import (
    load_config,
    get_all_file_paths,
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    DEFAULT_SKIP_TYPES,
    load_function_schema,
)
from file_handlers import process_file  # Updated import
from process_manager import DocumentationProcessManager

# Load environment variables from .env file early
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("documentation_generation.log")
    ],
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate and insert docstrings using Azure OpenAI.")
    parser.add_argument("repo_path", help="Path to the code repository")
    parser.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    parser.add_argument("--concurrency", help="Number of concurrent requests", type=int, default=5)
    parser.add_argument("--deployment-name", help="Deployment name for Azure OpenAI", required=True)  # Correct argument name
    parser.add_argument("--skip-types", help="Comma-separated list of file extensions to skip", default="")
    parser.add_argument("--project-info", help="Information about the project", default="")
    parser.add_argument("--style-guidelines", help="Documentation style guidelines", default="")
    parser.add_argument("--safe-mode", help="Run in safe mode (no files modified)", action="store_true")  # Removed unnecessary argument
    parser.add_argument("--log-level", help="Logging level", default="INFO")
    parser.add_argument("--schema", help="Path to function_schema.json", default="schemas/function_schema.json")
    parser.add_argument("--doc-output-dir", help="Directory to save documentation files", default="documentation")
    parser.add_argument("--project-id", help="Unique identifier for the project", required=True)
    return parser.parse_args()


async def main():
    """Main function to orchestrate the documentation generation process."""
    args = parse_arguments()

    # Configure logging
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    repo_path = args.repo_path
    config_path = args.config
    project_id = args.project_id  # Use project_id directly
    output_dir = args.doc_output_dir

    # Validate Azure OpenAI environment variables
    required_vars = {
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "API_VERSION": os.getenv("API_VERSION"),
        "AZURE_OPENAI_DEPLOYMENT": args.deployment_name,  # Use deployment name from arguments
    }

    missing_vars = [key for key, value in required_vars.items() if not value]
    if missing_vars:
        logger.critical(f"Missing required environment variables or arguments: {', '.join(missing_vars)}")
        sys.exit(1)

    if not os.path.isdir(repo_path):
        logger.critical(f"Invalid repository path: '{repo_path}'")
        sys.exit(1)

    try:
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

        # Initialize and use the DocumentationProcessManager
        manager = DocumentationProcessManager(
            repo_root=repo_path,
            output_dir=output_dir,
            azure_config=required_vars,
            function_schema=function_schema,
            max_concurrency=args.concurrency,
        )

        # Use process_manager to handle file processing
        results = await manager.process_files(
            task_id=project_id,
            file_paths=file_paths,
            skip_types=skip_types_set,
            project_info=project_info,
            style_guidelines=style_guidelines,
            safe_mode=args.safe_mode,  # Pass safe_mode here
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
