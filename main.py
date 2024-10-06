import os
import sys
import logging
import argparse
import asyncio

import aiohttp
import openai
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

from file_handlers import process_all_files
from utils import (
    load_config,
    get_all_file_paths,
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    DEFAULT_SKIP_TYPES,
    load_function_schema,
)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and insert comments/docstrings using OpenAI's GPT-4 API."
    )
    parser.add_argument("repo_path", help="Path to the code repository")
    parser.add_argument(
        "-c",
        "--config",
        help="Path to config.json",
        default="config.json"
    )
    parser.add_argument(
        "--concurrency",
        help="Number of concurrent requests",
        type=int,
        default=5
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output Markdown file",
        default="output.md"
    )
    parser.add_argument(
        "--model",
        help="OpenAI model to use (e.g., gpt-4)",
        default=None
    )
    parser.add_argument(
        "--deployment-name",
        help="Deployment name for Azure OpenAI",
        default=None
    )
    parser.add_argument(
        "--skip-types",
        help="Comma-separated list of file extensions to skip",
        default=""
    )
    parser.add_argument(
        "--project-info",
        help="Information about the project",
        default=""
    )
    parser.add_argument(
        "--style-guidelines",
        help="Documentation style guidelines to follow",
        default=""
    )
    parser.add_argument(
        "--safe-mode",
        help="Run in safe mode (no files will be modified)",
        action="store_true"
    )
    parser.add_argument(
        "--log-level",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        default="INFO"
    )
    parser.add_argument(
        "--schema",
        help="Path to function_schema.json",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "schemas",
            "function_schema.json"
        )
    )
    parser.add_argument(
        "--use-azure",
        help="Use Azure OpenAI instead of regular OpenAI API",
        action="store_true"
    )
    return parser.parse_args()


def configure_logging(log_level):
    """Configures logging based on the provided log level."""
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(module)s:%(funcName)s:"
        "%(lineno)d: %(message)s"
    )

    file_handler = RotatingFileHandler(
        "docs_generation.log", maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def validate_model_name(model_name: str, use_azure: bool) -> bool:
    """Validates the OpenAI model name format."""
    if use_azure:
        # Skip validation for Azure OpenAI; deployment IDs are user-defined
        return True
    else:
        valid_models = [
            "gpt-4",
            "gpt-4-0314",
            "gpt-4-32k",
            "gpt-4-32k-0314",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-2024-08-06",
            "gpt-4o",
        ]
        if model_name in valid_models:
            return True
        else:
            logger.warning(
                f"Unrecognized model name '{model_name}'. Proceeding with caution."
            )
            return False  # Change to False to enforce valid model names


async def main():
    """Main function to orchestrate documentation generation."""
    args = parse_arguments()

    # Configure logging based on the parsed log level
    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    logger.info("Starting Documentation Generation Tool.")
    logger.debug(f"Parsed arguments: {args}")

    # Assign arguments to variables for easier access
    repo_path = args.repo_path
    config_path = args.config
    concurrency = args.concurrency
    output_file = args.output
    project_info_arg = args.project_info
    style_guidelines_arg = args.style_guidelines
    safe_mode = args.safe_mode
    schema_path = args.schema
    use_azure = args.use_azure

    # Fetch API keys and endpoints from environment variables or command-line arguments
    AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
    api_base = os.getenv('ENDPOINT_URL')
    api_version = os.getenv('API_VERSION')
    deployment_id = args.deployment_name or os.getenv('DEPLOYMENT_NAME')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    model_name = args.model or os.getenv('MODEL_NAME')

    # Validate API keys and initialize client
    if use_azure:
        if not AZURE_OPENAI_API_KEY or not api_base:
            logger.critical(
                "AZURE_OPENAI_API_KEY or ENDPOINT_URL not set. "
                "Please set them in your environment or .env file."
            )
            sys.exit(1)
        if not deployment_id:
            logger.critical("DEPLOYMENT_NAME not set. Please set it in your environment or .env file.")
            sys.exit(1)
        logger.info("Using Azure OpenAI with Deployment ID: %s", deployment_id)
        # Configure the OpenAI library for Azure
        openai.api_type = "azure"
        openai.api_key = AZURE_OPENAI_API_KEY
        openai.api_base = api_base
        openai.api_version = api_version
        model_name = deployment_id  # Use deployment ID for Azure OpenAI
    else:
        if not OPENAI_API_KEY:
            logger.critical(
                "OPENAI_API_KEY not set. Please set it in your environment or .env file."
            )
            sys.exit(1)
        if not model_name:
            logger.error("Model name is not specified.")
            sys.exit(1)
        logger.info("Using OpenAI with Model: %s", model_name)
        openai.api_key = OPENAI_API_KEY

    logger.info(f"Repository Path: {repo_path}")
    logger.info(f"Configuration File: {config_path}")
    logger.info(f"Concurrency Level: {concurrency}")
    logger.info(f"Output Markdown File: {output_file}")
    logger.info(f"Model Name / Deployment ID: {model_name}")
    logger.info(f"Safe Mode: {'Enabled' if safe_mode else 'Disabled'}")
    logger.info(f"Function Schema Path: {schema_path}")

    if not validate_model_name(model_name, use_azure):
        logger.error(f"Invalid model name '{model_name}'. Exiting.")
        sys.exit(1)

    if not os.path.isdir(repo_path):
        logger.critical(
            f"Invalid repository path: '{repo_path}' is not a directory."
        )
        sys.exit(1)
    else:
        logger.debug(f"Repository path '{repo_path}' is valid.")

    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    excluded_files = set(DEFAULT_EXCLUDED_FILES)
    skip_types = set(DEFAULT_SKIP_TYPES)
    if args.skip_types:
        skip_types.update(
            ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}"
            for ext in args.skip_types.split(",") if ext.strip()
        )
        logger.debug(f"Updated skip_types: {skip_types}")

    # Load configuration
    project_info_config = ""
    style_guidelines_config = ""

    if not os.path.isfile(config_path):
        logger.warning(
            f"Configuration file '{config_path}' not found. "
            "Proceeding with default and command-line settings."
        )
    else:
        try:
            project_info_config, style_guidelines_config = load_config(
                config_path, excluded_dirs, excluded_files, skip_types
            )
            logger.debug(
                f"Loaded configurations from '{config_path}': "
                f"Project Info='{project_info_config}', "
                f"Style Guidelines='{style_guidelines_config}'"
            )
        except Exception as e:
            logger.error(f"Failed to load configuration from '{config_path}': {e}")
            sys.exit(1)

    project_info = project_info_arg or project_info_config
    style_guidelines = style_guidelines_arg or style_guidelines_config

    if project_info:
        logger.debug(f"Project Info: {project_info}")
    if style_guidelines:
        logger.debug(f"Style Guidelines: {style_guidelines}")

    # Load and validate function schema
    function_schema = load_function_schema(schema_path)

    # Get all file paths to process
    try:
        file_paths = get_all_file_paths(
            repo_path, excluded_dirs, excluded_files, skip_types
        )
        logger.info(f"Found {len(file_paths)} files to process.")
    except Exception as e:
        logger.error(f"Error getting file paths: {e}")
        sys.exit(1)

    # Create async HTTP session
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        semaphore = asyncio.Semaphore(concurrency)
        await process_all_files(
            session=session,
            file_paths=file_paths,
            skip_types=skip_types,
            semaphore=semaphore,
            model_name=model_name,
            function_schema=function_schema,
            repo_root=repo_path,
            project_info=project_info,
            style_guidelines=style_guidelines,
            safe_mode=safe_mode,
            output_file=output_file,
            use_azure=use_azure,
        )

    logger.info("Documentation generation completed successfully.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Documentation generation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
