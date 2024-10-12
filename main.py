import os
import sys
import logging
import argparse
import asyncio
import aiohttp
from dotenv import load_dotenv
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
    parser = argparse.ArgumentParser(
        description="Generate and insert comments/docstrings using Azure OpenAI's REST API."
    )
    parser.add_argument("repo_path", help="Path to the code repository")
    parser.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    parser.add_argument("--concurrency", help="Number of concurrent requests", type=int, default=5)
    parser.add_argument("-o", "--output", help="Output Markdown file", default="output.md")
    parser.add_argument("--deployment-name", help="Deployment name for Azure OpenAI", required=True)
    parser.add_argument("--skip-types", help="Comma-separated list of file extensions to skip", default="")
    parser.add_argument("--project-info", help="Information about the project", default="")
    parser.add_argument("--style-guidelines", help="Documentation style guidelines to follow", default="")
    parser.add_argument("--safe-mode", help="Run in safe mode (no files will be modified)", action="store_true")
    parser.add_argument("--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)", default="INFO")
    parser.add_argument("--schema", help="Path to function_schema.json", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "schemas", "function_schema.json"))
    return parser.parse_args()

def configure_logging(log_level):
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s"
    )

    file_handler = logging.FileHandler("docs_generation.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

async def main():
    args = parse_arguments()

    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    logger.info("Starting Documentation Generation Tool.")
    logger.debug(f"Parsed arguments: {args}")

    repo_path = args.repo_path
    config_path = args.config
    concurrency = args.concurrency
    output_file = args.output
    deployment_name = args.deployment_name
    skip_types = args.skip_types
    project_info_arg = args.project_info
    style_guidelines_arg = args.style_guidelines
    safe_mode = args.safe_mode
    schema_path = args.schema

    # Ensure necessary environment variables are set for Azure OpenAI Service
    AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
    AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_API_VERSION = os.getenv('API_VERSION')

    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, deployment_name]):
        logger.critical(
            "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, API_VERSION, or DEPLOYMENT_NAME not set. "
            "Please set them in your environment or .env file."
        )
        sys.exit(1)
    logger.info("Using Azure OpenAI with Deployment ID: %s", deployment_name)

    logger.info(f"Repository Path: {repo_path}")
    logger.info(f"Configuration File: {config_path}")
    logger.info(f"Concurrency Level: {concurrency}")
    logger.info(f"Output Markdown File: {output_file}")
    logger.info(f"Deployment Name: {deployment_name}")
    logger.info(f"Safe Mode: {'Enabled' if safe_mode else 'Disabled'}")
    logger.info(f"Function Schema Path: {schema_path}")

    if not os.path.isdir(repo_path):
        logger.critical(f"Invalid repository path: '{repo_path}' is not a directory.")
        sys.exit(1)
    else:
        logger.debug(f"Repository path '{repo_path}' is valid.")

    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    excluded_files = set(DEFAULT_EXCLUDED_FILES)
    skip_types_set = set(DEFAULT_SKIP_TYPES)
    if skip_types:
        skip_types_set.update(
            ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}"
            for ext in skip_types.split(",") if ext.strip()
        )
        logger.debug(f"Updated skip_types: {skip_types_set}")

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
                config_path, excluded_dirs, excluded_files, skip_types_set
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

    # Load function schema
    function_schema = load_function_schema(schema_path)

    try:
        file_paths = get_all_file_paths(
            repo_path, excluded_dirs, excluded_files, skip_types_set
        )
        logger.info(f"Found {len(file_paths)} files to process.")
    except Exception as e:
        logger.error(f"Error getting file paths: {e}")
        sys.exit(1)

    async with aiohttp.ClientSession(raise_for_status=True) as session:
        semaphore = asyncio.Semaphore(concurrency)
        await process_all_files(
            session=session,
            file_paths=file_paths,
            skip_types=skip_types_set,
            semaphore=semaphore,
            deployment_name=deployment_name,
            function_schema=function_schema,
            repo_root=repo_path,
            project_info=project_info,
            style_guidelines=style_guidelines,
            safe_mode=safe_mode,
            output_file=output_file,
            azure_api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_api_version=AZURE_OPENAI_API_VERSION,
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
