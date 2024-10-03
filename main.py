# main.py

import os
import sys
import argparse
import asyncio
import logging
import json
import aiohttp
import shutil
from logging.handlers import RotatingFileHandler
from file_handlers import process_all_files
from utils import (
    load_config,
    get_all_file_paths,
    OPENAI_API_KEY,
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    DEFAULT_SKIP_TYPES,
    function_schema,
    call_openai_api,  # Newly added for function calling
    load_json_schema,     # Newly added for loading JSON schemas
)

import aiofiles

# Configure logging
logger = logging.getLogger(__name__)

def configure_logging(log_level):
    """Configures logging based on the provided log level."""
    logger.setLevel(log_level)

    # Create formatter with module, function, and line number
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(funcName)s:%(lineno)d:%(message)s')

    # Create rotating file handler which logs debug and higher level messages
    file_handler = RotatingFileHandler('docs_generation.log', maxBytes=5*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

def validate_model_name(model_name: str) -> bool:
    """Validates the OpenAI model name format."""
    valid_models = [
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-32k",
        "gpt-4-32k-0314",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",  # Ensure this model supports function calling
        # Add other valid model names as needed
    ]
    if model_name in valid_models:
        return True
    else:
        logger.error(f"Invalid model name '{model_name}'. Please choose a valid OpenAI model.")
        return False

async def main():
    """Main function to orchestrate documentation generation."""
    parser = argparse.ArgumentParser(
        description="Generate and insert comments/docstrings using OpenAI's GPT-4 API."
    )
    parser.add_argument("repo_path", help="Path to the code repository")
    parser.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    parser.add_argument("--concurrency", help="Number of concurrent requests", type=int, default=5)
    parser.add_argument("-o", "--output", help="Output Markdown file", default="output.md")
    parser.add_argument("--model", help="OpenAI model to use (default: gpt-4)", default="gpt-4")
    parser.add_argument("--skip-types", help="Comma-separated list of file extensions to skip", default="")
    parser.add_argument("--project-info", help="Information about the project", default="")
    parser.add_argument("--style-guidelines", help="Documentation style guidelines to follow", default="")
    parser.add_argument("--safe-mode", help="Run in safe mode (no files will be modified)", action='store_true')
    parser.add_argument("--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)", default="INFO")
    parser.add_argument("--schema", help="Path to function_schema.json", default="function_schema.json")
    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    configure_logging(log_level)

    logger.info("Starting Documentation Generation Tool.")
    logger.debug(f"Parsed arguments: {args}")

    # Validate OpenAI API key
    if not OPENAI_API_KEY:
        logger.critical("OPENAI_API_KEY not set. Please set it in your environment or .env file.")
        sys.exit(1)
    else:
        logger.debug("OPENAI_API_KEY found.")

    repo_path = args.repo_path
    config_path = args.config
    concurrency = args.concurrency
    output_file = args.output
    model_name = args.model
    project_info_arg = args.project_info
    style_guidelines_arg = args.style_guidelines
    safe_mode = args.safe_mode
    schema_path = args.schema

    logger.info(f"Repository Path: {repo_path}")
    logger.info(f"Configuration File: {config_path}")
    logger.info(f"Concurrency Level: {concurrency}")
    logger.info(f"Output Markdown File: {output_file}")
    logger.info(f"OpenAI Model: {model_name}")
    logger.info(f"Safe Mode: {'Enabled' if safe_mode else 'Disabled'}")
    logger.info(f"Function Schema Path: {schema_path}")

    # Validate model name
    if not validate_model_name(model_name):
        sys.exit(1)

    if not os.path.isdir(repo_path):
        logger.critical(f"Invalid repository path: '{repo_path}' is not a directory.")
        sys.exit(1)
    else:
        logger.debug(f"Repository path '{repo_path}' is valid.")

    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    excluded_files = set(DEFAULT_EXCLUDED_FILES)
    skip_types = set(DEFAULT_SKIP_TYPES)
    if args.skip_types:
        skip_types.update(ext.strip() for ext in args.skip_types.split(','))
        logger.debug(f"Updated skip_types: {skip_types}")

    # Load configuration from a JSON file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from '{config_path}'")
    except Exception as e:
        logger.critical(f"Failed to load configuration from '{config_path}': {e}")
        sys.exit(1)

    # Initialize these variables with default values to avoid UnboundLocalError
    project_info_config = ''
    style_guidelines_config = ''

    # Check if the config file exists
    if not os.path.isfile(config_path):
        logger.warning(f"Configuration file '{config_path}' not found. Proceeding with default and command-line settings.")
    else:
        # Load additional configurations
        try:
            project_info_config, style_guidelines_config = load_config(config_path, excluded_dirs, excluded_files, skip_types)
            logger.debug(f"Loaded configurations from '{config_path}': Project Info='{project_info_config}', Style Guidelines='{style_guidelines_config}'")
        except Exception as e:
            logger.error(f"Failed to load configuration from '{config_path}': {e}")
            sys.exit(1)

    # Determine final project_info and style_guidelines
    project_info = project_info_arg or project_info_config
    style_guidelines = style_guidelines_arg or style_guidelines_config

    if project_info:
        logger.debug(f"Project Info: {project_info}")
    if style_guidelines:
        logger.debug(f"Style Guidelines: {style_guidelines}")

    # Load JSON schema for function calling
    function_schema_loaded = load_json_schema(schema_path)
    if not function_schema_loaded:
        logger.critical(f"Failed to load function schema from '{schema_path}'. Exiting.")
        sys.exit(1)

    # Get all file paths
    try:
        file_paths = get_all_file_paths(
            repo_path=repo_path,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            skip_types=skip_types
        )
        logger.info(f"Total files to process: {len(file_paths)}")
    except Exception as e:
        logger.error(f"Error retrieving file paths from '{repo_path}': {e}")
        sys.exit(1)

    if not file_paths:
        logger.warning("No files found to process. Exiting.")
        sys.exit(0)

    logger.info("Initializing output Markdown file.")
    # Clear and initialize the output file with a header asynchronously
    try:
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            await f.write("# Documentation Generation Report\n\n")
        logger.debug(f"Output file '{output_file}' initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize output file '{output_file}': {e}")
        sys.exit(1)

    # Initialize semaphore and locks
    semaphore = asyncio.Semaphore(concurrency)
    output_lock = asyncio.Lock()

    # Start the asynchronous processing
    logger.info("Starting asynchronous file processing.")
    try:
        async with aiohttp.ClientSession() as session:
            await process_all_files(
                session=session,
                file_paths=file_paths,
                skip_types=skip_types,
                output_file=output_file,
                semaphore=semaphore,
                model_name=model_name,
                function_schema=function_schema_loaded,
                repo_root=repo_path,
                project_info=project_info,
                style_guidelines=style_guidelines,
                safe_mode=safe_mode
            )
    except Exception as e:
        logger.critical(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Documentation generation completed successfully.")
    logger.info(f"Check the output file '{output_file}' for the generated documentation.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Documentation generation interrupted by user.")
        sys.exit(0)