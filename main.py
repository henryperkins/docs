# main.py

import os
import sys
import argparse
import asyncio
import logging
import json
from utils import (
    load_config,
    get_all_file_paths,
    OPENAI_API_KEY,
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    DEFAULT_SKIP_TYPES,
    function_schema,
)
from language_handlers import process_all_files

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler('docs_generation.log')
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter and add it to handlers
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def main():
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
    args = parser.parse_args()

    # Validate OpenAI API key
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set. Check environment or .env file.")
        sys.exit(1)

    repo_path = args.repo_path
    config_path = args.config
    concurrency = args.concurrency
    output_file = args.output
    model_name = args.model
    project_info_arg = args.project_info
    style_guidelines_arg = args.style_guidelines
    safe_mode = args.safe_mode

    if not os.path.isdir(repo_path):
        logger.error(f"Invalid repository path: '{repo_path}' is not a directory.")
        sys.exit(1)

    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    excluded_files = set(DEFAULT_EXCLUDED_FILES)
    skip_types = set(DEFAULT_SKIP_TYPES)
    if args.skip_types:
        skip_types.update(ext.strip() for ext in args.skip_types.split(','))

    # Check if config file exists
    if not os.path.isfile(config_path):
        logger.warning(f"Configuration file '{config_path}' not found. Proceeding with default and command-line settings.")
        project_info_config, style_guidelines_config = '', ''
    else:
        # Load additional configurations
        project_info_config, style_guidelines_config = load_config(config_path, excluded_dirs, excluded_files, skip_types)

    # Determine final project_info and style_guidelines
    project_info = project_info_arg or project_info_config
    style_guidelines = style_guidelines_arg or style_guidelines_config

    # Get all file paths
    file_paths = get_all_file_paths(repo_path, excluded_dirs, excluded_files)
    if not file_paths:
        logger.error("No files found to process.")
        sys.exit(1)

    logger.info(f"Starting documentation generation for {len(file_paths)} files.")

    # Clear the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Documentation Generation Report\n\n")
        logger.info(f"Cleared and initialized the output file '{output_file}'.")
    except Exception as e:
        logger.error(f"Failed to initialize output file '{output_file}': {e}")
        sys.exit(1)

    # Initialize semaphore and locks
    semaphore = asyncio.Semaphore(concurrency)
    output_lock = asyncio.Lock()

    # Start the asynchronous processing
    try:
        asyncio.run(
            process_all_files(
                file_paths=file_paths,
                skip_types=skip_types,
                output_file=output_file,
                semaphore=semaphore,
                output_lock=output_lock,
                model_name=model_name,
                function_schema=function_schema,
                repo_root=repo_path,  # Pass repo_root correctly
                project_info=project_info,
                style_guidelines=style_guidelines,
                safe_mode=safe_mode
            )
        )
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Documentation generation completed successfully.")


if __name__ == "__main__":
    main()
