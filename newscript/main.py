# main.py

import os
import sys
import argparse
import asyncio
import logging
from utils import (
    load_config,
    get_all_file_paths,
    OPENAI_API_KEY,
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    DEFAULT_SKIP_TYPES,
)
from language_handlers import process_all_files

# Configure logging
logging.basicConfig(
    filename='docs_generation.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to orchestrate documentation generation."""
    parser = argparse.ArgumentParser(
        description="Generate and insert comments/docstrings using OpenAI's GPT-4o-mini API."
    )
    parser.add_argument("repo_path", help="Path to the code repository")
    parser.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    parser.add_argument("--concurrency", help="Number of concurrent requests", type=int, default=5)
    parser.add_argument("-o", "--output", help="Output Markdown file", default="output.md")
    parser.add_argument("--model", help="OpenAI model to use (default: gpt-4o-mini)", default="gpt-4o-mini")
    parser.add_argument("--skip-types", help="Comma-separated list of file extensions to skip", default="")
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

    if not os.path.isdir(repo_path):
        logger.error(f"Invalid repository path: '{repo_path}' is not a directory.")
        sys.exit(1)

    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    excluded_files = set(DEFAULT_EXCLUDED_FILES)
    skip_types = set(DEFAULT_SKIP_TYPES)
    if args.skip_types:
        skip_types.update(args.skip_types.split(','))

    # Load additional configurations
    load_config(config_path, excluded_dirs, excluded_files, skip_types)

    # Get all file paths
    file_paths = get_all_file_paths(repo_path, excluded_dirs, excluded_files)
    if not file_paths:
        logger.error("No files found to process.")
        sys.exit(1)

    logger.info(f"Starting documentation generation for {len(file_paths)} files.")

    # Clear the output file
    open(output_file, 'w').close()

    # Initialize semaphore and locks
    semaphore = asyncio.Semaphore(concurrency)
    output_lock = asyncio.Lock()

    # Start the asynchronous processing
    try:
        asyncio.run(
            process_all_files(
                file_paths,
                skip_types,
                output_file,
                semaphore,
                output_lock,
                model_name
            )
        )
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)

    logger.info("Documentation generation completed successfully.")

if __name__ == "__main__":
    main()
