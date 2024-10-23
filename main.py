"""
main.py

This script orchestrates the documentation generation process.
"""

import aiohttp
import os
import sys
import logging
import argparse
import asyncio
import json
from dotenv import load_dotenv

from utils import (
    load_config,
    get_all_file_paths,
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    DEFAULT_SKIP_TYPES,
    load_function_schema,
)
from file_handlers import process_all_files

# Load environment variables
load_dotenv()

# Import Sentry SDK and integrations
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.aiohttp import AioHttpIntegration

logger = logging.getLogger(__name__)


def before_send(event, hint):
    """Sentry event handler."""
    if 'request' in event and 'data' in event['request'] and 'password' in event['request']['data']:
        event['request']['data']['password'] = '***REDACTED***'
    if event.get('exception'):
        exception_type = event['exception']['values'][0]['type']
        if exception_type in ['SomeNonCriticalException', 'IgnoredException']:
            return None
    return event


# Configure Sentry Logging Integration
logging_integration = LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)

# Initialize Sentry
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[logging_integration, AioHttpIntegration()],
    traces_sample_rate=0.2,
    environment=os.getenv("ENVIRONMENT", "production"),
    release=os.getenv("RELEASE_VERSION", "unknown"),
    before_send=before_send,
)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate and insert docstrings.")
    parser.add_argument("repo_path", help="Path to the code repository")
    parser.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    parser.add_argument("--concurrency", help="Number of concurrent requests", type=int, default=5)
    parser.add_argument("-o", "--output", help="Output Markdown file", default="output.md")
    parser.add_argument("--deployment-name", help="Deployment name for Azure OpenAI", required=True)
    parser.add_argument("--skip-types", help="Comma-separated list of file extensions to skip", default="")
    parser.add_argument("--project-info", help="Information about the project", default="")
    parser.add_argument("--style-guidelines", help="Documentation style guidelines", default="")
    parser.add_argument("--safe-mode", help="Run in safe mode (no files modified)", action="store_true")
    parser.add_argument("--log-level", help="Logging level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--schema", help="Path to function_schema.json", default="schemas/function_schema.json")
    parser.add_argument("--doc-output-dir", help="Directory to save documentation files", default="documentation")
    return parser.parse_args()


def configure_logging(log_level: str):
    """Configures logging."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
        handlers=[
            logging.FileHandler("docs_generation.log"),
            logging.StreamHandler(sys.stdout)
        ],
    )


async def main():
    """Main function."""
    args = parse_arguments()
    configure_logging(args.log_level)

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
    output_dir = args.doc_output_dir

    azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_openai_api_version = os.getenv('API_VERSION')

    if not all([azure_openai_api_key, azure_openai_endpoint, azure_openai_api_version, deployment_name]):
        logger.critical("Azure OpenAI environment variables not set.")
        sys.exit(1)

    logger.info(f"Repository Path: {repo_path}")
    logger.info(f"Configuration File: {config_path}")
    logger.info(f"Concurrency Level: {concurrency}")
    logger.info(f"Output Markdown File: {output_file}")
    logger.info(f"Deployment Name: {deployment_name}")
    logger.info(f"Safe Mode: {'Enabled' if safe_mode else 'Disabled'}")
    logger.info(f"Function Schema Path: {schema_path}")
    logger.info(f"Documentation Output Directory: {output_dir}")


    if not os.path.isdir(repo_path):
        logger.critical(f"Invalid repository path: '{repo_path}'")
        sys.exit(1)

    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    excluded_files = set(DEFAULT_EXCLUDED_FILES)
    skip_types_set = set(DEFAULT_SKIP_TYPES)
    if skip_types:
        skip_types_list = [ext.strip() for ext in skip_types.split(",") if ext.strip()]
        normalized_skip_types = {
            ext if ext.startswith(".") else f".{ext}" for ext in skip_types_list
        }
        skip_types_set.update(normalized_skip_types)
        logger.debug(f"Updated skip_types: {skip_types_set}")


    project_info_config = ""
    style_guidelines_config = ""

    if not os.path.isfile(config_path):
        logger.warning(
            f"Configuration file '{config_path}' not found. "
            "Proceeding with default and command-line settings."
        )
    else:
        project_info_config, style_guidelines_config = load_config(config_path, excluded_dirs, excluded_files, skip_types_set)

    project_info = project_info_arg or project_info_config
    style_guidelines = style_guidelines_arg or style_guidelines_config

    if project_info:
        logger.info(f"Project Info: {project_info}")
    if style_guidelines:
        logger.info(f"Style Guidelines: {style_guidelines}")


    try:
        function_schema = load_function_schema(schema_path)
    except (ValueError, FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.critical(f"Schema error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected schema loading error: {e}", exc_info=True)
        sys.exit(1)

    try:
        file_paths = get_all_file_paths(repo_path, excluded_dirs, excluded_files, skip_types_set)
    except Exception as e:
        logger.critical(f"Error retrieving file paths: {e}")
        sys.exit(1)

    with sentry_sdk.start_transaction(op="task", name="Documentation Generation"):
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            await process_all_files(
                session=session,
                file_paths=file_paths,
                skip_types=skip_types_set,
                semaphore=asyncio.Semaphore(concurrency),
                deployment_name=deployment_name,
                function_schema=function_schema,
                repo_root=repo_path,
                project_info=project_info,
                style_guidelines=style_guidelines,
                safe_mode=safe_mode,
                output_file=output_file,
                azure_api_key=azure_openai_api_key,
                azure_endpoint=azure_openai_endpoint,
                azure_api_version=azure_openai_api_version,
                output_dir=output_dir,
                schema_path = schema_path # Added schema_path
            )

    logger.info("Documentation generation completed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)