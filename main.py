# main.py

import sys
import logging
import argparse
import asyncio
import tempfile
import shutil
import subprocess
from dotenv import load_dotenv
from provider_config import load_provider_configs
from process_manager import DocumentationProcessManager
from utils import DEFAULT_EXCLUDED_FILES, DEFAULT_EXCLUDED_DIRS, DEFAULT_SKIP_TYPES, load_config, get_all_file_paths, setup_logging

logger = logging.getLogger(__name__)


def is_git_url(path: str) -> bool:
    """Check if the path looks like a git URL."""
    return (
        path.startswith("https://") or
        path.startswith("http://") or
        path.startswith("git@") or
        path.startswith("ssh://") or
        path.endswith(".git")
    )


def clone_repo(url: str) -> str:
    """Clone a git repo to a temp directory and return the path."""
    clone_dir = tempfile.mkdtemp(prefix="docgen_")
    logger.info(f"Cloning {url} to {clone_dir}...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, clone_dir],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Clone complete: {clone_dir}")
        return clone_dir
    except subprocess.CalledProcessError as e:
        shutil.rmtree(clone_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to clone {url}: {e.stderr.strip()}")


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and insert docstrings using Azure OpenAI, Gemini, or OpenAI models.")
    parser.add_argument("repo_path", help="Path to the code repository or a git URL (https/ssh)")
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
    cloned_dir = None

    # Auto-clone if a git URL is provided
    if is_git_url(repo_path):
        cloned_dir = clone_repo(repo_path)
        repo_path = cloned_dir

    config_path = args.config
    output_dir = args.doc_output_dir

    # Load provider configurations
    provider_configs = load_provider_configs()

    # Validate API configuration based on provider
    if args.provider not in provider_configs:
        logger.error(f"Unsupported provider: {args.provider}")
        sys.exit(1)

    try:
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

        file_paths = get_all_file_paths(
            repo_path, excluded_dirs, excluded_files, skip_types_set)

        # Initialize DocumentationProcessManager
        manager = DocumentationProcessManager(
            repo_root=repo_path,
            output_dir=output_dir,
            provider_configs=provider_configs,
            max_concurrency=args.concurrency,
            schema_path=args.schema,
        )

        # Build request
        from process_manager import DocumentationRequest
        request = DocumentationRequest(
            file_paths=file_paths,
            skip_types=list(skip_types_set),
            project_info=project_info,
            style_guidelines=style_guidelines,
            safe_mode=args.safe_mode,
            project_id=args.project_id,
            provider=args.provider,
            max_concurrency=args.concurrency
        )

        import uuid
        task_id = str(uuid.uuid4())
        results = await manager.process_files(request, task_id)

        logger.info(f"Documentation generation completed. Results: {results}")

    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if cloned_dir:
            logger.info(f"Cleaning up cloned repo: {cloned_dir}")
            shutil.rmtree(cloned_dir, ignore_errors=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
