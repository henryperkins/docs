## main.py

```python
"""
main.py

This script serves as the main entry point for generating and inserting docstrings into code repositories using Azure OpenAI. It handles command-line argument parsing, configuration loading, and orchestrates the documentation generation process.
"""

import aiohttp
import os
import sys
import logging
import argparse
import asyncio
import tracemalloc
import json
from dotenv import load_dotenv

from utils import (
    load_config,
    get_all_file_paths,
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    DEFAULT_SKIP_TYPES,
    load_function_schema,
    validate_schema,
)
from file_handlers import process_all_files

# Load environment variables from .env file early
load_dotenv()

# Enable tracemalloc for memory allocation tracking
tracemalloc.start()

# Import Sentry SDK and integrations
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.aiohttp import AioHttpIntegration

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
)
logger = logging.getLogger(__name__)

def before_send(event, hint):
    """
    Modify or filter events before sending to Sentry.

    Args:
        event (dict): The event data.
        hint (dict): Additional context about the event.

    Returns:
        dict or None: The modified event or None to drop the event.
    """
    if 'request' in event and 'data' in event['request'] and 'password' in event['request']['data']:
        event['request']['data']['password'] = '***REDACTED***'
    if event.get('exception'):
        exception_type = event['exception']['values'][0]['type']
        if exception_type in ['SomeNonCriticalException', 'IgnoredException']:
            return None
    return event

# Configure Sentry Logging Integration
logging_integration = LoggingIntegration(
    level=logging.INFO,
    event_level=logging.ERROR
)

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
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate and insert docstrings using Azure OpenAI.")
    parser.add_argument("repo_path", help="Path to the code repository")
    parser.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    parser.add_argument("--concurrency", help="Number of concurrent requests", type=int, default=5)
    parser.add_argument("-o", "--output", help="Output Markdown file", default="output.md")
    parser.add_argument("--deployment-name", help="Deployment name for Azure OpenAI", required=True)
    parser.add_argument("--skip-types", help="Comma-separated list of file extensions to skip", default="")
    parser.add_argument("--project-info", help="Information about the project", default="")
    parser.add_argument("--style-guidelines", help="Documentation style guidelines", default="")
    parser.add_argument("--safe-mode", help="Run in safe mode (no files modified)", action="store_true")
    parser.add_argument("--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)", default="INFO")
    parser.add_argument("--schema", help="Path to function_schema.json", default="schemas/function_schema.json")
    parser.add_argument("--doc-output-dir", help="Directory to save documentation files", default="documentation")
    return parser.parse_args()

def configure_logging(log_level: str):
    """
    Configures logging.

    Args:
        log_level (str): The logging level.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s")

    file_handler = logging.FileHandler("docs_generation.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

async def main():
    """
    Main function to orchestrate the documentation generation process.
    """
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

    # Ensure necessary environment variables are set for Azure OpenAI Service
    azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_openai_api_version = os.getenv('API_VERSION')

    if not all([azure_openai_api_key, azure_openai_endpoint, azure_openai_api_version, deployment_name]):
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
    logger.info(f"Documentation Output Directory: {output_dir}")

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
        project_info_config, style_guidelines_config = load_config(config_path, excluded_dirs, excluded_files, skip_types_set)

    project_info = project_info_arg or project_info_config
    style_guidelines = style_guidelines_arg or style_guidelines_config

    if project_info:
        logger.info(f"Project Info: {project_info}")
    if style_guidelines:
        logger.info(f"Style Guidelines: {style_guidelines}")

    # Load function schema
    try:
        function_schema = load_function_schema(schema_path)
        validate_schema(schema_path)
    except (ValueError, FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.critical(f"Schema error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during schema loading: {e}", exc_info=True)
        sys.exit(1)

    try:
        file_paths = get_all_file_paths(repo_path, excluded_dirs, excluded_files, skip_types_set)
    except Exception as e:
        logger.critical(f"Error retrieving file paths: {e}")
        sys.exit(1)

    # Start a Sentry transaction for the main documentation generation process
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
                output_dir=output_dir
            )

    logger.info("Documentation generation completed successfully.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
```

## file_handlers.py

```python
"""
file_handlers.py

This module contains asynchronous functions for processing individual files, extracting code structures, generating documentation via Azure OpenAI API calls, and inserting docstrings into the code. It also manages backups and uses the ContextManager to maintain persistent context across files.
"""
import os
import shutil
import logging
import aiohttp
import json
import asyncio
import jsonschema
from typing import Set, List, Dict, Any, Optional, Tuple
from jsonschema import validate, ValidationError
from language_functions import get_handler
from language_functions.base_handler import BaseHandler
from utils import (
    is_binary,
    get_language,
    clean_unused_imports_async,
    format_with_black_async,
)
from write_documentation_report import generate_documentation_prompt, write_documentation_report
from context_manager import ContextManager

logger = logging.getLogger(__name__)

# Initialize the ContextManager
context_manager = ContextManager(max_entries=100)


async def extract_code_structure(
    content: str, file_path: str, language: str, handler: BaseHandler
) -> Optional[Dict[str, Any]]:
    """
    Asynchronously extracts the code structure from the given content using the specified handler.

    Args:
        content (str): The source code content.
        file_path (str): Path to the source file.
        language (str): Programming language of the source code.
        handler (BaseHandler): The handler object for the specific language.

    Returns:
        Optional[Dict[str, Any]]: A dictionary representing the code structure or None if extraction fails.
    """
    logger.debug(f"Extracting code structure for '{file_path}' (language: {language})")
    try:
        structure = await asyncio.to_thread(handler.extract_structure, content, file_path)
        if not structure:
            logger.warning(f"No structure extracted from '{file_path}'")
            return None
        return structure
    except Exception as e:
        logger.error(f"Error extracting structure from '{file_path}': {e}", exc_info=True)
        return None


async def backup_and_write_new_content(file_path: str, new_content: str) -> None:
    """
    Creates a backup of the original file and writes new content to it.

    Args:
        file_path (str): Path to the file to update.
        new_content (str): The new content to write to the file.
    """
    backup_path = f"{file_path}.bak"
    try:
        if os.path.exists(backup_path):
            os.remove(backup_path)
            logger.debug(f"Removed existing backup at '{backup_path}'.")
        await asyncio.to_thread(shutil.copy, file_path, backup_path)
        logger.debug(f"Backup created at '{backup_path}'.")
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(new_content)
        logger.info(f"Inserted documentation into '{file_path}'.")
    except Exception as e:
        logger.error(f"Error writing to '{file_path}': {e}", exc_info=True)
        # Attempt to restore from backup
        if os.path.exists(backup_path):
            try:
                await asyncio.to_thread(shutil.copy, backup_path, file_path)
                os.remove(backup_path)
                logger.info(f"Restored original file from backup for '{file_path}'.")
            except Exception as restore_error:
                logger.error(f"Failed to restore backup for '{file_path}': {restore_error}", exc_info=True)
                

def validate_ai_response(response: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and potentially reformats the AI's response to match the required schema.

    Args:
        response (Dict[str, Any]): The AI's response.
        schema (Dict[str, Any]): The expected schema.

    Returns:
        Dict[str, Any]: The validated and potentially reformatted response.
    """
    try:
        validate(instance=response, schema=schema)
        return response
    except ValidationError as validation_error:
        logger.error(f"AI response does not match schema: {validation_error}")
        # Here you could implement logic to try to correct the response
        # For now, we'll just return None to indicate failure
        return None

async def fetch_documentation_rest(
    session: aiohttp.ClientSession,
    prompt: str,
    semaphore: asyncio.Semaphore,
    deployment_name: str,
    function_schema: Dict[str, Any],
    azure_api_key: str,
    azure_endpoint: str,
    azure_api_version: str,
    retry: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Fetches documentation from Azure OpenAI API using the provided prompt and schema.
    
    ... (rest of the docstring)
    """
    logger.debug(f"Fetching documentation using REST API for deployment: {deployment_name}")

    url = f"{azure_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={azure_api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_api_key,
    }

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "functions": function_schema.get("functions", []),
        "function_call": {"name": "generate_documentation"}
    }

    async def handle_api_error(response: aiohttp.ClientResponse) -> None:
        """Handle API errors and log appropriate messages."""
        error_text = await response.text()
        try:
            error_json = json.loads(error_text)
            error_message = error_json.get("error", {}).get("message", "Unknown error")
            logger.error(f"Azure OpenAI API request failed: {response.status} - {error_message}")
        except json.JSONDecodeError:
            logger.error(f"API request failed with non-JSON response: {response.status} - {error_text}")

    for attempt in range(retry):
        try:
            async with semaphore:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"API Response: {data}")

                        if "choices" in data and data["choices"]:
                            choice = data["choices"][0]
                            message = choice.get("message")
                            
                            if message and "function_call" in message:
                                function_call = message["function_call"]
                                if function_call.get("name") == "generate_documentation":
                                    arguments = function_call.get("arguments")
                                    try:
                                        documentation = json.loads(arguments)
                                        # Validate the AI's response against the schema
                                        validated_documentation = validate_ai_response(
                                            documentation, 
                                            function_schema["functions"][0]["parameters"]
                                        )
                                        if validated_documentation:
                                            logger.debug("Successfully validated documentation response")
                                            return validated_documentation
                                        else:
                                            logger.error("AI response validation failed")
                                            # Add more context to the prompt for retry
                                            payload["messages"][0]["content"] = (
                                                prompt + "\n\nPlease ensure your response exactly matches "
                                                "the provided schema and includes all required fields."
                                            )
                                            continue
                                    except json.JSONDecodeError as e:
                                        logger.error(f"Error decoding JSON: {e}")
                                        logger.error(f"Arguments Content: {arguments}")
                                else:
                                    logger.error(f"Unexpected function called: {function_call.get('name')}")
                            else:
                                logger.error("No function_call found in the response.")
                        else:
                            logger.error("No choices found in the API response.")
                    elif response.status == 429:  # Rate limit hit
                        retry_after = response.headers.get("Retry-After", str(2 ** attempt))
                        wait_time = int(retry_after)
                        logger.warning(f"Rate limited. Retrying after {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        await handle_api_error(response)

        except aiohttp.ClientError as e:
            logger.error(f"Network error during API request: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during API request: {e}", exc_info=True)

        if attempt < retry - 1:
            wait_time = 2 ** attempt
            logger.warning(f"Retrying after {wait_time} seconds... (Attempt {attempt + 1}/{retry})")
            await asyncio.sleep(wait_time)

    logger.error("All retry attempts to fetch documentation failed.")
    return None


async def handle_api_error(response):
    error_text = await response.text()
    try:
        error_json = json.loads(error_text)
        error_message = error_json.get("error", {}).get("message", "Unknown error")
        logger.error(f"Azure OpenAI API request failed: {response.status} - {error_message}")
    except json.JSONDecodeError:
        logger.error(f"API request failed with non-JSON response: {response.status} - {error_text}")


def validate_documentation(documentation: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validates the given documentation against the provided JSON schema.

    Args:
        documentation (Dict[str, Any]): The documentation to be validated.
        schema (Dict[str, Any]): The JSON schema to validate against.

    Returns:
        Optional[Dict[str, Any]]: The validated documentation if it passes the schema validation, 
        otherwise None if validation fails.
    """
    try:
        jsonschema.validate(instance=documentation, schema=schema)
        return documentation
    except jsonschema.exceptions.ValidationError as e:
        logger.error(f"Documentation validation failed: {e}")
        # You might want to add logic here to attempt to fix the documentation
        return None

async def process_file(
    session, file_path, skip_types, semaphore, deployment_name,
    function_schema, repo_root, project_info, style_guidelines, safe_mode,
    azure_api_key, azure_endpoint, azure_api_version, output_dir
) -> Optional[str]:
    """Main file processing function."""

    if not should_process_file(file_path, skip_types):
        return None

    content, language, handler = await _prepare_file(file_path, function_schema, skip_types)
    if not all([content, language, handler]):  # Check if all are not None/False
        return None

    code_structure = await _extract_code_structure(content, file_path, language, handler)
    if not code_structure:
        return None

    documentation = await _generate_documentation(
        session, semaphore, deployment_name, function_schema,
        repo_root, project_info, style_guidelines, code_structure, file_path,
        azure_api_key, azure_endpoint, azure_api_version
    )
    if not documentation:
        return None

    _update_documentation_metrics(documentation, code_structure)

    if not safe_mode:
        await _insert_and_validate_documentation(handler, content, documentation, file_path, language)

    return await _write_documentation_report(documentation, language, file_path, repo_root, output_dir)



async def _prepare_file(file_path: str, function_schema: Dict[str, Any], skip_types: Set[str]) -> tuple[Optional[str], Optional[str], Optional[BaseHandler]]:
    """Reads file content, gets language and handler."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None, None, None

    _, ext = os.path.splitext(file_path)
    extra_skip_types = {'.flake8', '.gitignore', '.env', '.pyc', '.pyo', '.pyd', '.git', '.d.ts'}
    if ext in extra_skip_types or ext in skip_types or not ext or "node_modules" in file_path:
        logger.debug(f"Skipping file '{file_path}' due to extension/location: {ext}")
        return None, None, None

    if is_binary(file_path):
        logger.debug(f"Skipping binary file: {file_path}")
        return None, None, None

    language = get_language(ext)
    if language == "plaintext":
        logger.debug(f"Skipping plaintext file: {file_path}")
        return None, None, None

    handler = get_handler(language, function_schema)
    if not handler:
        logger.debug(f"No handler available for language: {language}")
        return None, None, None

    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
        logger.debug(f"Successfully read content from {file_path}")
        return content, language, handler
    except UnicodeDecodeError:
        logger.warning(f"Skipping file due to encoding issues: {file_path}")
        return None, None, None
    except Exception as e:
        logger.error(f"Failed to read '{file_path}': {e}", exc_info=True)
        return None, None, None


async def _extract_code_structure(content: str, file_path: str, language: str, handler: BaseHandler) -> Optional[Dict[str, Any]]:
    """Extracts code structure and adds context."""

    try:
        code_structure = await extract_code_structure(content, file_path, language, handler)  # Assuming this function is defined elsewhere
        if not code_structure:
            return None

        try:
            critical_info = extract_critical_info(code_structure, file_path)  # Assuming this function is defined elsewhere
            context_manager.add_context(critical_info)
        except Exception as e:
            logger.error(f"Error extracting critical info: {e}", exc_info=True)
            critical_info = f"File: {file_path}\n# Failed to extract detailed information"
            context_manager.add_context(critical_info)
        return code_structure

    except Exception as e:
        logger.error(f"Error extracting structure: {e}", exc_info=True)
        return None



async def _generate_documentation(
    session, semaphore, deployment_name, function_schema,
    repo_root, project_info, style_guidelines, code_structure, file_path,
    azure_api_key, azure_endpoint, azure_api_version
) -> Optional[Dict[str, Any]]:
    """Generates documentation from Azure OpenAI."""
    try:
        persistent_context = "\n".join(context_manager.get_context())
        prompt = f"""
[Context Start]
{persistent_context}
[Context End]

{generate_documentation_prompt(
    file_name=os.path.basename(file_path),
    code_structure=code_structure,
    project_info=project_info,
    style_guidelines=style_guidelines,
    language=get_language(os.path.splitext(file_path)[1]),
    function_schema=function_schema
)}
        """.strip()

        documentation = await fetch_documentation_rest(  # Assuming this function is defined elsewhere
            session=session,
            prompt=prompt,
            semaphore=semaphore,
            deployment_name=deployment_name,
            function_schema=function_schema,
            azure_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
        )

        if not documentation:
            logger.error(f"Failed to generate documentation for '{file_path}'")
        return documentation

    except Exception as e:
        logger.error(f"Error generating documentation: {e}", exc_info=True)
        return None


def _update_documentation_metrics(documentation: Dict[str, Any], code_structure: Dict[str, Any]) -> None:
    """Updates documentation with metrics from code structure."""

    documentation.update({
        "halstead": code_structure.get("halstead", {}),
        "maintainability_index": code_structure.get("maintainability_index"),
        "variables": code_structure.get("variables", []),
        "constants": code_structure.get("constants", []),
        "changes_made": documentation.get("changes_made", [])
    })

    function_complexity = {func["name"]: func.get("complexity", 0) for func in code_structure.get("functions", [])}
    for func in documentation.get("functions", []):
        func["complexity"] = function_complexity.get(func["name"], 0)

    class_complexity = {}
    for cls in code_structure.get("classes", []):
        methods_complexity = {method["name"]: method.get("complexity", 0) for method in cls.get("methods", [])}
        class_complexity[cls["name"]] = methods_complexity

    for cls in documentation.get("classes", []):
        methods_complexity = class_complexity.get(cls["name"], {})
        for method in cls.get("methods", []):
            method["complexity"] = methods_complexity.get(method["name"], 0)


async def _insert_and_validate_documentation(handler: BaseHandler, content: str, documentation: Dict[str, Any], file_path: str, language: str) -> None:
    """Inserts documentation and validates the updated code."""
    try:
        new_content = await asyncio.to_thread(handler.insert_docstrings, content, documentation)

        if language.lower() == "python":
            new_content = await clean_unused_imports_async(new_content, file_path)
            new_content = await format_with_black_async(new_content)

        is_valid = await asyncio.to_thread(handler.validate_code, new_content, file_path)

        if is_valid:
            await backup_and_write_new_content(file_path, new_content)  # Assuming this function is defined elsewhere
            logger.info(f"Documentation inserted into '{file_path}'")
        else:
            logger.error(f"Code validation failed for '{file_path}'")

    except Exception as e:
        logger.error(f"Error processing documentation: {e}", exc_info=True)


async def _write_documentation_report(documentation: Dict[str, Any], language: str, file_path: str, repo_root: str, output_dir: str) -> Optional[str]:
    """Writes the documentation report to a file."""
    try:
        file_content = await write_documentation_report(
            documentation=documentation,
            language=language,
            file_path=file_path,
            repo_root=repo_root,
            output_dir=output_dir,
        )
        logger.info(f"Finished processing '{file_path}'")
        return file_content

    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        return None

def should_process_file(file_path: str, skip_types: Set[str]) -> bool:
    """
    Checks if a file should be processed based on path, extension and skip types.
    """

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    _, ext = os.path.splitext(file_path)

    # Combines all skip conditions for clarity
    if (
        os.path.islink(file_path) or
        any(part in file_path for part in ['node_modules', '.bin']) or
        file_path.endswith('.d.ts') or
        any(excluded in file_path for excluded in {'.git', '__pycache__', 'node_modules', '.bin', 'build', 'dist'}) or
        ext in {'.flake8', '.gitignore', '.env', '.pyc', '.pyo', '.pyd', '.git', '.d.ts'} or
        ext in skip_types or not ext
    ):
        logger.debug(f"Skipping file '{file_path}' due to extension/location: {ext}")
        return False

    return True

async def process_all_files(
    session: aiohttp.ClientSession,
    file_paths: List[str],
    skip_types: Set[str],
    semaphore: asyncio.Semaphore,
    deployment_name: str,
    function_schema: Dict[str, Any],
    repo_root: str,
    project_info: Optional[str],
    style_guidelines: Optional[str],
    safe_mode: bool = False,
    output_file: str = "output.md",
    azure_api_key: str = "",
    azure_endpoint: str = "",
    azure_api_version: str = "",
    output_dir: str = "documentation",
) -> None:
    logger.info("Starting process of all files.")
    tasks = [
        process_file(
            session=session,
            file_path=file_path,
            skip_types=skip_types,
            semaphore=semaphore,
            deployment_name=deployment_name,
            function_schema=function_schema,
            repo_root=repo_root,
            project_info=project_info,
            style_guidelines=style_guidelines,
            safe_mode=safe_mode,
            azure_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            output_dir=output_dir,
        )
        for file_path in file_paths
    ]

    documentation_contents = []
    for f in asyncio.as_completed(tasks):
        try:
            file_content = await f
            if file_content:
                documentation_contents.append(file_content)
        except Exception as e:
            logger.error(f"Error processing a file: {e}", exc_info=True)

    logger.info("Completed processing all files.")

    final_content = "\n\n".join(documentation_contents)

    if final_content:
        toc = generate_table_of_contents(final_content)
        report_content = (
            "# Documentation Generation Report\n\n## Table of Contents\n\n" + toc + "\n\n" + final_content
        )

        try:
            async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
                await f.write(report_content)
            logger.info(f"Documentation report written to '{output_file}'")
        except Exception as e:
            logger.error(f"Error writing final documentation to '{output_file}': {e}", exc_info=True)

    else:
        logger.warning("No documentation was generated.")

    logger.info("Documentation generation process completed.")


def extract_critical_info(code_structure: Dict[str, Any], file_path: str) -> str:
    """
    Extracts critical information from the code structure to be used as persistent context.

    Args:
        code_structure (Dict[str, Any]): The extracted code structure.
        file_path (str): Path to the source file.

    Returns:
        str: A formatted string containing critical context information.
    """
    info_lines = [f"File: {file_path}"]
    
    # Extract function signatures
    functions = code_structure.get("functions", [])
    for func in functions:
        if isinstance(func, dict):  # Check if func is a dictionary
            signature = f"def {func.get('name', 'unknown')}({', '.join(func.get('args', []))}):"
            doc = func.get('docstring', '').split('\n')[0]
            info_lines.append(f"{signature}  # {doc}")

    # Extract class definitions
    classes = code_structure.get("classes", [])
    for cls in classes:
        if isinstance(cls, dict):  # Check if cls is a dictionary
            class_info = f"class {cls.get('name', 'unknown')}:"
            doc = cls.get('docstring', '').split('\n')[0]
            info_lines.append(f"{class_info}  # {doc}")
            
            # Include methods
            for method in cls.get('methods', []):
                if isinstance(method, dict):  # Check if method is a dictionary
                    method_signature = f"    def {method.get('name', 'unknown')}({', '.join(method.get('args', []))}):"
                    method_doc = method.get('docstring', '').split('\n')[0]
                    info_lines.append(f"{method_signature}  # {method_doc}")

    # Extract important variables
    variables = code_structure.get("variables", [])
    for var in variables:
        if isinstance(var, dict):  # Check if var is a dictionary
            var_info = f"{var.get('name', 'unknown')} = "
            var_type = var.get('type', 'Unknown')
            var_desc = var.get('description', '').split('\n')[0]
            info_lines.append(f"{var_info}  # Type: {var_type}, {var_desc}")
        elif isinstance(var, str):  # Handle string variables
            info_lines.append(f"{var} = ")  # Just add the variable name if it's a string

    critical_info = "\n".join(info_lines)
    return critical_info
```

## metrics.py

```python
"""
metrics.py

This module provides functions for calculating various code metrics, including:
- Halstead complexity metrics
- Cyclomatic complexity
- Maintainability index
- Raw metrics (lines of code, blank lines, comment lines, etc.)
- Other code quality metrics (method length, argument count, etc.)
"""

import logging
import math
from typing import Dict, Any, Tuple, Optional
import ast

from radon.complexity import cc_visit, SCORE
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze

logger = logging.getLogger(__name__)


def calculate_halstead_metrics(code: str) -> Dict[str, Any]:
    """Calculates Halstead complexity metrics."""
    try:
        halstead_visitor = h_visit(code)
        total_metrics = halstead_visitor.total

        h1 = len(total_metrics.operators)
        h2 = len(total_metrics.operands)
        N1 = sum(total_metrics.operators.values())
        N2 = sum(total_metrics.operands.values())

        vocabulary = h1 + h2
        length = N1 + N2
        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (h1 * N2) / (2 * h2) if h2 > 0 else 0
        effort = difficulty * volume

        return {
            "volume": round(volume, 2),
            "difficulty": round(difficulty, 2),
            "effort": round(effort, 2),
            "vocabulary": vocabulary,
            "length": length,
            "distinct_operators": h1,
            "distinct_operands": h2,
            "total_operators": N1,
            "total_operands": N2,
            "operator_counts": dict(total_metrics.operators),
            "operand_counts": dict(total_metrics.operands),
        }

    except Exception as e:
        logger.error(f"Error calculating Halstead metrics: {e}")
        return {  # Return a dictionary of zeros on error
            "volume": 0,
            "difficulty": 0,
            "effort": 0,
            "vocabulary": 0,
            "length": 0,
            "distinct_operators": 0,
            "distinct_operands": 0,
            "total_operators": 0,
            "total_operands": 0,
            "operator_counts": {},
            "operand_counts": {},
        }


def calculate_cyclomatic_complexity(code: str) -> Tuple[Dict[str, int], int]:
    """Calculates cyclomatic complexity."""
    try:
        complexity_scores = cc_visit(code)
        function_complexity = {score.fullname: score.complexity for score in complexity_scores}
        total_complexity = sum(score.complexity for score in complexity_scores)
        return function_complexity, total_complexity
    except Exception as e:
        logger.error(f"Error calculating cyclomatic complexity: {e}")
        return {}, 0


def calculate_maintainability_index(code: str) -> Optional[float]:
    """Calculates maintainability index."""
    try:
        return mi_visit(code, True)
    except Exception as e:
        logger.error(f"Error calculating maintainability index: {e}")
        return None


def calculate_raw_metrics(code: str) -> Dict[str, int]:
    """Calculates raw metrics (LOC, comments, blank lines, etc.)."""
    try:
        raw_metrics = analyze(code)
        return {
            "loc": raw_metrics.loc,
            "lloc": raw_metrics.lloc,
            "sloc": raw_metrics.sloc,
            "comments": raw_metrics.comments,
            "multi": raw_metrics.multi,
            "blank": raw_metrics.blank,
        }
    except Exception as e:
        logger.error(f"Error calculating raw metrics: {e}")
        return {
            "loc": 0,
            "lloc": 0,
            "sloc": 0,
            "comments": 0,
            "multi": 0,
            "blank": 0,
        }

def calculate_code_quality_metrics(code: str) -> Dict[str, Any]:
    """Calculates code quality metrics."""
    try:
        tree = ast.parse(code)
        metrics = {
            "method_length": [],
            "argument_count": [],
            "nesting_level": [], # Now a list to store per-function nesting levels
            "max_nesting_level": 0 # Track the overall maximum nesting
        }

        class QualityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_nesting = 0
                self.max_nesting = 0

            def visit_FunctionDef(self, node):
                self.current_nesting = 0 # Reset for each function
                self.max_nesting = 0 # Reset for each function
                self.generic_visit(node)
                metrics["nesting_level"].append(self.max_nesting)
                metrics["method_length"].append(node.end_lineno - node.lineno + 1)
                metrics["argument_count"].append(len(node.args.args))
            
            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node) # Treat async functions the same

            def visit_If(self, node):
                self.current_nesting += 1
                self.max_nesting = max(self.max_nesting, self.current_nesting)
                self.generic_visit(node)
                self.current_nesting -= 1

            def visit_While(self, node):
                self.current_nesting += 1
                self.max_nesting = max(self.max_nesting, self.current_nesting)
                self.generic_visit(node)
                self.current_nesting -= 1

            def visit_For(self, node):
                self.current_nesting += 1
                self.max_nesting = max(self.max_nesting, self.current_nesting)
                self.generic_visit(node)
                self.current_nesting -= 1

            # Similar logic for other nesting structures (try, except, etc.)

        QualityVisitor().visit(tree)

        metrics["max_nesting_level"] = max(metrics["nesting_level"]) if metrics["nesting_level"] else 0
        metrics["avg_nesting_level"] = sum(metrics["nesting_level"]) / len(metrics["nesting_level"]) if metrics["nesting_level"] else 0
        # ... (Calculate averages for method_length and argument_count as before)

        return metrics

    except Exception as e:
        logger.error(f"Error calculating code quality metrics: {e}")
        return {
            "method_length": [],
            "argument_count": [],
            "nesting_level": 0,
            "avg_method_length": 0,
            "avg_argument_count": 0,
        }


def calculate_all_metrics(code: str) -> Dict[str, Any]:
    """Calculates all available metrics."""
    metrics = {}
    metrics["halstead"] = calculate_halstead_metrics(code)
    metrics["cyclomatic"] = calculate_cyclomatic_complexity(code) # Changed name for clarity
    metrics["maintainability_index"] = calculate_maintainability_index(code)
    metrics["raw"] = calculate_raw_metrics(code)
    metrics["quality"] = calculate_code_quality_metrics(code)  # New metrics
    return metrics
```

## utils.py

```python
"""
utils.py

This module provides utility functions for handling language and file operations, configuration management, code formatting, and cleanup. It includes functions for loading JSON schemas, managing file paths, and running code formatters like autoflake, black, and flake8.
"""

import os
import sys
import json
import logging
import aiohttp
import asyncio
import subprocess
from dotenv import load_dotenv
from typing import Any, Set, List, Optional, Dict, Tuple
from jsonschema import Draft7Validator, ValidationError, SchemaError

# Load environment variables
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("documentation_generation.log"),
        logging.StreamHandler(sys.stdout)
    ],
)
logger = logging.getLogger(__name__)

# ----------------------------
# Constants
# ----------------------------

DEFAULT_COMPLEXITY_THRESHOLDS = {"low": 10, "medium": 20, "high": 30}

DEFAULT_HALSTEAD_THRESHOLDS = {
    "volume": {"low": 100, "medium": 500, "high": 1000},
    "difficulty": {"low": 10, "medium": 20, "high": 30},
    "effort": {"low": 500, "medium": 1000, "high": 2000}
}

DEFAULT_MAINTAINABILITY_THRESHOLDS = {"low": 50, "medium": 70, "high": 85}

DEFAULT_EXCLUDED_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', '.idea', 'scripts'}
DEFAULT_EXCLUDED_FILES = {".DS_Store"}
DEFAULT_SKIP_TYPES = {".json", ".md", ".txt", ".csv", ".lock"}

LANGUAGE_MAPPING = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".go": "go",
    ".cpp": "cpp",
    ".c": "cpp",
    ".java": "java",
}

# ----------------------------
# Language and File Utilities
# ----------------------------

def get_language(ext: str) -> str:
    """
    Determines the programming language based on file extension.

    Args:
        ext (str): File extension.

    Returns:
        str: Corresponding programming language.
    """
    language = LANGUAGE_MAPPING.get(ext.lower(), "plaintext")
    logger.debug(f"Detected language for extension '{ext}': {language}")
    return language

def is_valid_extension(ext: str, skip_types: Set[str]) -> bool:
    """
    Checks if a file extension is valid (not in the skip list).

    Args:
        ext (str): File extension.
        skip_types (Set[str]): Set of file extensions to skip.

    Returns:
        bool: True if valid, False otherwise.
    """
    is_valid = ext.lower() not in skip_types
    logger.debug(f"Extension '{ext}' is valid: {is_valid}")
    return is_valid

def get_threshold(metric: str, key: str, default: int) -> int:
    """
    Retrieves the threshold value for a given metric and key from environment variables.

    Args:
        metric (str): The metric name.
        key (str): The threshold key (e.g., 'low', 'medium', 'high').
        default (int): The default value if the environment variable is not set or invalid.

    Returns:
        int: The threshold value.
    """
    try:
        return int(os.getenv(f"{metric.upper()}_{key.upper()}_THRESHOLD", default))
    except ValueError:
        logger.error(f"Invalid environment variable for {metric.upper()}_{key.upper()}_THRESHOLD")
        return default
    
def is_binary(file_path: str) -> bool:
    """
    Checks if a file is binary.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if binary, False otherwise.
    """
    try:
        with open(file_path, "rb") as file:
            return b"\0" in file.read(1024)
    except Exception as e:
        logger.error(f"Error checking if file is binary '{file_path}': {e}")
        return True

import os
import pathspec  # You'll need to add this to requirements.txt
from typing import List, Set

def load_gitignore(repo_path: str) -> pathspec.PathSpec:
    """
    Loads .gitignore patterns into a PathSpec object.
    
    Args:
        repo_path (str): Path to the repository root
        
    Returns:
        pathspec.PathSpec: Compiled gitignore patterns
    """
    gitignore_path = os.path.join(repo_path, '.gitignore')
    patterns = []
    
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
    
    return pathspec.PathSpec.from_lines(
        pathspec.patterns.GitWildMatchPattern, patterns
    )

def get_all_file_paths(repo_path: str, excluded_dirs: Set[str], excluded_files: Set[str], skip_types: Set[str]) -> List[str]:
    """
    Retrieves all file paths in the repository, excluding specified directories and files.

    Args:
        repo_path (str): Path to the repository.
        excluded_dirs (Set[str]): Set of directories to exclude.
        excluded_files (Set[str]): Set of files to exclude.
        skip_types (Set[str]): Set of file extensions to skip.

    Returns:
        List[str]: List of file paths.
    """
    file_paths = []
    normalized_excluded_dirs = {os.path.normpath(os.path.join(repo_path, d)) for d in excluded_dirs}
    
    # Add common node_modules patterns to excluded dirs
    node_modules_patterns = {
        'node_modules',
        '.bin',
        'node_modules/.bin',
        '**/node_modules/**/.bin',
        '**/node_modules/**/node_modules'
    }
    normalized_excluded_dirs.update({os.path.normpath(os.path.join(repo_path, d)) for d in node_modules_patterns})

    for root, dirs, files in os.walk(repo_path, topdown=True):
        # Skip node_modules and other excluded directories
        if any(excluded in root for excluded in ['node_modules', '.bin']):
            dirs[:] = []  # Skip processing subdirectories
            continue

        # Exclude directories
        dirs[:] = [
            d for d in dirs 
            if os.path.normpath(os.path.join(root, d)) not in normalized_excluded_dirs
            and not any(excluded in d for excluded in ['node_modules', '.bin'])
        ]

        for file in files:
            # Skip excluded files
            if file in excluded_files:
                continue
                
            # Skip specified file types
            file_ext = os.path.splitext(file)[1]
            if file_ext in skip_types:
                continue

            # Skip symlinks
            full_path = os.path.join(root, file)
            if os.path.islink(full_path):
                continue

            file_paths.append(full_path)
            
    logger.debug(f"Collected {len(file_paths)} files from '{repo_path}'.")
    return file_paths

def should_process_file(file_path: str) -> bool:
    """
    Determines if a file should be processed based on various criteria.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file should be processed, False otherwise.
    """
    # Skip symlinks
    if os.path.islink(file_path):
        return False

    # Skip node_modules related paths
    if any(part in file_path for part in ['node_modules', '.bin']):
        return False

    # Skip if file doesn't exist
    if not os.path.exists(file_path):
        return False

    return True

# ----------------------------
# Configuration Management
# ----------------------------

def load_json_schema(schema_path: str) -> Optional[dict]:
    """
    Loads a JSON schema from the specified path.

    Args:
        schema_path (str): Path to the JSON schema file.

    Returns:
        Optional[dict]: Loaded JSON schema or None if failed.
    """
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        logger.debug(f"Successfully loaded JSON schema from '{schema_path}'.")
        return schema
    except FileNotFoundError:
        logger.error(f"JSON schema file '{schema_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{schema_path}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading JSON schema from '{schema_path}': {e}")
        return None

def load_function_schema(schema_path: str) -> Dict[str, Any]:
    """
    Loads a function schema and validates it.

    Args:
        schema_path (str): Path to the schema file.

    Returns:
        Dict[str, Any]: The loaded and validated function schema.

    Raises:
        ValueError: If the schema is invalid or missing required keys.
    """
    schema = validate_schema(schema_path)
    if not schema:
        raise ValueError("Invalid or missing schema file.")  # Raise ValueError
    if 'functions' not in schema:
        raise ValueError("Schema missing 'functions' key.")  # Raise ValueError
    return schema

def load_config(config_path: str, excluded_dirs: Set[str], excluded_files: Set[str], skip_types: Set[str]) -> Tuple[str, str]:
    """
    Loads additional configurations from a config.json file.

    Args:
        config_path (str): Path to the config.json file.
        excluded_dirs (Set[str]): Set to update with excluded directories.
        excluded_files (Set[str]): Set to update with excluded files.
        skip_types (Set[str]): Set to update with file types to skip.

    Returns:
        Tuple[str, str]: Project information and style guidelines.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        project_info = config.get("project_info", "")
        style_guidelines = config.get("style_guidelines", "")
        excluded_dirs.update(config.get("excluded_dirs", []))
        excluded_files.update(config.get("excluded_files", []))
        skip_types.update(config.get("skip_types", []))
        logger.debug(f"Loaded configuration from '{config_path}'.")
        return project_info, style_guidelines
    except FileNotFoundError:
        logger.error(f"Config file '{config_path}' not found.")
        return "", ""
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{config_path}': {e}")
        return "", ""
    except Exception as e:
        logger.error(f"Unexpected error loading config file '{config_path}': {e}")
        return "", ""

# ----------------------------
# Code Formatting and Cleanup
# ----------------------------

async def clean_unused_imports_async(code: str, file_path: str) -> str:
    """
    Asynchronously removes unused imports and variables from the provided code using autoflake.

    Args:
        code (str): The source code to clean.
        file_path (str): The file path used for display purposes in autoflake.

    Returns:
        str: The cleaned code with unused imports and variables removed.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            'autoflake', '--remove-all-unused-imports', '--remove-unused-variables', '--stdin-display-name', file_path, '-',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=code.encode())
        
        if process.returncode != 0:
            logger.error(f"Autoflake failed:\n{stderr.decode()}")
            return code
        
        return stdout.decode()
    except Exception as e:
        logger.error(f'Error running Autoflake: {e}')
        return code

async def format_with_black_async(code: str) -> str:
    """
    Asynchronously formats the provided code using Black.

    Args:
        code (str): The source code to format.

    Returns:
        str: The formatted code.
    """
    process = await asyncio.create_subprocess_exec(
        'black', '--quiet', '-',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate(input=code.encode())
    if process.returncode == 0:
        return stdout.decode()
    else:
        logger.error(f"Black formatting failed: {stderr.decode()}")
        return code

async def run_flake8_async(file_path: str) -> Optional[str]:
    """
    Asynchronously runs Flake8 on the specified file to check for style violations.

    Args:
        file_path (str): The path to the file to be checked.

    Returns:
        Optional[str]: The output from Flake8 if there are violations, otherwise None.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            'flake8', file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            return stdout.decode() + stderr.decode()
        
        return None
    except Exception as e:
        logger.error(f'Error running Flake8: {e}')
        return None

# ----------------------------
# JavaScript/TypeScript Utilities
# ----------------------------

async def run_node_script_async(script_path: str, input_json: str) -> Optional[str]:
    """
    Runs a Node.js script asynchronously.

    Args:
        script_path (str): Path to the Node.js script.
        input_json (str): JSON string to pass as input to the script.

    Returns:
        Optional[str]: The output from the script if successful, None otherwise.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            'node', script_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate(input=input_json.encode())

        if proc.returncode != 0:
            logger.error(f"Node.js script '{script_path}' failed: {stderr.decode()}")
            return None

        return stdout.decode()
    except FileNotFoundError:
        logger.error("Node.js is not installed or not in PATH. Please install Node.js.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while running Node.js script '{script_path}': {e}")
        return None
        
def run_node_insert_docstrings(script_name: str, input_data: dict) -> Optional[str]:
    """
    Runs a Node.js script to insert docstrings and returns the modified code.

    Args:
        script_name (str): Name of the script to run.
        input_data (dict): Input data to pass to the script.

    Returns:
        Optional[str]: The modified code if successful, None otherwise.
    """
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'scripts', script_name)
        logger.debug(f"Running Node.js script: {script_path}")

        input_json = json.dumps(input_data)
        result = subprocess.run(
            ["node", script_path],
            input=input_json,
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"Successfully ran {script_path}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error(f"Node.js script {script_name} not found.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error running {script_name}: {e}")
        return None

# ----------------------------
# Documentation Generation
# ----------------------------

def validate_schema(schema_path: str) -> Optional[Dict[str, Any]]:
    """
    Validates a JSON schema and loads it if valid.

    Args:
        schema_path (str): Path to the JSON schema file.

    Returns:
        Optional[Dict[str, Any]]: The loaded schema if valid, None otherwise.
    """
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        try:
            Draft7Validator.check_schema(schema)
            logger.debug("Schema is valid.")
            return schema
        except (SchemaError, ValidationError) as e:
            logger.error(f"Invalid schema: {e}")
            return None
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading schema file: {e}")
        return None

def load_function_schema(schema_path: str) -> Dict[str, Any]:
    """
    Loads and validates the function schema.

    Args:
        schema_path (str): Path to the function schema file.

    Returns:
        Dict[str, Any]: The loaded and validated function schema.

    Raises:
        ValueError: If the schema is invalid or missing required keys.
    """
    try:
        schema = validate_schema(schema_path)
        if schema is None:
            raise ValueError("Schema validation failed. Check the schema file.")
        if "functions" not in schema:
            raise ValueError("Schema missing 'functions' key.")
        return schema
    except ValueError as e:
        logger.critical(f"Error loading schema: {e}")
        sys.exit(1)

# ----------------------------
# EOF
# ----------------------------
```

## context_manager.py

```python
"""
context_manager.py

This module defines the ContextManager class, which manages persistent context information for AI interactions. It uses a deque to store context entries and a threading lock to ensure thread-safe operations.
"""

import threading
from collections import deque
from typing import List

class ContextManager:
    """
    Manages persistent context information for AI interactions.
    """

    def __init__(self, max_entries: int = 100):
        """
        Initializes the ContextManager with a maximum number of entries it can hold.

        Args:
            max_entries (int): The maximum number of context entries allowed.
        """
        self.max_entries = max_entries
        self.context_entries = deque(maxlen=self.max_entries)  # Use deque for efficient appends and pops
        self.lock = threading.Lock()  # Ensure thread-safe operations

    def add_context(self, context_entry: str):
        """
        Adds a new context entry to the context manager.

        Args:
            context_entry (str): The context information to add.
        """
        with self.lock:  # Lock to ensure thread-safe access
            self.context_entries.append(context_entry)

    def get_context(self) -> List[str]:
        """
        Retrieves all current context entries.

        Returns:
            List[str]: A list of context entries.
        """
        with self.lock:  # Lock to ensure thread-safe access
            return list(self.context_entries)

    def clear_context(self):
        """
        Clears all context entries.
        """
        with self.lock:  # Lock to ensure thread-safe access
            self.context_entries.clear()

    def remove_context(self, context_reference: str):
        """
        Removes context entries that contain the specified reference.

        Args:
            context_reference (str): Reference string to identify context entries to remove.
        """
        with self.lock:  # Lock to ensure thread-safe access
            self.context_entries = deque(
                [entry for entry in self.context_entries if context_reference not in entry],
                maxlen=self.max_entries
            )

    def get_relevant_context(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieves the most relevant context entries based on the query.

        Args:
            query (str): The query to match against context entries.
            top_k (int): Number of top relevant entries to return.

        Returns:
            List[str]: A list of relevant context entries.
        """
        with self.lock:  # Lock to ensure thread-safe access
            matched_entries = [entry for entry in self.context_entries if query.lower() in entry.lower()]
            return matched_entries[:top_k]
```

## write_documentation_report.py

```python
"""
write_documentation_report.py

This module provides functions for generating documentation reports in Markdown format. It includes utilities for creating badges, formatting tables, generating summaries, and writing the final documentation report to a file.
"""

import aiofiles
import re
import json
import os
import textwrap
import logging
import sys
from typing import Optional, Dict, Any, List

# Import default thresholds and get_threshold function from utils
from utils import (
    DEFAULT_COMPLEXITY_THRESHOLDS,
    DEFAULT_HALSTEAD_THRESHOLDS,
    DEFAULT_MAINTAINABILITY_THRESHOLDS,
    get_threshold  # Import get_threshold function
)

logger = logging.getLogger(__name__)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

async def save_documentation_to_db(
    documentation: Dict[str, Any],
    project_id: str,
    file_path: str,
    api_url: str
) -> Optional[str]:
    """
    Saves the generated documentation to the MongoDB database through the API.

    Args:
        documentation (Dict[str, Any]): The documentation to save
        project_id (str): The project identifier
        file_path (str): Path to the source file
        api_url (str): The API endpoint URL

    Returns:
        Optional[str]: The ID of the created document, or None if the operation failed
    """
    try:
        # Prepare the documentation data
        doc_data = {
            "project_id": project_id,
            "file_path": file_path,
            "version": os.getenv("PROJECT_VERSION", "1.0.0"),
            "language": documentation.get("language", "unknown"),
            "summary": documentation.get("summary", ""),
            "classes": documentation.get("classes", []),
            "functions": documentation.get("functions", []),
            "metrics": documentation.get("metrics", {}),
            "generated_by": "documentation-generator"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{api_url}/documentation",
                json=doc_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    logger.info(f"Documentation saved to database for {file_path}")
                    return result.get("_id")
                else:
                    error_data = await response.json()
                    logger.error(f"Failed to save documentation: {error_data}")
                    return None

    except Exception as e:
        logger.error(f"Error saving documentation to database: {e}", exc_info=True)
        return None

async def update_documentation_in_db(
    documentation: Dict[str, Any],
    project_id: str,
    file_path: str,
    api_url: str
) -> bool:
    """
    Updates existing documentation in the MongoDB database.

    Args:
        documentation (Dict[str, Any]): The updated documentation
        project_id (str): The project identifier
        file_path (str): Path to the source file
        api_url (str): The API endpoint URL

    Returns:
        bool: True if the update was successful, False otherwise
    """
    try:
        doc_data = {
            "project_id": project_id,
            "file_path": file_path,
            "version": os.getenv("PROJECT_VERSION", "1.0.0"),
            "language": documentation.get("language", "unknown"),
            "summary": documentation.get("summary", ""),
            "classes": documentation.get("classes", []),
            "functions": documentation.get("functions", []),
            "metrics": documentation.get("metrics", {}),
            "generated_by": "documentation-generator"
        }

        async with aiohttp.ClientSession() as session:
            async with session.put(
                f"{api_url}/documentation/{project_id}/{file_path}",
                json=doc_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    logger.info(f"Documentation updated in database for {file_path}")
                    return True
                else:
                    error_data = await response.json()
                    logger.error(f"Failed to update documentation: {error_data}")
                    return False

    except Exception as e:
        logger.error(f"Error updating documentation in database: {e}", exc_info=True)
        return False
    
def generate_badge(metric_name: str, value: float, thresholds: Dict[str, int], logo: str = None) -> str:
    """
    Generates a dynamic badge for a given metric.

    Args:
        metric_name (str): The name of the metric.
        value (float): The metric value.
        thresholds (Dict[str, int]): The thresholds for determining badge color.
        logo (str, optional): The logo to include in the badge. Defaults to None.

    Returns:
        str: The Markdown string for the badge.
    """
    low, medium, high = thresholds["low"], thresholds["medium"], thresholds["high"]
    color = "green" if value <= low else "yellow" if value <= medium else "red"
    badge_label = metric_name.replace("_", " ").title()
    logo_part = f"&logo={logo}" if logo else ""
    return f"![{badge_label}](https://img.shields.io/badge/{badge_label}-{value:.2f}-{color}?style=flat-square{logo_part})"


def generate_all_badges(
    complexity: Optional[int] = None,
    halstead: Optional[dict] = None,
    mi: Optional[float] = None
) -> str:
    """
    Generates badges for all metrics, using dynamic thresholds.

    Args:
        complexity (Optional[int]): The complexity value.
        halstead (Optional[dict]): The Halstead metrics.
        mi (Optional[float]): The maintainability index.

    Returns:
        str: A string containing all generated badges.
    """
    # Fetch thresholds using get_threshold function
    thresholds = {
        "complexity": {
            "low": get_threshold("complexity", "low", DEFAULT_COMPLEXITY_THRESHOLDS["low"]),
            "medium": get_threshold("complexity", "medium", DEFAULT_COMPLEXITY_THRESHOLDS["medium"]),
            "high": get_threshold("complexity", "high", DEFAULT_COMPLEXITY_THRESHOLDS["high"]),
        },
        "halstead_volume": {
            "low": get_threshold("halstead_volume", "low", DEFAULT_HALSTEAD_THRESHOLDS["volume"]["low"]),
            "medium": get_threshold("halstead_volume", "medium", DEFAULT_HALSTEAD_THRESHOLDS["volume"]["medium"]),
            "high": get_threshold("halstead_volume", "high", DEFAULT_HALSTEAD_THRESHOLDS["volume"]["high"]),
        },
        "halstead_difficulty": {
            "low": get_threshold("halstead_difficulty", "low", DEFAULT_HALSTEAD_THRESHOLDS["difficulty"]["low"]),
            "medium": get_threshold("halstead_difficulty", "medium", DEFAULT_HALSTEAD_THRESHOLDS["difficulty"]["medium"]),
            "high": get_threshold("halstead_difficulty", "high", DEFAULT_HALSTEAD_THRESHOLDS["difficulty"]["high"]),
        },
        "halstead_effort": {
            "low": get_threshold("halstead_effort", "low", DEFAULT_HALSTEAD_THRESHOLDS["effort"]["low"]),
            "medium": get_threshold("halstead_effort", "medium", DEFAULT_HALSTEAD_THRESHOLDS["effort"]["medium"]),
            "high": get_threshold("halstead_effort", "high", DEFAULT_HALSTEAD_THRESHOLDS["effort"]["high"]),
        },
        "maintainability_index": {
            "low": get_threshold("maintainability_index", "low", DEFAULT_MAINTAINABILITY_THRESHOLDS["low"]),
            "medium": get_threshold("maintainability_index", "medium", DEFAULT_MAINTAINABILITY_THRESHOLDS["medium"]),
            "high": get_threshold("maintainability_index", "high", DEFAULT_MAINTAINABILITY_THRESHOLDS["high"]),
        },
    }

    badges = []

    if complexity is not None:
        badges.append(generate_badge("Complexity", complexity, thresholds["complexity"], logo="codeClimate"))

    if halstead:
        badges.append(generate_badge("Halstead Volume", halstead["volume"], thresholds["halstead_volume"], logo="stackOverflow"))
        badges.append(generate_badge("Halstead Difficulty", halstead["difficulty"], thresholds["halstead_difficulty"], logo="codewars"))
        badges.append(generate_badge("Halstead Effort", halstead["effort"], thresholds["halstead_effort"], logo="atlassian"))

    if mi is not None:
        badges.append(generate_badge("Maintainability Index", mi, thresholds["maintainability_index"], logo="codeclimate"))

    return " ".join(badges)


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """
    Formats a table in Markdown.

    Args:
        headers (List[str]): The table headers.
        rows (List[List[str]]): The table rows.

    Returns:
        str: The formatted Markdown table.
    """
    # Sanitize headers
    headers = [sanitize_text(header) for header in headers]

    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        # Sanitize each cell
        sanitized_row = [sanitize_text(cell) for cell in row]
        table += "| " + " | ".join(sanitized_row) + " |\n"
    return table

def truncate_description(description: str, max_length: int = 100) -> str:
    """
    Truncates a description to a maximum length.

    Args:
        description (str): The description to truncate.
        max_length (int, optional): The maximum length. Defaults to 100.

    Returns:
        str: The truncated description.
    """
    return (description[:max_length] + '...') if len(description) > max_length else description


def sanitize_text(text: str) -> str:
    """
    Sanitizes text for Markdown by escaping special characters.

    Args:
        text (str): The text to sanitize.

    Returns:
        str: The sanitized text.
    """
    markdown_special_chars = ['\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '#', '+', '-', '.', '!', '|']
    for char in markdown_special_chars:
        text = text.replace(char, f"\\{char}")
    return text.replace('\n', ' ').strip()


def generate_table_of_contents(content: str) -> str:
    """
    Generates a table of contents from Markdown headers.

    Args:
        content (str): The Markdown content.

    Returns:
        str: The generated table of contents.
    """
    toc = []
    for line in content.splitlines():
        if line.startswith("#"):
            level = line.count("#")
            title = line.lstrip("#").strip()
            anchor = re.sub(r'[^a-zA-Z0-9\s-]', '', title)
            anchor = re.sub(r'\s+', '-', anchor).lower()
            anchor = re.sub(r'-+', '-', anchor).strip('-')
            toc.append(f"{'  ' * (level - 1)}- [{title}](#{anchor})")
    return "\n".join(toc)


def format_methods(methods: List[Dict[str, Any]]) -> str:
    """
    Formats method information into a Markdown table.

    Args:
        methods (List[Dict[str, Any]]): The methods to format.

    Returns:
        str: The formatted Markdown table.
    """
    headers = ["Method Name", "Complexity", "Async", "Docstring"]
    rows = [
        [
            method.get("name", "N/A"),
            str(method.get("complexity", 0)),
            str(method.get("async", False)),
            truncate_description(sanitize_text(method.get("docstring", "")))
        ]
        for method in methods
    ]
    return format_table(headers, rows)


def format_classes(classes: List[Dict[str, Any]]) -> str:
    """
    Formats class information into a Markdown table.

    Args:
        classes (List[Dict[str, Any]]): The classes to format.

    Returns:
        str: The formatted Markdown table.
    """
    headers = ["Class Name", "Docstring"]
    rows = [
        [
            cls.get("name", "N/A"),
            truncate_description(sanitize_text(cls.get("docstring", "")))
        ]
        for cls in classes
    ]
    class_table = format_table(headers, rows)
    
    method_tables = []
    for cls in classes:
        if cls.get("methods"):
            method_tables.append(f"#### Methods for {cls.get('name')}\n")
            method_tables.append(format_methods(cls.get("methods", [])))
    
    return class_table + "\n\n" + "\n".join(method_tables)


def format_functions(functions: List[Dict[str, Any]]) -> str:
    """
    Formats function information into a Markdown table.

    Args:
        functions (List[Dict[str, Any]]): The functions to format.

    Returns:
        str: The formatted Markdown table or a message if no functions are found.
    """
    if not functions:
        return "No functions found."

    headers = ["Function Name", "Complexity", "Async", "Docstring"]
    rows = [
        [
            func.get("name", "N/A"),
            str(func.get("complexity", 0)),
            str(func.get("async", False)),
            truncate_description(sanitize_text(func.get("docstring", "")))
        ]
        for func in functions
    ]
    return format_table(headers, rows)

def extract_module_docstring(file_path: str) -> str:
    """
    Extracts the module docstring from a Python file.

    Args:
        file_path (str): Path to the Python source file.

    Returns:
        str: The module docstring if found, otherwise an empty string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        module = ast.parse(source)
        return ast.get_docstring(module) or ''
    except Exception as e:
        logger.error(f"Error extracting module docstring from '{file_path}': {e}", exc_info=True)
        return ''
    
def generate_summary(variables: List[Dict[str, Any]], constants: List[Dict[str, Any]]) -> str:
    """
    Generates a summary with tooltips for variables and constants.

    Args:
        variables (List[Dict[str, Any]]): The variables to include in the summary.
        constants (List[Dict[str, Any]]): The constants to include in the summary.

    Returns:
        str: The generated summary.
    """
    total_vars = len(variables)
    total_consts = len(constants)

    # Create tooltip content for variables
    var_tooltip = ""
    if total_vars > 0:
        var_tooltip = "\n".join([f"- {sanitize_text(var.get('name', 'N/A'))}" for var in variables])
        total_vars_display = f'<span title="{var_tooltip}">{total_vars}</span>'
    else:
        total_vars_display = str(total_vars)

    # Create tooltip content for constants
    const_tooltip = ""
    if total_consts > 0:
        const_tooltip = "\n".join([f"- {sanitize_text(const.get('name', 'N/A'))}" for const in constants])
        total_consts_display = f'<span title="{const_tooltip}">{total_consts}</span>'
    else:
        total_consts_display = str(total_consts)

    summary = f"### **Summary**\n\n- **Total Variables:** {total_vars_display}\n- **Total Constants:** {total_consts_display}\n"
    return summary


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a filename by replacing invalid characters.

    Args:
        filename (str): The filename to sanitize.

    Returns:
        str: The sanitized filename.
    """
    # Replace invalid characters with underscores
    invalid_chars = r'<>:"/\\|?*'
    return ''.join('_' if c in invalid_chars else c for c in filename)

def generate_documentation_prompt(file_name, code_structure, project_info, style_guidelines, language, function_schema):
    """
    Generates a prompt for documentation generation that is compatible with the provided schema.

    Args:
        file_name (str): The name of the file.
        code_structure (dict): The code structure.
        project_info (str): Information about the project.
        style_guidelines (str): The style guidelines.
        language (str): The programming language.
        function_schema (dict): The function schema.

    Returns:
        str: The generated prompt.
    """
    docstring_format = function_schema["functions"][0]["parameters"]["properties"]["docstring_format"]["enum"][0]
    
    prompt = f"""
    You are a code documentation generator. Your task is to generate documentation for the following {language} file: '{file_name}'.
    
    Project Info:
    {project_info}

    Style Guidelines:
    {style_guidelines}

    Use the {docstring_format} style for docstrings.

    The documentation should strictly follow this schema:
    {{
        "docstring_format": "{docstring_format}",
        "summary": "A detailed summary of the file.",
        "changes_made": ["List of changes made"],
        "functions": [
            {{
                "name": "Function name",
                "docstring": "Detailed description in {docstring_format} style",
                "args": ["List of argument names"],
                "async": true/false,
                "complexity": integer
            }}
        ],
        "classes": [
            {{
                "name": "Class name",
                "docstring": "Detailed description in {docstring_format} style",
                "methods": [
                    {{
                        "name": "Method name",
                        "docstring": "Detailed description in {docstring_format} style",
                        "args": ["List of argument names"],
                        "async": true/false,
                        "type": "instance/class/static",
                        "complexity": integer
                    }}
                ]
            }}
        ],
        "halstead": {{
            "volume": number,
            "difficulty": number,
            "effort": number
        }},
        "maintainability_index": number,
        "variables": [
            {{
                "name": "Variable name",
                "type": "Inferred data type",
                "description": "Description of the variable",
                "file": "File name",
                "line": integer,
                "link": "Link to definition",
                "example": "Example usage",
                "references": "References to the variable"
            }}
        ],
        "constants": [
            {{
                "name": "Constant name",
                "type": "Inferred data type",
                "description": "Description of the constant",
                "file": "File name",
                "line": integer,
                "link": "Link to definition",
                "example": "Example usage",
                "references": "References to the constant"
            }}
        ]
    }}

    Ensure that all required fields are included and properly formatted according to the schema.

    Given the following code structure:
    {json.dumps(code_structure, indent=2)}

    Generate detailed documentation that strictly follows the provided schema. Include the following:
    1. A comprehensive summary of the file's purpose and functionality.
    2. A list of recent changes or modifications made to the file.
    3. Detailed documentation for all functions, including their arguments, return types, and whether they are asynchronous.
    4. Comprehensive documentation for all classes and their methods, including inheritance information if applicable.
    5. Information about all variables and constants, including their types, descriptions, and usage examples.
    6. Accurate Halstead metrics (volume, difficulty, and effort) for the entire file.
    7. The maintainability index of the file.

    Ensure that all docstrings follow the {docstring_format} format and provide clear, concise, and informative descriptions.
    """
    return textwrap.dedent(prompt).strip()

async def write_documentation_report(
    documentation: Optional[dict],
    language: str,
    file_path: str,
    repo_root: str,
    output_dir: str
) -> Optional[str]:
    """
    Generates enhanced documentation with improved formatting and structure.
    """
    if not documentation:
        logger.warning(f"No documentation to write for '{file_path}'")
        return None

    try:
        os.makedirs(output_dir, exist_ok=True)
        relative_path = os.path.relpath(file_path, repo_root)
        
        # Generate table of contents
        toc = [
            "# Table of Contents\n",
            "1. [Overview](#overview)",
            "2. [Code Structure](#code-structure)",
            "3. [Dependencies](#dependencies)",
            "4. [Metrics](#metrics)",
            "\n---\n"
        ]

        # Generate Overview section
        overview = [
            "# Overview\n",
            f"**File:** `{os.path.basename(file_path)}`  ",
            f"**Language:** {language}  ",
            f"**Path:** `{relative_path}`  \n",
            "## Summary\n",
            f"{documentation.get('summary', 'No summary available.')}\n",
            "## Recent Changes\n",
            "\n".join(f"- {change}" for change in documentation.get('changes_made', ['No recent changes.'])),
            "\n"
        ]

        # Generate Code Structure section
        code_structure = [
            "# Code Structure\n",
            "## Classes\n"
        ]

        for cls in documentation.get('classes', []):
            code_structure.extend([
                f"### {cls['name']}\n",
                f"{cls['docstring']}\n",
                "#### Methods\n",
                "| Method | Description | Complexity |",
                "|--------|-------------|------------|",
            ])
            for method in cls.get('methods', []):
                desc = method['docstring'].split('\n')[0]
                code_structure.append(
                    f"| `{method['name']}` | {desc} | {method.get('complexity', 'N/A')} |"
                )
            code_structure.append("\n")

        # Generate Dependencies section
        dependencies = [
            "# Dependencies\n",
            "```mermaid",
            "graph TD;",
        ]
        
        # Create dependency graph
        dep_map = {}
        for dep in documentation.get('variables', []) + documentation.get('constants', []):
            if dep.get('type') == 'import':
                dep_name = dep['name']
                dep_refs = dep.get('references', [])
                dep_map[dep_name] = dep_refs
                dependencies.append(f"    {dep_name}[{dep_name}];")
        
        for dep, refs in dep_map.items():
            for ref in refs:
                if ref in dep_map:
                    dependencies.append(f"    {dep} --> {ref};")
        
        dependencies.extend([
            "```\n"
        ])

        # Generate Metrics section
        metrics = [
            "# Metrics\n",
            "## Code Quality\n",
            "| Metric | Value | Status |",
            "|--------|-------|--------|",
            f"| Maintainability | {documentation.get('maintainability_index', 0):.1f} | {get_metric_status(documentation.get('maintainability_index', 0))} |",
        ]

        if 'halstead' in documentation:
            halstead = documentation['halstead']
            metrics.extend([
                "## Complexity Metrics\n",
                "| Metric | Value |",
                "|--------|--------|",
                f"| Volume | {halstead.get('volume', 0):.1f} |",
                f"| Difficulty | {halstead.get('difficulty', 0):.1f} |",
                f"| Effort | {halstead.get('effort', 0):.1f} |",
            ])

        # Combine all sections
        content = "\n".join([
            *toc,
            *overview,
            *code_structure,
            *dependencies,
            *metrics
        ])

        # Write the documentation file
        safe_filename = sanitize_filename(os.path.basename(file_path))
        output_path = os.path.join(output_dir, f"{safe_filename}.md")
        
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(content)

        logger.info(f"Documentation written to {output_path}")
        return content

    except Exception as e:
        logger.error(f"Error writing documentation report: {e}", exc_info=True)
        return None

def get_metric_status(value: float) -> str:
    """Returns a status indicator based on metric value."""
    if value >= 80:
        return "✅ Good"
    elif value >= 60:
        return "⚠️ Warning"
    return "❌ Needs Improvement"

def sanitize_filename(filename: str) -> str:
    """Sanitizes filename by removing invalid characters."""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)
```

## language_functions/language_functions.py

```python
"""
language_functions.py

This module provides utility functions for handling different programming languages within the documentation generation pipeline.
It includes functions to retrieve the appropriate language handler and to insert docstrings/comments into source code based on AI-generated documentation.

Functions:
    - get_handler(language, function_schema): Retrieves the appropriate handler for a given programming language.
    - insert_docstrings(original_code, documentation, language, schema_path): Inserts docstrings/comments into the source code using the specified language handler.
"""

import json
import logging
import subprocess
from typing import Dict, Any, Optional

from .base_handler import BaseHandler
from .python_handler import PythonHandler
from .java_handler import JavaHandler
from .js_ts_handler import JSTsHandler
from .go_handler import GoHandler
from .cpp_handler import CppHandler
from .html_handler import HTMLHandler
from .css_handler import CSSHandler
from utils import load_function_schema  # Import for dynamic schema loading

logger = logging.getLogger(__name__)


def get_handler(language: str, function_schema: Dict[str, Any]) -> Optional[BaseHandler]:
    """
    Factory function to retrieve the appropriate language handler.

    This function matches the provided programming language with its corresponding handler class.
    If the language is supported, it returns an instance of the handler initialized with the given function schema.
    If the language is unsupported, it logs a warning and returns None.

    Args:
        language (str): The programming language of the source code (e.g., "python", "java", "javascript").
        function_schema (Dict[str, Any]): The schema defining functions and their documentation structure.

    Returns:
        Optional[BaseHandler]: An instance of the corresponding language handler or None if unsupported.
    """
    if function_schema is None:
        logger.error("Function schema is None. Cannot retrieve handler.")
        return None

    language = language.lower()
    if language == "python":
        return PythonHandler(function_schema)
    elif language == "java":
        return JavaHandler(function_schema)
    elif language in ["javascript", "js", "typescript", "ts"]:
        return JSTsHandler(function_schema)
    elif language == "go":
        return GoHandler(function_schema)
    elif language in ["cpp", "c++", "cxx"]:
        return CppHandler(function_schema)
    elif language in ["html", "htm"]:
        return HTMLHandler(function_schema)
    elif language == "css":
        return CSSHandler(function_schema)
    else:
        logger.warning(f"No handler available for language: {language}")
        return None


def insert_docstrings(
    original_code: str, 
    documentation: Dict[str, Any], 
    language: str, 
    schema_path: str  # schema_path is now required
) -> str:
    """
    Inserts docstrings/comments into code based on the specified programming language.

    This function dynamically loads the function schema from a JSON file, retrieves the appropriate
    language handler, and uses it to insert documentation comments into the original source code.
    If any errors occur during schema loading or docstring insertion, the original code is returned.

    Args:
        original_code (str): The original source code to be documented.
        documentation (Dict[str, Any]): Documentation details obtained from AI, typically including descriptions of functions, classes, and methods.
        language (str): The programming language of the source code (e.g., "python", "java", "javascript").
        schema_path (str): Path to the function schema JSON file, which defines the structure and expected documentation format.

    Returns:
        str: The source code with inserted documentation comments, or the original code if errors occur.
    """
    logger.debug(f"Processing docstrings for language: {language}")

    try:
        # Load the function schema from the provided schema path
        function_schema = load_function_schema(schema_path)
    except (ValueError, FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.error(f"Error loading function schema: {e}")
        return original_code  # Return original code on schema loading error
    except Exception as e:  # Catch-all for other unexpected errors
        logger.error(f"An unexpected error occurred during schema loading: {e}", exc_info=True)
        return original_code

    # Retrieve the appropriate handler for the specified language
    handler = get_handler(language, function_schema)
    if not handler:
        logger.warning(f"Unsupported language '{language}'. Skipping docstring insertion.")
        return original_code

    if documentation is None:
        logger.error("Documentation is None. Skipping docstring insertion.")
        return original_code

    try:
        # Use the handler to insert docstrings/comments into the original code
        updated_code = handler.insert_docstrings(original_code, documentation)
        logger.debug("Docstring insertion completed successfully.")
        return updated_code
    except Exception as e:
        logger.error(f"Error inserting docstrings: {e}", exc_info=True)
        return original_code  # Return original code on docstring insertion error

```

## language_functions/css_handler.py

```python
"""
css_handler.py

This module provides the `CSSHandler` class, which is responsible for handling CSS code files.
It includes methods for extracting the code structure, inserting comments, and validating CSS code.
The handler uses external JavaScript scripts for parsing and modifying the code.

The `CSSHandler` class extends the `BaseHandler` abstract class.
"""

import os
import logging
import subprocess
import json
from typing import Dict, Any, Optional

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class CSSHandler(BaseHandler):
    """Handler for the CSS programming language."""

    def __init__(self, function_schema: Dict[str, Any]):
        """
        Initializes the `CSSHandler` with a function schema.

        Args:
            function_schema (Dict[str, Any]): The schema defining functions for documentation generation.
        """
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """
        Extracts the structure of the CSS code, analyzing selectors, properties, and rules.

        This method runs an external JavaScript parser script that processes the CSS code and outputs
        a JSON structure representing the code elements.

        Args:
            code (str): The source code to analyze.
            file_path (str, optional): The file path for code reference. Defaults to None.

        Returns:
            Dict[str, Any]: A detailed structure of the CSS components.
        """
        try:
            # Path to the CSS parser script
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "css_parser.js")
            # Prepare input data for the parser
            input_data = {"code": code, "language": "css"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running CSS parser script: {script_path}")

            # Run the parser script using Node.js
            result = subprocess.run(["node", script_path], input=input_json, capture_output=True, text=True, check=True)

            # Parse the output JSON structure
            structure = json.loads(result.stdout)
            logger.debug(f"Extracted CSS code structure successfully from file: {file_path}")
            return structure

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running css_parser.js for file {file_path}: {e.stderr}")
            return {}

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing output from css_parser.js for file {file_path}: {e}")
            return {}

        except Exception as e:
            logger.error(f"Unexpected error extracting CSS structure from file {file_path}: {e}")
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts comments into CSS code based on the provided documentation.

        This method runs an external JavaScript inserter script that processes the code and documentation
        to insert comments.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The CSS code with inserted documentation.
        """
        logger.debug("Inserting comments into CSS code.")
        try:
            # Path to the CSS inserter script
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "css_inserter.js")
            # Prepare input data for the inserter
            input_data = {"code": code, "documentation": documentation, "language": "css"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running CSS inserter script: {script_path}")

            # Run the inserter script using Node.js
            result = subprocess.run(["node", script_path], input=input_json, capture_output=True, text=True, check=True)

            modified_code = result.stdout
            logger.debug("Completed inserting comments into CSS code.")
            return modified_code

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running css_inserter.js: {e.stderr}")
            return code

        except Exception as e:
            logger.error(f"Unexpected error inserting comments into CSS code: {e}")
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates CSS code for correctness using 'stylelint'.

        Args:
            code (str): The CSS code to validate.
            file_path (Optional[str]): The path to the CSS source file.

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        logger.debug("Starting CSS code validation.")
        try:
            # Use stylelint to validate CSS code
            process = subprocess.run(["stylelint", "--stdin"], input=code, capture_output=True, text=True)

            # Check the result of the validation
            if process.returncode != 0:
                logger.error(f"stylelint validation failed:\n{process.stderr}")
                return False
            else:
                logger.debug("stylelint validation passed.")
            return True

        except FileNotFoundError:
            logger.error(
                "stylelint is not installed or not found in PATH. Please install it using 'npm install -g stylelint'."
            )
            return False

        except Exception as e:
            logger.error(f"Unexpected error during CSS code validation: {e}")
            return False

```

## language_functions/html_handler.py

```python
"""
html_handler.py

This module provides the `HTMLHandler` class, which is responsible for handling HTML code files.
It includes methods for extracting the code structure, inserting comments, and validating HTML code.
The handler uses external JavaScript scripts for parsing and modifying the code.

The `HTMLHandler` class extends the `BaseHandler` abstract class.
"""

import os
import logging
import subprocess
import json
from typing import Dict, Any, Optional

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class HTMLHandler(BaseHandler):
    """Handler for the HTML language."""

    def __init__(self, function_schema: Dict[str, Any]):
        """
        Initializes the HTMLHandler with a function schema.

        Args:
            function_schema (Dict[str, Any]): The schema used for function operations.
        """
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """
        Extracts the structure of the HTML code, analyzing tags, attributes, and nesting.

        This method runs an external JavaScript parser script that processes the HTML code and outputs
        a JSON structure representing the code elements.

        Args:
            code (str): The source code to analyze.
            file_path (str, optional): The file path for code reference.

        Returns:
            Dict[str, Any]: A detailed structure of the HTML components.
        """
        try:
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "html_parser.js")
            input_data = {"code": code, "language": "html"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running HTML parser script: {script_path}")

            result = subprocess.run(["node", script_path], input=input_json, capture_output=True, text=True, check=True)

            structure = json.loads(result.stdout)
            logger.debug(f"Extracted HTML code structure successfully from file: {file_path}")
            return structure

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running html_parser.js for file {file_path}: {e.stderr}")
            return {}

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing output from html_parser.js for file {file_path}: {e}")
            return {}

        except Exception as e:
            logger.error(f"Unexpected error extracting HTML structure from file {file_path}: {e}")
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts comments into HTML code based on the provided documentation.

        This method runs an external JavaScript inserter script that processes the code and documentation
        to insert comments.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The source code with inserted documentation.
        """
        logger.debug("Inserting comments into HTML code.")
        try:
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "html_inserter.js")
            input_data = {"code": code, "documentation": documentation, "language": "html"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running HTML inserter script: {script_path}")

            result = subprocess.run(["node", script_path], input=input_json, capture_output=True, text=True, check=True)

            modified_code = result.stdout
            logger.debug("Completed inserting comments into HTML code.")
            return modified_code

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running html_inserter.js: {e.stderr}")
            return code

        except Exception as e:
            logger.error(f"Unexpected error inserting comments into HTML code: {e}")
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates HTML code for correctness using an HTML validator like 'tidy'.

        Args:
            code (str): The HTML code to validate.
            file_path (Optional[str]): The path to the HTML source file.

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        logger.debug("Starting HTML code validation.")
        try:
            # Using 'tidy' for HTML validation
            process = subprocess.run(["tidy", "-errors", "-quiet", "-utf8"], input=code, capture_output=True, text=True)

            if process.returncode > 0:
                logger.error(f"HTML validation failed:\n{process.stderr}")
                return False
            else:
                logger.debug("HTML validation passed.")
            return True

        except FileNotFoundError:
            logger.error("tidy is not installed or not found in PATH. Please install it for HTML validation.")
            return False

        except Exception as e:
            logger.error(f"Unexpected error during HTML code validation: {e}")
            return False

```

## language_functions/__init__.py

```python
"""
language_functions Package

This package provides language-specific handlers for extracting code structures,
inserting documentation comments (docstrings), and validating code across various
programming languages. It includes handlers for languages such as Python, Java,
JavaScript/TypeScript, Go, C++, HTML, and CSS.

Modules:
    - python_handler.py: Handler for Python code.
    - java_handler.py: Handler for Java code.
    - js_ts_handler.py: Handler for JavaScript and TypeScript code.
    - go_handler.py: Handler for Go code.
    - cpp_handler.py: Handler for C++ code.
    - html_handler.py: Handler for HTML code.
    - css_handler.py: Handler for CSS code.
    - base_handler.py: Abstract base class defining the interface for all handlers.

Functions:
    - get_handler(language, function_schema): Factory function to retrieve the appropriate language handler.

Example:
    ```python
    from language_functions import get_handler
    from utils import load_function_schema

    function_schema = load_function_schema('path/to/schema.json')
    handler = get_handler('python', function_schema)
    if handler:
        updated_code = handler.insert_docstrings(original_code, documentation)
    ```
"""

import logging
from typing import Dict, Any, Optional

from .python_handler import PythonHandler
from .java_handler import JavaHandler
from .js_ts_handler import JSTsHandler
from .go_handler import GoHandler
from .cpp_handler import CppHandler
from .html_handler import HTMLHandler
from .css_handler import CSSHandler
from .base_handler import BaseHandler

logger = logging.getLogger(__name__)


def get_handler(language: str, function_schema: Dict[str, Any]) -> Optional[BaseHandler]:
    """
    Factory function to retrieve the appropriate language handler.

    Args:
        language (str): The programming language of the source code.
        function_schema (Dict[str, Any]): The schema defining functions.

    Returns:
        Optional[BaseHandler]: An instance of the corresponding language handler or None if unsupported.
    """
    if function_schema is None:
        logger.error("Function schema is None. Cannot retrieve handler.")
        return None

    # Normalize the language string to lowercase to ensure case-insensitive matching
    language = language.lower()
    
    # Map of supported languages to their handlers
    handlers = {
        "python": PythonHandler,
        "java": JavaHandler,
        "javascript": JSTsHandler,
        "js": JSTsHandler,
        "typescript": JSTsHandler,
        "ts": JSTsHandler,
        "go": GoHandler,
        "cpp": CppHandler,
        "c++": CppHandler,
        "cxx": CppHandler,
        "html": HTMLHandler,
        "htm": HTMLHandler,
        "css": CSSHandler
    }

    handler_class = handlers.get(language)
    if handler_class:
        return handler_class(function_schema)
    else:
        logger.debug(f"No handler available for language: {language}")
        return None  # Return None instead of raising an exception
```

## language_functions/cpp_handler.py

```python
"""
cpp_handler.py

This module provides the `CppHandler` class, which is responsible for handling C++ code files.
It includes methods for extracting the code structure, inserting docstrings/comments, and validating C++ code.
The handler uses external C++ scripts for parsing and modifying the code.

The `CppHandler` class extends the `BaseHandler` abstract class.
"""

import os
import logging
import subprocess
import json
from typing import Dict, Any, Optional

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class CppHandler(BaseHandler):
    """Handler for the C++ programming language."""

    def __init__(self, function_schema: Dict[str, Any]):
        """
        Initializes the `CppHandler` with a function schema.

        Args:
            function_schema (Dict[str, Any]): The schema defining functions for documentation generation.
        """
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """
        Extracts the structure of the C++ code, analyzing classes, functions, and variables.

        This method runs an external C++ parser executable that processes the code and outputs a JSON structure
        representing the code elements.

        Args:
            code (str): The source code to analyze.
            file_path (str, optional): The file path for code reference. Defaults to None.

        Returns:
            Dict[str, Any]: A detailed structure of the code components.
        """
        try:
            # Path to the C++ parser script
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "cpp_parser.cpp")
            # The executable path after compilation
            executable_path = os.path.splitext(script_path)[0]  # Remove .cpp extension

            # Compile the C++ parser if not already compiled
            if not os.path.exists(executable_path):
                logger.debug(f"Compiling C++ parser script: {script_path}")
                compile_process = subprocess.run(
                    ["g++", script_path, "-o", executable_path], capture_output=True, text=True, check=True
                )
                if compile_process.returncode != 0:
                    logger.error(f"Compilation of cpp_parser.cpp failed: {compile_process.stderr}")
                    return {}

            # Prepare input data for the parser
            input_data = {"code": code, "language": "cpp"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running C++ parser executable: {executable_path}")

            # Run the parser executable
            result = subprocess.run([executable_path], input=input_json, capture_output=True, text=True, check=True)

            # Parse the output JSON structure
            structure = json.loads(result.stdout)
            logger.debug(f"Extracted C++ code structure successfully from file: {file_path}")
            return structure

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running cpp_parser executable for file {file_path}: {e.stderr}")
            return {}

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing output from cpp_parser for file {file_path}: {e}")
            return {}

        except Exception as e:
            logger.error(f"Unexpected error extracting C++ structure from file {file_path}: {e}")
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts comments into C++ code based on the provided documentation.

        This method runs an external C++ inserter executable that processes the code and documentation
        to insert comments.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The source code with inserted documentation.
        """
        logger.debug("Inserting comments into C++ code.")
        try:
            # Path to the C++ inserter script
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "cpp_inserter.cpp")
            # The executable path after compilation
            executable_path = os.path.splitext(script_path)[0]  # Remove .cpp extension

            # Compile the C++ inserter if not already compiled
            if not os.path.exists(executable_path):
                logger.debug(f"Compiling C++ inserter script: {script_path}")
                compile_process = subprocess.run(
                    ["g++", script_path, "-o", executable_path], capture_output=True, text=True, check=True
                )
                if compile_process.returncode != 0:
                    logger.error(f"Compilation of cpp_inserter.cpp failed: {compile_process.stderr}")
                    return code

            # Prepare input data for the inserter
            input_data = {"code": code, "documentation": documentation, "language": "cpp"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running C++ inserter executable: {executable_path}")

            # Run the inserter executable
            result = subprocess.run([executable_path], input=input_json, capture_output=True, text=True, check=True)

            modified_code = result.stdout
            logger.debug("Completed inserting comments into C++ code.")
            return modified_code

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running cpp_inserter executable: {e.stderr}")
            return code

        except Exception as e:
            logger.error(f"Unexpected error inserting comments into C++ code: {e}")
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates C++ code for syntax correctness using 'g++' with the '-fsyntax-only' flag.

        Args:
            code (str): The C++ code to validate.
            file_path (Optional[str]): The path to the C++ source file. Required for validation.

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        logger.debug("Starting C++ code validation.")
        if not file_path:
            logger.warning("File path not provided for C++ validation. Skipping validation.")
            return True  # Assuming no validation without a file

        try:
            # Write code to a temporary file for validation
            temp_file = f"{file_path}.temp.cpp"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(code)
            logger.debug(f"Wrote temporary C++ file for validation: {temp_file}")

            # Run 'g++ -fsyntax-only' to check syntax
            process = subprocess.run(["g++", "-fsyntax-only", temp_file], capture_output=True, text=True)

            # Check the result of the syntax check
            if process.returncode != 0:
                logger.error(f"g++ syntax validation failed for {file_path}:\n{process.stderr}")
                return False
            else:
                logger.debug("g++ syntax validation passed.")

            # Remove the temporary file
            os.remove(temp_file)
            return True

        except FileNotFoundError:
            logger.error("g++ is not installed or not found in PATH. Please install a C++ compiler.")
            return False

        except Exception as e:
            logger.error(f"Unexpected error during C++ code validation: {e}")
            return False

```

## language_functions/python_handler.py

```python
"""
python_handler.py

This module defines the PythonHandler class, which is responsible for extracting code structures, inserting docstrings, and validating Python code. It utilizes the radon library for complexity metrics and libcst for code transformations.
"""

import logging
import os
import tempfile
import subprocess
import ast
from typing import Dict, Any, Optional, List, Union
# from metrics import calculate_all_metrics
# from ..metrics import calculate_all_metrics
from metrics import calculate_all_metrics
# External dependencies
try:
    from radon.complexity import cc_visit
    from radon.metrics import h_visit, mi_visit
except ImportError:
    logging.error("radon is not installed. Please install it using 'pip install radon'.")
    raise

try:
    import libcst as cst
    from libcst import FunctionDef, ClassDef, SimpleStatementLine, Expr, SimpleString
except ImportError:
    logging.error("libcst is not installed. Please install it using 'pip install libcst'.")
    raise

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class PythonHandler(BaseHandler):
    """Handler for Python language."""

    def __init__(self, function_schema: Dict[str, Any]):
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str) -> Dict[str, Any]:
        """
        Extracts the structure of the Python code, analyzing functions, classes, and assignments.

        Args:
            code (str): The source code to analyze.
            file_path (str): The file path for code reference.

        Returns:
            Dict[str, Any]: A detailed structure of the code components.
        """
        try:
            tree = ast.parse(code)
            code_structure = {
                "classes": [],
                "functions": [],
                "variables": [],
                "constants": [],
                "decorators": [],
                "context_managers": [],
                "comprehensions": [],
            }   

            # Calculate all metrics
            metrics = calculate_all_metrics(code)
        
            # Add metrics to code structure
            code_structure.update({
                "halstead": metrics["halstead"],
                "complexity": metrics["complexity"],
                "maintainability_index": metrics["maintainability_index"]
            })
        
            # Store function complexity for use in visitor
            function_complexity = metrics["function_complexity"]

            class CodeVisitor(ast.NodeVisitor):
                """AST visitor for traversing Python code structures and extracting functional and class definitions."""

                def __init__(self, file_path: str):
                    """Initializes the CodeVisitor for traversing AST nodes."""
                    self.scope_stack = []
                    self.file_path = file_path
                    self.comments = self._extract_comments(code, tree)

                def _extract_comments(self, code: str, tree: ast.AST) -> Dict[int, List[str]]:
                    comments = {}
                    for lineno, line in enumerate(code.splitlines(), start=1):
                        stripped = line.strip()
                        if stripped.startswith("#"):
                            comment = stripped.lstrip("#").strip()
                            comments.setdefault(lineno, []).append(comment)
                    return comments

                def _get_method_type(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
                    """
                    Determines the type of method based on decorators and context.

                    Args:
                        node: The AST node for the method.

                    Returns:
                        str: The method type (instance, class, static, or async).
                    """
                    if isinstance(node, ast.AsyncFunctionDef):
                        return "async"

                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name):
                            if decorator.id == "classmethod":
                                return "class"
                            elif decorator.id == "staticmethod":
                                return "static"
                        elif isinstance(decorator, ast.Attribute):
                            # Handle cases like @decorators.classmethod
                            if decorator.attr in ["classmethod", "staticmethod"]:
                                return decorator.attr
                    return "instance"

                def visit_FunctionDef(self, node: ast.FunctionDef):
                    self._visit_function(node)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                    self._visit_function(node, is_async=True)

                def _visit_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool = False) -> None:
                    self.scope_stack.append(node)
                    full_name = ".".join([scope.name for scope in self.scope_stack if hasattr(scope, "name")])
                    complexity = function_complexity.get(full_name, 0)
                    decorators = [ast.unparse(d) for d in node.decorator_list] if hasattr(ast, "unparse") else []
                    docstring = ast.get_docstring(node) or ""
                    function_info = {
                        "name": node.name,
                        "docstring": docstring,
                        "args": [arg.arg for arg in node.args.args if arg.arg != "self"],
                        "async": is_async,
                        "complexity": complexity,
                        "decorators": decorators,
                    }
                    if not any(isinstance(parent, ast.ClassDef) for parent in self.scope_stack[:-1]):
                        code_structure["functions"].append(function_info)
                    self.generic_visit(node)
                    self.scope_stack.pop()

                def visit_ClassDef(self, node: ast.ClassDef):
                    self.scope_stack.append(node)
                    class_docstring = ast.get_docstring(node) or ""
                    class_info = {
                        "name": node.name,
                        "docstring": class_docstring,
                        "methods": [],
                        "decorators": [ast.unparse(d) for d in node.decorator_list] if hasattr(ast, "unparse") else [],
                    }
                    for body_item in node.body:
                        if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self.scope_stack.append(body_item)
                            full_method_name = ".".join(
                                [scope.name for scope in self.scope_stack if hasattr(scope, "name")]
                            )
                            complexity = function_complexity.get(full_method_name, 0)
                            decorators = (
                                [ast.unparse(d) for d in body_item.decorator_list] if hasattr(ast, "unparse") else []
                            )
                            method_docstring = ast.get_docstring(body_item) or ""
                            method_info = {
                                "name": body_item.name,
                                "docstring": method_docstring,
                                "args": [arg.arg for arg in body_item.args.args if arg.arg != "self"],
                                "async": isinstance(body_item, ast.AsyncFunctionDef),
                                "complexity": complexity,
                                "decorators": decorators,
                                "type": self._get_method_type(body_item),
                            }
                            class_info["methods"].append(method_info)
                            self.scope_stack.pop()
                    code_structure["classes"].append(class_info)
                    self.generic_visit(node)
                    self.scope_stack.pop()

                def visit_Assign(self, node: ast.Assign):
                    for target in node.targets:
                        self._process_target(target, node.value)
                    self.generic_visit(node)

                def _process_target(self, target: ast.AST, value: ast.AST) -> None:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        is_constant = var_name.isupper()
                        var_type = self._infer_type(value)
                        description = self._extract_description(target.lineno)
                        example = self._extract_example(target.lineno)
                        references = self._extract_references(target.lineno)
                        var_info = {
                            "name": var_name,
                            "type": var_type,
                            "description": description,
                            "file": os.path.basename(self.file_path),
                            "line": target.lineno,
                            "link": f"https://github.com/user/repo/blob/main/{self.file_path}#L{target.lineno}",
                            "example": example,
                            "references": references,
                        }
                        if is_constant:
                            code_structure["constants"].append(var_info)
                        else:
                            code_structure["variables"].append(var_info)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for elt in target.elts:
                            self._process_target(elt, value)
                    elif isinstance(target, ast.Attribute):
                        var_name = target.attr
                        is_constant = var_name.isupper()
                        var_type = self._infer_type(value)
                        description = self._extract_description(target.lineno)
                        example = self._extract_example(target.lineno)
                        references = self._extract_references(target.lineno)
                        var_info = {
                            "name": var_name,
                            "type": var_type,
                            "description": description,
                            "file": os.path.basename(self.file_path),
                            "line": target.lineno,
                            "link": f"https://github.com/user/repo/blob/main/{self.file_path}#L{target.lineno}",
                            "example": example,
                            "references": references,
                        }
                        if is_constant:
                            code_structure["constants"].append(var_info)
                        else:
                            code_structure["variables"].append(var_info)

                def _infer_type(self, value: ast.AST) -> str:
                    if isinstance(value, ast.Constant):
                        return type(value.value).__name__
                    elif isinstance(value, ast.List):
                        return "List"
                    elif isinstance(value, ast.Tuple):
                        return "Tuple"
                    elif isinstance(value, ast.Dict):
                        return "Dict"
                    elif isinstance(value, ast.Set):
                        return "Set"
                    elif isinstance(value, ast.Call):
                        return "Call"
                    elif isinstance(value, ast.BinOp):
                        return "BinOp"
                    elif isinstance(value, ast.UnaryOp):
                        return "UnaryOp"
                    elif isinstance(value, ast.Lambda):
                        return "Lambda"
                    elif isinstance(value, ast.Name):
                        return "Name"
                    else:
                        return "Unknown"

                def _extract_description(self, lineno: int) -> str:
                    comments = self.comments.get(lineno - 1, []) + self.comments.get(lineno, [])
                    if comments:
                        return " ".join(comments)
                    return "No description provided."

                def _extract_example(self, lineno: int) -> str:
                    comments = self.comments.get(lineno + 1, [])
                    if comments:
                        return " ".join(comments)
                    return "No example provided."

                def _extract_references(self, lineno: int) -> str:
                    comments = self.comments.get(lineno + 2, [])
                    if comments:
                        return " ".join(comments)
                    return "N/A"

                def visit_With(self, node: ast.With):
                    for item in node.items:
                        if isinstance(item.context_expr, ast.Call):
                            context_manager = ast.unparse(item.context_expr) if hasattr(ast, "unparse") else ""
                            code_structure.setdefault("context_managers", []).append(context_manager)
                    self.generic_visit(node)

                def visit_AsyncWith(self, node: ast.AsyncWith):
                    for item in node.items:
                        if isinstance(item.context_expr, ast.Call):
                            context_manager = ast.unparse(item.context_expr) if hasattr(ast, "unparse") else ""
                            code_structure.setdefault("context_managers", []).append(context_manager)
                    self.generic_visit(node)

                def visit_ListComp(self, node: ast.ListComp):
                    code_structure.setdefault("comprehensions", []).append("ListComprehension")
                    self.generic_visit(node)

                def visit_DictComp(self, node: ast.DictComp):
                    code_structure.setdefault("comprehensions", []).append("DictComprehension")
                    self.generic_visit(node)

                def visit_SetComp(self, node: ast.SetComp):
                    code_structure.setdefault("comprehensions", []).append("SetComprehension")
                    self.generic_visit(node)

                def visit_GeneratorExp(self, node: ast.GeneratorExp):
                    code_structure.setdefault("comprehensions", []).append("GeneratorExpression")
                    self.generic_visit(node)

            visitor = CodeVisitor(file_path)
            visitor.visit(tree)
            logger.debug(f"Extracted structure for '{file_path}': {code_structure}")
            return code_structure

        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e.text.strip()} at line {e.lineno}, offset {e.offset}")
            return {}
        except Exception as e:
            logger.error(f"Error extracting Python structure: {e}", exc_info=True)
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts docstrings into the Python code based on the provided documentation.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The source code with inserted documentation.
        """
        logger.debug("Starting docstring insertion for Python code (Google Style).")
        try:
            docstrings_mapping = {}
            for func_doc in documentation.get("functions", []):
                name = func_doc.get("name")
                if name:
                    docstrings_mapping[name] = self._format_google_docstring(func_doc)
            for class_doc in documentation.get("classes", []):
                class_name = class_doc.get("name")
                if class_name:
                    docstrings_mapping[class_name] = self._format_google_docstring(class_doc)
                    for method_doc in class_doc.get("methods", []):
                        method_name = method_doc.get("name")
                        if method_name:
                            full_method_name = f"{class_name}.{method_name}"
                            docstrings_mapping[full_method_name] = self._format_google_docstring(method_doc)

            class DocstringInserter(cst.CSTTransformer):
                def __init__(self, docstrings_mapping: Dict[str, str]):
                    self.docstrings_mapping = docstrings_mapping
                    self.scope_stack = []

                def visit_FunctionDef(self, node: FunctionDef):
                    self.scope_stack.append(node.name.value)

                def leave_FunctionDef(self, original_node: FunctionDef, updated_node: FunctionDef) -> FunctionDef:
                    full_name = ".".join(self.scope_stack)
                    docstring = self.docstrings_mapping.get(full_name)
                    if docstring and not original_node.get_docstring():
                        new_doc = SimpleStatementLine([Expr(SimpleString(f'"""{docstring}"""'))])
                        new_body = [new_doc] + list(updated_node.body.body)
                        updated_node = updated_node.with_changes(body=updated_node.body.with_changes(body=new_body))
                        logger.debug(f"Inserted docstring for function: {full_name}")
                    self.scope_stack.pop()
                    return updated_node

                def visit_ClassDef(self, node: ClassDef):
                    self.scope_stack.append(node.name.value)

                def leave_ClassDef(self, original_node: ClassDef, updated_node: ClassDef) -> ClassDef:
                    full_name = ".".join(self.scope_stack)
                    docstring = self.docstrings_mapping.get(full_name)
                    if docstring and not original_node.get_docstring():
                        new_doc = SimpleStatementLine([Expr(SimpleString(f'"""{docstring}"""'))])
                        new_body = [new_doc] + list(updated_node.body.body)
                        updated_node = updated_node.with_changes(body=updated_node.body.with_changes(body=new_body))
                        logger.debug(f"Inserted docstring for class: {full_name}")
                    self.scope_stack.pop()
                    return updated_node

            tree = cst.parse_module(code)
            inserter = DocstringInserter(docstrings_mapping)
            modified_tree = tree.visit(inserter)
            modified_code = modified_tree.code
            logger.debug("Docstring insertion completed successfully.")
            return modified_code
        except Exception as e:
            logger.error(f"Error inserting docstrings: {e}", exc_info=True)
            return code

    def _format_google_docstring(self, doc: Dict[str, Any]) -> str:
        """
        Formats a docstring in Google style.

        Args:
            doc (Dict[str, Any]): The documentation details.

        Returns:
            str: The formatted docstring.
        """
        docstring = f'{doc.get("docstring", "")}\n\n'

        arguments = doc.get("arguments", [])
        if arguments:
            docstring += "Args:\n"
            for arg in arguments:
                arg_name = arg.get("name", "unknown")
                arg_type = arg.get("type", "Any")
                arg_description = arg.get("description", "")
                default_value = arg.get("default_value")

                docstring += f"    {arg_name} ({arg_type}): {arg_description}"
                if default_value is not None:
                    docstring += f" (Default: {default_value})"
                docstring += "\n"

        return_type = doc.get("return_type")
        return_description = doc.get("return_description", "")
        if return_type:
            docstring += f"\nReturns:\n    {return_type}: {return_description}\n"

        return docstring.strip()

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates the modified Python code for syntax correctness.

        Args:
            code (str): The modified source code.
            file_path (Optional[str]): Path to the Python source file (optional).

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        logger.debug("Starting Python code validation.")
        try:
            ast.parse(code)
            logger.debug("Syntax validation passed.")
            if file_path:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
                    tmp.write(code)
                    temp_file = tmp.name
                try:
                    result = subprocess.run(
                        ["flake8", temp_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    if result.returncode != 0:
                        logger.error(f"Flake8 validation failed for {file_path}:\n{result.stdout}\n{result.stderr}")
                        return False
                    else:
                        logger.debug("Flake8 validation passed.")
                except FileNotFoundError:
                    logger.error(
                        "flake8 is not installed or not found in PATH. Please install it using 'pip install flake8'."
                    )
                    return False
                except subprocess.SubprocessError as e:
                    logger.error(f"Subprocess error during flake8 execution: {e}")
                    return False
                finally:
                    try:
                        os.remove(temp_file)
                    except OSError as e:
                        logger.error(f"Error deleting temporary file {temp_file}: {e}")
            else:
                logger.warning("File path not provided for flake8 validation. Skipping flake8.")
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error during validation: {e.text.strip()} at line {e.lineno}, offset {e.offset}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during code validation: {e}", exc_info=True)
            return False

```

## language_functions/java_handler.py

```python
"""
java_handler.py

This module provides the `JavaHandler` class, which is responsible for handling Java code files.
It includes methods for extracting the code structure, inserting Javadoc comments, and validating Java code.
The handler uses external JavaScript scripts for parsing and modifying the code.

The `JavaHandler` class extends the `BaseHandler` abstract class.
"""

import os
import logging
import subprocess
import json
from typing import Dict, Any, Optional

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class JavaHandler(BaseHandler):
    """Handler for the Java language."""

    def __init__(self, function_schema: Dict[str, Any]):
        """
        Initializes the JavaHandler with a function schema.

        Args:
            function_schema (Dict[str, Any]): The schema used for function operations.
        """
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """
        Extracts the structure of the Java code, analyzing classes, methods, and fields.

        This method runs an external JavaScript parser script that processes the Java code and outputs
        a JSON structure representing the code elements.

        Args:
            code (str): The source code to analyze.
            file_path (str, optional): The file path for code reference.

        Returns:
            Dict[str, Any]: A detailed structure of the code components.
        """
        try:
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "java_parser.js")
            input_data = {"code": code, "language": "java"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running Java parser script: {script_path}")

            result = subprocess.run(["node", script_path], input=input_json, capture_output=True, text=True, check=True)

            structure = json.loads(result.stdout)
            logger.debug(f"Extracted Java code structure successfully from file: {file_path}")
            return structure

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running java_parser.js for file {file_path}: {e.stderr}")
            return {}

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing output from java_parser.js for file {file_path}: {e}")
            return {}

        except Exception as e:
            logger.error(f"Unexpected error extracting Java structure from file {file_path}: {e}")
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts Javadoc comments into Java code based on the provided documentation.

        This method runs an external JavaScript inserter script that processes the code and documentation
        to insert comments.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The source code with inserted documentation.
        """
        logger.debug("Inserting Javadoc docstrings into Java code.")
        try:
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "java_inserter.js")
            input_data = {"code": code, "documentation": documentation, "language": "java"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running Java inserter script: {script_path}")

            result = subprocess.run(["node", script_path], input=input_json, capture_output=True, text=True, check=True)

            modified_code = result.stdout
            logger.debug("Completed inserting Javadoc docstrings into Java code.")
            return modified_code

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running java_inserter.js: {e.stderr}")
            return code

        except Exception as e:
            logger.error(f"Unexpected error inserting Javadoc docstrings: {e}")
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates Java code for syntax correctness using javac.

        Args:
            code (str): The Java code to validate.
            file_path (Optional[str]): The path to the Java source file.

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        logger.debug("Starting Java code validation.")
        if not file_path:
            logger.warning("File path not provided for javac validation. Skipping javac.")
            return True  # Assuming no validation without a file

        try:
            # Write code to a temporary file for validation
            temp_file = f"{file_path}.temp.java"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(code)
            logger.debug(f"Wrote temporary Java file for validation: {temp_file}")

            # Compile the temporary Java file
            process = subprocess.run(["javac", temp_file], capture_output=True, text=True)

            # Remove the temporary file and class file if compilation was successful
            if process.returncode != 0:
                logger.error(f"javac validation failed for {file_path}:\n{process.stderr}")
                return False
            else:
                logger.debug("javac validation passed.")
                os.remove(temp_file)
                # Remove the generated class file
                class_file = temp_file.replace(".java", ".class")
                if os.path.exists(class_file):
                    os.remove(class_file)
            return True

        except FileNotFoundError:
            logger.error("javac is not installed or not found in PATH. Please install the JDK.")
            return False

        except Exception as e:
            logger.error(f"Unexpected error during Java code validation: {e}")
            return False

```

## language_functions/base_handler.py

```python
"""
base_handler.py

This module defines the abstract base class `BaseHandler` for language-specific handlers.
Each handler is responsible for extracting code structure, inserting docstrings/comments,
and validating code for a specific programming language.

Classes:
    - BaseHandler: Abstract base class defining the interface for all language handlers.
"""

import abc
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseHandler(abc.ABC):
    """
    Abstract base class for language-specific handlers.

    Each handler must implement methods to extract the structure of the code,
    insert docstrings/comments, and validate the modified code.
    """

    @abc.abstractmethod
    def extract_structure(self, code: str, file_path: str) -> Dict[str, Any]:
        """
        Extracts the structure of the code (classes, functions, etc.).

        This method should parse the source code and identify key components such as
        classes, functions, methods, variables, and other relevant elements.

        Args:
            code (str): The source code to analyze.
            file_path (str): Path to the source file.

        Returns:
            Dict[str, Any]: A dictionary representing the code structure, including details
                            like classes, functions, variables, and their attributes.
        """

    @abc.abstractmethod
    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts docstrings/comments into the code based on the documentation.

        This method should take the original source code and the generated documentation,
        then insert the appropriate docstrings or comments into the code at the correct locations.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The source code with inserted documentation.
        """

    @abc.abstractmethod
    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates the modified code for syntax correctness.

        This method should ensure that the code remains syntactically correct after
        inserting docstrings/comments. It may involve compiling the code or running
        language-specific linters/validators.

        Args:
            code (str): The modified source code.
            file_path (Optional[str]): Path to the source file (optional).

        Returns:
            bool: True if the code is valid, False otherwise.
        """

```

## language_functions/go_handler.py

```python
"""
go_handler.py

This module provides the `GoHandler` class, which is responsible for handling Go language code files.
It includes methods for extracting the code structure, inserting comments, and validating Go code.
The handler uses external Go scripts for parsing and modifying the code.

The `GoHandler` class extends the `BaseHandler` abstract class.
"""

import os
import logging
import subprocess
import json
from typing import Dict, Any, Optional

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class GoHandler(BaseHandler):
    """Handler for the Go programming language."""

    def __init__(self, function_schema: Dict[str, Any]):
        """
        Initializes the `GoHandler` with a function schema.

        Args:
            function_schema (Dict[str, Any]): The schema defining functions for documentation generation.
        """
        self.function_schema = function_schema

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """
        Extracts the structure of the Go code, analyzing functions, types, and variables.

        This method runs an external Go parser script that processes the code and outputs
        a JSON structure representing the code elements.

        Args:
            code (str): The source code to analyze.
            file_path (str, optional): The file path for code reference. Defaults to None.

        Returns:
            Dict[str, Any]: A detailed structure of the code components.
        """
        try:
            # Path to the Go parser script
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "go_parser.go")
            # Prepare input data for the parser
            input_data = {"code": code, "language": "go"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running Go parser script: {script_path}")

            # Run the parser script
            result = subprocess.run(
                ["go", "run", script_path], input=input_json, capture_output=True, text=True, check=True
            )

            # Parse the output JSON structure
            structure = json.loads(result.stdout)
            logger.debug(f"Extracted Go code structure successfully from file: {file_path}")
            return structure

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running go_parser.go for file {file_path}: {e.stderr}")
            return {}

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing output from go_parser.go for file {file_path}: {e}")
            return {}

        except Exception as e:
            logger.error(f"Unexpected error extracting Go structure from file {file_path}: {e}")
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts comments into Go code based on the provided documentation.

        This method runs an external Go inserter script that processes the code and documentation
        to insert comments.

        Args:
            code (str): The original source code.
            documentation (Dict[str, Any]): Documentation details obtained from AI.

        Returns:
            str: The Go code with inserted documentation.
        """
        logger.debug("Inserting comments into Go code.")
        try:
            # Path to the Go inserter script
            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "go_inserter.go")
            # Prepare input data for the inserter
            input_data = {"code": code, "documentation": documentation, "language": "go"}
            input_json = json.dumps(input_data)
            logger.debug(f"Running Go inserter script: {script_path}")

            # Run the inserter script
            result = subprocess.run(
                ["go", "run", script_path], input=input_json, capture_output=True, text=True, check=True
            )

            modified_code = result.stdout
            logger.debug("Completed inserting comments into Go code.")
            return modified_code

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running go_inserter.go: {e.stderr}")
            return code

        except Exception as e:
            logger.error(f"Unexpected error inserting comments into Go code: {e}")
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates Go code for syntax correctness using 'go fmt' and 'go vet'.

        Args:
            code (str): The Go code to validate.
            file_path (Optional[str]): The path to the Go source file.

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        logger.debug("Starting Go code validation.")
        if not file_path:
            logger.warning("File path not provided for Go validation. Skipping validation.")
            return True  # Assuming no validation without a file

        try:
            # Write code to a temporary file for validation
            temp_file = f"{file_path}.temp.go"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(code)
            logger.debug(f"Wrote temporary Go file for validation: {temp_file}")

            # Run 'go fmt' to format the code and check syntax
            fmt_process = subprocess.run(["go", "fmt", temp_file], capture_output=True, text=True)

            # Check the result of 'go fmt'
            if fmt_process.returncode != 0:
                logger.error(f"go fmt validation failed for {file_path}:\n{fmt_process.stderr}")
                return False
            else:
                logger.debug("go fmt validation passed.")

            # Run 'go vet' to check for potential issues
            vet_process = subprocess.run(["go", "vet", temp_file], capture_output=True, text=True)

            # Check the result of 'go vet'
            if vet_process.returncode != 0:
                logger.error(f"go vet validation failed for {file_path}:\n{vet_process.stderr}")
                return False
            else:
                logger.debug("go vet validation passed.")

            # Remove the temporary file
            os.remove(temp_file)
            logger.debug(f"Removed temporary Go file: {temp_file}")

            return True

        except FileNotFoundError:
            logger.error("Go is not installed or not found in PATH. Please install Go.")
            return False

        except Exception as e:
            logger.error(f"Unexpected error during Go code validation: {e}")
            return False

```

## language_functions/js_ts_handler.py

```python
# js_ts_handler.py
import os
import logging
import subprocess
import json
import tempfile
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class JSDocStyle(Enum):
    JSDOC = "jsdoc"
    TSDOC = "tsdoc"

@dataclass
class MetricsResult:
    complexity: int
    maintainability: float
    halstead: Dict[str, float]
    function_metrics: Dict[str, Dict[str, Any]]

class JSTsHandler(BaseHandler):

    def __init__(self, function_schema: Dict[str, Any]):
        self.function_schema = function_schema
        self.script_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        try:
            is_typescript = self._is_typescript_file(file_path)
            parser_options = self._get_parser_options(is_typescript)
            input_data = {
                "code": code,
                "language": "typescript" if is_typescript else "javascript",
                "filePath": file_path or "unknown",
                "options": parser_options
            }

            # Get metrics
            metrics = self._calculate_metrics(code, is_typescript)
            if metrics is None:
                return self._get_empty_structure("Metrics calculation failed")

            # Run parser script
            structure = self._run_parser_script(input_data)
            if structure is None:
                return self._get_empty_structure("Parsing failed")

            structure.update({
                "halstead": metrics.halstead,
                "complexity": metrics.complexity,
                "maintainability_index": metrics.maintainability,
                "function_metrics": metrics.function_metrics
            })

            # React analysis
            if self._is_react_file(file_path):
                react_info = self._analyze_react_components(code, is_typescript)
                if react_info is not None:
                    structure["react_components"] = react_info

            return structure

        except Exception as e:
            logger.error(f"Error extracting structure: {str(e)}", exc_info=True)
            return self._get_empty_structure(f"Error: {str(e)}")

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        try:
            is_typescript = self._is_typescript_file(documentation.get("file_path"))
            doc_style = JSDocStyle.TSDOC if is_typescript else JSDocStyle.JSDOC

            input_data = {
                "code": code,
                "documentation": documentation,
                "language": "typescript" if is_typescript else "javascript",
                "options": {
                    "style": doc_style.value,
                    "includeTypes": is_typescript,
                    "preserveExisting": True
                }
            }

            updated_code = self._run_inserter_script(input_data)
            return updated_code if updated_code is not None else code

        except Exception as e:
            logger.error(f"Error inserting documentation: {str(e)}", exc_info=True)
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        try:
            if not file_path:
                logger.warning("File path not provided for validation")
                return True

            is_typescript = self._is_typescript_file(file_path)
            config_path = self._get_eslint_config(is_typescript)

            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.ts' if is_typescript else '.js',
                encoding='utf-8',
                delete=False
            ) as tmp:
                tmp.write(code)
                temp_path = tmp.name

            try:
                result = subprocess.run(
                    ["eslint", "--config", config_path, temp_path],
                    capture_output=True,
                    text=True
                )
                return result.returncode == 0
            finally:
                os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Validation error: {str(e)}", exc_info=True)
            return False

    def _calculate_metrics(self, code: str, is_typescript: bool) -> Optional[MetricsResult]:
        try:
            input_data = {
                "code": code,
                "options": {
                    "typescript": is_typescript,
                    "sourceType": "module",
                    "loc": True,
                    "cyclomatic": True,
                    "halstead": True,
                    "maintainability": True
                }
            }
            result = self._run_script(
                script_name="js_ts_metrics.js",
                input_data=input_data,
                error_message="Metrics calculation failed"
            )
            logger.debug(f"Metrics calculation result: {result}")

            if result is None or not isinstance(result, dict) or not all(key in result for key in ["complexity", "maintainability", "halstead", "functions"]):
                logger.error("Invalid metrics result format.")
                return None

            return MetricsResult(
                complexity=result.get("complexity", 0),
                maintainability=result.get("maintainability", 0.0),
                halstead=result.get("halstead", {}),
                function_metrics=result.get("functions", {})
            )

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
            return None

    def _analyze_react_components(self, code: str, is_typescript: bool) -> Optional[Dict[str, Any]]:
        try:
            input_data = {
                "code": code,
                "options": {
                    "typescript": is_typescript,
                    "plugins": ["jsx", "react"]
                }
            }
            result = self._run_script(
                script_name="react_analyzer.js",
                input_data=input_data,
                error_message="React analysis failed"
            )
            logger.debug(f"React analysis result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing React components: {str(e)}", exc_info=True)
            return None

    def _get_parser_options(self, is_typescript: bool) -> Dict[str, Any]:
        options = {
            "sourceType": "module",
            "plugins": [
                "jsx",
                "decorators-legacy",
                ["decorators", {"decoratorsBeforeExport": True}],
                "classProperties",
                "classPrivateProperties",
                "classPrivateMethods",
                "exportDefaultFrom",
                "exportNamespaceFrom",
                "dynamicImport",
                "nullishCoalescingOperator",
                "optionalChaining",
            ]
        }

        if is_typescript:
            options["plugins"].extend([
                "typescript"
            ])

        return options

    @staticmethod
    def _is_typescript_file(file_path: Optional[str]) -> bool:
        if not file_path:
            return False
        return file_path.lower().endswith(('.ts', '.tsx'))

    @staticmethod
    def _is_react_file(file_path: Optional[str]) -> bool:
        if not file_path:
            return False
        return file_path.lower().endswith(('.jsx', '.tsx'))

    def _get_eslint_config(self, is_typescript: bool) -> str:
        config_name = '.eslintrc.typescript.json' if is_typescript else '.eslintrc.json'
        return os.path.join(self.script_dir, config_name)

    def _get_empty_structure(self, reason: str = "") -> Dict[str, Any]:
        return {
            "classes": [],
            "functions": [],
            "variables": [],
            "constants": [],
            "imports": [],
            "exports": [],
            "react_components": [],
            "summary": f"Empty structure: {reason}" if reason else "Empty structure",
            "halstead": {
                "volume": 0,
                "difficulty": 0,
                "effort": 0
            },
            "complexity": 0,
            "maintainability_index": 0,
            "function_metrics": {}
        }

    def _run_parser_script(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self._run_script(
            script_name="js_ts_parser.js",
            input_data=input_data,
            error_message="Parsing failed"
        )

    def _run_inserter_script(self, input_data: Dict[str, Any]) -> Optional[str]:
        result = self._run_script(
            script_name="js_ts_inserter.js",
            input_data=input_data,
            error_message="Error running inserter"
        )
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return json.dumps(result)
        else:
            logger.error("Inserter script did not return code string.")
            return None

    def _run_script(self, script_name: str, input_data: Dict[str, Any], error_message: str) -> Any:
        try:
            script_path = os.path.join(self.script_dir, script_name)
            process = subprocess.run(
                ['node', script_path],
                input=json.dumps(input_data, ensure_ascii=False).encode('utf-8'),
                capture_output=True,
                text=True,
                check=True,
                timeout=60
            )
            if process.returncode != 0:
                logger.error(f"{error_message}: {process.stderr}")
                return None

            output = process.stdout.strip()
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                if script_name == "js_ts_inserter.js":
                    return output
                logger.error(f"{error_message}: Invalid JSON output")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"{error_message}: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            logger.error(f"{error_message}: Script timed out.")
            return None
        except Exception as e:
            logger.error(f"{error_message}: {e}")
            return None

```

## scripts/html_parser.js

```javascript
// scripts/html_parser.js

const fs = require('fs');
const cheerio = require('cheerio');
const Ajv = require('ajv');
const path = require('path');

// Initialize AJV for JSON schema validation
const ajv = new Ajv({ allErrors: true, strict: false });

// Load the unified function_schema.json
const schemaPath = path.join(__dirname, '../schemas/function_schema.json');
const functionSchema = JSON.parse(fs.readFileSync(schemaPath, 'utf-8'));
const validate = ajv.compile(functionSchema);

// Read input from stdin
let inputChunks = [];
process.stdin.on('data', (chunk) => {
  inputChunks.push(chunk);
});

process.stdin.on('end', () => {
  const inputData = inputChunks.join('');

  let parsedInput;
  try {
    parsedInput = JSON.parse(inputData);
  } catch (e) {
    console.error('Error parsing input JSON:', e.message);
    process.exit(1);
  }

  const { code, documentation, language } = parsedInput;

  if (language.toLowerCase() !== 'html') {
    console.error('Unsupported language:', language);
    process.exit(1);
  }

  const $ = cheerio.load(code, { xmlMode: false });

  // Initialize the structure object
  const structure = {
    summary: "", // To be filled externally or manually
    changes_made: [], // To be filled externally or manually
    functions: [], // Not typically applicable for HTML
    classes: [], // Not typically applicable for HTML
    halstead: {
      volume: 0,
      difficulty: 0,
      effort: 0
    },
    maintainability_index: 0,
    variables: [], // Not typically applicable for HTML
    constants: []  // Not typically applicable for HTML
  };

  // Traverse all elements and extract information
  $('*').each(function(i, elem) {
    const tagName = elem.tagName;
    const attributes = {};
    for (let attr in elem.attribs) {
      attributes[attr] = elem.attribs[attr];
    }
    const elementDoc = documentation.elements.find(e => e.tag === tagName);
    const docstring = elementDoc ? elementDoc.docstring : "";

    // Populate classes or functions if applicable
    // HTML does not have classes or functions, but you can treat certain tags as classes if needed

    // Add to structure.elements or other relevant fields
    // Since the unified schema does not have an "elements" field, consider mapping HTML elements to classes or variables if appropriate

    // For demonstration, we'll skip adding to classes or functions
  });

  // Note: HTML does not inherently have functions or classes. Documentation can focus on tags and their purposes.

  // Validate the structure against the schema
  const valid = validate(structure);
  if (!valid) {
    console.error('Validation errors:', validate.errors);
    process.exit(1);
  }

  // Output the structure as JSON
  console.log(JSON.stringify(structure, null, 2));
});

```

## scripts/js_ts_metrics.js

```javascript
// js_ts_metrics.js
const escomplex = require('typhonjs-escomplex');

process.stdin.on('data', async (data) => {
    try {
        const input = JSON.parse(data.toString());
        const code = input.code;
        const options = input.options;

        const analysis = escomplex.analyzeModule(code, options);

        const halstead = analysis.aggregate.halstead;
        const functionsMetrics = analysis.functions.reduce((acc, method) => {
            acc[method.name] = {
                complexity: method.cyclomatic,
                sloc: method.sloc,
                params: method.params,
                halstead: method.halstead
            };
            return acc;
        }, {});

        const result = {
            complexity: analysis.aggregate.cyclomatic,
            maintainability: analysis.maintainability,
            halstead: {
                volume: halstead.volume,
                difficulty: halstead.difficulty,
                effort: halstead.effort
            },
            functions: functionsMetrics
        };

        console.log(JSON.stringify(result));

    } catch (error) {
        console.error(`Metrics calculation error: ${error.message}`);
        const defaultMetrics = {
            complexity: 0,
            maintainability: 0,
            halstead: { volume: 0, difficulty: 0, effort: 0 },
            functions: {}
        };
        console.log(JSON.stringify(defaultMetrics));
    }
});

```

## scripts/cpp_inserter.cpp

```cpp
// scripts/cpp_inserter.cpp

#include <clang/AST/AST.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace clang;
using namespace clang::tooling;
using json = nlohmann::json;

// Structure Definitions
struct Function {
    std::string name;
    std::string docstring;
    std::vector<std::string> args;
    bool async;
    int complexity; // Placeholder
};

struct Class {
    std::string name;
    std::string docstring;
    std::vector<Function> methods;
};

struct Variable {
    std::string name;
    std::string type;
    std::string description;
    std::string file;
    int line;
    std::string link;
    std::string example;
    std::string references;
};

// Visitor Class
class CppASTVisitor : public RecursiveASTVisitor<CppASTVisitor> {
public:
    CppASTVisitor(ASTContext *Context, Rewriter &R, const std::unordered_map<std::string, std::string>& docMap)
        : Context(Context), TheRewriter(R), docMap(docMap) {}

    bool VisitCXXRecordDecl(CXXRecordDecl *Declaration) {
        if (Declaration->isThisDeclarationADefinition()) {
            std::string className = Declaration->getNameAsString();
            auto it = docMap.find(className);
            if (it != docMap.end()) {
                // Insert class docstring before the class declaration
                SourceLocation loc = Declaration->getBeginLoc();
                TheRewriter.InsertTextBefore(loc, "/**\n * " + it->second + "\n */\n");
            }

            for (auto method : Declaration->methods()) {
                std::string methodName = method->getNameAsString();
                std::string fullMethodName = className + "." + methodName;
                auto mit = docMap.find(fullMethodName);
                if (mit != docMap.end()) {
                    // Insert method docstring before the method declaration
                    SourceLocation loc = method->getBeginLoc();
                    TheRewriter.InsertTextBefore(loc, "/**\n * " + mit->second + "\n */\n");
                }
            }
        }
        return true;
    }

    bool VisitFunctionDecl(FunctionDecl *Declaration) {
        if (Declaration->isThisDeclarationADefinition() && !Declaration->isCXXClassMember()) {
            std::string funcName = Declaration->getNameAsString();
            auto it = docMap.find(funcName);
            if (it != docMap.end()) {
                // Insert function docstring before the function declaration
                SourceLocation loc = Declaration->getBeginLoc();
                TheRewriter.InsertTextBefore(loc, "/**\n * " + it->second + "\n */\n");
            }
        }
        return true;
    }

private:
    ASTContext *Context;
    Rewriter &TheRewriter;
    const std::unordered_map<std::string, std::string>& docMap;
};

// AST Consumer
class CppASTConsumer : public ASTConsumer {
public:
    CppASTConsumer(ASTContext *Context, Rewriter &R, const std::unordered_map<std::string, std::string>& docMap)
        : Visitor(Context, R, docMap) {}

    virtual void HandleTranslationUnit(ASTContext &Context) {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }

private:
    CppASTVisitor Visitor;
};

// Frontend Action
class CppFrontendAction : public ASTFrontendAction {
public:
    CppFrontendAction(const std::unordered_map<std::string, std::string>& docMap)
        : docMap(docMap) {}

    void EndSourceFileAction() override {
        SourceManager &SM = TheRewriter.getSourceMgr();
        llvm::outs() << TheRewriter.getEditBuffer(SM.getMainFileID()).Buf;
    }

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
        TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        return std::make_unique<CppASTConsumer>(&CI.getASTContext(), TheRewriter, docMap);
    }

private:
    Rewriter TheRewriter;
    std::unordered_map<std::string, std::string> docMap;
};

// Main Function
int main(int argc, const char **argv) {
    // Read input from stdin
    std::string input;
    std::string line;
    while (std::getline(std::cin, line)) {
        input += line + "\n";
    }

    json inputData;
    try {
        inputData = json::parse(input);
    } catch (json::parse_error &e) {
        std::cerr << "Error parsing input JSON: " << e.what() << std::endl;
        return 1;
    }

    std::string code = inputData["code"].get<std::string>();
    std::unordered_map<std::string, std::string> documentation;

    // Parse documentation into a map
    if (inputData.contains("documentation")) {
        json doc = inputData["documentation"];
        // Classes
        if (doc.contains("classes")) {
            for (auto &cls : doc["classes"]) {
                std::string className = cls["name"].get<std::string>();
                std::string classDoc = cls["docstring"].get<std::string>();
                documentation[className] = classDoc;
                // Methods
                if (cls.contains("methods")) {
                    for (auto &method : cls["methods"]) {
                        std::string methodName = method["name"].get<std::string>();
                        std::string fullMethodName = className + "." + methodName;
                        std::string methodDoc = method["docstring"].get<std::string>();
                        documentation[fullMethodName] = methodDoc;
                    }
                }
            }
        }
        // Functions
        if (doc.contains("functions")) {
            for (auto &func : doc["functions"]) {
                std::string funcName = func["name"].get<std::string>();
                std::string funcDoc = func["docstring"].get<std::string>();
                documentation[funcName] = funcDoc;
            }
        }
    }

    // Create temporary file
    std::string tempFile = "temp.cpp";
    std::ofstream ofs(tempFile);
    ofs << code;
    ofs.close();

    // Parse command-line options
    CommonOptionsParser OptionsParser(argc, argv, llvm::cl::GeneralCategory);
    ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

    // Run the Clang Tool with our FrontendAction
    CppFrontendAction action(documentation);
    int result = Tool.run(newFrontendActionFactory(&action).get());

    // Cleanup temporary file
    remove(tempFile.c_str());

    return result;
}

```

## scripts/html_inserter.js

```javascript
// scripts/html_inserter.js

const fs = require('fs');
const cheerio = require('cheerio');
const path = require('path');

// Read input from stdin
let inputChunks = [];
process.stdin.on('data', (chunk) => {
  inputChunks.push(chunk);
});

process.stdin.on('end', () => {
  const inputData = inputChunks.join('');

  let parsedInput;
  try {
    parsedInput = JSON.parse(inputData);
  } catch (e) {
    console.error('Error parsing input JSON:', e.message);
    process.exit(1);
  }

  const { code, documentation, language } = parsedInput;

  if (language.toLowerCase() !== 'html') {
    console.error('Unsupported language:', language);
    process.exit(1);
  }

  const $ = cheerio.load(code, { xmlMode: false });

  // Traverse documentation to insert comments
  if (documentation.classes) {
    documentation.classes.forEach(cls => {
      // HTML does not have classes in the OOP sense; skip or map as needed
    });
  }

  if (documentation.functions) {
    documentation.functions.forEach(func => {
      // HTML does not have functions; skip or map as needed
    });
  }

  if (documentation.variables) {
    documentation.variables.forEach(varObj => {
      // HTML does not have variables; skip or map as needed
    });
  }

  if (documentation.constants) {
    documentation.constants.forEach(constObj => {
      // HTML does not have constants; skip or map as needed
    });
  }

  // Insert docstrings as comments before specific tags
  if (documentation.elements) {
    documentation.elements.forEach(elemDoc => {
      const tag = elemDoc.tag;
      const docstring = elemDoc.docstring;
      if (docstring) {
        $(tag).each(function(i, elem) {
          // Insert comment before the element
          $(elem).before(`<!-- ${docstring} -->\n`);
        });
      }
    });
  }

  // Generate the modified HTML
  const modifiedHTML = $.html();
  console.log(modifiedHTML);
});

```

## scripts/go_inserter.go

```go
// scripts/go_inserter.go

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"strings"
)

// InputData represents the structure of input JSON
type InputData struct {
	Code         string                 `json:"code"`
	Documentation map[string]interface{} `json:"documentation"`
	Language     string                 `json:"language"`
}

// Function represents a function/method in the code
type Function struct {
	Name       string   `json:"name"`
	Docstring  string   `json:"docstring"`
	Args       []string `json:"args"`
	Async      bool     `json:"async"`
	Complexity int      `json:"complexity"` // Placeholder for cyclomatic complexity
}

// Class represents a class/type in the code
type Class struct {
	Name      string    `json:"name"`
	Docstring string    `json:"docstring"`
	Methods   []Function `json:"methods"`
}

// Variable represents a variable in the code
type Variable struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Description string `json:"description"`
	File        string `json:"file"`
	Line        int    `json:"line"`
	Link        string `json:"link"`
	Example     string `json:"example"`
	References  string `json:"references"`
}

// Structure represents the overall code structure
type Structure struct {
	Summary              string      `json:"summary"`
	ChangesMade          []string    `json:"changes_made"`
	Functions            []Function  `json:"functions"`
	Classes              []Class     `json:"classes"`
	Halstead             map[string]float64 `json:"halstead"`
	MaintainabilityIndex float64     `json:"maintainability_index"`
	Variables            []Variable  `json:"variables"`
	Constants            []Variable  `json:"constants"`
}

func main() {
	reader := bufio.NewReader(os.Stdin)
	inputBytes, err := reader.ReadBytes(0)
	if err != nil && err != os.EOF {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
		os.Exit(1)
	}

	var input InputData
	if err := json.Unmarshal(inputBytes, &input); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing input JSON: %v\n", err)
		os.Exit(1)
	}

	if strings.ToLower(input.Language) != "go" {
		fmt.Fprintf(os.Stderr, "Unsupported language: %s\n", input.Language)
		os.Exit(1)
	}

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "src.go", input.Code, parser.ParseComments)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing Go code: %v\n", err)
		os.Exit(1)
	}

	structure := Structure{
		Summary:              "", // To be filled externally or manually
		ChangesMade:          [], // To be filled externally or manually
		Functions:            []Function{},
		Classes:              []Class{},
		Halstead:             make(map[string]float64),
		MaintainabilityIndex: 0.0, // Placeholder
		Variables:            []Variable{},
		Constants:            []Variable{},
	}

	// Convert documentation map to Structure
	docBytes, err := json.Marshal(input.Documentation)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error marshaling documentation: %v\n", err)
		os.Exit(1)
	}
	if err := json.Unmarshal(docBytes, &structure); err != nil {
		fmt.Fprintf(os.Stderr, "Error unmarshaling documentation: %v\n", err)
		os.Exit(1)
	}

	// Traverse the AST to insert docstrings
	ast.Inspect(file, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.FuncDecl:
			// Insert function docstring
			for _, funcDoc := range structure.Functions {
				if node.Name.Name == funcDoc.Name {
					if funcDoc.Docstring != "" {
						if node.Doc == nil {
							node.Doc = &ast.CommentGroup{}
						}
						docComment := &ast.Comment{
							Text: fmt.Sprintf("// %s", funcDoc.Docstring),
						}
						node.Doc.List = append([]*ast.Comment{docComment}, node.Doc.List...)
					}
				}
			}
		case *ast.TypeSpec:
			// Insert type (struct) docstring
			for _, classDoc := range structure.Classes {
				if node.Name.Name == classDoc.Name {
					if node.Doc == nil {
						node.Doc = &ast.CommentGroup{}
					}
					docComment := &ast.Comment{
						Text: fmt.Sprintf("// %s", classDoc.Docstring),
					}
					node.Doc.List = append([]*ast.Comment{docComment}, node.Doc.List...)
					
					// Insert method docstrings
					if structType, ok := node.Type.(*ast.StructType); ok {
						// Methods are defined outside the struct; handle separately
					}
				}
			}
		}
		return true
	})

	// Note: Go does not have a direct equivalent to classes. Methods are associated with types (usually structs).
	// The above code handles inserting docstrings for functions and types.

	// Generate the modified code
	var modifiedCode strings.Builder
	if err := printer.Fprint(&modifiedCode, fset, file); err != nil {
		fmt.Fprintf(os.Stderr, "Error generating modified code: %v\n", err)
		os.Exit(1)
	}

	// Output the modified code
	fmt.Println(modifiedCode.String())
}

```

## scripts/js_ts_inserter.js

```javascript
// js_ts_inserter.js

const fs = require('fs');
const babelParser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;
const t = require('@babel/types');

function generateJSDoc(description, params = [], returns = '', examples = []) {
    const lines = ['/**', ` * ${description}`];

    params.forEach(param => {
        lines.push(` * @param {${param.type || 'any'}} ${param.name} - ${param.description || ''}`);
    });

    if (returns) {
        lines.push(` * @returns {${returns.type || 'any'}} - ${returns.description || ''}`);
    }

    examples.forEach(example => {
        lines.push(' * @example');
        lines.push(` * ${example}`);
    });

    lines.push(' */');
    return lines.join('\n');
}

function insertJSDoc(code, documentation, language) {
    const isTypeScript = language === 'typescript';

    // Parse the code into an AST
    const ast = babelParser.parse(code, {
        sourceType: 'module',
        plugins: [
            'jsx',
            isTypeScript ? 'typescript' : null,
            'classProperties',
            'decorators-legacy',
        ].filter(Boolean),
    });

    const docsMap = new Map();

    // Map documentation to code elements
    if (documentation.functions) {
        documentation.functions.forEach(func => {
            docsMap.set(func.name, func);
        });
    }
    if (documentation.classes) {
        documentation.classes.forEach(cls => {
            docsMap.set(cls.name, cls);
            if (cls.methods) {
                cls.methods.forEach(method => {
                    docsMap.set(`${cls.name}.${method.name}`, method);
                });
            }
        });
    }

    // Traverse the AST to insert comments
    traverse(ast, {
        enter(path) {
            const node = path.node;
            if (node.type === 'FunctionDeclaration' && node.id) {
                const doc = docsMap.get(node.id.name);
                if (doc) {
                    const jsDocComment = generateJSDoc(doc.description, doc.params, doc.returns, doc.examples);
                    node.leadingComments = node.leadingComments || [];
                    node.leadingComments.push({
                        type: 'CommentBlock',
                        value: jsDocComment.replace(/^\/\*\*|\*\/$/g, '').trim(),
                    });
                }
            } else if (node.type === 'ClassDeclaration' && node.id) {
                const doc = docsMap.get(node.id.name);
                if (doc) {
                    const jsDocComment = generateJSDoc(doc.description, [], null, doc.examples);
                    node.leadingComments = node.leadingComments || [];
                    node.leadingComments.push({
                        type: 'CommentBlock',
                        value: jsDocComment.replace(/^\/\*\*|\*\/$/g, '').trim(),
                    });
                }
                // Handle class methods
                node.body.body.forEach(element => {
                    if (
                        (element.type === 'ClassMethod' || element.type === 'ClassPrivateMethod') &&
                        element.key.type === 'Identifier'
                    ) {
                        const methodName = `${node.id.name}.${element.key.name}`;
                        const doc = docsMap.get(methodName);
                        if (doc) {
                            const jsDocComment = generateJSDoc(doc.description, doc.params, doc.returns, doc.examples);
                            element.leadingComments = element.leadingComments || [];
                            element.leadingComments.push({
                                type: 'CommentBlock',
                                value: jsDocComment.replace(/^\/\*\*|\*\/$/g, '').trim(),
                            });
                        }
                    }
                });
            }
        },
    });

    // Generate the modified code
    const output = generate(ast, { comments: true }, code);
    return output.code;
}

function main() {
    const input = fs.readFileSync(0, 'utf-8');
    const data = JSON.parse(input);
    const code = data.code;
    const documentation = data.documentation;
    const language = data.language || 'javascript';
    const modifiedCode = insertJSDoc(code, documentation, language);
    console.log(modifiedCode);
}

main();
```

## scripts/java_inserter.js

```javascript
// scripts/java_inserter.js

const fs = require('fs');
const javaParser = require('java-parser');
const path = require('path');

// Read input from stdin
let inputChunks = [];
process.stdin.on('data', (chunk) => {
  inputChunks.push(chunk);
});

process.stdin.on('end', () => {
  const inputData = inputChunks.join('');

  let parsedInput;
  try {
    parsedInput = JSON.parse(inputData);
  } catch (e) {
    console.error('Error parsing input JSON:', e.message);
    process.exit(1);
  }

  const { code, documentation, language } = parsedInput;

  if (language.toLowerCase() !== 'java') {
    console.error('Unsupported language:', language);
    process.exit(1);
  }

  let ast;
  try {
    ast = javaParser.parse(code);
  } catch (e) {
    console.error('Parsing error:', e.message);
    process.exit(1);
  }

  // Helper function to insert docstrings
  function insertDocstring(node, docstring) {
    if (node.documentation) {
      node.documentation = `/**\n * ${docstring}\n */`;
    } else {
      node.documentation = `/** ${docstring} */`;
    }
  }

  // Traverse documentation to insert into AST
  documentation.classes.forEach(clsDoc => {
    const cls = ast.children.find(child => child.node === 'ClassDeclaration' && child.name.identifier === clsDoc.name);
    if (cls) {
      insertDocstring(cls, clsDoc.docstring);
      cls.body.body.forEach(member => {
        if (member.node === 'MethodDeclaration') {
          const methodDoc = clsDoc.methods.find(m => m.name === member.name.identifier);
          if (methodDoc) {
            insertDocstring(member, methodDoc.docstring);
          }
        }
      });
    }
  });

  documentation.functions.forEach(funcDoc => {
    // Java functions are typically static methods; find and insert
    const cls = ast.children.find(child => child.node === 'ClassDeclaration');
    if (cls) {
      const method = cls.body.body.find(member => member.node === 'MethodDeclaration' && member.name.identifier === funcDoc.name);
      if (method) {
        insertDocstring(method, funcDoc.docstring);
      }
    }
  });

  // Note: java-parser does not support code generation. To output modified code,
  // consider using alternative libraries or integrating with Java tools that support AST modifications.

  // As a placeholder, output the original code
  // In a real-world scenario, you would need to use a Java code generation library
  // or implement a method to serialize the modified AST back to source code.
  console.log(code);
});

```

## scripts/js_ts_parser.js

```javascript
// Enhanced JavaScript/TypeScript parser with comprehensive analysis capabilities

const babelParser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const t = require('@babel/types');
const generate = require('@babel/generator').default;
const tsEstree = require('@typescript-eslint/typescript-estree');
const escomplex = require('typhonjs-escomplex');

class JSTSParser {
    constructor(options = {}) {
        this.options = {
            sourceType: 'module',
            errorRecovery: true,
            ...options
        };
    }

    parse(code, language = 'javascript', filePath = 'unknown') {
        try {
            const isTypeScript = language === 'typescript' || filePath.endsWith('.ts') || filePath.endsWith('.tsx');

            const ast = this._parseCode(code, isTypeScript);
            const structure = this._initializeStructure();

            const metrics = this._calculateMetrics(code, isTypeScript, filePath);
            Object.assign(structure, metrics);

            this._traverseAST(ast, structure, isTypeScript);
            return structure;

        } catch (error) {
            console.error(`Parse error in ${filePath}: ${error.message}`);
            return this._getEmptyStructure(error.message);
        }
    }

    _parseCode(code, isTypeScript) {
        const parserOptions = {
            sourceType: this.options.sourceType,
            plugins: this._getBabelPlugins(isTypeScript),
            errorRecovery: this.options.errorRecovery,
            tokens: true,
            ...this.options
        };

        try {
            if (isTypeScript) {
                return tsEstree.parse(code, { jsx: true, ...parserOptions });
            } else {
                return babelParser.parse(code, parserOptions);
            }
        } catch (error) {
            console.error("Parsing failed:", error);
            throw error;
        }
    }

    _calculateMetrics(code, isTypeScript, filePath) {
        try {
            const analysis = escomplex.analyzeModule(code, {
                sourceType: 'module',
                useTypeScriptEstree: isTypeScript,
                loc: true,
                newmi: true,
                skipCalculation: false
            });

            return {
                complexity: analysis.aggregate.cyclomatic,
                maintainability_index: analysis.maintainability,
                halstead: {
                    volume: analysis.aggregate.halstead.volume,
                    difficulty: analysis.aggregate.halstead.difficulty,
                    effort: analysis.aggregate.halstead.effort
                },
                function_metrics: analysis.methods.reduce((acc, method) => {
                    acc[method.name] = {
                        complexity: method.cyclomatic,
                        sloc: method.sloc,
                        params: method.params
                    };
                    return acc;
                }, {})
            };
        } catch (error) {
            console.error(`Metrics calculation error in ${filePath}: ${error.message}`);
            return {
                complexity: 0,
                maintainability_index: 0,
                halstead: { volume: 0, difficulty: 0, effort: 0 },
                function_metrics: {}
            };
        }
    }

    _traverseAST(ast, structure, isTypeScript) {
        traverse(ast, {
            ClassDeclaration: (path) => {
                structure.classes.push(this._extractClassInfo(path.node, path, isTypeScript));
            },
            FunctionDeclaration: (path) => {
                structure.functions.push(this._extractFunctionInfo(path.node, path, isTypeScript));
            },
            VariableDeclaration: (path) => {
                const declarations = this._extractVariableInfo(path.node, path, isTypeScript);
                const collection = path.node.kind === 'const' ? structure.constants : structure.variables;
                collection.push(...declarations);
            },
            ImportDeclaration: (path) => {
                structure.imports.push(this._extractImportInfo(path.node));
            },
            ExportDefaultDeclaration: (path) => {
                structure.exports.push(this._extractExportInfo(path.node, true));
            },
            ExportNamedDeclaration: (path) => {
                const exportInfo = this._extractExportInfo(path.node, false);
                if (Array.isArray(exportInfo)) {
                    structure.exports.push(...exportInfo);
                } else if (exportInfo) {
                    structure.exports.push(exportInfo);
                }
            },
            JSXElement: (path) => {
                if (this._isReactComponent(path)) {
                    structure.react_components.push(this._extractReactComponentInfo(path));
                }
            },
            TSInterfaceDeclaration: isTypeScript ? (path) => {
                structure.interfaces.push(this._extractInterfaceInfo(path.node));
            } : null,
            TSTypeAliasDeclaration: isTypeScript ? (path) => {
                structure.types.push(this._extractTypeAliasInfo(path.node));
            } : null,
            ArrowFunctionExpression: (path) => {
                structure.functions.push(this._extractFunctionInfo(path.node, path, isTypeScript));
            },
            ...this._getAdditionalVisitors(isTypeScript)
        });
    }

    _extractClassInfo(node, path, isTypeScript) {
        return {
            name: node.id.name,
            methods: node.body.body
                .filter(member => t.isClassMethod(member) || t.isClassPrivateMethod(member))
                .map(method => this._extractMethodInfo(method, isTypeScript)),
            properties: node.body.body
                .filter(member => t.isClassProperty(member) || t.isClassPrivateProperty(member))
                .map(prop => this._extractPropertyInfo(prop, isTypeScript)),

            superClass: node.superClass?.name,
            decorators: this._extractDecorators(node),
            docstring: this._extractDocstring(node),
            isAbstract: node.abstract || false,
            isExported: this._isExported(path),
            implements: isTypeScript ? this._extractImplementedInterfaces(node) : []
        };
    }

    _extractFunctionInfo(node, path, isTypeScript) {
        const functionName = node.id ? node.id.name : (node.key && node.key.name) || 'anonymous';
        const params = this._extractParameters(node.params, isTypeScript);
        const returnType = isTypeScript ? this._getTypeString(node.returnType) : null;
        const async = node.async || false;
        const generator = node.generator || false;

        return {
            name: functionName,
            params,
            returnType,
            docstring: this._extractDocstring(node),
            isExported: this._isExported(path),
            async: async,
            generator: generator,
            complexity: this.options.function_metrics && this.options.function_metrics[functionName] ? this.options.function_metrics[functionName].complexity : null
        };
    }

    _extractVariableInfo(node, path, isTypeScript) {
        return node.declarations.map(declarator => {
            const varName = declarator.id.name;
            const varType = isTypeScript ? this._getTypeString(declarator.id.typeAnnotation) : null;
            const defaultValue = this._getDefaultValue(declarator.init);

            return {
                name: varName,
                type: varType,
                defaultValue: defaultValue,
                docstring: this._extractDocstring(declarator),
                isExported: this._isExported(path)
            };
        });
    }

    _extractImportInfo(node) {
        const source = node.source.value;
        const specifiers = node.specifiers.map(specifier => {
            if (t.isImportSpecifier(specifier)) {
                return {
                    type: 'named',
                    imported: specifier.imported.name,
                    local: specifier.local.name,
                };
            } else if (t.isImportDefaultSpecifier(specifier)) {
                return {
                    type: 'default',
                    local: specifier.local.name
                };
            } else if (t.isImportNamespaceSpecifier(specifier)) {
                return {
                    type: 'namespace',
                    local: specifier.local.name
                };
            }
        });
        return { source, specifiers };
    }

    _extractExportInfo(node, isDefault) {
        if (isDefault) {
            const declaration = node.declaration;
            return {
                type: 'default',
                name: this._getDeclarationName(declaration),
                declaration: generate(declaration).code
            };
        } else if (node.declaration) {
            const declaration = node.declaration;
            const declarations = t.isVariableDeclaration(declaration) ? declaration.declarations : [declaration];
            return declarations.map(decl => ({
                type: 'named',
                name: this._getDeclarationName(decl),
                declaration: generate(decl).code
            }));
        } else if (node.specifiers && node.specifiers.length > 0) {
            return node.specifiers.map(specifier => ({
                type: 'named',
                exported: specifier.exported.name,
                local: specifier.local.name
            }));
        }
        return null;
    }

    _getDeclarationName(declaration) {
        if (t.isIdentifier(declaration)) {
            return declaration.name;
        } else if (t.isFunctionDeclaration(declaration) || t.isClassDeclaration(declaration)) {
            return declaration.id?.name || null;
        } else if (t.isVariableDeclarator(declaration)) {
            return declaration.id.name;
        }
        return null;
    }

    _extractInterfaceInfo(node) {
        const interfaceName = node.id.name;
        const properties = node.body.body.map(property => {
            return {
                name: property.key.name,
                type: this._getTypeString(property.typeAnnotation),
                docstring: this._extractDocstring(property)
            };
        });
        return { name: interfaceName, properties };
    }

    _extractTypeAliasInfo(node) {
        return {
            name: node.id.name,
            type: this._getTypeString(node.typeAnnotation)
        };
    }

    _extractReactComponentInfo(path) {
        const component = path.findParent(p =>
            t.isFunctionDeclaration(p) ||
            t.isArrowFunctionExpression(p) ||
            t.isClassDeclaration(p) ||
            t.isVariableDeclarator(p)
        );

        if (!component) return null;

        const componentName = this._getComponentName(component.node);
        const props = this._extractReactProps(component);
        const hooks = this._extractReactHooks(component);
        const state = this._extractReactState(component);
        const effects = this._extractReactEffects(component);
        const isExportedComponent = this._isExported(component);

        return {
            name: componentName,
            props,
            hooks,
            state,
            effects,
            docstring: this._extractDocstring(component.node),
            isExported: isExportedComponent,
            type: this._getReactComponentType(component.node)
        };
    }

    _getComponentName(node) {
        if (t.isVariableDeclarator(node)) {
            return node.id.name;
        } else if (t.isFunctionDeclaration(node) || t.isClassDeclaration(node)) {
            return node.id?.name || null;
        }
        return 'anonymous';
    }

    _getReactComponentType(node) {
        if (t.isClassDeclaration(node)) {
            return 'class';
        } else if (t.isFunctionDeclaration(node) || t.isArrowFunctionExpression(node)) {
            return 'function';
        } else if (t.isVariableDeclarator(node)) {
            return 'variable';
        }
        return null;
    }

    _extractReactProps(componentPath) {
        const component = componentPath.node;
        let props = [];

        if (t.isClassDeclaration(component)) {
            const constructor = component.body.body.find(member => t.isClassMethod(member) && member.kind === 'constructor');
            if (constructor && constructor.params.length > 0) {
                props = this._extractPropsFromParam(constructor.params[0]);
            }
        } else if (t.isFunctionDeclaration(component) || t.isArrowFunctionExpression(component)) {
            if (component.params.length > 0) {
                props = this._extractPropsFromParam(component.params[0]);
            }
        } else if (t.isVariableDeclarator(component)) {
            if (component.init && (t.isArrowFunctionExpression(component.init) || t.isFunctionExpression(component.init))) {
                if (component.init.params.length > 0) {
                    props = this._extractPropsFromParam(component.init.params[0]);
                }
            }
        }

        return props;
    }

    _extractPropsFromParam(param) {
        if (param.typeAnnotation) {
            const typeAnnotation = param.typeAnnotation.typeAnnotation;
            if (t.isTSTypeLiteral(typeAnnotation)) {
                return typeAnnotation.members.map(member => ({
                    name: member.key.name,
                    type: this._getTypeString(member.typeAnnotation),
                    required: !member.optional,
                    defaultValue: this._getDefaultValue(member)
                }));
            } else if (t.isTSTypeReference(typeAnnotation) && t.isIdentifier(typeAnnotation.typeName)) {
                return [{ name: param.name, type: typeAnnotation.typeName.name, required: !param.optional }];
            }
        } else if (t.isObjectPattern(param)) {
            return param.properties.map(prop => ({
                name: prop.key.name,
                type: this._getTypeString(prop.value?.typeAnnotation),
                required: true
            }));
        }
        return [];
    }

    _extractReactHooks(componentPath) {
        const hooks = [];
        componentPath.traverse({
            CallExpression(path) {
                if (t.isIdentifier(path.node.callee) && path.node.callee.name.startsWith('use')) {
                    const hookName = path.node.callee.name;
                    const dependencies = this._extractHookDependencies(path.node);
                    hooks.push({ name: hookName, dependencies });
                }
            }
        });
        return hooks;
    }

    _extractHookDependencies(node) {
        if (node.arguments && node.arguments.length > 1 && t.isArrayExpression(node.arguments[1])) {
            return node.arguments[1].elements.map(element => generate(element).code);
        }
        return [];
    }

    _extractReactEffects(componentPath) {
        const effects = [];
        componentPath.traverse({
            CallExpression(path) {
                if (t.isIdentifier(path.node.callee) && path.node.callee.name === 'useEffect') {
                    const dependencies = this._extractHookDependencies(path.node);
                    const cleanup = this._hasEffectCleanup(path.node);
                    effects.push({ dependencies, cleanup });
                }
            }
        });
        return effects;
    }

    _hasEffectCleanup(node) {
        if (node.arguments && node.arguments.length > 0 && t.isArrowFunctionExpression(node.arguments[0]) && node.arguments[0].body) {
            const body = node.arguments[0].body;
            return t.isBlockStatement(body) && body.body.some(statement => t.isReturnStatement(statement) && statement.argument !== null);
        }
        return false;
    }

    _extractReactState(componentPath) {
        const state = [];
        componentPath.traverse({
            CallExpression(path) {
                if (t.isMemberExpression(path.node.callee) &&
                    t.isIdentifier(path.node.callee.object, { name: 'React' }) &&
                    t.isIdentifier(path.node.callee.property, { name: 'useState' })) {

                    const initialValue = path.node.arguments[0];
                    state.push({
                        initialValue: generate(initialValue).code
                    });
                } else if (t.isIdentifier(path.node.callee, { name: 'useState' })) {
                    const initialValue = path.node.arguments[0];
                    state.push({
                        initialValue: generate(initialValue).code
                    });
                }
            }
        });
        return state;
    }

    _getDefaultValue(node) {
        if (!node) return null;
        return generate(node).code;
    }

    _getTypeString(typeAnnotation) {
        if (!typeAnnotation) return null;
        if (t.isTSTypeReference(typeAnnotation)) {
            return generate(typeAnnotation.typeName).code;
        } else if (t.isTSLiteralType(typeAnnotation)) {
            return generate(typeAnnotation.literal).code;
        } else if (t.isTSTypeAnnotation(typeAnnotation)) {
            return this._getTypeString(typeAnnotation.typeAnnotation);
        }
        return null;
    }

    _extractParameters(params, isTypeScript) {
        return params.map(param => {
            return {
                name: param.name,
                type: isTypeScript ? this._getTypeString(param.typeAnnotation) : null,
                defaultValue: this._getDefaultValue(param.defaultValue)
            };
        });
    }

    _extractReturnType(node) {
        return this._getTypeString(node.returnType);
    }

    _extractDecorators(node) {
        return (node.decorators || []).map(decorator => generate(decorator.expression).code);
    }

    _getAccessibility(node) {
        return node.accessibility || 'public';
    }

    _getMethodName(node) {
        if (node.key && t.isIdentifier(node.key)) {
            return node.key.name;
        } else if (node.key && t.isPrivateName(node.key)) {
            return `#${node.key.id.name}`;
        }
        return null;
    }

    _calculateMethodComplexity(node) {
        return null;
    }

    _isExported(path) {
        let parent = path.parentPath;
        while (parent) {
            if (parent.isExportNamedDeclaration() || parent.isExportDefaultDeclaration()) {
                return true;
            }
            parent = parent.parentPath;
        }
        return false;
    }

    _isReactComponent(path) {
        return t.isJSXIdentifier(path.node.openingElement.name);
    }

    _getBabelPlugins(isTypeScript) {
        const plugins = [
            'jsx',
            'decorators-legacy',
            ['decorators', { decoratorsBeforeExport: true }],
            'classProperties', 'classPrivateProperties', 'classPrivateMethods',
            'exportDefaultFrom', 'exportNamespaceFrom', 'dynamicImport',
            'nullishCoalescing', 'optionalChaining', 'asyncGenerators', 'bigInt',
            'classProperties', 'doExpressions', 'dynamicImport', 'exportDefaultFrom',
            'exportNamespaceFrom', 'functionBind', 'functionSent', 'importMeta',
            'logicalAssignment', 'numericSeparator', 'nullishCoalescingOperator',
            'optionalCatchBinding', 'optionalChaining', 'partialApplication',
            'throwExpressions', "pipelineOperator", "recordAndTuple"
        ];

        if (isTypeScript) {
            plugins.push('typescript');
        }
        return plugins;
    }

    _getAdditionalVisitors(isTypeScript) {
        if (isTypeScript) {
            return {
                TSEnumDeclaration(path) {
                    this.node.enums.push({
                        name: path.node.id.name,
                        members: path.node.members.map(member => ({
                            name: member.id.name,
                            initializer: member.initializer ? generate(member.initializer).code : null
                        }))
                    });
                },
                TSTypeAliasDeclaration(path) {
                    this.node.types.push({
                        name: path.node.id.name,
                        type: generate(path.node.typeAnnotation).code
                    });
                },
                TSInterfaceDeclaration(path) {
                    this.node.interfaces.push({
                        name: path.node.id.name,
                    });
                },
            };
        }
        return {};
    }

    _initializeStructure() {
        return {
            classes: [],
            functions: [],
            variables: [],
            constants: [],
            imports: [],
            exports: [],
            interfaces: [],
            types: [],
            enums: [],
            react_components: [],
            complexity: 0,
            maintainability_index: 0,
            halstead: { volume: 0, difficulty: 0, effort: 0 },
            summary: "",
            function_metrics: {}
        };
    }

    _getEmptyStructure(reason = '') {
        return {
            ...this._initializeStructure(),
            summary: `Empty structure: ${reason}`
        };
    }

    _extractDocstring(node) {
        const leadingComments = node.leadingComments || [];
        const docstringComment = leadingComments.find(comment => comment.type === 'CommentBlock' && comment.value.trim().startsWith('*'));
        return docstringComment ? docstringComment.value.replace(/^\*\s?/gm, '').trim() : '';
    }

    _extractPropertyInfo(node, isTypeScript) {
        const propertyName = node.key.name;
        const propertyType = isTypeScript ? this._getTypeString(node.typeAnnotation) : null;
        const defaultValue = this._getDefaultValue(node.value);
        const accessibility = this._getAccessibility(node);
        const isStatic = node.static || false;
        const decorators = this._extractDecorators(node);
        const docstring = this._extractDocstring(node);

        return {
            name: propertyName,
            type: propertyType,
            defaultValue: defaultValue,
            accessibility,
            isStatic,
            decorators,
            docstring
        };
    }

    _extractImplementedInterfaces(node) {
        return (node.implements || []).map(i => i.id.name);
    }
}

module.exports = JSTSParser;
```

## scripts/go_parser.go

```go
// scripts/go_parser.go

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
)

// InputData represents the structure of input JSON
type InputData struct {
	Code     string `json:"code"`
	Language string `json:"language"`
}

// Function represents a function/method in the code
type Function struct {
	Name       string   `json:"name"`
	Docstring  string   `json:"docstring"`
	Args       []string `json:"args"`
	Async      bool     `json:"async"`
	Complexity int      `json:"complexity"` // Placeholder for cyclomatic complexity
}

// Class represents a class/type in the code
type Class struct {
	Name       string    `json:"name"`
	Docstring  string    `json:"docstring"`
	Methods    []Function `json:"methods"`
}

// Variable represents a variable in the code
type Variable struct {
	Name       string `json:"name"`
	Type       string `json:"type"`
	Description string `json:"description"`
	File       string `json:"file"`
	Line       int    `json:"line"`
	Link       string `json:"link"`
	Example    string `json:"example"`
	References string `json:"references"`
}

// Structure represents the overall code structure
type Structure struct {
	Summary              string      `json:"summary"`
	ChangesMade          []string    `json:"changes_made"`
	Functions            []Function  `json:"functions"`
	Classes              []Class     `json:"classes"`
	Halstead             map[string]float64 `json:"halstead"`
	MaintainabilityIndex float64     `json:"maintainability_index"`
	Variables            []Variable  `json:"variables"`
	Constants            []Variable  `json:"constants"` // Reusing Variable struct for constants
}

func main() {
	reader := bufio.NewReader(os.Stdin)
	inputBytes, err := reader.ReadBytes(0)
	if err != nil && err != os.EOF {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
		os.Exit(1)
	}

	var input InputData
	if err := json.Unmarshal(inputBytes, &input); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing input JSON: %v\n", err)
		os.Exit(1)
	}

	if strings.ToLower(input.Language) != "go" {
		fmt.Fprintf(os.Stderr, "Unsupported language: %s\n", input.Language)
		os.Exit(1)
	}

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "src.go", input.Code, parser.ParseComments)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing Go code: %v\n", err)
		os.Exit(1)
	}

	structure := Structure{
		Summary:              "", // To be filled externally or manually
		ChangesMade:          [], // To be filled externally or manually
		Functions:            []Function{},
		Classes:              []Class{},
		Halstead:             make(map[string]float64),
		MaintainabilityIndex: 0.0, // Placeholder
		Variables:            []Variable{},
		Constants:            []Variable{},
	}

	// Traverse the AST
	ast.Inspect(file, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.FuncDecl:
			funcInfo := Function{
				Name:      node.Name.Name,
				Docstring: getDocstring(node.Doc),
				Args:      getFuncArgs(node.Type.Params),
				Async:     false, // Go does not have async functions
				Complexity: 1,    // Placeholder for cyclomatic complexity
			}
			structure.Functions = append(structure.Functions, funcInfo)
		case *ast.TypeSpec:
			// Assuming structs as classes
			if structType, ok := node.Type.(*ast.StructType); ok {
				classInfo := Class{
					Name:      node.Name.Name,
					Docstring: getDocstring(node.Doc),
					Methods:   []Function{},
				}
				// Traverse methods
				ast.Inspect(file, func(n ast.Node) bool {
					if fn, ok := n.(*ast.FuncDecl); ok {
						if fn.Recv != nil && len(fn.Recv.List) > 0 {
							receiver := exprToString(fn.Recv.List[0].Type)
							if strings.Contains(receiver, classInfo.Name) {
								methodInfo := Function{
									Name:      fn.Name.Name,
									Docstring: getDocstring(fn.Doc),
									Args:      getFuncArgs(fn.Type.Params),
									Async:     false,
									Complexity: 1,
								}
								classInfo.Methods = append(classInfo.Methods, methodInfo)
							}
						}
					}
					return true
				})
				structure.Classes = append(structure.Classes, classInfo)
			}
		case *ast.ValueSpec:
			for i, name := range node.Names {
				varType := exprToString(node.Type)
				varDesc := ""
				if node.Comment != nil && len(node.Comment.List) > i {
					varDesc = strings.TrimPrefix(node.Comment.List[i].Text, "//")
				}
				variable := Variable{
					Name:        name.Name,
					Type:        varType,
					Description: varDesc,
					File:        "Unknown", // Can be set if file info is available
					Line:        fset.Position(name.Pos()).Line,
					Link:        "Unknown", // Can be constructed based on repository
					Example:     "No example provided.",
					References:  "No references.",
				}
				if strings.ToUpper(name.Name) == name.Name {
					structure.Constants = append(structure.Constants, variable)
				} else {
					structure.Variables = append(structure.Variables, variable)
				}
			}
		}
		return true
	})

	// Placeholder for Halstead metrics and Maintainability Index
	// These require detailed analysis and are not implemented here

	// Validate the structure against the schema
	validateStruct, err := json.Marshal(structure)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error marshaling structure: %v\n", err)
		os.Exit(1)
	}

	var schema ValidationSchema
	schema = loadSchema()

	var data interface{}
	if err := json.Unmarshal(validateStruct, &data); err != nil {
		fmt.Fprintf(os.Stderr, "Error unmarshaling structure: %v\n", err)
		os.Exit(1)
	}

	valid := validate(data)
	if !valid {
		fmt.Fprintf(os.Stderr, "Validation errors: %v\n", validate.errors)
		os.Exit(1)
	}

	// Output the structure as JSON
	fmt.Println(string(validateStruct))
}

// Helper functions

func getDocstring(doc *ast.CommentGroup) string {
	if doc == nil {
		return ""
	}
	return strings.TrimSpace(doc.Text())
}

func getFuncArgs(params *ast.FieldList) []string {
	if params == nil {
		return []string{}
	}
	args := []string{}
	for _, field := range params.List {
		for _, name := range field.Names {
			args = append(args, name.Name)
		}
	}
	return args
}

func exprToString(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + exprToString(t.X)
	case *ast.SelectorExpr:
		return exprToString(t.X) + "." + t.Sel.Name
	case *ast.ArrayType:
		return "[]" + exprToString(t.Elt)
	default:
		return "unknown"
	}
}

// Placeholder for schema loading and validation
// Implement schema loading and validation as needed
type ValidationSchema struct{}

func loadSchema() ValidationSchema {
	// Implement schema loading if necessary
	return ValidationSchema{}
}

```

## scripts/cpp_parser.cpp

```cpp
// scripts/cpp_parser.cpp

#include <clang/AST/AST.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace clang;
using namespace clang::tooling;
using json = nlohmann::json;

// Structure Definitions
struct Function {
    std::string name;
    std::string docstring;
    std::vector<std::string> args;
    bool async;
    int complexity; // Placeholder
};

struct Class {
    std::string name;
    std::string docstring;
    std::vector<Function> methods;
};

struct Variable {
    std::string name;
    std::string type;
    std::string description;
    std::string file;
    int line;
    std::string link;
    std::string example;
    std::string references;
};

// Visitor Class
class CppASTVisitor : public RecursiveASTVisitor<CppASTVisitor> {
public:
    explicit CppASTVisitor(ASTContext *Context) : Context(Context) {}

    bool VisitCXXRecordDecl(CXXRecordDecl *Declaration) {
        if (Declaration->isThisDeclarationADefinition()) {
            Class cls;
            cls.name = Declaration->getNameAsString();
            cls.docstring = getDocstring(Declaration->getASTContext(), Declaration);

            for (auto method : Declaration->methods()) {
                Function func;
                func.name = method->getNameAsString();
                func.docstring = getDocstring(method->getASTContext(), method);
                func.async = false; // C++ does not have async functions
                func.complexity = 1; // Placeholder

                // Get function arguments
                for (auto param : method->parameters()) {
                    func.args.push_back(param->getNameAsString());
                }

                cls.methods.push_back(func);
            }

            classes.push_back(cls);
        }
        return true;
    }

    bool VisitVarDecl(VarDecl *Declaration) {
        if (Declaration->hasGlobalStorage()) {
            Variable var;
            var.name = Declaration->getNameAsString();
            var.type = Declaration->getType().getAsString();
            var.description = getDocstring(Declaration->getASTContext(), Declaration);
            var.file = Context->getSourceManager().getFilename(Declaration->getLocation()).str();
            var.line = Context->getSourceManager().getSpellingLineNumber(Declaration->getLocation());
            var.link = "Unknown"; // Construct based on repository URL
            var.example = "No example provided.";
            var.references = "No references.";

            // Determine if the variable is a constant (e.g., const or constexpr)
            QualType qt = Declaration->getType();
            if (qt.isConstQualified() || Declaration->isConstexpr()) {
                structure.constants.push_back(var);
            } else {
                structure.variables.push_back(var);
            }
        }
        return true;
    }

    json getJSON() {
        json j;
        j["summary"] = ""; // To be filled externally or manually
        j["changes_made"] = json::array(); // To be filled externally or manually
        j["functions"] = json::array(); // For standalone functions
        j["classes"] = json::array();
        j["halstead"] = {
            {"volume", 0.0},
            {"difficulty", 0.0},
            {"effort", 0.0}
        };
        j["maintainability_index"] = 0.0; // Placeholder
        j["variables"] = structure.variables;
        j["constants"] = structure.constants;

        for (const auto &cls : classes) {
            json jcls;
            jcls["name"] = cls.name;
            jcls["docstring"] = cls.docstring;
            jcls["methods"] = json::array();
            for (const auto &method : cls.methods) {
                json jmethod;
                jmethod["name"] = method.name;
                jmethod["docstring"] = method.docstring;
                jmethod["args"] = method.args;
                jmethod["async"] = method.async;
                jmethod["type"] = "instance"; // C++ does not have explicit method types
                jmethod["complexity"] = method.complexity;
                jcls["methods"].push_back(jmethod);
            }
            j["classes"].push_back(jcls);
        }

        // Add standalone functions
        // Placeholder: Implement extraction of standalone functions if necessary

        return j;
    }

private:
    ASTContext *Context;
    std::vector<Class> classes;

    struct InternalStructure {
        std::vector<Variable> variables;
        std::vector<Variable> constants;
    } structure;

    std::string getDocstring(ASTContext &Context, const Decl *Declaration) {
        std::string doc = "";
        RawComment *RC = Context.getRawCommentForAnyRedecl(Declaration);
        if (RC) {
            doc = RC->getRawText(Context.getSourceManager());
            // Clean up comment markers
            doc = std::regex_replace(doc, std::regex("^\\/\\/\\/\\s*"), "");
            doc = std::regex_replace(doc, std::regex("^\\/\\/\\s*"), "");
        }
        return doc;
    }
};

// AST Consumer
class CppASTConsumer : public ASTConsumer {
public:
    explicit CppASTConsumer(ASTContext *Context) : Visitor(Context) {}

    virtual void HandleTranslationUnit(ASTContext &Context) {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
        json j = Visitor.getJSON();
        std::cout << j.dump(4) << std::endl;
    }

private:
    CppASTVisitor Visitor;
};

// Frontend Action
class CppFrontendAction : public ASTFrontendAction {
public:
    virtual std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) {
        return std::make_unique<CppASTConsumer>(&CI.getASTContext());
    }
};

// Main Function
int main(int argc, const char **argv) {
    // Parse command-line options
    CommonOptionsParser OptionsParser(argc, argv, llvm::cl::GeneralCategory);
    ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

    // Run the Clang Tool
    int result = Tool.run(newFrontendActionFactory<CppFrontendAction>().get());

    return result;
}

```

## scripts/css_inserter.js

```javascript
// scripts/css_inserter.js

const fs = require('fs');
const css = require('css');
const path = require('path');

// Read input from stdin
let inputChunks = [];
process.stdin.on('data', (chunk) => {
  inputChunks.push(chunk);
});

process.stdin.on('end', () => {
  const inputData = inputChunks.join('');

  let parsedInput;
  try {
    parsedInput = JSON.parse(inputData);
  } catch (e) {
    console.error('Error parsing input JSON:', e.message);
    process.exit(1);
  }

  const { code, documentation, language } = parsedInput;

  if (language.toLowerCase() !== 'css') {
    console.error('Unsupported language:', language);
    process.exit(1);
  }

  let ast;
  try {
    ast = css.parse(code, { source: 'input.css' });
  } catch (e) {
    console.error('Parsing error:', e.message);
    process.exit(1);
  }

  // Insert comments based on documentation
  if (documentation.rules) {
    documentation.rules.forEach(docRule => {
      const selectors = docRule.selectors;
      const docstring = docRule.docstring;

      ast.stylesheet.rules.forEach(rule => {
        if (rule.type === 'rule') {
          const ruleSelectors = rule.selectors;
          const isMatch = selectors.some(sel => ruleSelectors.includes(sel));
          if (isMatch && docstring) {
            // Insert comment before the rule
            if (!rule.comments) {
              rule.comments = [];
            }
            rule.comments.unshift(docstring);
          }
        }
      });
    });
  }

  // Stringify the modified AST
  const modifiedCSS = css.stringify(ast);
  console.log(modifiedCSS);
});

```

## scripts/css_parser.js

```javascript
// scripts/css_parser.js

const fs = require('fs');
const css = require('css');
const Ajv = require('ajv');
const path = require('path');

// Initialize AJV for JSON schema validation
const ajv = new Ajv({ allErrors: true, strict: false });

// Load the unified function_schema.json
const schemaPath = path.join(__dirname, '../schemas/function_schema.json');
const functionSchema = JSON.parse(fs.readFileSync(schemaPath, 'utf-8'));
const validate = ajv.compile(functionSchema);

// Read input from stdin
let inputChunks = [];
process.stdin.on('data', (chunk) => {
  inputChunks.push(chunk);
});

process.stdin.on('end', () => {
  const inputData = inputChunks.join('');

  let parsedInput;
  try {
    parsedInput = JSON.parse(inputData);
  } catch (e) {
    console.error('Error parsing input JSON:', e.message);
    process.exit(1);
  }

  const { code, documentation, language } = parsedInput;

  if (language.toLowerCase() !== 'css') {
    console.error('Unsupported language:', language);
    process.exit(1);
  }

  let ast;
  try {
    ast = css.parse(code, { source: 'input.css' });
  } catch (e) {
    console.error('Parsing error:', e.message);
    process.exit(1);
  }

  // Initialize the structure object
  const structure = {
    summary: "", // To be filled externally or manually
    changes_made: [], // To be filled externally or manually
    functions: [], // Not typically applicable for CSS
    classes: [], // Not typically applicable for CSS
    halstead: {
      volume: 0,
      difficulty: 0,
      effort: 0
    },
    maintainability_index: 0,
    variables: [], // CSS Variables (Custom Properties) can be mapped here
    constants: []  // Not typically applicable for CSS
  };

  ast.stylesheet.rules.forEach(rule => {
    if (rule.type === 'rule') {
      const selectors = rule.selectors;
      const declarations = rule.declarations.map(decl => ({
        property: decl.property,
        value: decl.value
      }));

      // Find documentation for this rule
      const doc = documentation.rules.find(r => {
        // Simple matching; can be enhanced
        return r.selectors.some(sel => selectors.includes(sel));
      });

      if (doc && doc.docstring) {
        // Insert comment before the rule
        rule.comments = [`${doc.docstring}`];
      }

      // Variables (Custom Properties) handling
      selectors.forEach(sel => {
        if (sel.startsWith('--')) { // CSS Variable
          const varName = sel;
          rule.declarations.forEach(decl => {
            if (decl.property === varName) {
              const variableInfo = {
                name: varName,
                type: "CSS Variable",
                description: "No description provided.",
                file: "Unknown", // Can be set if file info is available
                line: decl.position ? decl.position.start.line : 0,
                link: "Unknown", // Can be constructed based on repository
                example: decl.value,
                references: "No references."
              };
              structure.variables.push(variableInfo);
            }
          });
        }
      });
    }
  });

  // Validate the structure against the schema
  const valid = validate(structure);
  if (!valid) {
    console.error('Validation errors:', validate.errors);
    process.exit(1);
  }

  // Output the structure as JSON
  console.log(JSON.stringify(structure, null, 2));
});

```

## scripts/java_parser.js

```javascript
// scripts/java_parser.js

const fs = require('fs');
const javaParser = require('java-parser');
const Ajv = require('ajv');
const path = require('path');

// Initialize AJV for JSON schema validation
const ajv = new Ajv({ allErrors: true, strict: false });

// Load the unified function_schema.json
const schemaPath = path.join(__dirname, '../schemas/function_schema.json');
const functionSchema = JSON.parse(fs.readFileSync(schemaPath, 'utf-8'));
const validate = ajv.compile(functionSchema);

// Read input from stdin
let inputChunks = [];
process.stdin.on('data', (chunk) => {
  inputChunks.push(chunk);
});

process.stdin.on('end', () => {
  const inputData = inputChunks.join('');

  let parsedInput;
  try {
    parsedInput = JSON.parse(inputData);
  } catch (e) {
    console.error('Error parsing input JSON:', e.message);
    process.exit(1);
  }

  const { code, language } = parsedInput;

  if (language.toLowerCase() !== 'java') {
    console.error('Unsupported language:', language);
    process.exit(1);
  }

  let ast;
  try {
    ast = javaParser.parse(code);
  } catch (e) {
    console.error('Parsing error:', e.message);
    process.exit(1);
  }

  // Initialize the structure object
  const structure = {
    summary: "", // To be filled externally or manually
    changes_made: [], // To be filled externally or manually
    functions: [],
    classes: [],
    halstead: {
      volume: 0,
      difficulty: 0,
      effort: 0
    },
    maintainability_index: 0,
    variables: [],
    constants: []
  };

  // Helper function to extract docstrings (comments)
  function getDocstring(node) {
    if (node.documentation) {
      return node.documentation.trim();
    }
    return "";
  }

  // Traverse the AST to extract classes and functions
  const classes = ast.children.filter(child => child.node === 'ClassDeclaration');

  classes.forEach(cls => {
    const classInfo = {
      name: cls.name.identifier,
      docstring: getDocstring(cls),
      methods: []
    };

    cls.body.body.forEach(member => {
      if (member.node === 'MethodDeclaration') {
        const methodInfo = {
          name: member.name.identifier,
          docstring: getDocstring(member),
          args: member.parameters.map(param => param.name.identifier),
          async: false, // Java does not have async methods; can be extended if using CompletableFuture or similar
          type: 'instance', // Default to instance method; can be extended based on modifiers
          complexity: 1 // Placeholder: Cyclomatic complexity calculation requires further implementation
        };

        // Determine if the method is static
        if (member.modifiers && member.modifiers.includes('static')) {
          methodInfo.type = 'static';
        }

        classInfo.methods.push(methodInfo);
      } else if (member.node === 'FieldDeclaration') {
        member.declarators.forEach(decl => {
          const variableInfo = {
            name: decl.id.identifier,
            type: member.type.name.identifier,
            description: getDocstring(member),
            file: "Unknown", // File information can be added if available
            line: decl.position.start.line,
            link: "Unknown", // Link can be constructed based on repository URL
            example: "No example provided.",
            references: "No references."
          };

          // Determine if the field is a constant (e.g., final)
          if (member.modifiers && member.modifiers.includes('final')) {
            structure.constants.push(variableInfo);
          } else {
            structure.variables.push(variableInfo);
          }
        });
      }
    });

    structure.classes.push(classInfo);
  });

  // Traverse the AST to extract standalone functions (if any)
  // Note: Java primarily uses classes, but static methods can be considered standalone
  classes.forEach(cls => {
    cls.body.body.forEach(member => {
      if (member.node === 'MethodDeclaration' && member.modifiers && member.modifiers.includes('static')) {
        const functionInfo = {
          name: member.name.identifier,
          docstring: getDocstring(member),
          args: member.parameters.map(param => param.name.identifier),
          async: false, // Java does not support async directly
          complexity: 1 // Placeholder for cyclomatic complexity
        };
        structure.functions.push(functionInfo);
      }
    });
  });

  // Placeholder for Halstead metrics and Maintainability Index
  // These require detailed analysis and are not implemented here
  // They can be integrated using additional tools or libraries

  // Validate the structure against the schema
  const valid = validate(structure);
  if (!valid) {
    console.error('Validation errors:', validate.errors);
    process.exit(1);
  }

  // Output the structure as JSON
  console.log(JSON.stringify(structure, null, 2));
});

```

