# process_manager.py
import os
import asyncio
import aiofiles
import aiohttp
import json
import logging
import sys
import uuid
from typing import List, Set, Dict, Any, Optional, Tuple
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils import load_function_schema, calculate_project_metrics, should_process_file
from file_handlers import process_file
from context_manager import ContextManager
from write_documentation_report import (
    write_documentation_report, 
    generate_markdown_content, 
    generate_documentation_prompt, 
    sanitize_filename
)

logger = logging.getLogger(__name__)


class DocumentationRequest(BaseModel):
    project_id: str
    file_paths: List[str]
    skip_types: Set[str] = set()
    project_info: str
    style_guidelines: str
    safe_mode: bool  # Add safe_mode to the request model


class DocumentationResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    results: Optional[Dict[str, Any]] = None


class DocumentationProcessManager:
    """
    Manages the documentation generation process for multiple files concurrently.

    Attributes:
        repo_root (Path): Root directory of the repository.
        output_dir (Path): Directory where documentation will be saved.
        azure_config (Dict[str, Any]): Configuration details for Azure OpenAI.
        function_schema (Dict[str, Any]): Schema for function documentation.
        semaphore (asyncio.Semaphore): Semaphore to control concurrency.
        context_manager (ContextManager): Manages contextual information.
        active_tasks (Dict[str, Dict[str, Any]]): Tracks active documentation tasks.
        safe_mode (bool): Indicates if safe mode is enabled.
    """

    def __init__(
        self,
        repo_root: str,
        output_dir: str,
        azure_config: Dict[str, Any],
        function_schema: Dict[str, Any],
        max_concurrency: int = 5,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.azure_config = azure_config
        self.function_schema = function_schema
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.context_manager = ContextManager()
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.safe_mode = False  # Initialize safe_mode

    async def process_files(
        self,
        task_id: str,
        file_paths: List[str],
        skip_types: Set[str],
        project_info: str,
        style_guidelines: str,
        safe_mode: bool,
    ) -> Dict[str, Any]:
        """Processes files, tracks progress, and updates frontend data."""

        self.safe_mode = safe_mode  # Store safe_mode in the instance

        self.active_tasks[task_id] = {
            "status": "running",
            "progress": 0.0,
            "results": {"successful": [], "failed": [], "skipped": []},
        }

        try:
            async with aiohttp.ClientSession() as session:
                tasks = [
                    asyncio.create_task(
                        self._process_single_file(
                            session=session,
                            file_path=file_path,
                            skip_types=skip_types,
                            project_info=project_info,
                            style_guidelines=style_guidelines,
                            task_id=task_id,
                        )
                    )
                    for file_path in file_paths
                    if self._should_process(file_path, skip_types)
                ]

                # Handle skipped files
                skipped_files = [
                    file_path for file_path in file_paths
                    if not self._should_process(file_path, skip_types)
                ]
                self.active_tasks[task_id]["results"]["skipped"].extend(skipped_files)
                total_files = len(file_paths)
                processed_files = 0  # Tracks processed files (successful + failed)

                for future in asyncio.as_completed(tasks):
                    try:
                        result = await future  # Await the future
                        if result:
                            self.active_tasks[task_id]["results"]["successful"].append(result)
                        else:
                            self.active_tasks[task_id]["results"]["failed"].append("Processing failed")
                    except Exception as e:
                        logger.error(f"Task failed: {e}", exc_info=True)
                        self.active_tasks[task_id]["results"]["failed"].append(str(e))

                    processed_files += 1
                    progress = (processed_files / total_files) * 100 if total_files > 0 else 100
                    self.active_tasks[task_id]["progress"] = progress
                    await self._update_frontend_data(task_id, processed_files, total_files)

            self.active_tasks[task_id]["status"] = "completed"
            return self.active_tasks[task_id]["results"]

        except Exception as e:
            logger.error(f"Error in process_files: {e}", exc_info=True)
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["results"]["failed"].append(str(e))
            return self.active_tasks[task_id]["results"]

    async def _process_single_file(
        self,
        session: aiohttp.ClientSession,
        file_path: str,
        skip_types: Set[str],
        project_info: str,
        style_guidelines: str,
        task_id: str,
    ) -> Optional[str]:
        """Processes a single file and generates documentation."""
        try:
            async with self.semaphore:
                file_context = self.context_manager.get_relevant_context(file_path) if self.context_manager.context_entries else ""
                enhanced_project_info = f"{project_info}\n\nFile Context:\n{file_context}"

                documentation = await process_file(
                    session=session,
                    file_path=file_path,
                    skip_types=skip_types,
                    semaphore=self.semaphore,  # Pass the semaphore
                    deployment_name=self.azure_config["AZURE_OPENAI_DEPLOYMENT"],
                    function_schema=self.function_schema,
                    repo_root=str(self.repo_root),
                    project_info=enhanced_project_info,
                    style_guidelines=style_guidelines,
                    safe_mode=self.safe_mode,  # Pass safe_mode here
                    azure_api_key=self.azure_config["AZURE_OPENAI_API_KEY"],
                    azure_endpoint=self.azure_config["AZURE_OPENAI_ENDPOINT"],
                    azure_api_version=self.azure_config["API_VERSION"],
                    output_dir=str(self.output_dir / task_id),
                    project_id=task_id,
                )

                if documentation:
                    self.context_manager.add_context(
                        f"File: {file_path}\nDocumentation Summary: {documentation.get('summary', '')}"
                    )
                    relative_filepath = Path(file_path).relative_to(self.repo_root)
                    return str(relative_filepath)  # Return the relative file path

            return None  # Return None if processing fails

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            return None

    async def _update_frontend_data(self, task_id: str, completed: int, total: int):
        """Updates frontend data with progress and results."""

        output_dir = self.output_dir / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        status_file = output_dir / "status.json"
        status_data = {
            "progress": (completed / total) * 100 if total > 0 else 100,
            "status": self.active_tasks[task_id]["status"],
            "completed": completed,
            "total": total,
            "successful": len(self.active_tasks[task_id]["results"]["successful"]),
            "failed": len(self.active_tasks[task_id]["results"]["failed"]),
            "skipped": len(self.active_tasks[task_id]["results"]["skipped"]),
        }

        try:
            async with aiofiles.open(status_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(status_data, indent=2))
            logger.debug(f"Updated status file: {status_file}")
        except Exception as e:
            logger.error(f"Error writing status file: {e}", exc_info=True)

        # Aggregate successful documentation
        successful_docs = []
        for result in self.active_tasks[task_id]["results"]["successful"]:
            doc_path = self.output_dir / task_id / f"{sanitize_filename(Path(result).stem)}.json"
            if doc_path.exists():
                try:
                    async with aiofiles.open(doc_path, "r", encoding="utf-8") as f:
                        content = await f.read()
                        successful_docs.append(json.loads(content))
                except Exception as e:
                    logger.error(f"Error loading successful documentation from {doc_path}: {e}", exc_info=True)
            else:
                logger.warning(f"Documentation file does not exist: {doc_path}")

        # Calculate project metrics based on successful documentation
        metrics = calculate_project_metrics(successful_docs)

        summary_data = {
            "files": [
                {
                    "path": result if isinstance(result, str) else "unknown",
                    "status": "success" if isinstance(result, str) else "failed",
                    "documentation": next((doc for doc in successful_docs if doc.get("file_path") == result), None) if isinstance(result, str) else None,
                }
                for result in self.active_tasks[task_id]["results"]["successful"] + self.active_tasks[task_id]["results"]["failed"]
            ],
            "metrics": metrics,
        }

        summary_file = output_dir / "summary.json"

        try:
            async with aiofiles.open(summary_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(summary_data, indent=2))
            logger.debug(f"Updated summary file: {summary_file}")
        except Exception as e:
            logger.error(f"Error writing summary file: {e}", exc_info=True)

    def _should_process(self, file_path: str, skip_types: Set[str]) -> bool:
        """Checks if a file should be processed."""
        should = should_process_file(file_path, skip_types)
        logger.debug(f"Should process '{file_path}': {should}")
        return should


# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development - adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DocumentationProcessManager with environment variables
repo_root_env = os.getenv("REPO_ROOT")
output_dir_env = os.getenv("OUTPUT_DIR", "documentation")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("API_VERSION")

if not all([repo_root_env, azure_deployment, azure_api_key, azure_endpoint, api_version]):
    logger.critical("Missing required environment variables. Please set REPO_ROOT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and API_VERSION.")
    sys.exit(1)

function_schema_path = os.getenv("FUNCTION_SCHEMA_PATH", "schemas/function_schema.json")
function_schema = load_function_schema(function_schema_path)

manager = DocumentationProcessManager(
    repo_root=repo_root_env,
    output_dir=output_dir_env,
    azure_config={
        "AZURE_OPENAI_DEPLOYMENT": azure_deployment,
        "AZURE_OPENAI_API_KEY": azure_api_key,
        "AZURE_OPENAI_ENDPOINT": azure_endpoint,
        "API_VERSION": api_version,
    },
    function_schema=function_schema,
    max_concurrency=5,  # You can make this configurable if needed
)


@app.post("/api/documentation/generate", response_model=DocumentationResponse)
async def generate_documentation(
    request: DocumentationRequest, background_tasks: BackgroundTasks
) -> DocumentationResponse:
    """
    Endpoint to trigger documentation generation.

    Args:
        request (DocumentationRequest): The documentation generation request payload.
        background_tasks (BackgroundTasks): FastAPI background tasks manager.

    Returns:
        DocumentationResponse: Response containing task ID, status, and progress.
    """
    task_id = str(uuid.uuid4())
    logger.info(f"Received documentation generation request. Task ID: {task_id}")
    background_tasks.add_task(
        manager.process_files,
        task_id=task_id,
        file_paths=request.file_paths,
        skip_types=request.skip_types,
        project_info=request.project_info,
        style_guidelines=request.style_guidelines,
        safe_mode=request.safe_mode,
    )
    return DocumentationResponse(task_id=task_id, status="started", progress=0.0)


@app.get("/api/documentation/status/{task_id}", response_model=DocumentationResponse)
async def get_documentation_status(task_id: str) -> DocumentationResponse:
    """
    Endpoint to check documentation generation status.

    Args:
        task_id (str): The unique task ID.

    Returns:
        DocumentationResponse: Current status and progress of the task.
    """
    task_info = manager.active_tasks.get(task_id)
    if not task_info:
        logger.warning(f"Status request for unknown Task ID: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")

    return DocumentationResponse(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info["progress"],
        results=task_info.get("results"),
    )


@app.get("/api/documentation/{project_id}", response_class=JSONResponse)
async def get_documentation(project_id: str, file_path: Optional[str] = None):
    """
    Endpoint to retrieve generated documentation.

    Args:
        project_id (str): The unique project ID.
        file_path (Optional[str]): Specific file path to retrieve documentation for.

    Returns:
        JSONResponse: The requested documentation content.
    """
    try:
        sanitized_project_id = sanitize_filename(project_id)
        output_dir = Path("documentation") / sanitized_project_id

        if file_path:
            sanitized_file_path = sanitize_filename(file_path)
            doc_file = output_dir / f"{sanitized_file_path}.json"
        else:
            doc_file = output_dir / "summary.json"

        if not doc_file.exists():
            logger.warning(f"Documentation file not found: {doc_file}")
            raise HTTPException(status_code=404, detail="Documentation not found")

        async with aiofiles.open(doc_file, "r", encoding="utf-8") as f:
            content = await f.read()
            return JSONResponse(content=json.loads(content))

    except HTTPException:
        raise  # Re-raise HTTPExceptions
    except Exception as e:
        logger.error(f"Error fetching documentation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/{project_id}/{file_path}", response_class=JSONResponse)
async def get_code_content(project_id: str, file_path: str):
    """
    Endpoint to retrieve code content for a given file.

    Args:
        project_id (str): The unique project ID.
        file_path (str): The relative file path within the project.

    Returns:
        JSONResponse: The code content of the specified file.
    """
    try:
        sanitized_project_id = sanitize_filename(project_id)
        sanitized_file_path = sanitize_filename(file_path)
        base_dir = Path(os.getenv("REPO_ROOT", "")).resolve() / sanitized_project_id
        full_file_path = (base_dir / sanitized_file_path).resolve()

        # Ensure the file is within the repo_root to prevent directory traversal
        if not full_file_path.is_file() or not full_file_path.is_relative_to(self.repo_root):
            logger.warning(f"Invalid code file request: {full_file_path}")
            raise HTTPException(status_code=404, detail="Code not found")

        async with aiofiles.open(full_file_path, "r", encoding="utf-8") as f:
            code_content = await f.read()

        return JSONResponse(content={"code": code_content})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading code: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
