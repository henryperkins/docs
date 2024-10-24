# process_manager.py
import os
import asyncio
import aiohttp
import json
import logging
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
from write_documentation_report import write_documentation_report, generate_markdown_content, generate_documentation_prompt, sanitize_filename

logger = logging.getLogger(__name__)


class DocumentationRequest(BaseModel):
    project_id: str
    file_paths: List[str]
    skip_types: Set[str] = set()
    project_info: str
    style_guidelines: str


class DocumentationResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    results: Optional[Dict[str, Any]] = None


class DocumentationProcessManager:

    def __init__(
        self,
        repo_root: str,
        output_dir: str,
        azure_config: Dict[str, Any],
        function_schema: Dict[str, Any],
        max_concurrency: int = 5,
    ):
        self.repo_root = Path(repo_root)
        self.output_dir = Path(output_dir)
        self.azure_config = azure_config
        self.function_schema = function_schema
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.context_manager = ContextManager()
        self.active_tasks: Dict[str, Dict[str, Any]] = {}

    async def process_files(
        self,
        task_id: str,
        file_paths: List[str],
        skip_types: Set[str],
        project_info: str,
        style_guidelines: str,
    ) -> Dict[str, Any]:
        """Processes files, tracks progress, and updates frontend data."""

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
                    if should_process_file(file_path, skip_types)  # Call should_process_file from utils
                ]

                # Handle skipped files *before* awaiting tasks
                for file_path in file_paths:
                    if not should_process_file(file_path, skip_types):
                        self.active_tasks[task_id]["results"]["skipped"].append(file_path)

                total_files = len(file_paths)
                completed_files = 0

                for future in asyncio.as_completed(tasks):
                    completed_files += 1
                    try:
                        result = await future
                        if result:
                            self.active_tasks[task_id]["results"]["successful"].append(result)
                        else:
                            self.active_tasks[task_id]["results"]["failed"].append(future.exception())
                    except Exception as e:
                        logger.error(f"Task failed: {e}", exc_info=True)
                        self.active_tasks[task_id]["results"]["failed"].append(e)

                    progress = (completed_files / total_files) * 100 if total_files > 0 else 100
                    self.active_tasks[task_id]["progress"] = progress
                    await self._update_frontend_data(task_id, completed_files, total_files)

            self.active_tasks[task_id]["status"] = "completed"
            return self.active_tasks[task_id]["results"]

        except Exception as e:
            logger.error(f"Error in process_files: {e}", exc_info=True)
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["results"]["failed"].append(e)
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
        """Processes a single file."""
        try:
            async with self.semaphore:
                file_context = self.context_manager.get_relevant_context(file_path) if self.context_manager.context_entries else ""
                enhanced_project_info = f"{project_info}\n\nFile Context:\n{file_context}"

                documentation = await process_file(
                    session=session,
                    file_path=file_path,
                    skip_types=skip_types,
                    semaphore=self.semaphore,
                    deployment_name=self.azure_config["AZURE_OPENAI_DEPLOYMENT"],
                    function_schema=self.function_schema,
                    repo_root=str(self.repo_root),
                    project_info=enhanced_project_info,
                    style_guidelines=style_guidelines,
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

                    await write_documentation_report( # Correct call to write_documentation_report
                        documentation=documentation,
                        language=documentation.get("language", ""),
                        file_path=str(relative_filepath),  # Pass relative file path
                        repo_root=str(self.repo_root),
                        output_dir=str(self.output_dir / task_id),
                        project_id=task_id,
                    )
                    return str(relative_filepath)

            return None

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

        async with aiofiles.open(status_file, "w") as f:
            await f.write(json.dumps(status_data, indent=2))

        summary_file = output_dir / "summary.json"

        successful_docs = []
        for result in self.active_tasks[task_id]["results"]["successful"]:
            try:
                with open(self.output_dir / task_id / f"{sanitize_filename(Path(result).stem)}.json", "r") as f:
                    successful_docs.append(json.load(f))
            except Exception as e:
                logger.error(f"Error loading successful documentation: {e}", exc_info=True)


        summary_data = {
            "files": [
                {
                    "path": result if isinstance(result, str) else "unknown",
                    "status": "success" if isinstance(result, str) else "failed",
                    "documentation": next((doc for doc in successful_docs if doc.get("file_path") == result), None) if isinstance(result, str) else None,
                }
                for result in self.active_tasks[task_id]["results"]["successful"] + self.active_tasks[task_id]["results"]["failed"]
            ],
            "metrics": calculate_project_metrics(successful_docs),
        }

        async with aiofiles.open(summary_file, "w") as f:
            await f.write(json.dumps(summary_data, indent=2))


    def _should_process(self, file_path: str, skip_types: Set[str]) -> bool:
        """Checks if a file should be processed."""
        return should_process_file(file_path, skip_types)


# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development - adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = DocumentationProcessManager(
    repo_root=os.getenv("REPO_ROOT", ""),
    output_dir=os.getenv("OUTPUT_DIR", "documentation"),
    azure_config={
        "AZURE_OPENAI_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "API_VERSION": os.getenv("API_VERSION"),
    },
    function_schema=load_function_schema("schemas/function_schema.json"),
)

@app.post("/api/documentation/generate")
async def generate_documentation(
    request: DocumentationRequest, background_tasks: BackgroundTasks
) -> DocumentationResponse:
    """Endpoint to trigger documentation generation."""
    task_id = str(uuid.uuid4())
    background_tasks.add_task(
        manager.process_files,
        task_id=task_id,
        file_paths=request.file_paths,
        skip_types=request.skip_types,
        project_info=request.project_info,
        style_guidelines=request.style_guidelines,
    )
    return DocumentationResponse(task_id=task_id, status="started", progress=0.0)


@app.get("/api/documentation/status/{task_id}")
async def get_documentation_status(task_id: str) -> DocumentationResponse:
    """Endpoint to check documentation generation status."""
    if (task_info := manager.active_tasks.get(task_id)) is None:  # Use walrus operator for cleaner code
        raise HTTPException(status_code=404, detail="Task not found")

    return DocumentationResponse(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info["progress"],
        results=task_info.get("results"),
    )


@app.get("/api/documentation/{project_id}")
async def get_documentation(project_id: str, file_path: Optional[str] = None):
    """Endpoint to retrieve generated documentation."""
    try:
        output_dir = Path("documentation") / project_id

        if file_path:
            doc_file = output_dir / f"{sanitize_filename(Path(file_path).stem)}.json"
        else:
            doc_file = output_dir / "summary.json"

        if not doc_file.exists():
            raise HTTPException(status_code=404, detail="Documentation not found")

        async with aiofiles.open(doc_file, "r", encoding="utf-8") as f:
            content = await f.read()
            return JSONResponse(content=json.loads(content))

    except HTTPException:
        raise  # Re-raise HTTPExceptions
    except Exception as e:
        logger.error(f"Error fetching documentation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/{project_id}/{file_path}") # Corrected path
async def get_code_content(project_id: str, file_path: str):
    """Endpoint to retrieve code content for a given file."""
    try:
        # Construct the full file path - use REPO_ROOT from environment
        base_dir = Path(os.getenv("REPO_ROOT", "")) / project_id
        full_file_path = base_dir / file_path

        if not full_file_path.exists() or not full_file_path.is_file():
            raise HTTPException(status_code=404, detail="Code not found")

        async with aiofiles.open(full_file_path, "r", encoding="utf-8") as f:
            code_content = await f.read()

        return JSONResponse(content={"code": code_content}) # Return as JSONResponse

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading code: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
