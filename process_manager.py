"""
process_manager.py

Documentation generation process manager with integrated token, chunk, and context management.
Coordinates file processing, model interactions, and documentation generation.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, validator
import threading
import uuid
import aiohttp

from provider_config import load_provider_configs, ProviderConfig
from token_utils import TokenManager
from chunk import ChunkManager
from dependency_analyzer import DependencyAnalyzer
from context import HierarchicalContextManager
from utils import setup_logging, load_json_schema, handle_api_error
from metrics import MetricsManager

logger = logging.getLogger(__name__)

class DocumentationRequest(BaseModel):
    """API request model."""
    file_paths: List[str]
    skip_types: Optional[List[str]] = []
    project_info: Optional[str] = ""
    style_guidelines: Optional[str] = ""
    safe_mode: Optional[bool] = False
    project_id: str
    provider: str = "azure"
    max_concurrency: Optional[int] = 5
    priority: Optional[str] = "normal"
    callback_url: Optional[str] = None

    @validator("provider")
    def validate_provider(cls, v):
        valid_providers = {"azure", "gemini", "openai"}
        if v not in valid_providers:
            raise ValueError(f"Provider must be one of {valid_providers}")
        return v

    @validator("priority")
    def validate_priority(cls, v):
        if v not in {"low", "normal", "high"}:
            raise ValueError("Priority must be low, normal, or high")
        return v

class DocumentationResponse(BaseModel):
    """API response model."""
    task_id: str
    status: str
    progress: float
    results: Optional[Dict[str, Any]] = None
    errors: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    estimated_completion: Optional[datetime] = None

_manager_instance = None
_manager_lock = threading.Lock()

def get_manager_instance(metrics_manager: MetricsManager) -> 'DocumentationProcessManager':
    """Gets or creates DocumentationProcessManager instance."""
    global _manager_instance
    with _manager_lock:
        if _manager_instance is None:
            provider_configs = load_provider_configs()
            _manager_instance = DocumentationProcessManager(
                repo_root=os.getenv("REPO_ROOT", "."),
                output_dir=os.getenv("OUTPUT_DIR", "./docs"),
                provider_configs=provider_configs,
                max_concurrency=int(os.getenv("MAX_CONCURRENCY", 5)),
                cache_dir=os.getenv("CACHE_DIR"),
                metrics_manager=metrics_manager
            )
    return _manager_instance

class APIHandler:
    """Handles API interactions with AI providers."""

    def __init__(self, config: ProviderConfig, session: aiohttp.ClientSession):
        self.config = config
        self.session = session

    async def call_provider_api(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Calls the provider API with retry and timeout logic."""
        retries = self.config.max_retries
        delay = self.config.retry_delay

        for attempt in range(retries):
            try:
                async with self.session.post(endpoint, json=payload, timeout=self.config.timeout) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientError as e:
                logger.warning(f"API call failed on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"API call failed after {retries} attempts: {e}")
                    raise

class DocumentationProcessManager:
    """Manages the documentation generation process using AI models."""

    def __init__(
        self,
        repo_root: str,
        output_dir: str,
        provider_configs: Dict[str, ProviderConfig],
        max_concurrency: int = 5,
        cache_dir: Optional[str] = None,
        metrics_manager: MetricsManager = None
    ):
        self.repo_root = Path(repo_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.provider_configs = provider_configs
        self.max_concurrency = max_concurrency
        self.metrics_manager = metrics_manager

        # Initialize managers
        self.chunk_manager = ChunkManager(max_tokens=4096, overlap=200)
        self.context_manager = HierarchicalContextManager(cache_dir=cache_dir)

        # Task tracking
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._task_status: Dict[str, Dict[str, Any]] = {}

    async def process_files(self, request: DocumentationRequest, task_id: str) -> Dict[str, Any]:
        """
        Processes files to generate documentation.

        Args:
            request: Documentation request containing file paths and other parameters.
            task_id: Unique identifier for the task.

        Returns:
            A dictionary containing the processing results and metrics.
        """
        results = []
        self._task_status[task_id] = {"status": "in_progress", "progress": 0.0}

        try:
            async with aiohttp.ClientSession() as session:
                api_handler = APIHandler(self.provider_configs[request.provider], session)

                total_files = len(request.file_paths)
                completed_files = 0

                for file_path in request.file_paths:
                    try:
                        with open(file_path, 'r') as f:
                            code = f.read()

                        # Create chunks using ChunkManager
                        chunks = self.chunk_manager.create_chunks(code, file_path, language="python")

                        # Add chunks to context manager
                        for chunk in chunks:
                            await self.context_manager.add_code_chunk(chunk)

                        # Analyze dependencies using DependencyAnalyzer
                        analyzer = DependencyAnalyzer()
                        dependencies = analyzer.analyze(code)

                        # Use TokenManager to count tokens
                        token_result = TokenManager.count_tokens(code)
                        logger.info(f"Token count for {file_path}: {token_result.token_count}")

                        # Example API call
                        api_response = await api_handler.call_provider_api(
                            endpoint=self.provider_configs[request.provider].endpoint,
                            payload={"data": "example"}
                        )
                        logger.info(f"API response for {file_path}: {api_response}")

                        results.append({
                            "file_path": file_path,
                            "success": True,
                            "dependencies": dependencies,
                            "token_count": token_result.token_count,
                            "api_response": api_response
                        })

                        # Record successful processing
                        processing_time = (datetime.now() - start_time).total_seconds()
                        self.metrics_manager.record_file_processing(success=True, processing_time=processing_time)

                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        results.append({
                            "file_path": file_path,
                            "success": False,
                            "error": str(e)
                        })
                        processing_time = (datetime.now() - start_time).total_seconds()
                        self.metrics_manager.record_file_processing(success=False, processing_time=processing_time, error_type=str(e))

                    # Update progress
                    completed_files += 1
                    progress = (completed_files / total_files) * 100
                    self._task_status[task_id]["progress"] = progress

        except Exception as e:
            logger.error(f"Critical error in process_files: {e}")
            self._task_status[task_id] = {"status": "failed", "progress": 100.0}
            raise

        # Finalize task status
        self._task_status[task_id] = {
            "status": "completed",
            "progress": 100.0,
            "results": results
        }

        # Remove task from active tasks
        self._active_tasks.pop(task_id, None)

        return self._task_status[task_id]

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the status of a specific task."""
        return self._task_status.get(task_id)

    async def cancel_task(self, task_id: str) -> None:
        """Cancels a running task."""
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            task.cancel()
            self._task_status[task_id] = {
                "status": "cancelled",
                "progress": 100.0
            }
            logger.info(f"Task {task_id} has been cancelled.")
        else:
            logger.warning(f"Task {task_id} not found or already completed.")

    async def cleanup(self) -> None:
        """Cleans up resources."""
        try:
            # Cancel active tasks
            for task_id, task in self._active_tasks.items():
                task.cancel()
                self._task_status[task_id] = {
                    "status": "cancelled",
                    "progress": 100.0
                }

            # Clear context manager
            await self.context_manager.clear_context()

            # Clear task tracking
            self._active_tasks.clear()
            self._task_status.clear()

            logger.info("Cleanup completed successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# FastAPI app setup
app = FastAPI(title="Documentation Generator API")

@app.post("/api/documentation/generate", response_model=DocumentationResponse)
async def generate_documentation(
    request: DocumentationRequest,
    background_tasks: BackgroundTasks
) -> DocumentationResponse:
    try:
        task_id = str(uuid.uuid4())
        metrics_manager = MetricsManager()
        manager = get_manager_instance(metrics_manager)

        background_tasks.add_task(
            manager.process_files,
            request=request,
            task_id=task_id
        )

        return DocumentationResponse(
            task_id=task_id,
            status="started",
            progress=0.0,
            estimated_completion=None  # This can be calculated based on some logic
        )

    except Exception as e:
        logger.error(f"Error in generate_documentation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Documentation generation failed: {str(e)}"
        )

@app.get("/api/documentation/status/{task_id}", response_model=DocumentationResponse)
async def get_status(task_id: str) -> Dict[str, Any]:
    try:
        manager = get_manager_instance(MetricsManager())
        status = await manager.get_task_status(task_id)
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )
        return status

    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn

    setup_logging(
        log_file=os.getenv("LOG_FILE"),
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("DEBUG", False))
    )
