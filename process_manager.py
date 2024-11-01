"""
process_manager.py

Enhanced documentation generation process manager with improved provider handling,
metrics tracking, and error management. Coordinates file processing, model
interactions, and documentation generation across multiple AI providers.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, validator
import threading
import uuid

from utils import (
    FileHandler,
    setup_logging,
    get_all_file_paths,
    load_json_schema,
    handle_api_error  # New utility function for error handling
)
from file_handlers import (
    FileProcessor,
    APIHandler,
    ProcessingResult,
    ChunkManager,
    HierarchicalContextManager
)
from metrics_combined import (
    ProviderMetrics,
    ProcessingMetrics
)
from write_documentation_report import DocumentationGenerator

logger = logging.getLogger(__name__)

class ProviderConfig(BaseModel):
    """Enhanced provider configuration with validation."""
    name: str
    endpoint: str
    api_key: str
    deployment_name: Optional[str] = None
    api_version: Optional[str] = None
    model_name: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_enabled: bool = True
    timeout: float = 30.0
    chunk_overlap: int = 200  # Token overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size in tokens
    max_parallel_chunks: int = 3

    @validator("temperature")
    def validate_temperature(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v

    @validator("max_tokens")
    def validate_max_tokens(cls, v):
        if v < 1 or v > 8192:
            raise ValueError("max_tokens must be between 1 and 8192")
        return v

class DocumentationRequest(BaseModel):
    """Enhanced API request model."""
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
    """Enhanced API response model."""
    task_id: str
    status: str
    progress: float
    results: Optional[Dict[str, Any]] = None
    errors: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    estimated_completion: Optional[datetime] = None

_manager_instance = None
_manager_lock = threading.Lock()

def get_manager_instance() -> 'DocumentationProcessManager':
    """Gets or creates DocumentationProcessManager instance."""
    global _manager_instance
    with _manager_lock:
        if _manager_instance is None:
            # Load provider configurations
            provider_configs = {
                name: ProviderConfig(**config)
                for name, config in load_provider_configs().items()
            }

            _manager_instance = DocumentationProcessManager(
                repo_root=os.getenv("REPO_ROOT", "."),
                output_dir=os.getenv("OUTPUT_DIR", "./docs"),
                provider_configs=provider_configs,
                max_concurrency=int(os.getenv("MAX_CONCURRENCY", 5)),
                cache_dir=os.getenv("CACHE_DIR")
            )

    return _manager_instance

def load_provider_configs() -> Dict[str, Dict[str, Any]]:
    """Loads provider configurations from environment/files."""
    import os
    import yaml
    import json
    from pathlib import Path

    config_file = os.getenv('PROVIDER_CONFIG_FILE')

    if config_file:
        config_path = Path(config_file)
        if not config_path.is_file():
            raise FileNotFoundError(f"Provider config file not found: {config_file}")

        # Try to read the file
        try:
            with open(config_path, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config = yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    config = json.load(f)
                else:
                    raise ValueError("Unsupported config file format. Must be .yaml or .json")
        except Exception as e:
            raise ValueError(f"Failed to load provider config file: {e}")

    else:
        # Use default configuration
        config = {
            "azure": {
                "name": "azure",
                "endpoint": os.getenv("AZURE_ENDPOINT", "https://your-azure-endpoint.com"),
                "api_key": os.getenv("AZURE_API_KEY", "your-azure-api-key"),
                "deployment_name": os.getenv("AZURE_DEPLOYMENT_NAME", "your-deployment-name"),
                "api_version": os.getenv("AZURE_API_VERSION", "2023-01-01"),
                "model_name": os.getenv("AZURE_MODEL_NAME", "gpt-3"),
                "max_tokens": int(os.getenv("AZURE_MAX_TOKENS", "4096")),
                "temperature": float(os.getenv("AZURE_TEMPERATURE", "0.7")),
                "max_retries": int(os.getenv("AZURE_MAX_RETRIES", "3")),
                "retry_delay": float(os.getenv("AZURE_RETRY_DELAY", "1.0")),
                "cache_enabled": os.getenv("AZURE_CACHE_ENABLED", "True") == "True",
                "timeout": float(os.getenv("AZURE_TIMEOUT", "30.0")),
                "chunk_overlap": int(os.getenv("AZURE_CHUNK_OVERLAP", "200")),
                "min_chunk_size": int(os.getenv("AZURE_MIN_CHUNK_SIZE", "100")),
                "max_parallel_chunks": int(os.getenv("AZURE_MAX_PARALLEL_CHUNKS", "3"))
            },
            "gemini": {
                "name": "gemini",
                "endpoint": os.getenv("GEMINI_ENDPOINT", "https://your-gemini-endpoint.com"),
                "api_key": os.getenv("GEMINI_API_KEY", "your-gemini-api-key"),
                "model_name": os.getenv("GEMINI_MODEL_NAME", "gemini-model"),
                "max_tokens": int(os.getenv("GEMINI_MAX_TOKENS", "4096")),
                "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
                "max_retries": int(os.getenv("GEMINI_MAX_RETRIES", "3")),
                "retry_delay": float(os.getenv("GEMINI_RETRY_DELAY", "1.0")),
                "cache_enabled": os.getenv("GEMINI_CACHE_ENABLED", "True") == "True",
                "timeout": float(os.getenv("GEMINI_TIMEOUT", "30.0")),
                "chunk_overlap": int(os.getenv("GEMINI_CHUNK_OVERLAP", "200")),
                "min_chunk_size": int(os.getenv("GEMINI_MIN_CHUNK_SIZE", "100")),
                "max_parallel_chunks": int(os.getenv("GEMINI_MAX_PARALLEL_CHUNKS", "3"))
            },
            "openai": {
                "name": "openai",
                "endpoint": os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1/engines"),
                "api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key"),
                "model_name": os.getenv("OPENAI_MODEL_NAME", "gpt-3"),
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                "max_retries": int(os.getenv("OPENAI_MAX_RETRIES", "3")),
                "retry_delay": float(os.getenv("OPENAI_RETRY_DELAY", "1.0")),
                "cache_enabled": os.getenv("OPENAI_CACHE_ENABLED", "True") == "True",
                "timeout": float(os.getenv("OPENAI_TIMEOUT", "30.0")),
                "chunk_overlap": int(os.getenv("OPENAI_CHUNK_OVERLAP", "200")),
                "min_chunk_size": int(os.getenv("OPENAI_MIN_CHUNK_SIZE", "100")),
                "max_parallel_chunks": int(os.getenv("OPENAI_MAX_PARALLEL_CHUNKS", "3"))
            }
        }

    # Return the configuration
    return config

def calculate_estimated_completion(
    request: DocumentationRequest
) -> datetime:
    """Calculates estimated completion time."""
    import os
    from datetime import datetime, timedelta

    # For simplicity, we can assume that processing each file takes some base time plus some time proportional to its size.

    base_time_per_file = 5  # seconds
    time_per_kb = 0.1  # seconds per kilobyte

    total_time = 0.0

    for file_path in request.file_paths:
        try:
            file_size = os.path.getsize(file_path)  # in bytes
            file_time = base_time_per_file + (file_size / 1024) * time_per_kb
            total_time += file_time
        except Exception as e:
            # If file size can't be determined, use default time
            total_time += base_time_per_file

    estimated_completion_time = datetime.now() + timedelta(seconds=total_time)

    return estimated_completion_time

class DocumentationProcessManager:
    """Enhanced documentation process manager with improved error handling."""

    def __init__(
        self,
        repo_root: str,
        output_dir: str,
        provider_configs: Dict[str, ProviderConfig],
        function_schema: Optional[Dict[str, Any]] = None,
        max_concurrency: int = 5,
        cache_dir: Optional[str] = None
    ):
        """
        Initializes the documentation process manager.

        Args:
            repo_root: Repository root path
            output_dir: Output directory
            provider_configs: Configuration for AI providers
            function_schema: Documentation schema
            max_concurrency: Maximum concurrent operations
            cache_dir: Directory for caching
        """
        self.repo_root = Path(repo_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.provider_configs = provider_configs
        self.function_schema = function_schema
        self.max_concurrency = max_concurrency

        # Initialize metrics
        self.metrics = ProcessingMetrics()

        # Initialize semaphores
        self.api_semaphore = asyncio.Semaphore(max_concurrency)
        self.file_semaphore = asyncio.Semaphore(max_concurrency * 2)

        # Initialize managers
        self.context_manager = HierarchicalContextManager(
            cache_dir=cache_dir
        )
        self.doc_generator = DocumentationGenerator()

        # Initialize thread pool for CPU-bound operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max(os.cpu_count() - 1, 1)
        )

        # Task tracking
        self._active_tasks: Dict[str, Dict[asyncio.Task, str]] = {}
        self._task_progress: Dict[str, float] = {}
        self._task_status: Dict[str, str] = {}

        logger.info(
            f"Initialized DocumentationProcessManager with providers: "
            f"{', '.join(provider_configs.keys())}"
        )

    async def process_files(
        self,
        request: DocumentationRequest,
        task_id: str
    ) -> Dict[str, Any]:
        """
        Processes multiple files with improved task management and error handling.

        Args:
            request: Documentation request
            task_id: Unique task identifier

        Returns:
            Dict[str, Any]: Processing results and metrics
        """
        try:
            self.metrics = ProcessingMetrics()
            self.metrics.total_files = len(request.file_paths)

            # Store task status
            self._task_status[task_id] = "in_progress"
            self._task_progress[task_id] = 0.0

            # Get provider configuration
            provider_config = self.provider_configs.get(request.provider)
            if not provider_config:
                raise ValueError(f"Unsupported provider: {request.provider}")

            # Initialize ProviderMetrics for the provider
            provider_metrics = self.metrics.get_provider_metrics(request.provider)

            # Initialize processing components
            async with aiohttp.ClientSession() as session:
                api_handler = APIHandler(
                    session=session,
                    config=provider_config,
                    semaphore=self.api_semaphore,
                    provider_metrics=provider_metrics  # Pass ProviderMetrics
                )

                file_processor = FileProcessor(
                    context_manager=self.context_manager,
                    api_handler=api_handler,
                    provider_config=provider_config,
                    provider_metrics=provider_metrics  # Pass ProviderMetrics
                )

                # Process files with priority handling
                prioritized_files = self._prioritize_files(
                    request.file_paths,
                    request.priority
                )

                results = []
                total_files = len(prioritized_files)
                completed = 0  # Initialize completed files counter

                # Create tasks for all files
                tasks = {}
                for file_path in prioritized_files:
                    task = asyncio.create_task(
                        self._process_file(
                            file_path=file_path,
                            processor=file_processor,
                            request=request,
                            timeout=provider_config.timeout
                        )
                    )
                    tasks[task] = file_path  # Map task to file path

                # Keep track of active tasks
                self._active_tasks[task_id] = tasks

                for future in asyncio.as_completed(tasks.keys()):
                    file_path = tasks[future]
                    try:
                        result = await future
                        results.append(result)
                        self.metrics.record_file_result(
                            success=result["success"],
                            processing_time=result.get("processing_time", 0.0),
                            error_type=result.get("error_type")
                        )
                    except asyncio.CancelledError:
                        logger.warning(f"Processing of {file_path} was cancelled.")
                        self.metrics.record_file_result(
                            success=False,
                            processing_time=0.0,
                            error_type="CancelledError"
                        )
                        results.append({
                            "file_path": file_path,
                            "success": False,
                            "error": "Processing was cancelled",
                            "error_type": "CancelledError"
                        })
                    except Exception as e:
                        error_type = type(e).__name__
                        logger.error(f"Error processing {file_path}: {str(e)}")
                        self.metrics.record_file_result(
                            success=False,
                            processing_time=0.0,
                            error_type=error_type
                        )
                        results.append({
                            "file_path": file_path,
                            "success": False,
                            "error": str(e),
                            "error_type": error_type
                        })
                    finally:
                        completed += 1  # Increment completed files counter
                        # Update progress based on files processed
                        progress = (completed / total_files) * 100
                        self._update_task_progress(task_id, progress)

                        # Send progress callback if configured
                        if request.callback_url:
                            await self._send_progress_callback(
                                request.callback_url,
                                progress,
                                completed,
                                total_files
                            )

                # Update final metrics
                self.metrics.end_time = datetime.now()

                # Mark task as completed
                self._task_status[task_id] = "completed"
                self._task_progress[task_id] = 100.0
                # Remove from active tasks
                del self._active_tasks[task_id]

                return {
                    "task_id": task_id,
                    "status": "completed",
                    "results": results,
                    "metrics": self.metrics.get_summary()
                }

        except Exception as e:
            logger.error(f"Critical error in process_files: {str(e)}")
            # Mark task as failed
            self._task_status[task_id] = "failed"
            self._task_progress[task_id] = 100.0
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]
            raise

    async def _process_file(
        self,
        file_path: str,
        processor: FileProcessor,
        request: DocumentationRequest,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Processes a single file with improved error handling and timeouts."""
        start_time = datetime.now()

        try:
            if timeout:
                result = await asyncio.wait_for(
                    processor.process_file(
                        file_path=file_path,
                        skip_types=set(request.skip_types),
                        project_info=request.project_info,
                        style_guidelines=request.style_guidelines,
                        repo_root=str(self.repo_root),
                        output_dir=str(self.output_dir),
                        provider=request.provider,
                        project_id=request.project_id,
                        safe_mode=request.safe_mode
                    ),
                    timeout=timeout
                )
            else:
                result = await processor.process_file(
                    file_path=file_path,
                    skip_types=set(request.skip_types),
                    project_info=request.project_info,
                    style_guidelines=request.style_guidelines,
                    repo_root=str(self.repo_root),
                    output_dir=str(self.output_dir),
                    provider=request.provider,
                    project_id=request.project_id,
                    safe_mode=request.safe_mode
                )

            processing_time = (
                datetime.now() - start_time
            ).total_seconds()

            return {
                "file_path": file_path,
                "success": result.success,
                "content": result.content,
                "error": result.error,
                "processing_time": processing_time,
                "retries": result.retries
            }

        except asyncio.TimeoutError:
            logger.error(f"Processing of {file_path} timed out.")
            return {
                "file_path": file_path,
                "success": False,
                "error": "Processing timed out",
                "error_type": "TimeoutError",
                "processing_time": (
                    datetime.now() - start_time
                ).total_seconds()
            }
        except asyncio.CancelledError:
            logger.warning(f"Processing of {file_path} was cancelled.")
            return {
                "file_path": file_path,
                "success": False,
                "error": "Processing was cancelled",
                "error_type": "CancelledError",
                "processing_time": (
                    datetime.now() - start_time
                ).total_seconds()
            }
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return {
                "file_path": file_path,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time": (
                    datetime.now() - start_time
                ).total_seconds()
            }

    def _prioritize_files(
        self,
        file_paths: List[str],
        priority: str
    ) -> List[str]:
        """Prioritizes files based on various factors."""
        if priority == "low":
            return file_paths

        try:
            # Calculate file scores
            file_scores = []
            for file_path in file_paths:
                score = 0

                # Check file size
                size = Path(file_path).stat().st_size
                score += min(size / 1024, 100)  # Size score (max 100)

                # Check modification time
                mtime = Path(file_path).stat().st_mtime
                age_hours = (
                    datetime.now().timestamp() - mtime
                ) / 3600
                score += max(0, 100 - age_hours)  # Age score

                # Add file type priority
                ext = Path(file_path).suffix.lower()
                type_scores = {
                    '.py': 100,
                    '.js': 90,
                    '.ts': 90,
                    '.java': 85,
                    '.cpp': 85,
                    '.h': 80
                }
                score += type_scores.get(ext, 50)

                file_scores.append((score, file_path))

            # Sort by score (descending for high priority)
            file_scores.sort(reverse=priority == "high")
            return [f[1] for f in file_scores]

        except Exception as e:
            logger.warning(f"Error in file prioritization: {e}")
            return file_paths

    def _update_task_progress(
        self,
        task_id: str,
        progress: float
    ) -> None:
        """Updates task progress."""
        self._task_progress[task_id] = progress
        if progress >= 100:
            self._task_status[task_id] = "completed"
        else:
            self._task_status[task_id] = "in_progress"

    async def _send_progress_callback(
        self,
        callback_url: str,
        progress: float,
        processed: int,
        total: int
    ) -> None:
        """Sends progress callback to specified URL."""
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    callback_url,
                    json={
                        "progress": progress,
                        "processed_files": processed,
                        "total_files": total,
                        "timestamp": datetime.now().isoformat()
                    }
                )
        except Exception as e:
            logger.warning(f"Error sending progress callback: {e}")

    async def get_task_status(
        self,
        task_id: str
    ) -> Optional[Dict[str, Any]]:
        """Gets status of a specific task."""
        if task_id not in self._task_status:
            return None

        return {
            "task_id": task_id,
            "status": self._task_status[task_id],
            "progress": self._task_progress.get(task_id, 0.0)
        }

    async def cancel_task(self, task_id: str) -> None:
        """Cancels a running task."""
        if task_id in self._active_tasks:
            tasks = self._active_tasks[task_id]
            for task in tasks.keys():
                task.cancel()
            self._task_status[task_id] = "cancelled"
            self._task_progress[task_id] = 100.0
            del self._active_tasks[task_id]
            logger.info(f"Task {task_id} has been cancelled.")
        else:
            logger.warning(f"Task {task_id} not found or already completed.")

    async def cleanup(self) -> None:
        """Cleans up resources."""
        try:
            # Cancel active tasks
            for task_dict in self._active_tasks.values():
                for task in task_dict.keys():
                    task.cancel()

            # Cleanup thread pool
            self.thread_pool.shutdown(wait=True)

            # Clear context manager
            await self.context_manager.clear_context()

            # Clear task tracking
            self._active_tasks.clear()
            self._task_progress.clear()
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
    """API endpoint to generate documentation."""
    try:
        # Generate a unique task ID
        task_id = str(uuid.uuid4())

        # Load provider configurations
        provider_configs = {
            name: ProviderConfig(**config)
            for name, config in load_provider_configs().items()
        }

        manager = DocumentationProcessManager(
            repo_root=os.getenv("REPO_ROOT"),
            output_dir=os.getenv("OUTPUT_DIR"),
            provider_configs=provider_configs,
            max_concurrency=request.max_concurrency or 5,
            cache_dir=os.getenv("CACHE_DIR")
        )

        # Start processing task
        background_tasks.add_task(
            manager.process_files,
            request=request,
            task_id=task_id  # Pass task_id to process_files
        )

        return DocumentationResponse(
            task_id=task_id,
            status="started",
            progress=0.0,
            estimated_completion=calculate_estimated_completion(request)
        )

    except Exception as e:
        logger.error(f"Error in generate_documentation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Documentation generation failed: {str(e)}"
        )

@app.get(
    "/api/documentation/status/{task_id}",
    response_model=DocumentationResponse
)
async def get_status(task_id: str) -> Dict[str, Any]:
    """API endpoint to get documentation generation status."""
    try:
        # Get manager instance
        manager = get_manager_instance()

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

    # Setup logging
    setup_logging(
        log_file=os.getenv("LOG_FILE"),
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )

    # Start API server
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("DEBUG", False))
