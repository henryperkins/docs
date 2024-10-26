"""
process_manager.py

Manages the documentation generation process, handling both Azure and Gemini providers.
Coordinates file processing, model interactions, and documentation generation.
"""

import asyncio
import logging
import json
import os
import sys
import uuid
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from time import perf_counter

import aiohttp
import aiofiles
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from azure_model import AzureModel
from gemini_model import GeminiModel
from context_manager import ContextManager
from file_handlers import process_file
from utils import sanitize_filename, should_process_file, load_function_schema
from results import FileProcessingResult

logger = logging.getLogger(__name__)
app = FastAPI()

@dataclass
class TaskStatus:
    """Status information for a documentation generation task."""
    task_id: str
    status: str = "pending"
    progress: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    completion_time: Optional[datetime] = None
    error: Optional[str] = None
    processed_files: int = 0
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0

class DocumentationProcessManager:
    """
    Manages the documentation generation process for multiple files.
    Supports both Azure and Gemini providers.
    """
    
    def __init__(
        self,
        repo_root: str,
        output_dir: str,
        provider: str,
        azure_config: Optional[Dict[str, str]] = None,
        gemini_config: Optional[Dict[str, str]] = None,
        function_schema: Dict[str, Any] = None,
        max_concurrency: int = 5
    ):
        """
        Initialize the documentation process manager.

        Args:
            repo_root: Repository root path
            output_dir: Output directory for documentation
            provider: AI provider to use ("azure" or "gemini")
            azure_config: Configuration for Azure OpenAI
            gemini_config: Configuration for Gemini
            function_schema: Schema for documentation structure
            max_concurrency: Maximum concurrent file processing
        """
        self.repo_root = Path(repo_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.provider = provider
        self.function_schema = function_schema
        self.semaphore = asyncio.Semaphore(max_concurrency)
        
        # Initialize appropriate client
        if provider == "azure":
            if not azure_config:
                raise ValueError("Azure configuration required when using Azure provider")
            self.client = AzureModel(
                api_key=azure_config["AZURE_OPENAI_API_KEY"],
                endpoint=azure_config["AZURE_OPENAI_ENDPOINT"],
                deployment_name=azure_config["AZURE_OPENAI_DEPLOYMENT"],
                api_version=azure_config["API_VERSION"]
            )
        elif provider == "gemini":
            if not gemini_config:
                raise ValueError("Gemini configuration required when using Gemini provider")
            self.client = GeminiModel(
                api_key=gemini_config["GEMINI_API_KEY"],
                endpoint=gemini_config["GEMINI_ENDPOINT"]
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.context_manager = ContextManager()
        self.active_tasks: Dict[str, TaskStatus] = {}

    async def process_files(
        self,
        task_id: str,
        file_paths: List[str],
        skip_types: Set[str],
        project_info: str,
        style_guidelines: str,
        safe_mode: bool,
    ) -> Dict[str, Any]:
        """
        Process multiple files with documentation generation.

        Args:
            task_id: Unique task identifier
            file_paths: List of files to process
            skip_types: File extensions to skip
            project_info: Project documentation info
            style_guidelines: Documentation style guidelines
            safe_mode: If True, don't modify files

        Returns:
            Dict[str, Any]: Processing results and metrics
        """
        task = TaskStatus(
            task_id=task_id,
            status="running",
            total_files=len(file_paths)
        )
        self.active_tasks[task_id] = task

        try:
            async with aiohttp.ClientSession() as session:
                tasks = []
                for file_path in file_paths:
                    if should_process_file(file_path, skip_types):
                        tasks.append(
                            self._process_single_file(
                                session=session,
                                file_path=file_path,
                                skip_types=skip_types,
                                project_info=project_info,
                                style_guidelines=style_guidelines,
                                task_id=task_id,
                            )
                        )
                    else:
                        task.processed_files += 1
                        task.progress = task.processed_files / task.total_files * 100

                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                successful_results = []
                failed_results = []
                
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Task failed: {result}")
                        task.failed_files += 1
                    elif isinstance(result, FileProcessingResult):
                        if result.success:
                            successful_results.append(result)
                            task.successful_files += 1
                        else:
                            failed_results.append(result)
                            task.failed_files += 1

                # Update task status
                task.completion_time = datetime.now()
                task.status = "completed"
                task.processed_files = len(results)
                task.progress = 100.0

                return {
                    "successful": [r.file_path for r in successful_results],
                    "failed": [{
                        "file": r.file_path,
                        "error": r.error
                    } for r in failed_results],
                    "metrics": {
                        "total_files": task.total_files,
                        "processed": task.processed_files,
                        "success_rate": (task.successful_files / task.total_files * 100 
                                       if task.total_files > 0 else 0),
                        "execution_time": (task.completion_time - task.start_time).total_seconds(),
                        "errors": task.failed_files
                    }
                }

        except Exception as e:
            logger.error(f"Error in process_files: {e}", exc_info=True)
            task.status = "failed"
            task.error = str(e)
            return {
                "successful": [],
                "failed": [],
                "metrics": {
                    "total_files": task.total_files,
                    "processed": task.processed_files,
                    "success_rate": 0,
                    "execution_time": (datetime.now() - task.start_time).total_seconds(),
                    "errors": task.failed_files
                }
            }

    async def _process_single_file(
        self,
        session: aiohttp.ClientSession,
        file_path: str,
        skip_types: Set[str],
        project_info: str,
        style_guidelines: str,
        task_id: str,
    ) -> FileProcessingResult:
        """
        Process a single file.

        Args:
            session: aiohttp session
            file_path: Path to the file
            skip_types: File extensions to skip
            project_info: Project documentation info
            style_guidelines: Documentation style guidelines
            task_id: Task identifier

        Returns:
            FileProcessingResult: Result of processing the file
        """
        try:
            result = await process_file(
                session=session,
                file_path=file_path,
                skip_types=skip_types,
                semaphore=self.semaphore,
                provider=self.provider,
                client=self.client,
                function_schema=self.function_schema,
                repo_root=str(self.repo_root),
                project_info=project_info,
                style_guidelines=style_guidelines,
                output_dir=str(self.output_dir / task_id),
                project_id=task_id,
                context_manager=self.context_manager
            )

            # Update task status
            task = self.active_tasks[task_id]
            task.processed_files += 1
            task.progress = task.processed_files / task.total_files * 100

            return result

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            return FileProcessingResult(
                file_path=file_path,
                success=False,
                error=str(e)
            )

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a documentation generation task.

        Args:
            task_id: Task identifier

        Returns:
            Optional[Dict[str, Any]]: Task status information
        """
        task = self.active_tasks.get(task_id)
        if not task:
            return None

        return {
            "status": task.status,
            "progress": task.progress,
            "processed_files": task.processed_files,
            "total_files": task.total_files,
            "successful_files": task.successful_files,
            "failed_files": task.failed_files,
            "error": task.error,
            "start_time": task.start_time.isoformat(),
            "completion_time": task.completion_time.isoformat() if task.completion_time else None
        }

    def cleanup_task(self, task_id: str) -> None:
        """
        Clean up task information.

        Args:
            task_id: Task identifier to clean up
        """
        self.active_tasks.pop(task_id, None)


class DocumentationRequest(BaseModel):
    file_paths: List[str]
    skip_types: Optional[List[str]] = []
    project_info: Optional[str] = ""
    style_guidelines: Optional[str] = ""
    safe_mode: Optional[bool] = False

class DocumentationResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    results: Optional[Dict[str, Any]] = None


class PerformanceMonitor:
    """Monitors performance metrics for different operations"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self._start_times = {}
        self._lock = asyncio.Lock()

    @contextmanager
    def measure(self, operation: str):
        """Context manager to measure operation execution time"""
        try:
            start_time = perf_counter()
            yield
        finally:
            duration = perf_counter() - start_time
            asyncio.create_task(self._record_metric(operation, duration))

    async def _record_metric(self, operation: str, duration: float):
        """Records a metric measurement with thread safety"""
        async with self._lock:
            self.metrics[operation].append(duration)

    def get_summary(self) -> Dict[str, Any]:
        """Generates a summary of all recorded metrics"""
        summary = {}
        for operation, durations in self.metrics.items():
            if durations:
                summary[operation] = {
                    'count': len(durations),
                    'avg_duration': statistics.mean(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'total_duration': sum(durations),
                    'std_dev': statistics.stdev(durations) if len(durations) > 1 else 0
                }
        return summary

    async def reset(self):
        """Resets all metrics"""
        async with self._lock:
            self.metrics.clear()
            self._start_times.clear()

class MetricsAggregator:
    """Aggregates and analyzes metrics across multiple files"""

    def __init__(self):
        self.file_metrics: Dict[str, MetricsResult] = {}
        self.global_metrics = {
            'total_complexity': 0,
            'total_maintainability': 0,
            'total_files': 0,
            'processed_files': 0,
            'error_count': 0,
            'warning_count': 0
        }
        self._lock = asyncio.Lock()

    async def add_metrics(self, metrics_result: MetricsResult):
        """Adds metrics for a file to the aggregator"""
        async with self._lock:
            self.file_metrics[metrics_result.file_path] = metrics_result
            self.global_metrics['total_files'] += 1
            
            if metrics_result.success:
                self.global_metrics['processed_files'] += 1
                if metrics_result.metrics:
                    self._update_global_metrics(metrics_result.metrics)
            else:
                self.global_metrics['error_count'] += 1

    def _update_global_metrics(self, metrics: Dict[str, Any]):
        """Updates global metrics with file metrics"""
        self.global_metrics['total_complexity'] += metrics.get('complexity', 0)
        self.global_metrics['total_maintainability'] += metrics.get('maintainability_index', 0)

    def get_summary(self) -> Dict[str, Any]:
        """Generates a comprehensive metrics summary"""
        processed = self.global_metrics['processed_files']
        return {
            'overall_metrics': {
                'avg_complexity': (
                    self.global_metrics['total_complexity'] / processed 
                    if processed > 0 else 0
                ),
                'avg_maintainability': (
                    self.global_metrics['total_maintainability'] / processed 
                    if processed > 0 else 0
                ),
                'success_rate': (
                    (processed / self.global_metrics['total_files']) * 100 
                    if self.global_metrics['total_files'] > 0 else 0
                )
            },
            'file_counts': {
                'total': self.global_metrics['total_files'],
                'processed': processed,
                'failed': self.global_metrics['error_count']
            },
            'issues': {
                'errors': self.global_metrics['error_count'],
                'warnings': self.global_metrics['warning_count']
            }
        }

    def get_problematic_files(self) -> List[Dict[str, Any]]:
        """Identifies files with concerning metrics"""
        problematic_files = []
        for file_path, result in self.file_metrics.items():
            if not (result.success and result.metrics):
                continue

            issues = []
            metrics = result.metrics

            # Check maintainability
            if metrics.get('maintainability_index', 100) < 20:
                issues.append({
                    'type': 'maintainability',
                    'value': metrics['maintainability_index'],
                    'threshold': 20
                })

            # Check complexity
            if metrics.get('complexity', 0) > 15:
                issues.append({
                    'type': 'complexity',
                    'value': metrics['complexity'],
                    'threshold': 15
                })

            if issues:
                problematic_files.append({
                    'file_path': file_path,
                    'issues': issues
                })

        return problematic_files


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