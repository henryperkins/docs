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
from dataclasses import dataclass, field
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from time import perf_counter
import statistics
from collections import defaultdict
from contextlib import contextmanager

from utils import load_function_schema, calculate_project_metrics, should_process_file
from file_handlers import process_file
from context_manager import ContextManager
from write_documentation_report import (
    write_documentation_report, 
    generate_markdown_content, 
    generate_documentation_prompt, 
    sanitize_filename
)
from metrics import MetricsResult, MetricsAnalyzer

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

@dataclass
class FileProcessingResult:
    """Class to store file processing results and metrics."""
    file_path: str
    success: bool
    metrics_result: Optional[MetricsResult] = None
    error: Optional[str] = None
    documentation: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)

class DocumentationProcessManager:
    """Enhanced DocumentationProcessManager with better metrics handling"""
    
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
        self.safe_mode = False
        self.metrics_analyzer = MetricsAnalyzer()
        self.performance_monitor = PerformanceMonitor()
        self.metrics_aggregator = MetricsAggregator()  # Added MetricsAggregator initialization

    async def process_files(
        self,
        task_id: str,
        file_paths: List[str],
        skip_types: Set[str],
        project_info: str,
        style_guidelines: str,
        safe_mode: bool,
    ) -> Dict[str, Any]:
        """Enhanced process_files with metrics aggregation"""
        
        start_time = perf_counter()
        self.safe_mode = safe_mode
        self.active_tasks[task_id] = {
            "status": "running",
            "progress": 0.0,
            "results": {
                "successful": [],
                "failed": [],
                "skipped": []
            },
            "metrics": {
                "total_files": len(file_paths),
                "processed": 0,
                "success_rate": 0.0,
                "execution_time": 0.0,
                "warnings": 0,
                "errors": 0
            }
        }
        try:
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self._process_single_file(
                        session=session,
                        file_path=file_path,
                        skip_types=skip_types,
                        project_info=project_info,
                        style_guidelines=style_guidelines,
                        task_id=task_id,
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
                # Process files and collect results
                for future in asyncio.as_completed(tasks):
                    try:
                        result = await future
                        self._handle_processing_result(result, task_id)
                        await self._update_frontend_data(task_id)
                    except Exception as e:
                        logger.error(f"Task failed: {e}", exc_info=True)
                        self.metrics_analyzer.error_count += 1
                # Calculate final metrics and update task status
                await self._finalize_task(task_id, start_time)
                return self.active_tasks[task_id]["results"]
        except Exception as e:
            logger.error(f"Error in process_files: {e}", exc_info=True)
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = str(e)
            return self.active_tasks[task_id]["results"]

    async def _process_single_file(
        self,
        session: aiohttp.ClientSession,
        file_path: str,
        skip_types: Set[str],
        project_info: str,
        style_guidelines: str,
        task_id: str,
    ) -> FileProcessingResult:
        """Enhanced single file processing with performance monitoring"""
        
        start_time = perf_counter()
        
        try:
            with self.performance_monitor.measure(f"process_file_{Path(file_path).suffix}"):
                result = await process_file(
                    session=session,
                    file_path=file_path,
                    skip_types=skip_types,
                    semaphore=self.semaphore,
                    deployment_name=self.azure_config["AZURE_OPENAI_DEPLOYMENT"],
                    function_schema=self.function_schema,
                    repo_root=str(self.repo_root),
                    project_info=project_info,
                    style_guidelines=style_guidelines,
                    safe_mode=self.safe_mode,
                    azure_api_key=self.azure_config["AZURE_OPENAI_API_KEY"],
                    azure_endpoint=self.azure_config["AZURE_OPENAI_ENDPOINT"],
                    azure_api_version=self.azure_config["API_VERSION"],
                    output_dir=str(self.output_dir / task_id),
                    project_id=task_id,
                )
                if result and result.metrics_result:
                    await self.metrics_aggregator.add_metrics(result.metrics_result)  # Updated to use MetricsAggregator
                return result
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            return FileProcessingResult(
                file_path=file_path,
                success=False,
                error=str(e)
            )

    def _handle_processing_result(self, result: FileProcessingResult, task_id: str):
        """Handles the result of file processing"""
        if not result:
            return

        task_results = self.active_tasks[task_id]["results"]
        task_metrics = self.active_tasks[task_id]["metrics"]

        if result.success:
            task_results["successful"].append(result.file_path)
        else:
            task_results["failed"].append({
                "file": result.file_path,
                "error": result.error
            })
            task_metrics["errors"] += 1

        task_metrics["processed"] += 1
        task_metrics["success_rate"] = (
            len(task_results["successful"]) / task_metrics["total_files"] * 100
            if task_metrics["total_files"] > 0 else 0
        )

    async def _update_frontend_data(self, task_id: str):
        """Updates frontend data with enhanced metrics"""
        task_info = self.active_tasks[task_id]
        output_dir = self.output_dir / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_summary = self.metrics_aggregator.get_summary()  # Updated to use MetricsAggregator
        problematic_files = self.metrics_aggregator.get_problematic_files()  # Updated to use MetricsAggregator
        performance_summary = self.performance_monitor.get_summary()
        
        status_data = {
            "progress": (
                task_info["metrics"]["processed"] / 
                task_info["metrics"]["total_files"] * 100
                if task_info["metrics"]["total_files"] > 0 else 100
            ),
            "status": task_info["status"],
            "metrics": {
                "processed_files": task_info["metrics"]["processed"],
                "total_files": task_info["metrics"]["total_files"],
                "success_rate": task_info["metrics"]["success_rate"],
                "errors": task_info["metrics"]["errors"],
                "warnings": task_info["metrics"]["warnings"],
                "execution_metrics": metrics_summary,
                "performance_metrics": performance_summary
            },
            "results": {
                "successful": len(task_info["results"]["successful"]),
                "failed": len(task_info["results"]["failed"]),
                "skipped": len(task_info["results"]["skipped"])
            },
            "metrics_summary": metrics_summary,  # Added metrics summary
            "problematic_files": problematic_files,  # Added problematic files
            "performance_metrics": performance_summary
        }
        try:
            status_file = output_dir / "status.json"
            async with aiofiles.open(status_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(status_data, indent=2))
        except Exception as e:
            logger.error(f"Error writing status file: {e}", exc_info=True)

    async def _finalize_task(self, task_id: str, start_time: float):
        """Finalizes task processing and generates summary"""
        task_info = self.active_tasks[task_id]
        task_info["status"] = "completed"
        task_info["metrics"]["execution_time"] = perf_counter() - start_time
        # Generate final metrics summary
        metrics_summary = self.metrics_analyzer.get_summary()
        performance_summary = self.performance_monitor.get_summary()
        summary_data = {
            "task_id": task_id,
            "completion_time": datetime.now().isoformat(),
            "execution_time": task_info["metrics"]["execution_time"],
            "files": {
                "total": task_info["metrics"]["total_files"],
                "processed": task_info["metrics"]["processed"],
                "successful": len(task_info["results"]["successful"]),
                "failed": len(task_info["results"]["failed"]),
                "skipped": len(task_info["results"]["skipped"])
            },
            "metrics_summary": metrics_summary,
            "performance_summary": performance_summary,
            "warnings": task_info["metrics"]["warnings"],
            "errors": task_info["metrics"]["errors"]
        }
        try:
            summary_file = self.output_dir / task_id / "summary.json"
            async with aiofiles.open(summary_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(summary_data, indent=2))
        except Exception as e:
            logger.error(f"Error writing summary file: {e}", exc_info=True)
            
    def _should_process(self, file_path: str, skip_types: Set[str]) -> bool:
        """
        Determines if a file should be processed based on file type and path.

        Args:
            file_path (str): Path to the file to check
            skip_types (Set[str]): Set of file extensions to skip

        Returns:
            bool: True if the file should be processed, False otherwise
        """
        try:
            # Use the utility function from utils.py
            should_process = should_process_file(file_path, skip_types)
            logger.debug(f"Should process '{file_path}': {should_process}")
            
            # Additional project-specific checks
            if should_process:
                file_path = Path(file_path)
                
                # Check if file is within repo_root
                try:
                    file_path.relative_to(self.repo_root)
                except ValueError:
                    logger.warning(f"File '{file_path}' is outside repository root")
                    return False
                
                # Check file size (optional)
                if file_path.stat().st_size > 1_000_000:  # 1MB limit
                    logger.warning(f"File '{file_path}' exceeds size limit")
                    return False
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if should process '{file_path}': {e}")
            return False

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
