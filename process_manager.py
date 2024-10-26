"""
process_manager.py

Manages the documentation generation process, handling both Azure and Gemini providers.
Coordinates file processing, model interactions, and documentation generation with
enhanced error handling, metrics tracking, and parallel processing.
"""

import asyncio
import logging
import json
import os
import sys
from typing import Dict, Any, List, Optional, Set, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import aiofiles
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from azure_model import AzureModel
from gemini_model import GeminiModel
from context_manager import HierarchicalContextManager
from code_chunk import CodeChunk
from metrics import MetricsResult, calculate_code_metrics
from token_utils import TokenManager, TokenizerModel
from utils import sanitize_filename, should_process_file, load_function_schema
from write_documentation_report import write_documentation_report

logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Tracks processing metrics and performance data."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_files: int = 0
    processed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    successful_chunks: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    api_errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of processing metrics."""
        duration = (
            (self.end_time - self.start_time).total_seconds()
            if self.end_time
            else (datetime.now() - self.start_time).total_seconds()
        )
        
        return {
            "duration_seconds": duration,
            "files": {
                "total": self.total_files,
                "processed": self.processed_files,
                "successful": self.successful_files,
                "failed": self.failed_files,
                "success_rate": (
                    self.successful_files / self.total_files * 100
                    if self.total_files > 0
                    else 0
                )
            },
            "chunks": {
                "total": self.total_chunks,
                "successful": self.successful_chunks,
                "success_rate": (
                    self.successful_chunks / self.total_chunks * 100
                    if self.total_chunks > 0
                    else 0
                )
            },
            "tokens": {
                "total": self.total_tokens,
                "average_per_chunk": (
                    self.total_tokens / self.total_chunks
                    if self.total_chunks > 0
                    else 0
                )
            },
            "api": {
                "total_calls": self.api_calls,
                "errors": self.api_errors,
                "error_rate": (
                    self.api_errors / self.api_calls * 100
                    if self.api_calls > 0
                    else 0
                )
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": (
                    self.cache_hits / (self.cache_hits + self.cache_misses) * 100
                    if (self.cache_hits + self.cache_misses) > 0
                    else 0
                )
            }
        }

@dataclass
class ProcessingResult:
    """Stores the result of processing a file or chunk."""
    id: str
    success: bool
    error: Optional[str] = None
    documentation: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0

class DocumentationProcessManager:
    """Manages the documentation generation process."""

    def __init__(
        self,
        repo_root: str,
        output_dir: str,
        provider: str = "azure",
        azure_config: Optional[Dict[str, str]] = None,
        gemini_config: Optional[Dict[str, str]] = None,
        openai_config: Optional[Dict[str, str]] = None,  # New: OpenAI configuration
        function_schema: Optional[Dict[str, Any]] = None,
        max_concurrency: int = 5,
        max_retries: int = 3,
        cache_dir: Optional[str] = None,
        token_model: TokenizerModel = TokenizerModel.GPT4
    ):
        """
        Initializes the documentation process manager.

        Args:
            repo_root: Repository root path
            output_dir: Output directory for documentation
            provider: AI provider ("azure", "gemini", or "openai")
            azure_config: Azure OpenAI configuration
            gemini_config: Gemini configuration
            openai_config: OpenAI configuration
            function_schema: Documentation schema
            max_concurrency: Maximum concurrent operations
            max_retries: Maximum retry attempts
            cache_dir: Directory for caching
            token_model: Tokenizer model to use
        """
        self.repo_root = Path(repo_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.provider = provider.lower()
        self.function_schema = function_schema
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.token_model = token_model

        # Initialize semaphores for concurrency control
        self.api_semaphore = asyncio.Semaphore(max_concurrency)
        self.file_semaphore = asyncio.Semaphore(max_concurrency * 2)

        # Initialize metrics
        self.metrics = ProcessingMetrics()

        # Initialize context manager
        self.context_manager = HierarchicalContextManager(
            cache_dir=cache_dir,
            token_model=token_model
        )

        # Initialize appropriate client based on provider
        if self.provider == "azure":
            if not azure_config:
                raise ValueError("Azure configuration required")
            self.client = AzureModel(**azure_config)
        elif self.provider == "gemini":
            if not gemini_config:
                raise ValueError("Gemini configuration required")
            self.client = GeminiModel(**gemini_config)
        elif self.provider == "openai":  # New: OpenAI provider setup
            if not openai_config:
                raise ValueError("OpenAI configuration required")
            self.client = OpenAIModel(**openai_config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Initialize thread pool for CPU-bound operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max(os.cpu_count() - 1, 1)
        )

        logger.info(
            f"Initialized DocumentationProcessManager with {provider} provider"
        )

    async def process_files(
        self,
        file_paths: List[str],
        skip_types: Set[str],
        project_info: str,
        style_guidelines: str,
        safe_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Processes multiple files with documentation generation.

        Args:
            file_paths: List of files to process
            skip_types: File extensions to skip
            project_info: Project documentation info
            style_guidelines: Documentation style guidelines
            safe_mode: If True, don't modify files

        Returns:
            Dict[str, Any]: Processing results and metrics
        """
        try:
            self.metrics = ProcessingMetrics()
            self.metrics.total_files = len(file_paths)

            async with aiohttp.ClientSession() as session:
                # Process files in parallel with controlled concurrency
                tasks = []
                for file_path in file_paths:
                    if should_process_file(file_path, skip_types):
                        task = self._process_single_file(
                            session=session,
                            file_path=file_path,
                            project_info=project_info,
                            style_guidelines=style_guidelines,
                            safe_mode=safe_mode
                        )
                        tasks.append(task)
                    else:
                        logger.debug(f"Skipping file: {file_path}")

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                successful_results = []
                failed_results = []

                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"File processing failed: {result}")
                        self.metrics.failed_files += 1
                    elif isinstance(result, ProcessingResult):
                        if result.success:
                            successful_results.append(result)
                            self.metrics.successful_files += 1
                        else:
                            failed_results.append(result)
                            self.metrics.failed_files += 1

                self.metrics.processed_files = len(results)
                self.metrics.end_time = datetime.now()

                # Generate final report
                return {
                    "results": {
                        "successful": [
                            {
                                "id": r.id,
                                "documentation": r.documentation,
                                "metrics": r.metrics,
                                "processing_time": r.processing_time,
                                "warnings": r.warnings
                            }
                            for r in successful_results
                        ],
                        "failed": [
                            {
                                "id": r.id,
                                "error": r.error,
                                "processing_time": r.processing_time
                            }
                            for r in failed_results
                        ]
                    },
                    "metrics": self.metrics.get_summary()
                }

        except Exception as e:
            logger.error(f"Critical error in process_files: {str(e)}", exc_info=True)
            raise


    async def _process_single_file(
        self,
        session: aiohttp.ClientSession,
        file_path: str,
        project_info: str,
        style_guidelines: str,
        safe_mode: bool
    ) -> ProcessingResult:
        """
        Processes a single file with chunking and documentation generation.
        
        Args:
            session: aiohttp session
            file_path: Path to the file
            project_info: Project documentation info
            style_guidelines: Documentation style guidelines
            safe_mode: If True, don't modify files
            
        Returns:
            ProcessingResult: Processing result
        """
        start_time = datetime.now()
        
        try:
            async with self.file_semaphore:
                # Read file content
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()

                # Calculate initial metrics
                metrics_result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    calculate_code_metrics,
                    content,
                    file_path
                )

                # Create chunks
                chunks = await self._create_chunks(
                    content=content,
                    file_path=file_path,
                    metrics=metrics_result.metrics if metrics_result.success else None
                )

                self.metrics.total_chunks += len(chunks)

                # Process chunks with controlled concurrency
                chunk_results = await asyncio.gather(*[
                    self._process_chunk(
                        session=session,
                        chunk=chunk,
                        project_info=project_info,
                        style_guidelines=style_guidelines
                    )
                    for chunk in chunks
                ], return_exceptions=True)

                # Combine results
                successful_chunks = []
                failed_chunks = []
                warnings = []

                for result in chunk_results:
                    if isinstance(result, Exception):
                        failed_chunks.append(str(result))
                    elif isinstance(result, ProcessingResult):
                        if result.success:
                            successful_chunks.append(result)
                            self.metrics.successful_chunks += 1
                        else:
                            failed_chunks.append(result)
                        warnings.extend(result.warnings)

                # Combine documentation
                combined_documentation = await self._combine_documentation(
                    successful_chunks,
                    metrics_result
                )

                # Write documentation report
                if not safe_mode:
                    await write_documentation_report(
                        documentation=combined_documentation,
                        language=self._get_language(file_path),
                        file_path=file_path,
                        repo_root=str(self.repo_root),
                        output_dir=str(self.output_dir)
                    )

                processing_time = (datetime.now() - start_time).total_seconds()

                return ProcessingResult(
                    id=file_path,
                    success=len(successful_chunks) > 0,
                    documentation=combined_documentation,
                    metrics=metrics_result.metrics if metrics_result.success else None,
                    warnings=warnings,
                    processing_time=processing_time
                )

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
            return ProcessingResult(
                id=file_path,
                success=False,
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    async def _create_chunks(
        self,
        content: str,
        file_path: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> List[CodeChunk]:
        """Creates and validates code chunks."""
        try:
            language = self._get_language(file_path)
            if not language:
                raise ValueError(f"Unsupported file type: {file_path}")

            chunks = []
            token_count = 0

            # Create initial chunks
            raw_chunks = self._chunk_code(content, language)

            # Process and validate each chunk
            for chunk in raw_chunks:
                try:
                    # Validate chunk size
                    chunk_tokens = TokenManager.count_tokens(
                        chunk.chunk_content,
                        model=self.token_model
                    ).token_count

                    if chunk_tokens > 4096:  # Max token limit
                        # Split large chunks
                        sub_chunks = self._split_large_chunk(chunk)
                        chunks.extend(sub_chunks)
                        token_count += sum(
                            TokenManager.count_tokens(sc.chunk_content).token_count
                            for sc in sub_chunks
                        )
                    else:
                        chunks.append(chunk)
                        token_count += chunk_tokens

                except Exception as e:
                    logger.warning(f"Error processing chunk in {file_path}: {str(e)}")
                    continue

            self.metrics.total_tokens += token_count
            return chunks

        except Exception as e:
            logger.error(f"Error creating chunks for {file_path}: {str(e)}")
            raise

    async def _process_chunk(
        self,
        session: aiohttp.ClientSession,
        chunk: CodeChunk,
        project_info: str,
        style_guidelines: str
    ) -> ProcessingResult:
        """
        Processes a single code chunk with retries and error handling.
        
        Args:
            session: aiohttp session
            chunk: Code chunk to process
            project_info: Project documentation info
            style_guidelines: Documentation style guidelines
            
        Returns:
            ProcessingResult: Processing result
        """
        start_time = datetime.now()
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            try:
                async with self.api_semaphore:
                    # Check cache first
                    cached_doc = await self.context_manager.get_documentation_for_chunk(
                        chunk.chunk_id
                    )

                    if cached_doc:
                        self.metrics.cache_hits += 1
                        return ProcessingResult(
                            id=chunk.chunk_id,
                            success=True,
                            documentation=cached_doc,
                            processing_time=(
                                datetime.now() - start_time
                            ).total_seconds()
                        )

                    self.metrics.cache_misses += 1

                    # Generate documentation
                    self.metrics.api_calls += 1
                    documentation = await self._generate_documentation(
                        session=session,
                        chunk=chunk,
                        project_info=project_info,
                        style_guidelines=style_guidelines
                    )

                    # Cache the result
                    await self.context_manager.add_doc_chunk(
                        chunk.chunk_id,
                        documentation
                    )
                    return ProcessingResult(
                        id=chunk.chunk_id,
                        success=True,
                        documentation=documentation,
                        processing_time=(datetime.now() - start_time).total_seconds()
                    )

            except Exception as e:
                attempt += 1
                last_error = str(e)
                self.metrics.api_errors += 1
                
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Retry {attempt}/{self.max_retries} for chunk "
                        f"{chunk.chunk_id} after {wait_time}s. Error: {str(e)}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to process chunk {chunk.chunk_id} "
                        f"after {self.max_retries} attempts: {str(e)}"
                    )

        return ProcessingResult(
            id=chunk.chunk_id,
            success=False,
            error=last_error,
            processing_time=(datetime.now() - start_time).total_seconds()
        )

    async def _generate_documentation(
        self,
        session: aiohttp.ClientSession,
        chunk: CodeChunk,
        project_info: str,
        style_guidelines: str
    ) -> Dict[str, Any]:
        """
        Generates documentation for a code chunk using the configured AI provider.
        
        Args:
            session: aiohttp session
            chunk: Code chunk
            project_info: Project info
            style_guidelines: Style guidelines
            
        Returns:
            Dict[str, Any]: Generated documentation
        """
        # Get context for the chunk
        context_chunks = await self.context_manager.get_related_chunks(
            chunk.chunk_id
        )
        
        # Prepare prompt
        prompt = self._prepare_prompt(
            chunk=chunk,
            context_chunks=context_chunks,
            project_info=project_info,
            style_guidelines=style_guidelines
        )
        
        # Generate documentation using appropriate provider
        if self.provider == "azure":
            return await self.client.generate_documentation(prompt)
        else:  # gemini
            return await self.client.generate_documentation(prompt)

    def _prepare_prompt(
        self,
        chunk: CodeChunk,
        context_chunks: List[CodeChunk],
        project_info: str,
        style_guidelines: str
    ) -> List[Dict[str, str]]:
        """Prepares the prompt for documentation generation."""
        context = self._format_context(context_chunks)
        
        return [
            {
                "role": "system",
                "content": f"""Project Information:
{project_info}

Style Guidelines:
{style_guidelines}"""
            },
            {
                "role": "user",
                "content": f"""Related Code Context:
{context}

Current Code:
{chunk.chunk_content}"""
            }
        ]

    def _format_context(self, chunks: List[CodeChunk]) -> str:
        """Formats context chunks into a readable string."""
        context_parts = []
        for chunk in chunks:
            context_parts.append(f"""
# From {chunk.get_context_string()}:
{chunk.chunk_content}
""")
        return "\n".join(context_parts)

    async def _combine_documentation(
        self,
        chunk_results: List[ProcessingResult],
        metrics_result: MetricsResult
    ) -> Dict[str, Any]:
        """
        Combines documentation from multiple chunks intelligently.
        
        Args:
            chunk_results: Results from chunk processing
            metrics_result: Code metrics
            
        Returns:
            Dict[str, Any]: Combined documentation
        """
        combined = {
            "functions": [],
            "classes": [],
            "variables": [],
            "constants": [],
            "imports": [],
            "metrics": metrics_result.metrics if metrics_result.success else {},
            "summary": "",
            "warnings": []
        }
        
        # Track seen items to avoid duplication
        seen_functions = set()
        seen_classes = set()
        seen_vars = set()
        
        for result in chunk_results:
            if not result.success or not result.documentation:
                continue
                
            doc = result.documentation
            
            # Combine functions
            for func in doc.get("functions", []):
                if func["name"] not in seen_functions:
                    combined["functions"].append(func)
                    seen_functions.add(func["name"])
            
            # Combine classes
            for cls in doc.get("classes", []):
                if cls["name"] not in seen_classes:
                    combined["classes"].append(cls)
                    seen_classes.add(cls["name"])
            
            # Combine variables and constants
            for var in doc.get("variables", []):
                if var["name"] not in seen_vars:
                    combined["variables"].append(var)
                    seen_vars.add(var["name"])
                    
            for const in doc.get("constants", []):
                if const["name"] not in seen_vars:
                    combined["constants"].append(const)
                    seen_vars.add(const["name"])
            
            # Combine imports
            combined["imports"].extend(
                imp for imp in doc.get("imports", [])
                if imp not in combined["imports"]
            )
            
            # Combine warnings
            combined["warnings"].extend(result.warnings)
        
        # Generate overall summary
        combined["summary"] = self._generate_summary(combined)
        
        return combined

    def _generate_summary(self, documentation: Dict[str, Any]) -> str:
        """Generates an overall summary of the documentation."""
        summary_parts = []
        
        # Add basic statistics
        summary_parts.append(f"""Code Overview:
- {len(documentation['functions'])} functions
- {len(documentation['classes'])} classes
- {len(documentation['variables'])} variables
- {len(documentation['constants'])} constants
""")
        
        # Add metrics summary if available
        metrics = documentation.get("metrics", {})
        if metrics:
            summary_parts.append(f"""
Code Metrics:
- Maintainability Index: {metrics.get('maintainability_index', 'N/A')}
- Cyclomatic Complexity: {metrics.get('complexity', 'N/A')}
- Total Lines: {metrics.get('loc', {}).get('total', 'N/A')}
""")
        
        # Add warnings if any
        if documentation["warnings"]:
            summary_parts.append("\nWarnings:")
            for warning in documentation["warnings"]:
                summary_parts.append(f"- {warning}")
        
        return "\n".join(summary_parts)

    def _get_language(self, file_path: str) -> Optional[str]:
        """Determines the programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".hpp": "cpp",
            ".h": "cpp",
            ".c": "c",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".swift": "swift",
            ".kt": "kotlin",
            ".rs": "rust",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sql": "sql",
            ".sh": "shell",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".md": "markdown",
            ".xml": "xml"
        }
        return language_map.get(ext)

    async def cleanup(self):
        """Cleanup resources."""
        self.thread_pool.shutdown(wait=True)
        await self.context_manager.clear_context()
        logger.info("Cleanup completed")

    def _split_large_chunk(self, chunk: CodeChunk) -> List[CodeChunk]:
        """
        Splits a large chunk into smaller ones based on token limits.
        
        Args:
            chunk: Large code chunk to split
            
        Returns:
            List[CodeChunk]: List of smaller chunks
        """
        try:
            # Get possible split points
            split_points = chunk.get_possible_split_points()
            if not split_points:
                logger.warning(f"No valid split points found for chunk {chunk.chunk_id}")
                return [chunk]

            sub_chunks = []
            current_start = chunk.start_line
            
            for split_point in split_points:
                # Check if splitting here would create valid-sized chunks
                first_part = "\n".join(
                    chunk.chunk_content.splitlines()[:split_point - current_start]
                )
                
                if TokenManager.count_tokens(first_part).token_count <= 4096:
                    try:
                        new_chunks = chunk.split(split_point)
                        sub_chunks.extend(new_chunks)
                        current_start = split_point
                    except ValueError as e:
                        logger.warning(f"Failed to split at point {split_point}: {e}")
                        continue

            return sub_chunks if sub_chunks else [chunk]

        except Exception as e:
            logger.error(f"Error splitting chunk {chunk.chunk_id}: {e}")
            return [chunk]

    async def get_processing_status(self) -> Dict[str, Any]:
        """
        Gets current processing status and metrics.
        
        Returns:
            Dict[str, Any]: Current status and metrics
        """
        return {
            "metrics": self.metrics.get_summary(),
            "context_manager": {
                "metrics": await self.context_manager.get_metrics(),
                "cache_status": self._get_cache_status()
            },
            "provider_status": await self._check_provider_status()
        }

    def _get_cache_status(self) -> Dict[str, Any]:
        """Gets cache status information."""
        if not self._cache_dir:
            return {"enabled": False}
            
        try:
            cache_size = sum(
                f.stat().st_size
                for f in Path(self._cache_dir).glob('**/*')
                if f.is_file()
            )
            
            return {
                "enabled": True,
                "size_bytes": cache_size,
                "size_mb": cache_size / (1024 * 1024),
                "file_count": len(list(Path(self._cache_dir).glob('**/*')))
            }
        except Exception as e:
            logger.error(f"Error getting cache status: {e}")
            return {"enabled": True, "error": str(e)}

    async def _check_provider_status(self) -> Dict[str, Any]:
        """Checks AI provider status."""
        try:
            if self.provider == "azure":
                # Simple test call to Azure
                test_prompt = [{"role": "user", "content": "test"}]
                response = await self.client.generate_documentation(test_prompt)
                return {
                    "provider": "azure",
                    "status": "available" if response else "error",
                    "api_calls": self.metrics.api_calls,
                    "error_rate": (
                        self.metrics.api_errors / self.metrics.api_calls * 100
                        if self.metrics.api_calls > 0
                        else 0
                    )
                }
            else:  # gemini
                # Similar test for Gemini
                test_prompt = [{"role": "user", "content": "test"}]
                response = await self.client.generate_documentation(test_prompt)
                return {
                    "provider": "gemini",
                    "status": "available" if response else "error",
                    "api_calls": self.metrics.api_calls,
                    "error_rate": (
                        self.metrics.api_errors / self.metrics.api_calls * 100
                        if self.metrics.api_calls > 0
                        else 0
                    )
                }
        except Exception as e:
            return {
                "provider": self.provider,
                "status": "error",
                "error": str(e)
            }

    async def optimize_performance(self):
        """Optimizes processing performance based on metrics."""
        metrics = self.metrics.get_summary()
        
        # Adjust concurrency based on error rate
        if metrics["api"]["error_rate"] > 10:  # More than 10% errors
            self.max_concurrency = max(1, self.max_concurrency - 1)
            self.api_semaphore = asyncio.Semaphore(self.max_concurrency)
            logger.info(f"Reduced concurrency to {self.max_concurrency}")
            
        # Adjust cache size based on hit rate
        if metrics["cache"]["hit_rate"] < 50:  # Less than 50% cache hits
            await self.context_manager.optimize_cache()
            logger.info("Optimized cache")
            
        # Log optimization results
        logger.info(
            f"Performance optimization completed. "
            f"New concurrency: {self.max_concurrency}"
        )

class DocumentationRequest(BaseModel):
    """API request model for documentation generation."""
    file_paths: List[str]
    skip_types: Optional[List[str]] = []
    project_info: Optional[str] = ""
    style_guidelines: Optional[str] = ""
    safe_mode: Optional[bool] = False

class DocumentationResponse(BaseModel):
    """API response model for documentation generation."""
    task_id: str
    status: str
    progress: float
    results: Optional[Dict[str, Any]] = None

# FastAPI app setup
app = FastAPI(title="Documentation Generator API")

@app.post("/api/documentation/generate", response_model=DocumentationResponse)
async def generate_documentation(
    request: DocumentationRequest,
    background_tasks: BackgroundTasks
) -> DocumentationResponse:
    """API endpoint to generate documentation."""
    manager = DocumentationProcessManager(
        repo_root=os.getenv("REPO_ROOT"),
        output_dir=os.getenv("OUTPUT_DIR"),
        provider=os.getenv("AI_PROVIDER", "azure"),
        azure_config={
            "api_key": os.getenv("AZURE_API_KEY"),
            "endpoint": os.getenv("AZURE_ENDPOINT"),
            "deployment": os.getenv("AZURE_DEPLOYMENT")
        }
    )
    
    task_id = str(uuid.uuid4())
    background_tasks.add_task(
        manager.process_files,
        file_paths=request.file_paths,
        skip_types=set(request.skip_types),
        project_info=request.project_info,
        style_guidelines=request.style_guidelines,
        safe_mode=request.safe_mode
    )
    
    return DocumentationResponse(
        task_id=task_id,
        status="started",
        progress=0.0
    )

@app.get("/api/documentation/status/{task_id}")
async def get_status(task_id: str) -> Dict[str, Any]:
    """API endpoint to get documentation generation status."""
    # Implementation depends on how you want to track tasks
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
