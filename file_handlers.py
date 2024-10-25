"""
file_handlers.py

Handles file processing and documentation generation using Azure OpenAI,
with support for chunking, caching, and parallel processing.
"""

import asyncio
import logging
import aiohttp
import aiofiles
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from contextlib import asynccontextmanager

from code_chunk import CodeChunk
from context_manager import HierarchicalContextManager, ChunkNotFoundError
from utils import (
    chunk_code, ChunkTooLargeError, get_language,
    calculate_prompt_tokens, should_process_file
)
from write_documentation_report import (
    generate_documentation_prompt,
    write_documentation_report
)

logger = logging.getLogger(__name__)

@dataclass
class FileProcessingResult:
    """
    Stores the result of processing a file.
    
    Attributes:
        file_path: Path to the processed file
        success: Whether processing succeeded
        error: Error message if processing failed
        documentation: Generated documentation if successful
        metrics: Metrics about the processing
        chunk_count: Number of chunks processed
        successful_chunks: Number of successfully processed chunks
        timestamp: When the processing completed
    """
    file_path: str
    success: bool
    error: Optional[str] = None
    documentation: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    chunk_count: int = 0
    successful_chunks: int = 0
    timestamp: datetime = datetime.now()

@dataclass
class ChunkProcessingResult:
    """
    Stores the result of processing a single chunk.
    
    Attributes:
        chunk_id: ID of the processed chunk
        success: Whether processing succeeded
        documentation: Generated documentation if successful
        error: Error message if processing failed
        retries: Number of retry attempts made
    """
    chunk_id: str
    success: bool
    documentation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retries: int = 0

@asynccontextmanager
async def get_aiohttp_session():
    """Creates and manages an aiohttp session."""
    async with aiohttp.ClientSession() as session:
        yield session

async def process_all_files(
    session: aiohttp.ClientSession,
    file_paths: List[str],
    skip_types: Set[str],
    semaphore: asyncio.Semaphore,
    deployment_name: str,
    function_schema: Dict[str, Any],
    repo_root: str,
    project_info: str,
    style_guidelines: str,
    safe_mode: bool,
    azure_api_key: str,
    azure_endpoint: str,
    azure_api_version: str,
    output_dir: str,
    max_parallel_chunks: int = 3,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Process multiple files with integrated context management.
    
    Args:
        session: aiohttp session for API calls
        file_paths: List of files to process
        skip_types: File extensions to skip
        semaphore: Controls concurrent API requests
        deployment_name: Azure OpenAI deployment name
        function_schema: Schema for documentation generation
        repo_root: Root directory of the repository
        project_info: Project documentation info
        style_guidelines: Documentation style guidelines
        safe_mode: If True, don't modify files
        azure_api_key: Azure OpenAI API key
        azure_endpoint: Azure OpenAI endpoint
        azure_api_version: Azure OpenAI API version
        output_dir: Directory for output files
        max_parallel_chunks: Maximum chunks to process in parallel
        max_retries: Maximum retry attempts per chunk
        
    Returns:
        Dict[str, Any]: Processing results and metrics
    """
    try:
        # Initialize context manager and metrics
        context_manager = HierarchicalContextManager(
            cache_dir=Path(output_dir) / ".cache"
        )
        logger.info("Initialized HierarchicalContextManager")
        
        total_files = len(file_paths)
        results = []
        start_time = datetime.now()

        # Process files
        for index, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing file {index}/{total_files}: {file_path}")
            
            try:
                result = await process_file(
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
                    context_manager=context_manager,
                    max_parallel_chunks=max_parallel_chunks,
                    max_retries=max_retries
                )
                results.append(result)
                
                logger.info(
                    f"Completed file {file_path}: "
                    f"Success={result.success}, "
                    f"Chunks={result.chunk_count}"
                )
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
                results.append(FileProcessingResult(
                    file_path=file_path,
                    success=False,
                    error=str(e)
                ))

        # Calculate final metrics
        end_time = datetime.now()
        successful_files = sum(1 for r in results if r.success)
        total_chunks = sum(r.chunk_count for r in results)
        successful_chunks = sum(r.successful_chunks for r in results)
        
        return {
            "results": results,
            "metrics": {
                "total_files": total_files,
                "successful_files": successful_files,
                "failed_files": total_files - successful_files,
                "total_chunks": total_chunks,
                "successful_chunks": successful_chunks,
                "execution_time": (end_time - start_time).total_seconds()
            }
        }

    except Exception as e:
        logger.error(f"Critical error in process_all_files: {str(e)}", exc_info=True)
        raise

async def process_file(
    session: aiohttp.ClientSession,
    file_path: str,
    skip_types: Set[str],
    semaphore: asyncio.Semaphore,
    deployment_name: str,
    function_schema: Dict[str, Any],
    repo_root: str,
    project_info: str,
    style_guidelines: str,
    safe_mode: bool,
    azure_api_key: str,
    azure_endpoint: str,
    azure_api_version: str,
    output_dir: str,
    context_manager: HierarchicalContextManager,
    max_parallel_chunks: int = 3,
    max_retries: int = 3
) -> FileProcessingResult:
    """
    Process a single file using context-aware chunking.
    
    Args:
        session: aiohttp session for API calls
        file_path: Path to the file to process
        ... (other parameters match process_all_files)
        
    Returns:
        FileProcessingResult: Results of processing the file
    """
    try:
        if not should_process_file(file_path, skip_types):
            return FileProcessingResult(
                file_path=file_path,
                success=False,
                error="File type excluded"
            )

        # Read file content
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()

        # Determine language and create chunks
        language = get_language(file_path)
        if not language:
            return FileProcessingResult(
                file_path=file_path,
                success=False,
                error="Unsupported language"
            )

        try:
            chunks = chunk_code(content, file_path, language)
            logger.info(f"Split {file_path} into {len(chunks)} chunks")
        except ChunkTooLargeError as e:
            logger.warning(f"File contains chunks that are too large: {str(e)}")
            return FileProcessingResult(
                file_path=file_path,
                success=False,
                error=f"Chunk too large: {str(e)}"
            )

        # Add chunks to context manager
        for chunk in chunks:
            try:
                await context_manager.add_code_chunk(chunk)
            except ValueError as e:
                logger.warning(f"Couldn't add chunk to context: {str(e)}")
                continue

        # Process chunks in parallel groups
        chunk_results = []
        for i in range(0, len(chunks), max_parallel_chunks):
            group = chunks[i:i + max_parallel_chunks]
            
            tasks = [
                process_chunk_with_retry(
                    chunk=chunk,
                    session=session,
                    semaphore=semaphore,
                    deployment_name=deployment_name,
                    function_schema=function_schema,
                    project_info=project_info,
                    style_guidelines=style_guidelines,
                    context_manager=context_manager,
                    azure_api_key=azure_api_key,
                    azure_endpoint=azure_endpoint,
                    azure_api_version=azure_api_version,
                    max_retries=max_retries
                )
                for chunk in group
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for chunk, result in zip(group, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process chunk {chunk.chunk_id}: {str(result)}")
                    chunk_results.append(ChunkProcessingResult(
                        chunk_id=chunk.chunk_id,
                        success=False,
                        error=str(result)
                    ))
                else:
                    chunk_results.append(result)

        # Combine documentation from successful chunks
        successful_docs = [r.documentation for r in chunk_results if r.success]
        if not successful_docs:
            return FileProcessingResult(
                file_path=file_path,
                success=False,
                error="No chunks processed successfully",
                chunk_count=len(chunks),
                successful_chunks=0
            )

        combined_documentation = combine_chunk_documentation(chunk_results, chunks)

        # Write documentation report
        report_result = await write_documentation_report(
            documentation=combined_documentation,
            language=language,
            file_path=file_path,
            repo_root=repo_root,
            output_dir=output_dir
        )

        return FileProcessingResult(
            file_path=file_path,
            success=True,
            documentation=report_result,
            chunk_count=len(chunks),
            successful_chunks=sum(1 for r in chunk_results if r.success)
        )

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        return FileProcessingResult(
            file_path=file_path,
            success=False,
            error=str(e)
        )

async def process_chunk_with_retry(
    chunk: CodeChunk,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    deployment_name: str,
    function_schema: Dict[str, Any],
    project_info: str,
    style_guidelines: str,
    context_manager: HierarchicalContextManager,
    azure_api_key: str,
    azure_endpoint: str,
    azure_api_version: str,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> ChunkProcessingResult:
    """
    Process a chunk with automatic retries and exponential backoff.
    
    Args:
        chunk: The code chunk to process
        ... (other parameters match process_file)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff
        
    Returns:
        ChunkProcessingResult: Results of processing the chunk
    """
    last_error = None
    attempt = 0
    
    while attempt < max_retries:
        try:
            return await process_chunk(
                chunk=chunk,
                session=session,
                semaphore=semaphore,
                deployment_name=deployment_name,
                function_schema=function_schema,
                project_info=project_info,
                style_guidelines=style_guidelines,
                context_manager=context_manager,
                azure_api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                azure_api_version=azure_api_version
            )
        except Exception as e:
            last_error = e
            attempt += 1
            
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                logger.warning(
                    f"Retry {attempt}/{max_retries} for chunk {chunk.chunk_id} "
                    f"after {delay}s delay. Error: {str(e)}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"Final attempt failed for chunk {chunk.chunk_id}: {str(e)}"
                )
    
    return ChunkProcessingResult(
        chunk_id=chunk.chunk_id,
        success=False,
        error=str(last_error),
        retries=attempt
    )

async def process_chunk(
    chunk: CodeChunk,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    deployment_name: str,
    function_schema: Dict[str, Any],
    project_info: str,
    style_guidelines: str,
    context_manager: HierarchicalContextManager,
    azure_api_key: str,
    azure_endpoint: str,
    azure_api_version: str
) -> ChunkProcessingResult:
    """
    Process a single code chunk.
    
    Args:
        chunk: The code chunk to process
        ... (other parameters match process_file)
        
    Returns:
        ChunkProcessingResult: Results of processing the chunk
    """
    try:
        # Generate documentation with context
        prompt = generate_documentation_prompt(
            chunk=chunk,
            context_manager=context_manager,
            project_info=project_info,
            style_guidelines=style_guidelines,
            function_schema=function_schema
        )

        # Get documentation from Azure OpenAI
        documentation = await fetch_documentation_rest(
            session=session,
            prompt=prompt,
            semaphore=semaphore,
            deployment_name=deployment_name,
            function_schema=function_schema,
            azure_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version
        )

        # Store documentation in context manager
        await context_manager.add_doc_chunk(chunk.chunk_id, documentation)

        return ChunkProcessingResult(
            chunk_id=chunk.chunk_id,
            success=True,
            documentation=documentation
        )

    except Exception as e:
        logger.error(f"Error processing chunk {chunk.chunk_id}: {str(e)}")
        return ChunkProcessingResult(
            chunk_id=chunk.chunk_id,
            success=False,
            error=str(e)
        )

def combine_chunk_documentation(
    chunk_results: List[ChunkProcessingResult],
    chunks: List[CodeChunk]
) -> Dict[str, Any]:
    """
    Combines documentation from multiple chunks intelligently.
    
    Preserves the structure and relationships between different code elements
    while avoiding duplication and maintaining proper organization.
    
    Args:
        chunk_results: Results from processing chunks
        chunks: Original code chunks
        
    Returns:
        Dict[str, Any]: Combined documentation
    """
    combined = {
        "functions": [],
        "classes": {},
        "variables": [],
        "constants": [],
        "summary": "",
        "metrics": {},
        "structure": {
            "imports": [],
            "module_level": [],
            "classes": [],
            "functions": []
        }
    }
    
    # Group chunks by class
    class_chunks: Dict[str, List[CodeChunk]] = {}
    for chunk in chunks:
        if chunk.class_name:
            base_name = chunk.class_name.split('_part')[0]
            class_chunks.setdefault(base_name, []).append(chunk)
    
    # Process results
    for result in chunk_results:
        if not result.success:
            continue
            
        doc = result.documentation
        chunk = next(c for c in chunks if c.chunk_id == result.chunk_id)
        
        # Handle functions
        if chunk.function_name and not chunk.class_name:
            # Add function documentation
            combined["functions"].extend(doc.get("functions", []))
            # Add to structure
            combined["structure"]["functions"].append({
                "name": chunk.function_name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "is_async": chunk.is_async,
                "decorators": chunk.decorator_list,
                "docstring": doc.get("functions", [{}])[0].get("docstring", "")
            })
        
        # Handle classes
        if chunk.class_name:
            base_name = chunk.class_name.split('_part')[0]
            if base_name not in combined["classes"]:
                combined["classes"][base_name] = {
                    "name": base_name,
                    "docstring": "",
                    "methods": [],
                    "class_variables": [],
                    "start_line": chunk.start_line,
                    "decorators": chunk.decorator_list,
                    "bases": [],  # For inheritance
                    "metrics": {}
                }
            
            class_doc = combined["classes"][base_name]
            
            # Update class documentation
            if "classes" in doc:
                for cls in doc["classes"]:
                    if not class_doc["docstring"]:
                        class_doc["docstring"] = cls.get("docstring", "")
                    
                    # Merge methods
                    for method in cls.get("methods", []):
                        existing = next(
                            (m for m in class_doc["methods"] 
                             if m["name"] == method["name"]),
                            None
                        )
                        if existing:
                            existing.update(method)
                        else:
                            class_doc["methods"].append(method)
                    
                    # Track inheritance
                    if "bases" in cls and cls["bases"]:
                        class_doc["bases"].extend(
                            base for base in cls["bases"]
                            if base not in class_doc["bases"]
                        )
                    
                    # Merge class variables
                    class_doc["class_variables"].extend(
                        var for var in cls.get("class_variables", [])
                        if not any(v["name"] == var["name"] 
                                 for v in class_doc["class_variables"])
                    )
        
        # Handle variables and constants
        for var in doc.get("variables", []):
            if not any(v["name"] == var["name"] for v in combined["variables"]):
                combined["variables"].append(var)
                
        for const in doc.get("constants", []):
            if not any(c["name"] == const["name"] for c in combined["constants"]):
                combined["constants"].append(const)
        
        # Merge metrics carefully
        for key, value in doc.get("metrics", {}).items():
            if key not in combined["metrics"]:
                combined["metrics"][key] = value
            elif isinstance(value, (int, float)):
                combined["metrics"][key] = max(
                    combined["metrics"][key], value
                )
            elif isinstance(value, dict):
                if key not in combined["metrics"]:
                    combined["metrics"][key] = {}
                combined["metrics"][key].update(value)
        
        # Handle imports
        if "imports" in doc:
            combined["structure"]["imports"].extend(
                imp for imp in doc["imports"]
                if imp not in combined["structure"]["imports"]
            )
        
        # Combine summaries intelligently
        if doc.get("summary"):
            summary_part = doc["summary"].strip()
            if summary_part:
                if combined["summary"]:
                    # Try to avoid duplication in summaries
                    if summary_part not in combined["summary"]:
                        combined["summary"] += "\n\n"
                        combined["summary"] += summary_part
                else:
                    combined["summary"] = summary_part
    
    # Post-process
    # Convert classes dict to list
    combined["classes"] = list(combined["classes"].values())
    
    # Sort everything by line number
    combined["functions"].sort(key=lambda x: x.get("start_line", 0))
    combined["classes"].sort(key=lambda x: x.get("start_line", 0))
    combined["variables"].sort(key=lambda x: x.get("line", 0))
    combined["constants"].sort(key=lambda x: x.get("line", 0))
    
    # Generate structure summary
    combined["structure"]["summary"] = generate_structure_summary(combined)
    
    # Add overall metrics
    combined["metrics"]["total_lines"] = max(
        (c.end_line for c in chunks),
        default=0
    )
    combined["metrics"]["chunk_count"] = len(chunks)
    combined["metrics"]["success_rate"] = (
        sum(1 for r in chunk_results if r.success) / len(chunk_results)
        if chunk_results else 0
    )
    
    return combined

def generate_structure_summary(doc: Dict[str, Any]) -> str:
    """
    Generates a summary of the code structure.
    
    Args:
        doc: Combined documentation dictionary
        
    Returns:
        str: Formatted structure summary
    """
    parts = []
    
    # Add import summary if present
    if doc["structure"]["imports"]:
        parts.append("Imports:")
        for imp in doc["structure"]["imports"]:
            parts.append(f"  - {imp}")
    
    # Add class summary
    if doc["classes"]:
        parts.append("\nClasses:")
        for cls in doc["classes"]:
            parts.append(f"  - {cls['name']}")
            if cls.get("bases"):
                parts.append(f"    Inherits from: {', '.join(cls['bases'])}")
            if cls.get("methods"):
                parts.append("    Methods:")
                for method in cls["methods"]:
                    decorator_str = ""
                    if method.get("decorators"):
                        decorator_str = f" [{', '.join(method['decorators'])}]"
                    async_str = "async " if method.get("is_async") else ""
                    parts.append(
                        f"      - {async_str}{method['name']}{decorator_str}"
                    )
    
    # Add function summary
    if doc["functions"]:
        parts.append("\nFunctions:")
        for func in doc["functions"]:
            decorator_str = ""
            if func.get("decorators"):
                decorator_str = f" [{', '.join(func['decorators'])}]"
            async_str = "async " if func.get("is_async") else ""
            parts.append(f"  - {async_str}{func['name']}{decorator_str}")
    
    # Add variable summary
    if doc["variables"] or doc["constants"]:
        parts.append("\nModule-level variables:")
        for var in doc["variables"]:
            parts.append(f"  - {var['name']}: {var.get('type', 'unknown')}")
        for const in doc["constants"]:
            parts.append(
                f"  - {const['name']} (constant): {const.get('type', 'unknown')}"
            )
    
    return "\n".join(parts)

async def fetch_documentation_rest(
    session: aiohttp.ClientSession,
    prompt: List[Dict[str, str]],
    semaphore: asyncio.Semaphore,
    deployment_name: str,
    function_schema: Dict[str, Any],
    azure_api_key: str,
    azure_endpoint: str,
    azure_api_version: str,
    retry_count: int = 3,
    retry_delay: float = 1.0
) -> Dict[str, Any]:
    """
    Fetches documentation from Azure OpenAI API with retry logic.
    
    Args:
        session: aiohttp session
        prompt: List of messages forming the prompt
        semaphore: Controls concurrent API requests
        deployment_name: Azure OpenAI deployment name
        function_schema: Schema for documentation generation
        azure_api_key: Azure OpenAI API key
        azure_endpoint: Azure OpenAI endpoint
        azure_api_version: Azure OpenAI API version
        retry_count: Number of retry attempts
        retry_delay: Base delay between retries
        
    Returns:
        Dict[str, Any]: Generated documentation
        
    Raises:
        Exception: If all retry attempts fail
    """
    url = f"{azure_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={azure_api_version}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {azure_api_key}",
    }

    payload = {
        "messages": prompt,
        "max_tokens": 1500,
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1,
        "stop": None,
        "functions": function_schema["functions"],
        "function_call": {"name": "generate_documentation"}
    }

    for attempt in range(retry_count):
        try:
            async with semaphore:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        choice = data.get("choices", [{}])[0]
                        message = choice.get("message", {})
                        
                        if "function_call" in message:
                            function_call = message["function_call"]
                            arguments = function_call.get("arguments", "{}")
                            try:
                                return json.loads(arguments)
                            except json.JSONDecodeError as e:
                                logger.error(f"Error parsing function arguments: {e}")
                                raise
                        else:
                            logger.error("No function call in response")
                            raise ValueError("No function call in response")
                            
                    elif response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get("Retry-After", retry_delay))
                        logger.warning(f"Rate limited. Retrying after {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                        
                    elif response.status == 401:
                        raise ValueError("Unauthorized. Check API key and endpoint.")
                        
                    else:
                        error_text = await response.text()
                        raise ValueError(
                            f"API request failed with status {response.status}: "
                            f"{error_text}"
                        )
                        
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < retry_count - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(
                    f"Network error (attempt {attempt + 1}/{retry_count}). "
                    f"Retrying in {wait_time}s: {str(e)}"
                )
                await asyncio.sleep(wait_time)
            else:
                raise

    raise Exception(f"All {retry_count} attempts failed")