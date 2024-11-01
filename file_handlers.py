"""
file_handlers.py

Contains classes and functions for handling file processing, API interactions,
chunk management, and context management for the documentation generation process.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from datetime import datetime
import aiohttp

from utils import (
    should_process_file,
    get_language,
    FileHandler,
    CodeFormatter,
    TokenManager,
    ChunkAnalyzer,
    ProcessingResult,
    CodeChunk,
    ChunkValidationError,
    ChunkTooLargeError,
    ChunkingError,
    HierarchicalContextManager,
    write_documentation_report,
    handle_api_error  # New utility function for error handling
)
from metrics_combined import (
    ProviderMetrics,
    ProcessingMetrics,
    MetricsCalculator
)
from chunk_manager import ChunkManager  # Import the new ChunkManager

logger = logging.getLogger(__name__)

class APIHandler:
    """Enhanced API interaction handler with better error handling and rate limiting."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        config: 'ProviderConfig',
        semaphore: asyncio.Semaphore,
        provider_metrics: 'ProviderMetrics'
    ):
        self.session = session
        self.config = config
        self.semaphore = semaphore
        self.provider_metrics = provider_metrics
        self._rate_limit_tokens = {}  # Track rate limits per endpoint
        self._rate_limit_lock = asyncio.Lock()

    async def fetch_completion(
        self,
        prompt: List[Dict[str, str]],
        provider: str
    ) -> ProcessingResult:
        """
        Fetches completion with enhanced error handling and rate limiting.

        Args:
            prompt: Formatted prompt messages
            provider: AI provider name

        Returns:
            ProcessingResult: Processing result
        """
        start_time = datetime.now()
        attempt = 0
        last_error = None

        while attempt < self.config.max_retries:
            try:
                # Check rate limits
                await self._wait_for_rate_limit(provider)

                async with self.semaphore:
                    # Record API call attempt
                    self.provider_metrics.api_calls += 1

                    # Make API request
                    result = await self._make_api_request(prompt, provider)

                    # Calculate latency
                    latency = (datetime.now() - start_time).total_seconds()
                    self.provider_metrics.update_latency(latency)

                    # Update total tokens used
                    tokens_used = self._extract_tokens_used(result)
                    self.provider_metrics.total_tokens += tokens_used

                    # Update rate limit tracking
                    await self._update_rate_limits(provider, result)

                    processing_time = (
                        datetime.now() - start_time
                    ).total_seconds()

                    return ProcessingResult(
                        success=True,
                        content=result,
                        processing_time=processing_time
                    )

            except Exception as e:
                should_retry = handle_api_error(e, attempt, self.config.max_retries)
                last_error = e

            if should_retry and attempt < self.config.max_retries - 1:
                attempt += 1
                self.provider_metrics.retry_count += 1
                delay = self._calculate_retry_delay(attempt, str(last_error))
                logger.warning(
                    f"Retry {attempt}/{self.config.max_retries} "
                    f"after {delay}s. Error: {type(last_error).__name__}"
                )
                await asyncio.sleep(delay)
                continue
            else:
                error_msg = f"API request failed: {type(last_error).__name__}"
                logger.error(error_msg)
                break

        processing_time = (datetime.now() - start_time).total_seconds()
        return ProcessingResult(
            success=False,
            error=str(last_error),
            retries=attempt,
            processing_time=processing_time
        )

    async def _make_api_request(
        self,
        prompt: List[Dict[str, str]],
        provider: str
    ) -> Dict[str, Any]:
        """Makes actual API request based on provider."""
        try:
            if provider == "azure":
                return await self._fetch_azure_completion(prompt)
            elif provider == "gemini":
                return await self._fetch_gemini_completion(prompt)
            elif provider == "openai":
                return await self._fetch_openai_completion(prompt)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            logger.error(f"API request error for {provider}: {str(e)}")
            raise

    async def _fetch_azure_completion(
        self,
        prompt: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Fetches completion from Azure OpenAI service."""
        headers = {
            "Content-Type": "application/json",
            "api-key": self.config.api_key
        }
        params = {
            "api-version": self.config.api_version
        }
        payload = {
            "messages": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        url = f"{self.config.endpoint}/openai/deployments/{self.config.deployment_name}/chat/completions"

        async with self.session.post(
            url, headers=headers, params=params, json=payload, timeout=self.config.timeout
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def _fetch_gemini_completion(
        self,
        prompt: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Fetches completion from Gemini AI service."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        payload = {
            "messages": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "model": self.config.model_name
        }
        url = f"{self.config.endpoint}/v1/chat/completions"

        async with self.session.post(
            url, headers=headers, json=payload, timeout=self.config.timeout
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def _fetch_openai_completion(
        self,
        prompt: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Fetches completion from OpenAI service."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        payload = {
            "messages": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "model": self.config.model_name
        }
        url = f"{self.config.endpoint}/v1/chat/completions"

        async with self.session.post(
            url, headers=headers, json=payload, timeout=self.config.timeout
        ) as response:
            response.raise_for_status()
            return await response.json()

    def _extract_tokens_used(self, response: Dict[str, Any]) -> int:
        """Extracts the number of tokens used from the API response."""
        usage = response.get('usage', {})
        return usage.get('total_tokens', 0)

    async def _wait_for_rate_limit(self, provider: str) -> None:
        """Waits if rate limit is reached."""
        async with self._rate_limit_lock:
            if provider in self._rate_limit_tokens:
                tokens, reset_time = self._rate_limit_tokens[provider]
                if tokens <= 0 and datetime.now() < reset_time:
                    wait_time = (reset_time - datetime.now()).total_seconds()
                    logger.warning(
                        f"Rate limit reached for {provider}. "
                        f"Waiting {wait_time:.1f}s"
                    )
                    self.provider_metrics.rate_limit_hits += 1
                    await asyncio.sleep(wait_time)

    async def _update_rate_limits(
        self,
        provider: str,
        response: Dict[str, Any]
    ) -> None:
        """Updates rate limit tracking based on response headers."""
        async with self._rate_limit_lock:
            # Extract rate limit info from response headers
            headers = response.get("headers", {})
            remaining = int(headers.get("x-ratelimit-remaining", 1))
            reset = int(headers.get("x-ratelimit-reset", 0))

            if reset > 0:
                reset_time = datetime.fromtimestamp(reset)
                self._rate_limit_tokens[provider] = (remaining, reset_time)

    def _calculate_retry_delay(
        self,
        attempt: int,
        error_message: str
    ) -> float:
        """Calculates retry delay with exponential backoff and jitter."""
        base_delay = self.config.retry_delay
        max_delay = min(base_delay * (2 ** attempt), 60)  # Cap at 60 seconds

        # Add jitter (±25% of base delay)
        import random
        jitter = random.uniform(-0.25, 0.25) * base_delay

        # Increase delay for rate limit errors
        if "rate limit" in error_message.lower():
            max_delay *= 1.5

        return max(0.1, min(max_delay + jitter, 60))

class FileProcessor:
    """Enhanced file processing with improved error handling and metrics."""

    def __init__(
        self,
        context_manager: HierarchicalContextManager,
        api_handler: APIHandler,
        provider_config: 'ProviderConfig',
        provider_metrics: 'ProviderMetrics'
    ):
        self.context_manager = context_manager
        self.api_handler = api_handler
        self.provider_config = provider_config
        self.provider_metrics = provider_metrics
        self.chunk_manager = ChunkManager(self.provider_config)
        self.metrics_calculator = MetricsCalculator()

    async def process_file(
        self,
        file_path: str,
        skip_types: Set[str],
        project_info: str,
        style_guidelines: str,
        repo_root: str,
        output_dir: str,
        provider: str,
        project_id: str,
        safe_mode: bool = False
    ) -> ProcessingResult:
        """Processes a single file with enhanced error handling."""
        start_time = datetime.now()

        try:
            # Basic validation
            if not should_process_file(file_path, skip_types):
                return ProcessingResult(
                    success=False,
                    error="File type excluded",
                    processing_time=0.0
                )

            # Read file content
            content = await FileHandler.read_file(file_path)
            if content is None:
                return ProcessingResult(
                    success=False,
                    error="Failed to read file",
                    processing_time=0.0
                )

            # Get language and validate
            language = get_language(file_path)
            if not language:
                return ProcessingResult(
                    success=False,
                    error="Unsupported language",
                    processing_time=0.0
                )

            # Create and process chunks
            try:
                chunks = self.chunk_manager.create_chunks(
                    content,
                    file_path,
                    language
                )
                self.provider_metrics.total_chunks += len(chunks)
            except ChunkingError as e:
                return ProcessingResult(
                    success=False,
                    error=f"Chunking error: {str(e)}",
                    processing_time=(
                        datetime.now() - start_time
                    ).total_seconds()
                )

            # Process chunks
            chunk_results = await self._process_chunks(
                chunks=chunks,
                project_info=project_info,
                style_guidelines=style_guidelines,
                provider=provider
            )

            # Combine documentation
            combined_doc = await self._combine_documentation(
                chunk_results=chunk_results,
                file_path=file_path,
                language=language
            )

            if not combined_doc:
                return ProcessingResult(
                    success=False,
                    error="Failed to combine documentation",
                    processing_time=(
                        datetime.now() - start_time
                    ).total_seconds()
                )

            # Write documentation if not in safe mode
            if not safe_mode:
                doc_result = await write_documentation_report(
                    documentation=combined_doc,
                    language=language,
                    file_path=file_path,
                    repo_root=repo_root,
                    output_dir=output_dir,
                    project_id=project_id
                )

                if not doc_result:
                    return ProcessingResult(
                        success=False,
                        error="Failed to write documentation",
                        processing_time=(
                            datetime.now() - start_time
                        ).total_seconds()
                    )

            return ProcessingResult(
                success=True,
                content=combined_doc,
                processing_time=(
                    datetime.now() - start_time
                ).total_seconds()
            )

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=(
                    datetime.now() - start_time
                ).total_seconds()
            )

    async def _process_chunks(
        self,
        chunks: List[CodeChunk],
        project_info: str,
        style_guidelines: str,
        provider: str
    ) -> List[ProcessingResult]:
        """Processes chunks with improved parallel handling."""
        results = []
        max_parallel_chunks = self.provider_config.max_parallel_chunks

        for i in range(0, len(chunks), max_parallel_chunks):
            chunk_group = chunks[i:i + max_parallel_chunks]
            tasks = [
                self._process_chunk(
                    chunk=chunk,
                    project_info=project_info,
                    style_guidelines=style_guidelines,
                    provider=provider
                )
                for chunk in chunk_group
            ]

            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            for chunk, result in zip(chunk_group, group_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process chunk: {str(result)}")
                    results.append(ProcessingResult(
                        success=False,
                        error=str(result)
                    ))
                else:
                    results.append(result)
                    if result.success and result.content:
                        # Store successful results in context manager
                        try:
                            await self.context_manager.add_doc_chunk(
                                chunk.chunk_id,
                                result.content
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to store chunk result: {str(e)}"
                            )
                        # Update successful chunks count
                        self.provider_metrics.successful_chunks += 1

        return results

    async def _process_chunk(
        self,
        chunk: CodeChunk,
        project_info: str,
        style_guidelines: str,
        provider: str
    ) -> ProcessingResult:
        """Processes a single code chunk."""
        try:
            prompt = self._build_prompt(
                chunk=chunk,
                project_info=project_info,
                style_guidelines=style_guidelines
            )

            result = await self.api_handler.fetch_completion(
                prompt=prompt,
                provider=provider
            )

            if result.success:
                # Extract content from API response
                content = self._extract_content(result.content)
                return ProcessingResult(
                    success=True,
                    content=content,
                    processing_time=result.processing_time
                )
            else:
                return ProcessingResult(
                    success=False,
                    error=result.error,
                    processing_time=result.processing_time
                )

        except Exception as e:
            logger.error(f"Error processing chunk {chunk.chunk_id}: {str(e)}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=0.0
            )

    def _build_prompt(
        self,
        chunk: CodeChunk,
        project_info: str,
        style_guidelines: str
    ) -> List[Dict[str, str]]:
        """Builds the prompt for the AI model."""
        prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please generate documentation for the following code:\n\n{chunk.content}"}
        ]
        if project_info:
            prompt.append({"role": "user", "content": f"Project information:\n{project_info}"})
        if style_guidelines:
            prompt.append({"role": "user", "content": f"Style guidelines:\n{style_guidelines}"})
        return prompt

    def _extract_content(self, api_response: Dict[str, Any]) -> str:
        """Extracts content from the API response."""
        choices = api_response.get('choices', [])
        if choices:
            return choices[0].get('message', {}).get('content', '')
        return ''

    async def _combine_documentation(
        self,
        chunk_results: List[ProcessingResult],
        file_path: str,
        language: str
    ) -> str:
        """Combines documentation from chunk results."""
        documentation = ""
        for result in chunk_results:
            if result.success and result.content:
                documentation += result.content + "\n\n"
        return documentation.strip()
