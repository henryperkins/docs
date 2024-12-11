import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Set
from pathlib import Path
import aiohttp
import random

from provider_config import ProviderConfig
from token_utils import TokenManager
from chunk import ChunkManager
from dependency_analyzer import DependencyAnalyzer
from context import HierarchicalContextManager
from utils import should_process_file, get_language, FileHandler, write_documentation_report, handle_api_error
from metrics import MetricsManager, ProcessingResult

logger = logging.getLogger(__name__)


class APIHandler:
    """Handles API interactions with error handling and rate limiting."""

    def __init__(self, session: aiohttp.ClientSession, config: ProviderConfig, semaphore: asyncio.Semaphore, metrics_manager: MetricsManager):
        self.session = session
        self.config = config
        self.semaphore = semaphore
        self.metrics_manager = metrics_manager
        self._rate_limit_tokens = {}
        self._rate_limit_lock = asyncio.Lock()

    async def fetch_completion(self, prompt: List[Dict[str, str]], provider: str) -> ProcessingResult:
        """Fetches completion with error handling and rate limiting."""
        start_time = datetime.now()
        attempt = 0
        last_error = None

        while attempt < self.config.max_retries:
            try:
                await self._wait_for_rate_limit(provider)
                async with self.semaphore:
                    result = await self._make_api_request(prompt, provider)
                    latency = (datetime.now() - start_time).total_seconds()
                    tokens_used = TokenManager.count_tokens(result).token_count
                    self.metrics_manager.record_api_call(
                        provider, latency, tokens_used, success=True)
                    await self._update_rate_limits(provider, result)

                    processing_time = (
                        datetime.now() - start_time).total_seconds()
                    return ProcessingResult(success=True, content=result, processing_time=processing_time)

            except Exception as e:
                should_retry = handle_api_error(
                    e, attempt, self.config.max_retries)
                last_error = e

            if should_retry and attempt < self.config.max_retries - 1:
                attempt += 1
                delay = self._calculate_retry_delay(attempt, str(last_error))
                logger.warning(
                    f"Retry {attempt}/{self.config.max_retries} after {delay}s. Error: {type(last_error).__name__}")
                await asyncio.sleep(delay)
                continue
            else:
                logger.error(
                    f"API request failed: {type(last_error).__name__}")
                break

        processing_time = (datetime.now() - start_time).total_seconds()
        self.metrics_manager.record_api_call(
            provider, processing_time, 0, success=False, error_type=str(last_error))
        return ProcessingResult(success=False, error=str(last_error), retries=attempt, processing_time=processing_time)

    async def _make_api_request(self, prompt: List[Dict[str, str]], provider: str) -> Dict[str, Any]:
        """Makes actual API request based on provider."""
        fetch_method = {
            "azure": self._fetch_azure_completion,
            "gemini": self._fetch_gemini_completion,
            "openai": self._fetch_openai_completion
        }.get(provider)

        if fetch_method:
            return await fetch_method(prompt)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _fetch_azure_completion(self, prompt: List[Dict[str, str]]) -> Dict[str, Any]:
        """Fetches completion from Azure OpenAI service."""
        return await self._fetch_completion_generic(prompt, self.config.endpoint, self.config.deployment_name, self.config.api_key, self.config.api_version)

    async def _fetch_gemini_completion(self, prompt: List[Dict[str, str]]) -> Dict[str, Any]:
        """Fetches completion from Gemini AI service."""
        return await self._fetch_completion_generic(prompt, self.config.endpoint, "", self.config.api_key, "")

    async def _fetch_openai_completion(self, prompt: List[Dict[str, str]]) -> Dict[str, Any]:
        """Fetches completion from OpenAI service."""
        return await self._fetch_completion_generic(prompt, self.config.endpoint, "", self.config.api_key, "")

    async def _fetch_completion_generic(self, prompt: List[Dict[str, str]], endpoint: str, deployment_name: str, api_key: str, api_version: str) -> Dict[str, Any]:
        """Generic method to fetch completion from any AI service."""
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {api_key}"}
        params = {"api-version": api_version} if api_version else {}
        payload = {"messages": prompt, "max_tokens": self.config.max_tokens,
                   "temperature": self.config.temperature}
        url = f"{endpoint}/v1/chat/completions"

        async with self.session.post(url, headers=headers, params=params, json=payload, timeout=self.config.timeout) as response:
            response.raise_for_status()
            return await response.json()

    async def _wait_for_rate_limit(self, provider: str) -> None:
        """Waits if rate limit is reached."""
        async with self._rate_limit_lock:
            if provider in self._rate_limit_tokens:
                tokens, reset_time = self._rate_limit_tokens[provider]
                if tokens <= 0 and datetime.now() < reset_time:
                    wait_time = (reset_time - datetime.now()).total_seconds()
                    logger.warning(
                        f"Rate limit reached for {provider}. Waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)

    async def _update_rate_limits(self, provider: str, response: Dict[str, Any]) -> None:
        """Updates rate limit tracking based on response headers."""
        async with self._rate_limit_lock:
            headers = response.get("headers", {})
            remaining = int(headers.get("x-ratelimit-remaining", 1))
            reset = int(headers.get("x-ratelimit-reset", 0))

            if reset > 0:
                reset_time = datetime.fromtimestamp(reset)
                self._rate_limit_tokens[provider] = (remaining, reset_time)

    def _calculate_retry_delay(self, attempt: int, error_message: str) -> float:
        """Calculates retry delay with exponential backoff and jitter."""
        base_delay = self.config.retry_delay
        max_delay = min(base_delay * (2 ** attempt), 60)
        jitter = random.uniform(-0.25, 0.25) * base_delay

        if "rate limit" in error_message.lower():
            max_delay *= 1.5

        return max(0.1, min(max_delay + jitter, 60))


class FileProcessor:
    """Processes files and generates documentation."""

    def __init__(self, context_manager: HierarchicalContextManager, api_handler: APIHandler, provider_config: ProviderConfig, metrics_manager: MetricsManager):
        self.context_manager = context_manager
        self.api_handler = api_handler
        self.provider_config = provider_config
        self.metrics_manager = metrics_manager
        self.chunk_manager = ChunkManager(
            max_tokens=provider_config.max_tokens, overlap=provider_config.chunk_overlap)
        self.dependency_analyzer = DependencyAnalyzer()

    async def process_file(self, file_path: str, skip_types: Set[str], project_info: str, style_guidelines: str, repo_root: str, output_dir: str, provider: str, project_id: str, safe_mode: bool = False) -> ProcessingResult:
        """Processes a single file and generates documentation."""
        start_time = datetime.now()

        try:
            if not should_process_file(file_path, skip_types):
                return ProcessingResult(success=False, error="File type excluded", processing_time=0.0)

            content = await FileHandler.read_file(file_path)
            if content is None:
                return ProcessingResult(success=False, error="Failed to read file", processing_time=0.0)

            language = get_language(file_path)
            if not language:
                return ProcessingResult(success=False, error="Unsupported language", processing_time=0.0)

            # Create chunks using ChunkManager
            chunks = self.chunk_manager.create_chunks(
                content, file_path, language)
            self.metrics_manager.processing_metrics.total_chunks += len(chunks)

            # Add chunks to context manager
            for chunk in chunks:
                await self.context_manager.add_code_chunk(chunk)

            # Analyze dependencies using DependencyAnalyzer
            dependencies = self.dependency_analyzer.analyze(content)

            chunk_results = await self._process_chunks(chunks, project_info, style_guidelines, provider)

            combined_doc = await self._combine_documentation(chunk_results, file_path, language)

            if not combined_doc:
                return ProcessingResult(success=False, error="Failed to combine documentation", processing_time=(datetime.now() - start_time).total_seconds())

            if not safe_mode:
                doc_result = await write_documentation_report(documentation=combined_doc, language=language, file_path=file_path, repo_root=repo_root, output_dir=output_dir, project_id=project_id)

                if not doc_result:
                    return ProcessingResult(success=False, error="Failed to write documentation", processing_time=(datetime.now() - start_time).total_seconds())

            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics_manager.record_file_processing(
                success=True, processing_time=processing_time)

            return ProcessingResult(success=True, content=combined_doc, processing_time=processing_time)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics_manager.record_file_processing(
                success=False, processing_time=processing_time, error_type=str(e))
            return ProcessingResult(success=False, error=str(e), processing_time=processing_time)

    async def _process_chunks(self, chunks: List[CodeChunk], project_info: str, style_guidelines: str, provider: str) -> List[ProcessingResult]:
        """Processes chunks in parallel."""
        results = []
        max_parallel_chunks = self.provider_config.max_parallel_chunks

        for i in range(0, len(chunks), max_parallel_chunks):
            chunk_group = chunks[i:i + max_parallel_chunks]
            tasks = [self._process_chunk(
                chunk, project_info, style_guidelines, provider) for chunk in chunk_group]

            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            for chunk, result in zip(chunk_group, group_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process chunk: {str(result)}")
                    results.append(ProcessingResult(
                        success=False, error=str(result)))
                else:
                    results.append(result)
                    if result.success and result.content:
                        try:
                            await self.context_manager.add_code_chunk(chunk)
                        except Exception as e:
                            logger.warning(
                                f"Failed to store chunk result: {str(e)}")
                        self.metrics_manager.processing_metrics.successful_chunks += 1

        return results

    async def _process_chunk(self, chunk: CodeChunk, project_info: str, style_guidelines: str, provider: str) -> ProcessingResult:
        """Processes a single code chunk."""
        try:
            prompt = self._build_prompt(chunk, project_info, style_guidelines)

            result = await self.api_handler.fetch_completion(prompt, provider)

            if result.success:
                content = self._extract_content(result.content)
                return ProcessingResult(success=True, content=content, processing_time=result.processing_time)
            else:
                return ProcessingResult(success=False, error=result.error, processing_time=result.processing_time)

        except Exception as e:
            logger.error(f"Error processing chunk {chunk.chunk_id}: {str(e)}")
            return ProcessingResult(success=False, error=str(e), processing_time=0.0)

    def _build_prompt(self, chunk: CodeChunk, project_info: str, style_guidelines: str) -> List[Dict[str, str]]:
        """Builds the prompt for the AI model."""
        prompt = [{"role": "system", "content": "You are a helpful assistant."}, {
            "role": "user", "content": f"Please generate documentation for the following code:\n\n{chunk.chunk_content}"}]
        if project_info:
            prompt.append(
                {"role": "user", "content": f"Project information:\n{project_info}"})
        if style_guidelines:
            prompt.append(
                {"role": "user", "content": f"Style guidelines:\n{style_guidelines}"})
        return prompt

    def _extract_content(self, api_response: Dict[str, Any]) -> str:
        """Extracts content from the API response."""
        choices = api_response.get('choices', [])
        if choices:
            return choices[0].get('message', {}).get('content', '')
        return ''

    async def _combine_documentation(self, chunk_results: List[ProcessingResult], file_path: str, language: str) -> str:
        """Combines documentation from chunk results."""
        documentation = ""
        for result in chunk_results:
            if result.success and result.content:
                documentation += result.content + "\n\n"
        return documentation.strip()
