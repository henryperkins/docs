# metrics.py

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

# Custom Exceptions
class MetricsCalculationError(Exception):
    """Base exception for metrics calculation errors."""
    pass

@dataclass
class ComplexityMetrics:
    """Container for complexity metrics with default values."""
    cyclomatic_complexity: float = 0.0
    maintainability_index: float = 100.0
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    halstead_effort: float = 0.0
    halstead_bugs: float = 0.0
    halstead_time: float = 0.0
    type_hint_coverage: float = 0.0

@dataclass
class ProviderMetrics:
    """Tracks metrics for an AI provider."""
    api_calls: int = 0
    api_errors: int = 0
    total_tokens: int = 0
    average_latency: float = 0.0
    rate_limit_hits: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retry_count: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)
    successful_chunks: int = 0

    def update_latency(self, latency: float) -> None:
        """Updates average latency with new value."""
        if self.api_calls == 0:
            self.average_latency = latency
        else:
            self.average_latency = (
                (self.average_latency * (self.api_calls - 1) + latency) /
                self.api_calls
            )

    def record_error(self, error_type: str) -> None:
        """Records an API error."""
        self.api_errors += 1
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of provider metrics."""
        error_rate = (self.api_errors / self.api_calls * 100) if self.api_calls > 0 else 0
        tokens_per_call = (self.total_tokens / self.api_calls) if self.api_calls > 0 else 0
        hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0
        retries_per_call = (self.retry_count / self.api_calls) if self.api_calls > 0 else 0

        return {
            "api_calls": self.api_calls,
            "api_errors": self.api_errors,
            "error_rate": error_rate,
            "average_latency": self.average_latency,
            "total_tokens": self.total_tokens,
            "tokens_per_call": tokens_per_call,
            "rate_limit_hits": self.rate_limit_hits,
            "cache_efficiency": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": hit_rate
            },
            "retries": {
                "total": self.retry_count,
                "per_call": retries_per_call
            },
            "error_breakdown": self.error_types,
            "successful_chunks": self.successful_chunks
        }

@dataclass
class ProcessingMetrics:
    """Enhanced processing metrics with detailed tracking."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_files: int = 0
    processed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    successful_chunks: int = 0
    provider_metrics: Dict[str, ProviderMetrics] = field(default_factory=lambda: {})
    error_types: Dict[str, int] = field(default_factory=dict)
    processing_times: List[float] = field(default_factory=list)

    def record_file_result(
        self,
        success: bool,
        processing_time: float,
        error_type: Optional[str] = None
    ) -> None:
        """Records file processing result."""
        self.processed_files += 1
        if success:
            self.successful_files += 1
        else:
            self.failed_files += 1
            if error_type:
                self.error_types[error_type] = (
                    self.error_types.get(error_type, 0) + 1
                )
        self.processing_times.append(processing_time)

    def get_provider_metrics(self, provider: str) -> ProviderMetrics:
        """Gets or creates metrics for a provider."""
        if provider not in self.provider_metrics:
            self.provider_metrics[provider] = ProviderMetrics()
        return self.provider_metrics[provider]

    def get_summary(self) -> Dict[str, Any]:
        """Returns a comprehensive metrics summary."""
        duration = (
            (self.end_time - self.start_time).total_seconds()
            if self.end_time
            else (datetime.now() - self.start_time).total_seconds()
        )
        return {
            "duration": {
                "seconds": duration,
                "formatted": str(datetime.now() - self.start_time)
            },
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
            "processing_times": {
                "average": (
                    sum(self.processing_times) / len(self.processing_times)
                    if self.processing_times
                    else 0
                ),
                "min": min(self.processing_times) if self.processing_times else 0,
                "max": max(self.processing_times) if self.processing_times else 0
            },
            "providers": {
                provider: metrics.get_summary()
                for provider, metrics in self.provider_metrics.items()
            },
            "errors": {
                "total": self.failed_files,
                "types": self.error_types,
                "rate": (
                    self.failed_files / self.processed_files * 100
                    if self.processed_files > 0
                    else 0
                )
            }
        }

class MetricsManager:
    """Centralized manager for metrics calculation and tracking."""

    def __init__(self):
        self.processing_metrics = ProcessingMetrics()

    def record_api_call(self, provider: str, latency: float, tokens: int, success: bool, error_type: Optional[str] = None):
        """Records an API call for a specific provider."""
        provider_metrics = self.processing_metrics.get_provider_metrics(provider)
        provider_metrics.api_calls += 1
        if success:
            provider_metrics.update_latency(latency)
            provider_metrics.total_tokens += tokens
        else:
            provider_metrics.record_error(error_type or "Unknown")

    def record_file_processing(self, success: bool, processing_time: float, error_type: Optional[str] = None):
        """Records the result of processing a file."""
        self.processing_metrics.record_file_result(success, processing_time, error_type)

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of all metrics."""
        return self.processing_metrics.get_summary()
