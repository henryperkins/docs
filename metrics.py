"""
metrics.py - Enhanced version with improved error handling, debugging,
thread pool management, specific exception handling in the decorator,
and metrics validation.
"""

import ast
import logging
import traceback
import asyncio
import aiofiles
import json
from typing import Dict, Any, TypedDict, Optional, Union, List, Callable, TypeVar
from dataclasses import dataclass
from datetime import datetime
from functools import wraps, partial
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
import os

from radon.metrics import h_visit, mi_visit
from radon.complexity import cc_visit, ComplexityVisitor

# Configure logging
logger = logging.getLogger(__name__)

DEFAULT_EMPTY_METRICS = {
    "maintainability_index": 0.0,
    "cyclomatic": 0.0,
    "halstead": {
        "h1": 0,
        "h2": 0,
        "N1": 0,
        "N2": 0,
        "vocabulary": 0,
        "length": 0,
        "calculated_length": 0.0,
        "volume": 0.0,
        "difficulty": 0.0,
        "effort": 0.0,
        "time": 0.0,
        "bugs": 0.0,
    },
    "function_complexity": {},
    "raw": None,
    "quality": None,
}

# Custom exception classes
class MetricsCalculationError(Exception):
    """Base exception for metrics calculation errors"""
    pass

class HalsteadCalculationError(MetricsCalculationError):
    """Exception for Halstead metrics calculation errors"""
    pass

class ComplexityCalculationError(MetricsCalculationError):
    """Exception for complexity calculation errors"""
    pass

@dataclass
class MetricsResult:
    """Data class for storing metrics calculation results"""
    file_path: str
    timestamp: datetime
    execution_time: float
    success: bool
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

# Create a global thread pool with a limited number of threads
MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

T = TypeVar('T')  # Generic type variable for the return value

def safe_metric_calculation(default_value: T = None, metric_name: str = "metric") -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for safe metric calculation with specific error handling."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except ValueError as e:
                error_message = f"ValueError during {metric_name} calculation: {e}"
                logger.error(error_message)
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                return default_value
            except TypeError as e:
                error_message = f"TypeError during {metric_name} calculation: {e}"
                logger.error(error_message)
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                return default_value
            except Exception as e:
                error_message = f"Unexpected error during {metric_name} calculation: {e}"
                logger.error(error_message)
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                return default_value
        return wrapper
    return decorator

def log_execution_time(func):
    """Decorator to log function execution time"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = perf_counter()
        try:
            result = await func(*args, **kwargs)
            execution_time = perf_counter() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper

class HalsteadMetrics(TypedDict):
    """TypedDict for Halstead metrics"""
    h1: int
    h2: int
    N1: int
    N2: int
    vocabulary: int
    length: int
    calculated_length: float
    volume: float
    difficulty: float
    effort: float
    time: float
    bugs: float

class CodeMetrics(TypedDict):
    """TypedDict for code metrics"""
    maintainability_index: float
    cyclomatic: float
    halstead: HalsteadMetrics
    function_complexity: Dict[str, float]
    raw: Any
    quality: Any

@dataclass
class MetricsThresholds:
    """Thresholds for different metrics"""
    maintainability_low: float = 20.0
    complexity_high: int = 15
    halstead_effort_high: float = 1000000.0

@safe_metric_calculation(default_value=HalsteadMetrics(h1=0, h2=0, N1=0, N2=0, vocabulary=0, length=0, calculated_length=0.0, volume=0.0, difficulty=0.0, effort=0.0, time=0.0, bugs=0.0), metric_name="Halstead Metrics")
def calculate_halstead_metrics(code: str) -> HalsteadMetrics:
    """Calculates Halstead complexity metrics."""

    halstead_reports = h_visit(code)

    if not halstead_reports:
        raise HalsteadCalculationError("No Halstead metrics found")

    metrics = halstead_reports[0] if isinstance(halstead_reports, list) else halstead_reports

    return HalsteadMetrics(
        h1=metrics.h1, h2=metrics.h2, N1=metrics.N1, N2=metrics.N2,
        vocabulary=metrics.vocabulary, length=metrics.length,
        calculated_length=metrics.calculated_length, volume=metrics.volume,
        difficulty=metrics.difficulty, effort=metrics.effort, time=metrics.time,
        bugs=metrics.bugs
    )

@safe_metric_calculation(default_value=0.0, metric_name="Maintainability Index")
def calculate_maintainability_index(code: str) -> float:
    """Calculates the Maintainability Index."""
    mi_value = mi_visit(code, multi=False)
    return float(mi_value)

@safe_metric_calculation(default_value={}, metric_name="Cyclomatic Complexity")
def calculate_complexity(code: str) -> Dict[str, float]:
    """Calculates Cyclomatic Complexity."""
    complexity_visitor = ComplexityVisitor.from_code(code)
    function_complexity = {}
    for block in complexity_visitor.functions + complexity_visitor.classes:
        function_complexity[block.name] = float(block.complexity)
    return function_complexity

@log_execution_time
async def calculate_code_metrics(
    code: str,
    file_path: Optional[str] = None,
    language: str = "python"
) -> Union[CodeMetrics, MetricsResult]:
    """Calculates code metrics, handling different languages, with optimized thread pooling and validation."""
    start_time = perf_counter()
    logger.info(f"Starting metrics calculation for {file_path or 'unknown file'}")

    try:
        if not isinstance(code, str):
            raise ValueError(f"Invalid code type: {type(code)}")

        if not code.strip():
            raise ValueError("Empty code provided")

        metrics_data = {}

        if language.lower() == "python":
            halstead_func = partial(calculate_halstead_metrics, code)
            maintainability_func = partial(calculate_maintainability_index, code)
            complexity_func = partial(calculate_complexity, code)

            loop = asyncio.get_running_loop()
            try:
                halstead_task = loop.run_in_executor(thread_pool, halstead_func)
                maintainability_task = loop.run_in_executor(thread_pool, maintainability_func)
                complexity_task = loop.run_in_executor(thread_pool, complexity_func)

                metrics_data['halstead'], metrics_data['maintainability_index'], metrics_data['function_complexity'] = await asyncio.gather(
                    halstead_task, maintainability_task, complexity_task
                )

                complexities = list(metrics_data['function_complexity'].values())
                metrics_data['cyclomatic'] = sum(complexities) / len(complexities) if complexities else 0.0

            except asyncio.TimeoutError as e:
                logger.error(f"Metrics calculation timed out: {e}")
                metrics_data['halstead'] = HalsteadMetrics(h1=0, h2=0, N1=0, N2=0, vocabulary=0, length=0, calculated_length=0.0, volume=0.0, difficulty=0.0, effort=0.0, time=0.0, bugs=0.0)
                metrics_data['maintainability_index'] = 0.0
                metrics_data['function_complexity'] = {}
                metrics_data['cyclomatic'] = 0.0
            except Exception as e:
                logger.error(f"Error calculating Python metrics: {e}", exc_info=True)
                metrics_data['halstead'] = HalsteadMetrics(h1=0, h2=0, N1=0, N2=0, vocabulary=0, length=0, calculated_length=0.0, volume=0.0, difficulty=0.0, effort=0.0, time=0.0, bugs=0.0)
                metrics_data['maintainability_index'] = 0.0
                metrics_data['function_complexity'] = {}
                metrics_data['cyclomatic'] = 0.0
        else:
            metrics_data['halstead'] = HalsteadMetrics(h1=0, h2=0, N1=0, N2=0, vocabulary=0, length=0, calculated_length=0.0, volume=0.0, difficulty=0.0, effort=0.0, time=0.0, bugs=0.0)
            metrics_data['maintainability_index'] = 0.0
            metrics_data['function_complexity'] = {}
            metrics_data['cyclomatic'] = 0.0

        validated_metrics = validate_metrics(metrics_data)
        quality_score = calculate_quality_score(validated_metrics)
        validated_metrics['quality'] = quality_score

        validated_metrics['raw'] = {
            'code_size': len(code),
            'num_functions': len(metrics_data.get('function_complexity', {})),
            'calculation_time': perf_counter() - start_time
        }

        logger.info(f"Metrics calculation completed for {file_path or 'unknown file'}")
        logger.debug(f"Calculated metrics: {validated_metrics}")

        return MetricsResult(
            file_path=file_path or "unknown",
            timestamp=datetime.now(),
            execution_time=perf_counter() - start_time,
            success=True,
            metrics=validated_metrics
        )

    except Exception as e:
        error_msg = f"Metrics calculation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return MetricsResult(
            file_path=file_path or "unknown",
            timestamp=datetime.now(),
            execution_time=perf_counter() - start_time,
            success=False,
            error=error_msg
        )

def validate_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Validates calculated metrics against expected ranges and logs warnings."""
    validated = metrics.copy()

    halstead = validated.get('halstead', {})
    for metric, value in halstead.items():
        if value < 0:
            logger.warning(f"Invalid Halstead {metric}: {value} (expected non-negative)")
            validated['halstead'][metric] = 0

    mi = validated.get('maintainability_index', 0.0)
    if not 0 <= mi <= 100:
        logger.warning(f"Invalid Maintainability Index: {mi} (expected 0-100)")
        validated['maintainability_index'] = max(0, min(mi, 100))

    cyclomatic = validated.get('cyclomatic', 0.0)
    if cyclomatic < 1:
        logger.warning(f"Unusually low Cyclomatic Complexity: {cyclomatic} (expected >= 1)")
    if cyclomatic > 50:
        logger.warning(f"Very high Cyclomatic Complexity: {cyclomatic} (consider refactoring)")

    function_complexity = validated.get('function_complexity', {})
    for func, complexity in function_complexity.items():
        if complexity < 0:
            logger.warning(f"Invalid complexity for function '{func}': {complexity} (expected non-negative)")
            validated['function_complexity'][func] = 0
        if complexity > 30:
            logger.warning(f"High complexity for function '{func}': {complexity} (consider refactoring)")

    return validated

def calculate_quality_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates a normalized quality score based on metrics."""
    quality_scores = {
        'maintainability': normalize_score(metrics['maintainability_index'], 0, 100, 0.4),
        'complexity': normalize_score(metrics['cyclomatic'], 1, 30, 0.3, inverse=True),
        'halstead_effort': normalize_score(metrics['halstead'].get('effort', 0), 0, 1000000, 0.3, inverse=True)
    }
    quality_scores['overall'] = sum(quality_scores.values()) / len(quality_scores)
    return quality_scores

def normalize_score(value: float, min_val: float, max_val: float, weight: float = 1.0, inverse: bool = False) -> float:
    """Normalizes a metric value to a 0-1 scale."""
    try:
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0, normalized))
        if inverse:
            normalized = 1.0 - normalized
        return normalized * weight
    except Exception as e:
        logger.error(f"Error normalizing score: {e}")
        return 0.0

def get_default_halstead_metrics() -> HalsteadMetrics:
    """Returns default Halstead metrics."""
    return HalsteadMetrics(h1=0, h2=0, N1=0, N2=0, vocabulary=0, length=0, calculated_length=0.0, volume=0.0, difficulty=0.0, effort=0.0, time=0.0, bugs=0.0)

class MetricsAnalyzer:
    """Analyzes and aggregates metrics across multiple files."""

    def __init__(self, thresholds: Optional[MetricsThresholds] = None):
        self.metrics_history: List[MetricsResult] = []
        self.error_count = 0
        self.warning_count = 0
        self.thresholds = thresholds or MetricsThresholds()

    def add_result(self, result: MetricsResult):
        """Adds a metrics result and updates error/warning counts."""

        # Remove the run_coroutine_threadsafe call
        # result = asyncio.run_coroutine_threadsafe(result, asyncio.get_event_loop()).result()

        self.metrics_history.append(result)
        if not result.success:
            self.error_count += 1
            logger.error(f"Metrics calculation failed for {result.file_path}: {result.error}")
        elif result.metrics:  # Only check for warnings if metrics are available
            self._check_metrics_warnings(result)

    def _check_metrics_warnings(self, result: MetricsResult):
        """Checks for and logs warnings about concerning metric values."""
        metrics = result.metrics
        if metrics['maintainability_index'] < self.thresholds.maintainability_low:
            self.warning_count += 1
            logger.warning(f"Very low maintainability index ({metrics['maintainability_index']:.2f}) in {result.file_path}")
        for func, complexity in metrics['function_complexity'].items():
            if complexity > self.thresholds.complexity_high:
                self.warning_count += 1
                logger.warning(f"High cyclomatic complexity ({complexity}) in function '{func}' in {result.file_path}")
        if metrics['halstead']['effort'] > self.thresholds.halstead_effort_high:
            self.warning_count += 1
            logger.warning(f"High Halstead effort ({metrics['halstead']['effort']:.2f}) in {result.file_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Generates a summary of processed metrics."""
        successful_results = [r for r in self.metrics_history if r.success and r.metrics]
        avg_metrics = self._calculate_average_metrics(successful_results) if successful_results else None
        return {
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'processed_files': len(self.metrics_history),
            'success_rate': len(successful_results) / len(self.metrics_history) * 100 if self.metrics_history else 0,
            'average_metrics': avg_metrics,
            'execution_times': self._calculate_execution_time_summary()
        }

    def _calculate_average_metrics(self, results: List[MetricsResult]) -> Dict[str, Any]:
        """Calculates average metrics from successful results."""
        try:
            avg_maintainability = sum(r.metrics['maintainability_index'] for r in results) / len(results)
            avg_complexity = sum(sum(r.metrics['function_complexity'].values()) / max(1, len(r.metrics['function_complexity'])) for r in results) / len(results)
            avg_halstead = {
                metric: sum(r.metrics['halstead'][metric] for r in results) / len(results)
                for metric in ['volume', 'difficulty', 'effort']
            }
            return {
                'maintainability_index': avg_maintainability,
                'cyclomatic_complexity': avg_complexity,
                'halstead': avg_halstead
            }
        except Exception as e:
            logger.error(f"Error calculating average metrics: {e}")
            return None  # Or return a dictionary with default values

    def _calculate_execution_time_summary(self) -> Dict[str, float]:
        """Calculates execution time summary statistics."""
        if not self.metrics_history:
            return {'min': 0.0, 'max': 0.0, 'avg': 0.0}  # Return default values if no history
        times = [r.execution_time for r in self.metrics_history]
        return {
            'min': min(times),
            'max': max(times),
            'avg': sum(times) / len(times)
        }

    def get_problematic_files(self) -> List[Dict[str, Any]]:
        """
        Identifies files with concerning metrics based on provided thresholds.

        Returns:
            List of problematic files and their issues.
        """
        problematic_files = []

        for result in self.metrics_history:
            if not (result.success and result.metrics):
                continue

            issues = []
            metrics = result.metrics

            if metrics['maintainability_index'] < self.thresholds.maintainability_low:
                issues.append({
                    'type': 'maintainability',
                    'value': metrics['maintainability_index'],
                    'threshold': self.thresholds.maintainability_low
                })

            high_complexity_functions = [
                (func, complexity)
                for func, complexity in metrics['function_complexity'].items()
                if complexity > self.thresholds.complexity_high
            ]
            if high_complexity_functions:
                issues.append({
                    'type': 'complexity',
                    'functions': high_complexity_functions,
                    'threshold': self.thresholds.complexity_high
                })

            if metrics['halstead']['effort'] > self.thresholds.halstead_effort_high:
                issues.append({
                    'type': 'halstead_effort',
                    'value': metrics['halstead']['effort'],
                    'threshold': self.thresholds.halstead_effort_high
                })

            if issues:
                problematic_files.append({
                    'file_path': result.file_path,
                    'issues': issues
                })

        return problematic_files



# ... (End of metrics.py)
