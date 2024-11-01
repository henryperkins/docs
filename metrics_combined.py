# combined_metrics.py

"""
combined_metrics.py - Consolidated module for calculating and analyzing code metrics.

Aggregates functionalities from metrics.py, metrics_utils.py, and utils.py into a single module.
"""

import ast
import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union

import difflib
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from git import Repo, GitError, InvalidGitRepositoryError, NoSuchPathError
from radon.complexity import cc_visit, ComplexityVisitor
from radon.metrics import h_visit, mi_visit
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# --------------------------
# Custom Exceptions
# --------------------------

class MetricsCalculationError(Exception):
    """Base exception for metrics calculation errors."""
    pass

class HalsteadCalculationError(MetricsCalculationError):
    """Exception for Halstead metrics calculation errors."""
    pass

class ComplexityCalculationError(MetricsCalculationError):
    """Exception for cyclomatic complexity calculation errors."""
    pass

class BareRepositoryError(Exception):
    """Custom exception for bare repositories."""
    pass

class TokenizationError(Exception):
    """Exception raised for tokenization errors."""
    pass

class TokenAllocationError(Exception):
    """Exception raised for token allocation errors."""
    pass

class MetricsError(Exception):
    """General metrics error."""
    pass

class FileOperationError(Exception):
    """Exception raised for file operation errors."""
    pass

class CoverageFormatError(Exception):
    """Exception raised for coverage format errors."""
    pass

# --------------------------
# Data Classes
# --------------------------

@dataclass
class MetricsResult:
    """Data class for storing metrics calculation results."""
    file_path: str
    timestamp: datetime
    execution_time: float
    success: bool
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

@dataclass
class MetricsThresholds:
    """Thresholds for different metrics."""
    complexity_high: int = 15
    complexity_warning: int = 10
    maintainability_low: float = 20.0
    halstead_effort_high: float = 1000000.0
    code_churn_high: int = 1000
    code_churn_warning: int = 500
    code_duplication_high: float = 30.0  # Percentage
    code_duplication_warning: float = 10.0

    @classmethod
    def from_dict(cls, thresholds_dict: Dict[str, Any]) -> 'MetricsThresholds':
        """Creates a MetricsThresholds instance from a dictionary with validation."""
        validated_dict = {}
        for field_name, field_type in cls.__annotations__.items():
            if field_name in thresholds_dict:
                value = thresholds_dict[field_name]
                if not isinstance(value, field_type):  # Simplified type checking
                    try:
                        validated_dict[field_name] = field_type(value)  # Attempt type conversion
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Invalid value for threshold '{field_name}': {value} - {e}") from e
                else:
                    validated_dict[field_name] = value

        # Additional validation for percentage fields
        percentage_fields = ['code_duplication_high', 'code_duplication_warning']
        for pf in percentage_fields:
            if pf in validated_dict:
                pct = validated_dict[pf]
                if not (0.0 <= pct <= 100.0):
                    raise ValueError(f"Threshold '{pf}' must be between 0 and 100.")

        return cls(**validated_dict)

    @classmethod
    def load_from_file(cls, config_path: str) -> 'MetricsThresholds':
        """Loads metric thresholds from a JSON configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                thresholds_dict = json.load(f)
                logger.debug(f"Loaded thresholds from {config_path}: {thresholds_dict}")
            return cls.from_dict(thresholds_dict)
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
        except ValueError as e:  # Catch validation errors
            logger.error(f"Invalid threshold values in config: {e}")
            raise

@dataclass
class CodeMetadata:
    """Metadata extracted from code."""
    function_name: str
    variable_names: List[str]
    complexity: float
    halstead_volume: float
    maintainability_index: float

@dataclass
class TokenBudget:
    """Represents token budget allocation for context retrieval."""
    total_limit: int
    reserved_tokens: int = 100  # Reserve for metadata
    allocation_strategy: Enum = auto()

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
class CoverageData:
    """Container for coverage metrics."""
    line_rate: float = 0.0
    branch_rate: float = 0.0
    complexity: float = 0.0
    timestamp: str = ""
    source_file: str = ""

@dataclass
class TokenResult:
    """Contains token analysis results."""
    tokens: List[int]
    token_count: int
    encoding_name: str
    special_tokens: Optional[Dict[str, int]] = None
    error: Optional[str] = None

# --------------------------
# Decorators
# --------------------------

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

def log_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        try:
            result = func(*args, **kwargs)
            execution_time = perf_counter() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper

# --------------------------
# Metric Calculation Functions
# --------------------------

@safe_metric_calculation(default_value={}, metric_name="Halstead Metrics")
def calculate_halstead_metrics(code: str) -> Dict[str, Union[int, float]]:
    """Calculates Halstead complexity metrics."""
    halstead_reports = h_visit(code)

    if not halstead_reports:
        raise HalsteadCalculationError("No Halstead metrics found")

    metrics = halstead_reports[0] if isinstance(halstead_reports, list) else halstead_reports

    return {
        'h1': metrics.h1,
        'h2': metrics.h2,
        'N1': metrics.N1,
        'N2': metrics.N2,
        'vocabulary': metrics.vocabulary,
        'length': metrics.length,
        'calculated_length': metrics.calculated_length,
        'volume': metrics.volume,
        'difficulty': metrics.difficulty,
        'effort': metrics.effort,
        'time': metrics.time,
        'bugs': metrics.bugs
    }

@safe_metric_calculation(default_value=0.0, metric_name="Maintainability Index")
def calculate_maintainability_index(code: str) -> float:
    """Calculates the Maintainability Index."""
    mi_value = mi_visit(code, multi=False)
    return float(mi_value)

@safe_metric_calculation(default_value={}, metric_name="Cyclomatic Complexity")
def calculate_cyclomatic_complexity(code: str) -> Dict[str, float]:
    """Calculates Cyclomatic Complexity."""
    complexity_visitor = ComplexityVisitor.from_code(code)
    function_complexity = {}
    for block in complexity_visitor.functions + complexity_visitor.classes:
        function_complexity[block.name] = float(block.complexity)
    return function_complexity

@log_execution_time
def calculate_code_metrics(code: str, file_path: Optional[str] = None, language: str = "python") -> MetricsResult:
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
            metrics_data['halstead'] = calculate_halstead_metrics(code)
            metrics_data['maintainability_index'] = calculate_maintainability_index(code)
            metrics_data['function_complexity'] = calculate_cyclomatic_complexity(code)

            complexities = list(metrics_data['function_complexity'].values())
            metrics_data['cyclomatic_complexity'] = sum(complexities) / len(complexities) if complexities else 0.0
        else:
            # Placeholder for other languages
            metrics_data['halstead'] = {}
            metrics_data['maintainability_index'] = 0.0
            metrics_data['function_complexity'] = {}
            metrics_data['cyclomatic_complexity'] = 0.0
            logger.warning(f"Metrics calculation for language '{language}' is not implemented.")

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

    except MetricsCalculationError as e:
        error_msg = f"Metrics calculation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return MetricsResult(
            file_path=file_path or "unknown",
            timestamp=datetime.now(),
            execution_time=perf_counter() - start_time,
            success=False,
            error=error_msg
        )
    except Exception as e:
        error_msg = f"Unexpected error during metrics calculation: {str(e)}"
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
        if isinstance(value, (int, float)) and value < 0:
            logger.warning(f"Invalid Halstead {metric}: {value} (expected non-negative)")
            validated['halstead'][metric] = 0

    mi = validated.get('maintainability_index', 0.0)
    if not (0 <= mi <= 100):
        logger.warning(f"Invalid Maintainability Index: {mi} (expected 0-100)")
        validated['maintainability_index'] = max(0, min(mi, 100))

    cyclomatic = validated.get('cyclomatic_complexity', 0.0)
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
        'complexity': normalize_score(metrics['cyclomatic_complexity'], 1, 30, 0.3, inverse=True),
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

# --------------------------
# Metrics Analyzer
# --------------------------

class MetricsAnalyzer:
    """Analyzes and aggregates metrics across multiple files."""

    def __init__(self, thresholds: Optional[MetricsThresholds] = None):
        self.metrics_history: List[MetricsResult] = []
        self.error_count = 0
        self.warning_count = 0
        self.thresholds = thresholds or MetricsThresholds()

    def add_result(self, result: MetricsResult):
        """Adds a metrics result and updates error/warning counts."""
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
        if metrics['halstead'].get('effort', 0) > self.thresholds.halstead_effort_high:
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
            'success_rate': (len(successful_results) / len(self.metrics_history) * 100) if self.metrics_history else 0,
            'average_metrics': avg_metrics,
            'execution_times': self._calculate_execution_time_summary()
        }

    def _calculate_average_metrics(self, results: List[MetricsResult]) -> Dict[str, Any]:
        """Calculates average metrics from successful results."""
        try:
            avg_maintainability = sum(r.metrics['maintainability_index'] for r in results) / len(results)
            avg_complexity = sum(
                sum(r.metrics['function_complexity'].values()) / max(1, len(r.metrics['function_complexity']))
                for r in results
            ) / len(results)
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
            return {}  # Or return a dictionary with default values

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

            if metrics['halstead'].get('effort', 0) > self.thresholds.halstead_effort_high:
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
