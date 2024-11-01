# metrics_combined.py

import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar
import ast

# Import necessary modules for metrics calculation (radon or lizard)
try:
    import lizard
    USE_LIZARD = True
except ImportError:
    import radon.complexity as radon_cc
    import radon.metrics as radon_metrics
    USE_LIZARD = False

logger = logging.getLogger(__name__)

# Custom Exceptions
class MetricsCalculationError(Exception):
    """Base exception for metrics calculation errors."""
    pass

class HalsteadCalculationError(MetricsCalculationError):
    """Exception for Halstead metrics calculation errors."""
    pass

class ComplexityCalculationError(MetricsCalculationError):
    """Exception for cyclomatic complexity calculation errors."""
    pass

# Data Classes
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

# Decorators
T = TypeVar('T')  # Generic type variable for the return value

def safe_metric_calculation(default_value: T = None, metric_name: str = "metric") -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for safe metric calculation with specific error handling."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
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

# Enhanced Metrics Calculator
class EnhancedMetricsCalculator:
    """Enhanced metrics calculator with robust error handling."""

    def __init__(self):
        self.using_lizard = USE_LIZARD
        logger.info(f"Using {'lizard' if self.using_lizard else 'radon'} for metrics calculation")

    def calculate_metrics(self, code: str, file_path: Optional[str] = None, language: Optional[str] = None) -> Dict[str, Any]:
        """Calculates comprehensive code metrics."""
        try:
            metrics = ComplexityMetrics()
            if self.using_lizard:
                metrics = self._calculate_lizard_metrics(code)
            else:
                metrics = self._calculate_radon_metrics(code)

            # Add language-specific metrics if available
            if language:
                self._add_language_metrics(metrics, code, language)

            return self._prepare_metrics_output(metrics)

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return self._prepare_metrics_output(ComplexityMetrics())

    def _calculate_maintainability_index(self, code: str) -> float:
        """Calculates maintainability index using a modified version of the SEI formula."""
        try:
            # Count lines of code (excluding empty lines and comments)
            lines = [line.strip() for line in code.splitlines() if line.strip() and not line.strip().startswith('#')]
            loc = len(lines)
            if loc == 0:
                return 100.0

            # Calculate average line length as a complexity factor
            avg_line_length = sum(len(line) for line in lines) / loc

            # Count control structures as a basic complexity measure
            control_structures = len(re.findall(r'\b(if|else|elif|for|while|try|except|with)\b', code))

            # Basic Halstead volume approximation
            operators = len(re.findall(r'[\+\-\*/=<>!&|%]+|and|or|not|in|is', code))
            operands = len(re.findall(r'\b[a-zA-Z_]\w*\b', code))

            # Modified SEI formula
            vol = (operators + operands) * log2(operators + operands) if operators + operands > 0 else 0
            cc = control_structures / loc

            mi = 171 - 5.2 * log2(vol + 1) - 0.23 * cc * 100 - 16.2 * log2(loc)

            # Normalize to 0-100 scale
            return max(0.0, min(100.0, mi))

        except Exception as e:
            logger.error(f"Error calculating maintainability index: {str(e)}")
            return 100.0

    def _calculate_lizard_metrics(self, code: str) -> ComplexityMetrics:
        """Calculates metrics using lizard."""
        try:
            analysis = lizard.analyze_file.analyze_source_code("temp.py", code)

            # Calculate average complexity
            total_complexity = sum(func.cyclomatic_complexity for func in analysis.function_list)
            avg_complexity = (total_complexity / len(analysis.function_list) if analysis.function_list else 0.0)

            return ComplexityMetrics(
                cyclomatic_complexity=avg_complexity,
                maintainability_index=self._calculate_maintainability_index(code),
                halstead_volume=analysis.nloc,  # Using NLOC as a proxy
                halstead_difficulty=avg_complexity,  # Using complexity as proxy
                halstead_effort=analysis.nloc * avg_complexity
            )

        except Exception as e:
            logger.error(f"Error in lizard metrics calculation: {str(e)}")
            return ComplexityMetrics()

    def _calculate_radon_metrics(self, code: str) -> ComplexityMetrics:
        """Calculates metrics using radon with robust error handling."""
        metrics = ComplexityMetrics()

        try:
            # Calculate cyclomatic complexity
            try:
                cc_blocks = radon_cc.cc_visit(code)
                total_cc = sum(block.complexity for block in cc_blocks)
                metrics.cyclomatic_complexity = (total_cc / len(cc_blocks) if cc_blocks else 0.0)
            except Exception as e:
                logger.warning(f"Error calculating cyclomatic complexity: {str(e)}")

            # Calculate maintainability index
            try:
                mi_result = radon_metrics.mi_visit(code, multi=False)
                if isinstance(mi_result, (int, float)):
                    metrics.maintainability_index = float(mi_result)
            except Exception as e:
                logger.warning(f"Error calculating maintainability index: {str(e)}")

            # Calculate Halstead metrics
            try:
                h_visit_result = radon_metrics.h_visit(code)

                # Handle different return types from h_visit
                if isinstance(h_visit_result, (list, tuple)) and h_visit_result:
                    h_metrics = h_visit_result[0]
                elif hasattr(h_visit_result, 'h1'):  # Single object
                    h_metrics = h_visit_result
                else:
                    raise ValueError("Invalid Halstead metrics format")

                # Safely extract Halstead metrics
                metrics.halstead_volume = getattr(h_metrics, 'volume', 0.0)
                metrics.halstead_difficulty = getattr(h_metrics, 'difficulty', 0.0)
                metrics.halstead_effort = getattr(h_metrics, 'effort', 0.0)
                metrics.halstead_bugs = getattr(h_metrics, 'bugs', 0.0)
                metrics.halstead_time = getattr(h_metrics, 'time', 0.0)

            except Exception as e:
                logger.warning(f"Error calculating Halstead metrics: {str(e)}")

        except Exception as e:
            logger.error(f"Error in radon metrics calculation: {str(e)}")

        return metrics

    def _add_language_metrics(self, metrics: ComplexityMetrics, code: str, language: str) -> None:
        """Adds language-specific metrics if available."""
        try:
            if language in LANGUAGE_MAPPING:
                lang_info = LANGUAGE_MAPPING[language]

                # Add language-specific calculations here
                if lang_info["supports_type_hints"]:
                    # Example: Count type hints in Python
                    if language == "python":
                        import ast
                        try:
                            tree = ast.parse(code)
                            type_hints = sum(
                                1 for node in ast.walk(tree)
                                if isinstance(node, ast.AnnAssign)
                                or (isinstance(node, ast.FunctionDef) and node.returns)
                            )
                            metrics.type_hint_coverage = type_hints
                        except Exception as e:
                            logger.warning(f"Error analyzing type hints: {str(e)}")

        except Exception as e:
            logger.warning(f"Error adding language metrics: {str(e)}")

    def _prepare_metrics_output(self, metrics: ComplexityMetrics) -> Dict[str, Any]:
        """Prepares the final metrics output dictionary."""
        return {
            "timestamp": datetime.now().isoformat(),
            "complexity": metrics.cyclomatic_complexity,
            "maintainability_index": metrics.maintainability_index,
            "halstead": {
                "volume": metrics.halstead_volume,
                "difficulty": metrics.halstead_difficulty,
                "effort": metrics.halstead_effort,
                "bugs": metrics.halstead_bugs,
                "time": metrics.halstead_time
            }
        }

    def get_summary_metrics(self, code: str) -> Dict[str, Any]:
        """Gets summary metrics for a given code."""
        try:
            tree = ast.parse(code)
            metrics = {
                'complexity_score': self._get_node_complexity(tree),
                'dependency_count': len(self._get_node_dependencies(tree)),
                'critical_score': self._get_critical_score(tree),
                'maintenance_score': self._get_maintenance_score(tree),
                'usage_patterns': self._get_usage_count(tree)
            }
            return metrics
        except Exception as e:
            logger.error(f"Error getting summary metrics: {str(e)}")
            return {}

    def _get_node_complexity(self, node: ast.AST) -> float:
        """Calculates complexity score for a node."""
        complexity = 1.0
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1.0
            elif isinstance(child, ast.Try):
                complexity += 0.5
            elif isinstance(child, ast.BoolOp):
                complexity += 0.3 * len(child.values)
            elif isinstance(child, ast.Compare):
                complexity += 0.2 * len(child.ops)
        return min(complexity, 10.0)

    def _get_node_dependencies(self, node: ast.AST) -> Set[str]:
        """Gets dependencies for a node."""
        dependencies = set()
        class DependencyVisitor(ast.NodeVisitor):
            def visit_Name(self, node: ast.Name):
                if isinstance(node.ctx, ast.Load):
                    dependencies.add(node.id)
            def visit_Attribute(self, node: ast.Attribute):
                if isinstance(node.ctx, ast.Load):
                    dependencies.add(node.attr)
        DependencyVisitor().visit(node)
        return dependencies

    def _get_usage_count(self, node: ast.AST) -> int:
        """Gets usage count for variables in node."""
        usage_count = 0
        class UsageVisitor(ast.NodeVisitor):
            def visit_Name(self, node: ast.Name):
                nonlocal usage_count
                if isinstance(node.ctx, ast.Load):
                    usage_count += 1
        UsageVisitor().visit(node)
        return usage_count

    def _get_critical_score(self, node: ast.AST) -> float:
        """Calculates criticality score for a node."""
        score = 0.0
        critical_patterns = {
            'error': 0.8,
            'exception': 0.8,
            'validate': 0.6,
            'check': 0.5,
            'verify': 0.5,
            'assert': 0.7,
            'security': 0.9,
            'auth': 0.9
        }
        class CriticalityVisitor(ast.NodeVisitor):
            def visit_Name(self, node: ast.Name):
                nonlocal score
                for pattern, value in critical_patterns.items():
                    if pattern in node.id.lower():
                        score = max(score, value)
            def visit_Str(self, node: ast.Str):
                nonlocal score
                for pattern, value in critical_patterns.items():
                    if pattern in node.s.lower():
                        score = max(score, value * 0.5)
        CriticalityVisitor().visit(node)
        return score

    def _get_maintenance_score(self, node: ast.AST) -> float:
        """Calculates maintenance score based on code quality indicators."""
        score = 0.0
        class MaintenanceVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef):
                nonlocal score
                if ast.get_docstring(node):
                    score += 0.3
                if len(node.args.args) > 5:
                    score -= 0.2
            def visit_ClassDef(self, node: ast.ClassDef):
                nonlocal score
                if ast.get_docstring(node):
                    score += 0.3
                if len(node.bases) > 2:
                    score -= 0.2
            def visit_Try(self, node: ast.Try):
                nonlocal score
                score += 0.2
        MaintenanceVisitor().visit(node)
        return max(0.0, min(1.0, score))

@log_execution_time
def calculate_code_metrics(code: str, file_path: Optional[str] = None, language: str = "python") -> Dict[str, Any]:
    """Calculates code metrics, leveraging EnhancedMetricsCalculator."""
    start_time = perf_counter()
    logger.info(f"Starting metrics calculation for {file_path or 'unknown file'}")

    try:
        if not isinstance(code, str):
            raise ValueError(f"Invalid code type: {type(code)}")

        if not code.strip():
            raise ValueError("Empty code provided")

        metrics_calculator = EnhancedMetricsCalculator()
        metrics_data = metrics_calculator.calculate_metrics(code, file_path, language)

        logger.info(f"Metrics calculation completed for {file_path or 'unknown file'}")
        logger.debug(f"Calculated metrics: {metrics_data}")

        return metrics_data

    except MetricsCalculationError as e:
        error_msg = f"Metrics calculation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error during metrics calculation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}

# Additional functions for metrics validation and quality score calculation can be added here if needed.

# Example usage in another file:
# from metrics_combined import calculate_code_metrics, EnhancedMetricsCalculator

# metrics = calculate_code_metrics("def example(): pass", "example.py", "python")
# print(metrics)
