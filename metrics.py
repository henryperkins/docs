"""
metrics.py - Enhanced version with improved error handling and debugging
"""

import ast
import logging
import traceback
import asyncio
import aiofiles
import json
from typing import Dict, Any, TypedDict, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from time import perf_counter

from radon.metrics import h_visit, mi_visit
from radon.complexity import cc_visit, ComplexityVisitor

# Configure logging
logger = logging.getLogger(__name__)

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
    h1: int  # Number of distinct operators
    h2: int  # Number of distinct operands
    N1: int  # Total operators
    N2: int  # Total operands
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

def safe_metric_calculation(default_value: Any = None):
    """Decorator for safe metric calculation with error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                return default_value
        return wrapper
    return decorator

@safe_metric_calculation(default_value=HalsteadMetrics(
    h1=0, h2=0, N1=0, N2=0, vocabulary=0, length=0,
    calculated_length=0.0, volume=0.0, difficulty=0.0,
    effort=0.0, time=0.0, bugs=0.0
))
def calculate_halstead_metrics(code: str) -> HalsteadMetrics:
    """
    Calculates Halstead complexity metrics with enhanced error handling.
    
    Args:
        code (str): Source code to analyze
        
    Returns:
        HalsteadMetrics: Calculated Halstead metrics
        
    Raises:
        HalsteadCalculationError: If calculation fails
    """
    logger.debug("Starting Halstead metrics calculation")
    
    try:
        halstead_reports = h_visit(code)
        
        if not halstead_reports:
            logger.warning("No Halstead metrics found in code")
            raise HalsteadCalculationError("No Halstead metrics found")

        metrics = halstead_reports[0] if isinstance(halstead_reports, list) else halstead_reports
        
        # Validate metrics
        for key in ['h1', 'h2', 'N1', 'N2']:
            if not hasattr(metrics, key) or getattr(metrics, key) < 0:
                raise HalsteadCalculationError(f"Invalid {key} value in Halstead metrics")

        halstead_metrics = HalsteadMetrics(
            h1=metrics.h1,
            h2=metrics.h2,
            N1=metrics.N1,
            N2=metrics.N2,
            vocabulary=metrics.vocabulary,
            length=metrics.length,
            calculated_length=metrics.calculated_length,
            volume=metrics.volume,
            difficulty=metrics.difficulty,
            effort=metrics.effort,
            time=metrics.time,
            bugs=metrics.bugs,
        )
        
        logger.debug(f"Halstead metrics calculated successfully: {halstead_metrics}")
        return halstead_metrics

    except Exception as e:
        logger.error(f"Error calculating Halstead metrics: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HalsteadCalculationError(f"Failed to calculate Halstead metrics: {str(e)}")

@safe_metric_calculation(default_value=0.0)
def calculate_maintainability_index(code: str) -> float:
    """
    Calculates the Maintainability Index with enhanced error handling.
    
    Args:
        code (str): Source code to analyze
        
    Returns:
        float: Maintainability Index value
    """
    logger.debug("Starting Maintainability Index calculation")
    
    try:
        mi_value = mi_visit(code, multi=False)
        
        if not isinstance(mi_value, (int, float)) or mi_value < 0:
            raise ValueError(f"Invalid Maintainability Index value: {mi_value}")
            
        logger.debug(f"Maintainability Index calculated: {mi_value}")
        return float(mi_value)
        
    except Exception as e:
        logger.error(f"Error calculating Maintainability Index: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return 0.0

@safe_metric_calculation(default_value={})
def calculate_complexity(code: str) -> Dict[str, float]:
    """
    Calculates Cyclomatic Complexity with enhanced error handling and validation.
    
    Args:
        code (str): Source code to analyze
        
    Returns:
        Dict[str, float]: Complexity metrics for each function
    """
    logger.debug("Starting Cyclomatic Complexity calculation")
    
    try:
        complexity_visitor = ComplexityVisitor.from_code(code)
        
        function_complexity = {}
        for block in complexity_visitor.functions + complexity_visitor.classes:
            if not hasattr(block, 'name') or not hasattr(block, 'complexity'):
                logger.warning(f"Invalid block structure found: {block}")
                continue
                
            complexity = float(block.complexity)
            if complexity < 0:
                logger.warning(f"Invalid complexity value for {block.name}: {complexity}")
                continue
                
            function_complexity[block.name] = complexity
            
            # Log high complexity functions
            if complexity > 10:
                logger.warning(f"High complexity function found: {block.name} ({complexity})")
        
        logger.debug(f"Complexity calculation completed: {function_complexity}")
        return function_complexity
        
    except Exception as e:
        logger.error(f"Error calculating complexity: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return {}


@log_execution_time
async def calculate_code_metrics(
    code: str,
    file_path: Optional[str] = None,
    language: str = "python"
) -> Union[CodeMetrics, MetricsResult]:
    """Calculates code metrics, handling different languages."""
    start_time = perf_counter()
    logger.info(f"Starting metrics calculation for {file_path or 'unknown file'}")

    try:
        # Input validation
        if not isinstance(code, str):
            raise ValueError(f"Invalid code type: {type(code)}")

        if not code.strip():
            raise ValueError("Empty code provided")

        metrics_data = {}

        if language.lower() == "python":
            try:  # Combined try-except block for Python metrics
                metrics_data['halstead'] = await asyncio.wait_for(
                    asyncio.to_thread(calculate_halstead_metrics, code, language),
                    timeout=30.0
                )
                metrics_data['maintainability_index'] = await asyncio.wait_for(
                    asyncio.to_thread(calculate_maintainability_index, code, language),
                    timeout=10.0
                )
                metrics_data['function_complexity'] = await asyncio.wait_for(
                    asyncio.to_thread(calculate_complexity, code, language),
                    timeout=20.0
                )
                complexities = list(metrics_data['function_complexity'].values())
                metrics_data['cyclomatic'] = (
                    sum(complexities) / len(complexities) if complexities else 0.0
                )
            except asyncio.TimeoutError as e:
                logger.error(f"Metrics calculation timed out: {e}")
                metrics_data['halstead'] = get_default_halstead_metrics()
                metrics_data['maintainability_index'] = 0.0
                metrics_data['function_complexity'] = {}
                metrics_data['cyclomatic'] = 0.0
            except Exception as e:  # Catch all other exceptions for Python metrics
                logger.error(f"Error calculating Python metrics: {e}", exc_info=True)
                metrics_data['halstead'] = get_default_halstead_metrics()
                metrics_data['maintainability_index'] = 0.0
                metrics_data['function_complexity'] = {}
                metrics_data['cyclomatic'] = 0.0                
        else:  # For non-Python languages
            metrics_data['halstead'] = get_default_halstead_metrics()
            metrics_data['maintainability_index'] = 0.0
            metrics_data['function_complexity'] = {}
            metrics_data['cyclomatic'] = 0.0

        # Validate metrics
        validated_metrics = validate_metrics(metrics_data)

        # Calculate quality score
        quality_score = calculate_quality_score(validated_metrics)
        validated_metrics['quality'] = quality_score

        # Add raw metrics for debugging
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
        logger.error(error_msg, exc_info=True)  # Include exception info in logs
        return MetricsResult(
            file_path=file_path or "unknown",
            timestamp=datetime.now(),
            execution_time=perf_counter() - start_time,
            success=False,
            error=error_msg
        )

def validate_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and sanitizes calculated metrics."""
    validated = {}
    
    # Validate Halstead metrics
    halstead = metrics.get('halstead', {})
    validated['halstead'] = {
        key: max(0.0, float(value))
        for key, value in halstead.items()
        if isinstance(value, (int, float))
    }

    # Validate Maintainability Index (0-100 scale)
    mi = metrics.get('maintainability_index', 0.0)
    validated['maintainability_index'] = max(0.0, min(100.0, float(mi)))

    # Validate complexity metrics
    validated['function_complexity'] = {
        str(func): max(0.0, float(complexity))
        for func, complexity in metrics.get('function_complexity', {}).items()
    }
    
    validated['cyclomatic'] = max(0.0, float(metrics.get('cyclomatic', 0.0)))

    return validated

def calculate_quality_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates a normalized quality score based on metrics.
    
    Args:
        metrics (Dict[str, Any]): Validated metrics
        
    Returns:
        Dict[str, Any]: Quality scores for different aspects
    """
    quality_scores = {
        'maintainability': normalize_score(
            metrics['maintainability_index'],
            min_val=0,
            max_val=100,
            weight=0.4
        ),
        'complexity': normalize_score(
            metrics['cyclomatic'],
            min_val=1,
            max_val=30,
            weight=0.3,
            inverse=True
        ),
        'halstead_effort': normalize_score(
            metrics['halstead'].get('effort', 0),
            min_val=0,
            max_val=1000000,
            weight=0.3,
            inverse=True
        )
    }
    
    # Calculate overall score
    quality_scores['overall'] = sum(quality_scores.values()) / len(quality_scores)
    
    return quality_scores

def normalize_score(
    value: float,
    min_val: float,
    max_val: float,
    weight: float = 1.0,
    inverse: bool = False
) -> float:
    """
    Normalizes a metric value to a 0-1 scale.
    
    Args:
        value (float): Raw metric value
        min_val (float): Minimum expected value
        max_val (float): Maximum expected value
        weight (float): Weight factor for the score
        inverse (bool): If True, lower values are better
        
    Returns:
        float: Normalized score between 0 and 1
    """
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
    """Returns default Halstead metrics for error cases."""
    return HalsteadMetrics(
        h1=0, h2=0, N1=0, N2=0,
        vocabulary=0, length=0,
        calculated_length=0.0,
        volume=0.0, difficulty=0.0,
        effort=0.0, time=0.0,
        bugs=0.0
    )

class MetricsAnalyzer:
    """
    Class for analyzing and aggregating metrics across multiple files.
    """
    
    def __init__(self):
        self.metrics_history: List[MetricsResult] = []
        self.error_count = 0
        self.warning_count = 0

    def add_result(self, result: MetricsResult):
        """
        Adds a metrics result to the history and updates error/warning counts.
        
        Args:
            result (MetricsResult): Metrics calculation result
        """
        self.metrics_history.append(result)
        
        if not result.success:
            self.error_count += 1
            logger.error(f"Metrics calculation failed for {result.file_path}: {result.error}")
        
        if result.success and result.metrics:
            self._check_metrics_warnings(result)

    def _check_metrics_warnings(self, result: MetricsResult):
        """
        Checks for concerning metrics values and logs warnings.
        
        Args:
            result (MetricsResult): Metrics calculation result
        """
        metrics = result.metrics
        if not metrics:
            return

        # Check maintainability
        if metrics['maintainability_index'] < 20:
            self.warning_count += 1
            logger.warning(
                f"Very low maintainability index ({metrics['maintainability_index']:.2f}) "
                f"in {result.file_path}"
            )

        # Check complexity
        for func, complexity in metrics['function_complexity'].items():
            if complexity > 15:
                self.warning_count += 1
                logger.warning(
                    f"High cyclomatic complexity ({complexity}) in function '{func}' "
                    f"in {result.file_path}"
                )

        # Check Halstead metrics
        if metrics['halstead']['effort'] > 1000000:
            self.warning_count += 1
            logger.warning(
                f"High Halstead effort ({metrics['halstead']['effort']:.2f}) "
                f"in {result.file_path}"
            )

    def get_summary(self) -> Dict[str, Any]:
        """
        Generates a summary of all processed metrics.
        
        Returns:
            Dict[str, Any]: Summary statistics and aggregated metrics
        """
        if not self.metrics_history:
            return {
                'error_count': 0,
                'warning_count': 0,
                'processed_files': 0,
                'success_rate': 0.0,
                'average_metrics': None
            }

        successful_results = [r for r in self.metrics_history if r.success and r.metrics]
        
        if not successful_results:
            return {
                'error_count': self.error_count,
                'warning_count': self.warning_count,
                'processed_files': len(self.metrics_history),
                'success_rate': 0.0,
                'average_metrics': None
            }

        # Calculate average metrics
        avg_metrics = self._calculate_average_metrics(successful_results)

        return {
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'processed_files': len(self.metrics_history),
            'success_rate': len(successful_results) / len(self.metrics_history) * 100,
            'average_metrics': avg_metrics,
            'execution_times': {
                'min': min(r.execution_time for r in self.metrics_history),
                'max': max(r.execution_time for r in self.metrics_history),
                'avg': sum(r.execution_time for r in self.metrics_history) / len(self.metrics_history)
            }
        }

    def _calculate_average_metrics(self, results: List[MetricsResult]) -> Dict[str, Any]:
        """
        Calculates average metrics across all successful results.
        
        Args:
            results (List[MetricsResult]): List of successful metric results
            
        Returns:
            Dict[str, Any]: Averaged metrics
        """
        try:
            avg_maintainability = sum(
                r.metrics['maintainability_index'] for r in results
            ) / len(results)

            avg_complexity = sum(
                sum(r.metrics['function_complexity'].values()) / 
                max(1, len(r.metrics['function_complexity'])) 
                for r in results
            ) / len(results)

            avg_halstead = {
                metric: sum(
                    r.metrics['halstead'][metric] for r in results
                ) / len(results)
                for metric in ['volume', 'difficulty', 'effort']
            }

            return {
                'maintainability_index': avg_maintainability,
                'cyclomatic_complexity': avg_complexity,
                'halstead': avg_halstead
            }

        except Exception as e:
            logger.error(f"Error calculating average metrics: {e}")
            return None

    def get_problematic_files(self) -> List[Dict[str, Any]]:
        """
        Identifies files with concerning metrics.
        
        Returns:
            List[Dict[str, Any]]: List of problematic files and their issues
        """
        problematic_files = []
        
        for result in self.metrics_history:
            if not (result.success and result.metrics):
                continue

            issues = []
            metrics = result.metrics

            # Check maintainability
            if metrics['maintainability_index'] < 20:
                issues.append({
                    'type': 'maintainability',
                    'value': metrics['maintainability_index'],
                    'threshold': 20
                })

            # Check complexity
            high_complexity_functions = [
                (func, complexity) 
                for func, complexity in metrics['function_complexity'].items() 
                if complexity > 15
            ]
            if high_complexity_functions:
                issues.append({
                    'type': 'complexity',
                    'functions': high_complexity_functions
                })

            # Check Halstead effort
            if metrics['halstead']['effort'] > 1000000:
                issues.append({
                    'type': 'halstead_effort',
                    'value': metrics['halstead']['effort'],
                    'threshold': 1000000
                })

            if issues:
                problematic_files.append({
                    'file_path': result.file_path,
                    'issues': issues
                })

        return problematic_files

if __name__ == "__main__":
    async def process_files(files: List[str]):
        """Process multiple files and aggregate their metrics."""
        results = []
        metrics_analyzer = MetricsAnalyzer()
        
        async with aiohttp.ClientSession() as session:
            for file_path in files:
                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        code = await f.read()
                        result = await calculate_code_metrics(code, file_path)
                        metrics_analyzer.add_result(result)
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue

        # Generate summary report
        summary = metrics_analyzer.get_summary()
        problematic_files = metrics_analyzer.get_problematic_files()

        # Write results to JSON
        output = {
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'problematic_files': problematic_files,
            'results': [
                {
                    'file_path': r.file_path,
                    'success': r.success,
                    'metrics': r.metrics if r.success else None,
                    'error': r.error if not r.success else None,
                    'execution_time': r.execution_time
                }
                for r in results
            ]
        }

        # Save results
        try:
            async with aiofiles.open('metrics_report.json', 'w', encoding='utf-8') as f:
                await f.write(json.dumps(output, indent=2, default=str))
            logger.info("Metrics report generated successfully")
        except Exception as e:
            logger.error(f"Error saving metrics report: {e}")

    # Production usage
    if len(sys.argv) > 1:
        files_to_process = sys.argv[1:]
        asyncio.run(process_files(files_to_process))
    else:
        logger.error("No files provided for processing")
        sys.exit(1)    
