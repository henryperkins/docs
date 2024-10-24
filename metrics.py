import ast
import logging
from typing import Dict, Any, TypedDict

from radon.metrics import h_visit, mi_visit
from radon.complexity import cc_visit, ComplexityVisitor

logger = logging.getLogger(__name__)

class HalsteadMetrics(TypedDict):
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
    maintainability_index: float
    cyclomatic: float
    halstead: HalsteadMetrics
    function_complexity: Dict[str, float]
    raw: Any
    quality: Any

def calculate_halstead_metrics(code: str) -> HalsteadMetrics:
    """Calculates Halstead complexity metrics."""
    logger.debug("Starting Halstead metrics calculation.")
    default_halstead_metrics = HalsteadMetrics(
        h1=0,
        h2=0,
        N1=0,
        N2=0,
        vocabulary=0,
        length=0,
        calculated_length=0.0,
        volume=0.0,
        difficulty=0.0,
        effort=0.0,
        time=0.0,
        bugs=0.0,
    )

    try:
        halstead_reports = h_visit(code)
        logger.debug(f"Halstead reports: {halstead_reports}")

        if not halstead_reports:
            logger.warning("No Halstead metrics found.")
            return default_halstead_metrics

        if isinstance(halstead_reports, list):
            metrics = halstead_reports[0]
        else:
            metrics = halstead_reports

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
        logger.debug(f"Calculated Halstead metrics: {halstead_metrics}")
        return halstead_metrics

    except Exception as e:
        logger.error(f"Error calculating Halstead metrics: {e}", exc_info=True)
        return default_halstead_metrics

def calculate_maintainability_index(code: str) -> float:
    """Calculates the Maintainability Index."""
    logger.debug("Starting Maintainability Index calculation.")
    try:
        maintainability_index = mi_visit(code, False)
        logger.debug(f"Calculated Maintainability Index: {maintainability_index}")
        return maintainability_index
    except Exception as e:
        logger.error(f"Error calculating Maintainability Index: {e}")
        return 0.0

def calculate_complexity(code: str) -> Dict[str, float]:
    """Calculates the Cyclomatic Complexity for each function in the code."""
    logger.debug("Starting Cyclomatic Complexity calculation.")
    try:
        complexity_visitor = ComplexityVisitor.from_code(code)
        function_complexity = {
            block.name: block.complexity for block in complexity_visitor.functions + complexity_visitor.classes
        }
        logger.debug(f"Calculated Cyclomatic Complexity: {function_complexity}")

        if not function_complexity:
            logger.warning("No cyclomatic complexity metrics found.")

        return function_complexity

    except Exception as e:
        logger.error(f"Error calculating Cyclomatic Complexity: {e}")
        return {}

def calculate_code_metrics(code: str) -> CodeMetrics:
    """Calculates various code metrics."""
    logger.debug("Starting code metrics calculation.")
    if isinstance(code, bytes):  # Ensure code is a string
        code = code.decode("utf-8")

    halstead_metrics = calculate_halstead_metrics(code)
    maintainability_index = calculate_maintainability_index(code)
    function_complexity = calculate_complexity(code)

    code_metrics: CodeMetrics = {
        "maintainability_index": maintainability_index,
        "cyclomatic": sum(function_complexity.values()) / len(function_complexity) if function_complexity else 0.0,
        "halstead": halstead_metrics,
        "function_complexity": function_complexity,
        "raw": None,  # Placeholder for any raw data if needed
        "quality": None,  # Placeholder for quality metrics if needed
    }

    logger.info(f"Aggregated Code Metrics: {code_metrics}")
    return code_metrics

# Example usage for testing purposes
if __name__ == "__main__":
    sample_code = """
def add(a, b):
    return a + b

class Calculator:
    def subtract(self, a, b):
        if a > b:
            return a - b
        else:
            return b - a
    """

    metrics = calculate_code_metrics(sample_code)
    import json
    print(json.dumps(metrics, indent=2))