"""
metrics.py

This module provides functions for calculating various code metrics, including:
- Halstead complexity metrics
- Cyclomatic complexity
- Maintainability index
- Raw metrics (lines of code, blank lines, comment lines, etc.)
- Other code quality metrics (method length, argument count, etc.)
"""

import logging
import math
from typing import Dict, Any, Tuple, Optional
import ast

from radon.complexity import cc_visit, SCORE
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze

logger = logging.getLogger(__name__)


def calculate_halstead_metrics(code: str) -> Dict[str, Any]:
    """Calculates Halstead complexity metrics."""
    try:
        halstead_visitor = h_visit(code)
        total_metrics = halstead_visitor.total

        h1 = len(total_metrics.operators)
        h2 = len(total_metrics.operands)
        N1 = sum(total_metrics.operators.values())
        N2 = sum(total_metrics.operands.values())

        vocabulary = h1 + h2
        length = N1 + N2
        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (h1 * N2) / (2 * h2) if h2 > 0 else 0
        effort = difficulty * volume

        return {
            "volume": round(volume, 2),
            "difficulty": round(difficulty, 2),
            "effort": round(effort, 2),
            "vocabulary": vocabulary,
            "length": length,
            "distinct_operators": h1,
            "distinct_operands": h2,
            "total_operators": N1,
            "total_operands": N2,
            "operator_counts": dict(total_metrics.operators),
            "operand_counts": dict(total_metrics.operands),
        }

    except Exception as e:
        logger.error(f"Error calculating Halstead metrics: {e}")
        return {  # Return a dictionary of zeros on error
            "volume": 0,
            "difficulty": 0,
            "effort": 0,
            "vocabulary": 0,
            "length": 0,
            "distinct_operators": 0,
            "distinct_operands": 0,
            "total_operators": 0,
            "total_operands": 0,
            "operator_counts": {},
            "operand_counts": {},
        }


def calculate_cyclomatic_complexity(code: str) -> Tuple[Dict[str, int], int]:
    """Calculates cyclomatic complexity."""
    try:
        complexity_scores = cc_visit(code)
        function_complexity = {score.fullname: score.complexity for score in complexity_scores}
        total_complexity = sum(score.complexity for score in complexity_scores)
        return function_complexity, total_complexity
    except Exception as e:
        logger.error(f"Error calculating cyclomatic complexity: {e}")
        return {}, 0

def calculate_maintainability_index(code: str) -> Optional[float]:
    """Calculates maintainability index."""
    try:
        return mi_visit(code, True)
    except Exception as e:
        logger.error(f"Error calculating maintainability index: {e}")
        return None


def calculate_raw_metrics(code: str) -> Dict[str, int]:
    """Calculates raw metrics (LOC, comments, blank lines, etc.)."""
    try:
        raw_metrics = analyze(code)
        return {
            "loc": raw_metrics.loc,
            "lloc": raw_metrics.lloc,
            "sloc": raw_metrics.sloc,
            "comments": raw_metrics.comments,
            "multi": raw_metrics.multi,
            "blank": raw_metrics.blank,
        }
    except Exception as e:
        logger.error(f"Error calculating raw metrics: {e}")
        return {
            "loc": 0,
            "lloc": 0,
            "sloc": 0,
            "comments": 0,
            "multi": 0,
            "blank": 0,
        }

def calculate_code_quality_metrics(code: str) -> Dict[str, Any]:
    """Calculates code quality metrics."""
    try:
        tree = ast.parse(code)
        metrics = {
            "method_length": [],
            "argument_count": [],
            "nesting_level": [],
            "function_complexity": {},  # Initialize here
        }

        class QualityVisitor(ast.NodeVisitor):

            def __init__(self):
                self.current_nesting = 0
                self.max_nesting = 0
                self.function_name = "" # Track current function name

            def visit_FunctionDef(self, node):
                self.current_nesting = 0
                self.max_nesting = 0
                self.function_name = node.name # Set current function name
                self.generic_visit(node)
                metrics["nesting_level"].append(self.max_nesting)
                metrics["method_length"].append(node.end_lineno - node.lineno + 1)
                metrics["argument_count"].append(len(node.args.args))
                metrics["function_complexity"][self.function_name] = self.max_nesting # Assign complexity

            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)  # Treat async functions the same

            def visit_If(self, node):
                self.current_nesting += 1
                self.max_nesting = max(self.max_nesting, self.current_nesting)
                self.generic_visit(node)
                self.current_nesting -= 1

            def visit_While(self, node):
                self.current_nesting += 1
                self.max_nesting = max(self.max_nesting, self.current_nesting)
                self.generic_visit(node)
                self.current_nesting -= 1

            def visit_For(self, node):
                self.current_nesting += 1
                self.max_nesting = max(self.max_nesting, self.current_nesting)
                self.generic_visit(node)
                self.current_nesting -= 1

            # ... (Add similar logic for other nesting structures: try, except, etc.)

        QualityVisitor().visit(tree)

        metrics["max_nesting_level"] = max(metrics["nesting_level"]) if metrics["nesting_level"] else 0
        metrics["avg_nesting_level"] = sum(metrics["nesting_level"]) / len(metrics["nesting_level"]) if metrics["nesting_level"] else 0
        metrics["avg_method_length"] = sum(metrics["method_length"]) / len(metrics["method_length"]) if metrics["method_length"] else 0
        metrics["avg_argument_count"] = sum(metrics["argument_count"]) / len(metrics["argument_count"]) if metrics["argument_count"] else 0


        return metrics

    except Exception as e:
        logger.error(f"Error calculating code quality metrics: {e}")
        return {
            "method_length": [],
            "argument_count": [],
            "nesting_level": [],
            "avg_method_length": 0,
            "avg_argument_count": 0,
            "max_nesting_level": 0,
            "avg_nesting_level": 0,
            "function_complexity": {}, # Return empty dictionary on error
        }


def calculate_all_metrics(code: str) -> Dict[str, Any]:
    """Calculates all available metrics."""
    metrics = {}
    try:
        metrics["halstead"] = calculate_halstead_metrics(code)
    except Exception as e:
        logger.error(f"Error calculating Halstead metrics: {e}")
        metrics["halstead"] = {  # Provide default values on error
            "volume": 0,
            "difficulty": 0,
            "effort": 0,
            "vocabulary": 0,
            "length": 0,
            "distinct_operators": 0,
            "distinct_operands": 0,
            "total_operators": 0,
            "total_operands": 0,
            "operator_counts": {},
            "operand_counts": {},
        }

    try:
        function_complexity, total_complexity = calculate_cyclomatic_complexity(code)
        metrics["function_complexity"] = function_complexity  # Store function-specific complexity
        metrics["cyclomatic"] = total_complexity  # Store total complexity
    except Exception as e:
        logger.error(f"Error calculating cyclomatic complexity: {e}")
        metrics["function_complexity"] = {} # Default on error
        metrics["cyclomatic"] = 0 # Default on error


    try:
        metrics["maintainability_index"] = calculate_maintainability_index(code)
    except Exception as e:
        logger.error(f"Error calculating maintainability index: {e}")
        metrics["maintainability_index"] = 0 # Default on error

    try:
        metrics["raw"] = calculate_raw_metrics(code)
    except Exception as e:
        logger.error(f"Error calculating raw metrics: {e}")
        metrics["raw"] = { # Default on error
            "loc": 0,
            "lloc": 0,
            "sloc": 0,
            "comments": 0,
            "multi": 0,
            "blank": 0,
        }

    try:
        metrics["quality"] = calculate_code_quality_metrics(code)
    except Exception as e:
        logger.error(f"Error calculating code quality metrics: {e}")
        metrics["quality"] = { # Default on error
            "method_length": [],
            "argument_count": [],
            "nesting_level": [],
            "avg_method_length": 0,
            "avg_argument_count": 0,
            "max_nesting_level": 0,
            "avg_nesting_level": 0,
            "function_complexity": {},
        }

    return metrics