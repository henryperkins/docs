"""
metrics.py

This module provides functions for calculating various code metrics including:
- Halstead complexity metrics
- Cyclomatic complexity
- Maintainability index
- Other code quality metrics

It serves as a centralized location for all metric calculations used throughout the project.
"""

import logging
import math
from typing import Dict, Any, Tuple, Optional

from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit

logger = logging.getLogger(__name__)

def calculate_halstead_metrics(code: str) -> Dict[str, Any]:
    """
    Calculates Halstead complexity metrics for the given code.

    Args:
        code (str): The source code to analyze.

    Returns:
        Dict[str, Any]: Dictionary containing Halstead metrics:
            - volume: Program volume
            - difficulty: Program difficulty
            - effort: Programming effort
            - vocabulary: Program vocabulary
            - length: Program length
            - distinct_operators: Number of distinct operators
            - distinct_operands: Number of distinct operands
            - total_operators: Total number of operators
            - total_operands: Total number of operands
    """
    try:
        # Get the metrics from radon
        halstead_visit = h_visit(code)
        
        # The h_visit() function returns a list of HalsteadVisitor objects
        # We'll take the first one (module level) if available
        if not halstead_visit:
            raise ValueError("No Halstead metrics available")
            
        # Get the first visitor
        visitor = halstead_visit[0]
        
        # Access the h1, h2, N1, N2 properties directly
        h1 = visitor.h1  # distinct operators
        h2 = visitor.h2  # distinct operands
        N1 = visitor.N1  # total operators
        N2 = visitor.N2  # total operands
        
        # Calculate derived metrics
        vocabulary = h1 + h2
        length = N1 + N2
        
        # Handle edge cases to avoid math errors
        if vocabulary > 0:
            volume = length * math.log2(vocabulary)
        else:
            volume = 0
            
        if h2 > 0:
            difficulty = (h1 * N2) / (2 * h2)
        else:
            difficulty = 0
            
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
            "total_operands": N2
        }
    except Exception as e:
        logger.error(f"Error calculating Halstead metrics: {e}")
        return {
            "volume": 0,
            "difficulty": 0,
            "effort": 0,
            "vocabulary": 0,
            "length": 0,
            "distinct_operators": 0,
            "distinct_operands": 0,
            "total_operators": 0,
            "total_operands": 0
        }


def calculate_complexity_metrics(code: str) -> Tuple[Dict[str, int], int]:
    """
    Calculates cyclomatic complexity metrics for the given code.

    Args:
        code (str): The source code to analyze.

    Returns:
        Tuple[Dict[str, int], int]: A tuple containing:
            - Dictionary mapping function names to their complexity scores
            - Total complexity score for the entire code
    """
    try:
        complexity_scores = cc_visit(code)
        function_complexity = {score.fullname: score.complexity for score in complexity_scores}
        total_complexity = sum(score.complexity for score in complexity_scores)
        return function_complexity, total_complexity
    except Exception as e:
        logger.error(f"Error calculating complexity metrics: {e}")
        return {}, 0

def calculate_maintainability_index(code: str) -> Optional[float]:
    """
    Calculates the maintainability index for the given code.

    Args:
        code (str): The source code to analyze.

    Returns:
        Optional[float]: The maintainability index score or None if calculation fails.
    """
    try:
        return mi_visit(code, True)
    except Exception as e:
        logger.error(f"Error calculating maintainability index: {e}")
        return None

def calculate_all_metrics(code: str) -> Dict[str, Any]:
    """
    Calculates all available metrics for the given code.

    Args:
        code (str): The source code to analyze.

    Returns:
        Dict[str, Any]: Dictionary containing all calculated metrics:
            - halstead: Halstead complexity metrics
            - complexity: Cyclomatic complexity metrics
            - maintainability_index: Maintainability index score
            - function_complexity: Individual function complexity scores
    """
    metrics = {}
    
    # Calculate Halstead metrics
    metrics["halstead"] = calculate_halstead_metrics(code)
    
    # Calculate complexity metrics
    function_complexity, total_complexity = calculate_complexity_metrics(code)
    metrics["complexity"] = total_complexity
    metrics["function_complexity"] = function_complexity
    
    # Calculate maintainability index
    metrics["maintainability_index"] = calculate_maintainability_index(code)
    
    return metrics
