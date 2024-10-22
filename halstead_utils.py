import logging
import math
from typing import Dict, Any
from radon.metrics import h_visit

logger = logging.getLogger(__name__)

def calculate_halstead_metrics_safe(code: str) -> Dict[str, Any]:
    """
    Safely calculates Halstead metrics with proper error handling.
    
    Args:
        code (str): The source code to analyze.
        
    Returns:
        Dict[str, Any]: Dictionary containing Halstead metrics with safe default values if calculation fails.
    """
    try:
        halstead_visitor = h_visit(code)
        
        # Access the 'total' attribute for file-level metrics
        total_metrics = halstead_visitor.total

        # Basic metrics
        h1 = len(total_metrics.operators)  # Access operators from total_metrics
        h2 = len(total_metrics.operands)
        N1 = sum(total_metrics.operators.values())
        N2 = sum(total_metrics.operands.values())
        
        # Derived metrics
        vocabulary = h1 + h2
        length = N1 + N2
        
        # Safe calculation of volume
        volume = (length * math.log2(vocabulary)) if vocabulary > 0 else 0
        
        # Safe calculation of difficulty
        difficulty = ((h1 * N2) / (2 * h2)) if h2 > 0 else 0
        
        # Calculate effort
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
            "operator_counts": dict(total_metrics.operators),  # Access operators from total_metrics
            "operand_counts": dict(total_metrics.operands)
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
            "total_operands": 0,
            "operator_counts": {},
            "operand_counts": {}
        }
