"""
metrics_utils.py - Utility functions for metrics handling
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MetricsThresholds:
    """Thresholds for different metrics"""
    complexity_high: int = 15
    complexity_warning: int = 10
    maintainability_low: float = 20.0
    halstead_effort_high: float = 1000000.0

def get_metric_severity(
    metric_name: str,
    value: float,
    thresholds: MetricsThresholds
) -> str:
    """Determines the severity level of a metric value."""
    if metric_name == "complexity":
        if value >= thresholds.complexity_high:
            return "high"
        elif value >= thresholds.complexity_warning:
            return "warning"
    elif metric_name == "maintainability_index":
        if value < thresholds.maintainability_low:
            return "low"
    elif metric_name == "halstead_effort":
        if value > thresholds.halstead_effort_high:
            return "high"
    return "normal"

def format_metric_value(
    metric_name: str,
    value: float
) -> str:
    """Formats metric values for display."""
    if metric_name in ["maintainability_index", "complexity"]:
        return f"{value:.2f}"
    elif metric_name == "halstead_effort":
        return f"{value:,.0f}"
    return str(value)    
