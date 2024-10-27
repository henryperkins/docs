"""
metrics_utils.py - Utility functions for metrics handling.

Provides functionalities to calculate various code metrics, including
complexity, maintainability, Halstead metrics (using radon), code churn,
and code duplication. Thresholds for these metrics are configurable.
"""

import os
import json
import logging
import difflib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Set, Tuple, Union, Optional, Iterable
from pathlib import Path

# For code churn
from git import Repo, GitError, InvalidGitRepositoryError, NoSuchPathError

# Configure logging
logger = logging.getLogger(__name__)

# Constants for duplication check
DEFAULT_MIN_DUPLICATION_BLOCK_SIZE = 5  # Lines

class HalsteadError(Exception):
    """Custom exception for Halstead calculation errors."""
    pass

class BareRepositoryError(Exception):
    """Custom exception for bare repositories."""
    pass


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
            raise  # Re-raise the exception after logging
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise  # Re-raise
        except ValueError as e: # Catch validation errors
            logger.error(f"Invalid threshold values in config: {e}")
            raise


def get_metric_severity(metric_name: str, value: Union[int, float], thresholds: MetricsThresholds) -> str:
    """Determines the severity level of a metric value."""
    logger.debug(f"Evaluating severity for metric '{metric_name}' with value {value}")
    severity = "normal"

    if metric_name == "complexity":
        if value >= thresholds.complexity_high:
            severity = "high"
        elif value >= thresholds.complexity_warning:
            severity = "warning"
    elif metric_name == "maintainability_index":
        if value < thresholds.maintainability_low:
            severity = "low"
    elif metric_name == "halstead_effort":
        if value > thresholds.halstead_effort_high:
            severity = "high"
    elif metric_name == "code_churn":
        if value >= thresholds.code_churn_high:
            severity = "high"
        elif value >= thresholds.code_churn_warning:
            severity = "warning"
    elif metric_name == "code_duplication":
        if value >= thresholds.code_duplication_high:
            severity = "high"
        elif value >= thresholds.code_duplication_warning:
            severity = "warning"

    logger.debug(f"Severity determined: {severity}")
    return severity


def format_metric_value(metric_name: str, value: Union[int, float]) -> str:
    """Formats metric values for display."""
    logger.debug(f"Formatting metric '{metric_name}' with value {value}")

    if metric_name in ["maintainability_index", "complexity"]:
        formatted = f"{value:.2f}"
    elif metric_name == "halstead_effort":
        formatted = f"{value:,.0f}"
    elif metric_name == "code_churn":
        formatted = f"{int(value)} lines"
    elif metric_name == "code_duplication":
        formatted = f"{value:.2f}%"
    else:
        formatted = str(value)

    logger.debug(f"Formatted value: {formatted}")
    return formatted


def calculate_code_churn(repo_path: str, since: datetime = datetime.now() - timedelta(days=30)) -> int:
    """Calculates code churn."""
    logger.debug(f"Calculating code churn for repository at '{repo_path}' since {since.isoformat()}")

    try:
        repo = Repo(repo_path)
    except (NoSuchPathError, InvalidGitRepositoryError) as e:
        logger.error(f"Error accessing Git repository: {e}")
        raise ValueError(f"Invalid Git repository: {repo_path}") from e

    if repo.bare:
        logger.error("Cannot calculate churn for a bare repository.")
        raise BareRepositoryError("Repository is bare.")

    churn = 0
    try:
        for commit in repo.iter_commits(since=since):
            insertions = commit.stats.total.get('insertions', 0)
            deletions = commit.stats.total.get('deletions', 0)
            churn += insertions + deletions
            logger.debug(f"Commit {commit.hexsha}: +{insertions} -{deletions}")
        logger.info(f"Total code churn: {churn} lines")
        return churn
    except GitError as e:
        logger.error(f"Git error while calculating churn: {e}")
        raise  # Re-raise after logging


def calculate_code_duplication(
    repo_path: str,
    extensions: Optional[Iterable[str]] = None,
    min_duplication_block_size: int = DEFAULT_MIN_DUPLICATION_BLOCK_SIZE
) -> float:
    """Calculates code duplication percentage."""

    logger.debug(f"Calculating code duplication for repository at '{repo_path}'")

    duplicated_lines = 0
    total_lines = 0
    file_content_batches = []

    if extensions is None:
        extensions = {".py"}

    for root, _, files in os.walk(repo_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.readlines()
                        # Create batches of lines
                        for i in range(0, len(content), 1000):
                            batch = content[i:i + 1000]
                            file_content_batches.append((file_path, batch))
                            total_lines += len(batch)
                            logger.debug(f"Read {len(batch)} lines (batch) from {file_path}")

                except (IOError, UnicodeDecodeError) as e:
                    logger.warning(f"Skipped file '{file_path}' due to error: {e}")
                    continue  # Skip unreadable files


    num_batches = len(file_content_batches)
    for i in range(num_batches):
        for j in range(i + 1, num_batches):  # Compare each batch with every other batch
            _, batch1 = file_content_batches[i]
            _, batch2 = file_content_batches[j]

            seq = difflib.SequenceMatcher(None, batch1, batch2)
            for block in seq.get_matching_blocks():
                if block.size >= min_duplication_block_size:
                    duplicated_lines += block.size
                    logger.debug(f"Found duplicated block of size {block.size} between batches {i} and {j}")

    duplication_percentage = (duplicated_lines / total_lines * 100) if total_lines > 0 else 0.0
    logger.info(f"Total duplicated lines: {duplicated_lines} out of {total_lines} lines")
    logger.info(f"Duplication percentage: {duplication_percentage:.2f}%")

    return duplication_percentage

