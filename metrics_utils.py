"""
metrics_utils.py - Utility module to support metrics-related operations.

Provides helper functions and classes that assist in metrics calculations,
including metadata extraction, embedding calculations, code churn and duplication analysis,
and formatting and severity determination of metrics.
"""

import os
import json
import logging
import difflib
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps, partial
from time import perf_counter
from typing import Dict, Any, Optional, Union, List, Callable, TypeVar, Iterable

from concurrent.futures import ThreadPoolExecutor

# External dependencies
from git import Repo, GitError, InvalidGitRepositoryError, NoSuchPathError
from radon.metrics import h_visit, mi_visit
from radon.complexity import cc_visit, ComplexityVisitor
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.decomposition import PCA
import ast
import numpy as np

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

class BareRepositoryError(Exception):
    """Custom exception for bare repositories."""
    pass

# --------------------------
# Data Classes
# --------------------------

@dataclass
class CodeMetadata:
    """Metadata extracted from code."""
    function_name: str
    variable_names: List[str]
    complexity: float
    halstead_volume: float
    maintainability_index: float

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
# Helper Functions
# --------------------------

def get_metric_severity(metric_name: str, value: Union[int, float], thresholds: MetricsThresholds) -> str:
    """
    Determines the severity level of a metric value based on thresholds.
    """
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
    """
    Formats metric values for display.
    """
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

# --------------------------
# Embedding Calculators
# --------------------------

class MultiLayerEmbeddingCalculator:
    """Generates multi-layer embeddings from code using transformer models."""

    def __init__(self, model_name: str = 'microsoft/codebert-base', pca_components: int = 384):
        logger.debug(f"Initializing MultiLayerEmbeddingCalculator with model '{model_name}' and PCA components {pca_components}")
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pca = PCA(n_components=pca_components)
        self.layer_weights = [0.1, 0.2, 0.3, 0.4]
        self._fit_pca()

    def _fit_pca(self, sample_embeddings: Optional[List[np.ndarray]] = None):
        """
        Fit PCA on a sample of embeddings to initialize PCA transformation.
        This method should be called with representative data to capture variance.
        """
        # Placeholder for fitting PCA. In practice, you should fit on a large representative dataset.
        logger.debug("Fitting PCA on sample embeddings.")
        if sample_embeddings:
            combined = np.vstack(sample_embeddings)
            self.pca.fit(combined)
            logger.debug("PCA fitting completed.")
        else:
            # Fit PCA with random data if no samples provided (not ideal)
            random_data = np.random.rand(100, self.pca.n_components * 10)
            self.pca.fit(random_data)
            logger.debug("PCA fitted with random data as placeholder.")

    def calculate_multi_layer_embedding(self, code: str) -> np.ndarray:
        """
        Calculate a multi-layer embedding for the given code snippet.
        """
        logger.debug("Calculating multi-layer embedding.")
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_four_layers = outputs.hidden_states[-4:]
        combined_embedding = torch.zeros_like(last_four_layers[0][0])
        for layer, weight in zip(last_four_layers, self.layer_weights):
            combined_embedding += weight * layer[0]

        combined_embedding = torch.mean(combined_embedding, dim=0)
        embedding_np = combined_embedding.numpy()
        reduced_embedding = self.pca.transform(embedding_np.reshape(1, -1))

        logger.debug("Multi-layer embedding calculated and reduced.")
        return reduced_embedding.flatten()

class EnhancedEmbeddingCalculator:
    """Combines various embeddings and metadata into a unified embedding vector."""

    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        code_model: str = 'microsoft/codebert-base',
        pca_components: int = 384
    ):
        logger.debug(f"Initializing EnhancedEmbeddingCalculator with embedding model '{embedding_model}' and code model '{code_model}'")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.multi_layer_calculator = MultiLayerEmbeddingCalculator(code_model, pca_components)
        self.metadata_weights = {
            'content': 0.3,
            'multi_layer': 0.3,
            'function_name': 0.1,
            'variable_names': 0.1,
            'complexity': 0.05,
            'halstead_volume': 0.05,
            'maintainability_index': 0.1
        }

    def calculate_enhanced_embedding(self, code: str, metadata: CodeMetadata) -> np.ndarray:
        """
        Combine various embeddings and metadata into a single normalized embedding vector.
        """
        logger.debug("Calculating enhanced embedding.")
        content_embedding = self.embedding_model.encode(code)
        multi_layer_embedding = self.multi_layer_calculator.calculate_multi_layer_embedding(code)
        function_name_embedding = self.embedding_model.encode(metadata.function_name)
        variable_names_embedding = self.embedding_model.encode(" ".join(metadata.variable_names))

        complexity_norm = self._normalize_value(metadata.complexity, 0, 50)
        halstead_volume_norm = self._normalize_value(metadata.halstead_volume, 0, 1000)
        maintainability_index_norm = self._normalize_value(metadata.maintainability_index, 0, 100)

        combined_embedding = (
            self.metadata_weights['content'] * content_embedding +
            self.metadata_weights['multi_layer'] * multi_layer_embedding +
            self.metadata_weights['function_name'] * function_name_embedding +
            self.metadata_weights['variable_names'] * variable_names_embedding +
            self.metadata_weights['complexity'] * complexity_norm * np.ones_like(content_embedding) +
            self.metadata_weights['halstead_volume'] * halstead_volume_norm * np.ones_like(content_embedding) +
            self.metadata_weights['maintainability_index'] * maintainability_index_norm * np.ones_like(content_embedding)
        )

        norm = np.linalg.norm(combined_embedding)
        if norm == 0:
            logger.warning("Combined embedding norm is zero. Returning zero vector.")
            return combined_embedding
        normalized_embedding = combined_embedding / norm

        logger.debug("Enhanced embedding calculated and normalized.")
        return normalized_embedding

    def _normalize_value(self, value: float, min_value: float, max_value: float) -> float:
        """Normalize a value to a 0-1 range based on provided min and max."""
        normalized = (value - min_value) / (max_value - min_value) if max_value > min_value else 0.0
        logger.debug(f"Normalized value: {normalized} (value: {value}, min: {min_value}, max: {max_value})")
        return normalized

    def set_metadata_weights(self, new_weights: Dict[str, float]) -> None:
        """Update the weights for metadata features."""
        logger.debug(f"Setting new metadata weights: {new_weights}")
        if not np.isclose(sum(new_weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")
        self.metadata_weights.update(new_weights)
        logger.debug("Metadata weights updated.")

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        similarity = np.dot(embedding1, embedding2)
        logger.debug(f"Calculated similarity: {similarity}")
        return similarity

class EmbeddingManager:
    """Manages embedding calculations and similarity assessments."""

    def __init__(self, embedding_calculator: Optional[EnhancedEmbeddingCalculator] = None):
        self.embedding_calculator = embedding_calculator or EnhancedEmbeddingCalculator()

    def get_embedding(self, code: str, metadata: CodeMetadata) -> np.ndarray:
        """Generates an enhanced embedding for the given code and metadata."""
        embedding = self.embedding_calculator.calculate_enhanced_embedding(code, metadata)
        logger.debug(f"Generated embedding: {embedding}")
        return embedding

    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculates the similarity between two embeddings."""
        similarity = self.embedding_calculator.calculate_similarity(embedding1, embedding2)
        logger.debug(f"Similarity between embeddings: {similarity}")
        return similarity

# --------------------------
# Code Churn and Duplication Analysis
# --------------------------

class CodeChurnAnalyzer:
    """Analyzes code churn in a Git repository."""

    def __init__(self, repo_path: str, since_days: int = 30):
        self.repo_path = repo_path
        self.since = datetime.now() - timedelta(days=since_days)
        self.repo = self._load_repo()

    def _load_repo(self) -> Repo:
        """Loads the Git repository."""
        try:
            repo = Repo(self.repo_path)
            if repo.bare:
                logger.error("Cannot analyze churn for a bare repository.")
                raise BareRepositoryError("Repository is bare.")
            logger.debug(f"Loaded Git repository from '{self.repo_path}'")
            return repo
        except (NoSuchPathError, InvalidGitRepositoryError) as e:
            logger.error(f"Error accessing Git repository: {e}")
            raise ValueError(f"Invalid Git repository: {self.repo_path}") from e

    def calculate_code_churn(self) -> int:
        """
        Calculates code churn by summing insertions and deletions in Git commits since a given date.
        """
        logger.debug(f"Calculating code churn since {self.since.isoformat()}")
        churn = 0
        try:
            commits = list(self.repo.iter_commits(since=self.since))
            logger.debug(f"Found {len(commits)} commits since {self.since.isoformat()}")
            for commit in commits:
                insertions = commit.stats.total.get('insertions', 0)
                deletions = commit.stats.total.get('deletions', 0)
                churn += insertions + deletions
                logger.debug(f"Commit {commit.hexsha}: +{insertions} -{deletions}")
            logger.info(f"Total code churn: {churn} lines")
            return churn
        except GitError as e:
            logger.error(f"Git error while calculating churn: {e}")
            raise

def calculate_code_duplication(
    repo_path: str,
    extensions: Optional[Iterable[str]] = None,
    min_duplication_block_size: int = 5
) -> float:
    """
    Calculates code duplication percentage by comparing file contents using difflib.

    Note: This implementation has O(n^2) time complexity and may be inefficient for large repositories.
    Consider using specialized tools like jscpd or SonarQube for better performance.
    """
    logger.debug(f"Calculating code duplication for repository at '{repo_path}' with extensions {extensions}.")

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
    logger.debug(f"Total batches to compare: {num_batches}")

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

# --------------------------
# Metadata Extraction
# --------------------------

def extract_metadata(code: str, metrics: Dict[str, Any]) -> CodeMetadata:
    """
    Extract metadata such as function name and variable names from code using AST.
    """
    logger.debug("Extracting metadata from code using AST.")
    function_name = ""
    variable_names = []

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                for arg in node.args.args:
                    variable_names.append(arg.arg)
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                variable_names.append(target.id)
    except SyntaxError as e:
        logger.warning(f"Syntax error while parsing code: {e}")

    complexity = metrics.get('cyclomatic_complexity', 0)
    halstead_volume = metrics.get('halstead', {}).get('volume', 0.0)
    maintainability_index = metrics.get('maintainability_index', 0.0)

    metadata = CodeMetadata(
        function_name=function_name,
        variable_names=variable_names,
        complexity=complexity,
        halstead_volume=halstead_volume,
        maintainability_index=maintainability_index
    )

    logger.debug(f"Metadata extracted: {metadata}")
    return metadata

# --------------------------
# Embedding Integration
# --------------------------

def calculate_embedding_metrics(code: str, embedding_calculator: EnhancedEmbeddingCalculator, metadata: CodeMetadata) -> np.ndarray:
    """
    Calculates the enhanced embedding for a given code snippet using provided metadata.
    """
    logger.debug("Calculating embedding metrics.")
    embedding = embedding_calculator.calculate_enhanced_embedding(code, metadata)
    logger.debug(f"Enhanced embedding calculated: {embedding}")
    return embedding

# --------------------------
# Example Usage
# --------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Metrics Utilities Example Usage")
    parser.add_argument('--repo_path', type=str, required=True, help='Path to the Git repository.')
    parser.add_argument('--thresholds', type=str, default='thresholds.json', help='Path to thresholds JSON file.')
    args = parser.parse_args()

    # Load thresholds
    try:
        thresholds = MetricsThresholds.load_from_file(args.thresholds)
    except Exception as e:
        logger.error(f"Failed to load thresholds: {e}")
        thresholds = MetricsThresholds()  # Use default thresholds

    # Initialize analyzers and calculators
    churn_analyzer = CodeChurnAnalyzer(repo_path=args.repo_path, since_days=30)
    embedding_calculator = EnhancedEmbeddingCalculator()
    embedding_manager = EmbeddingManager(embedding_calculator=embedding_calculator)

    # Calculate code churn
    try:
        churn = churn_analyzer.calculate_code_churn()
        severity = get_metric_severity("code_churn", churn, thresholds)
        formatted_churn = format_metric_value("code_churn", churn)
        logger.info(f"Code Churn: {formatted_churn} (Severity: {severity})")
    except Exception as e:
        logger.error(f"Error calculating code churn: {e}")

    # Calculate code duplication
    try:
        duplication = calculate_code_duplication(args.repo_path)
        severity = get_metric_severity("code_duplication", duplication, thresholds)
        formatted_duplication = format_metric_value("code_duplication", duplication)
        logger.info(f"Code Duplication: {formatted_duplication} (Severity: {severity})")
    except Exception as e:
        logger.error(f"Error calculating code duplication: {e}")

    # Example: Calculate embedding for a specific file
    example_file_path = os.path.join(args.repo_path, 'example.py')  # Replace with an actual file path
    if os.path.exists(example_file_path):
        try:
            with open(example_file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                # Assume that metrics.py has been used to calculate metrics for this code
                # Here, we simulate metrics for demonstration purposes
                metrics = {
                    'cyclomatic_complexity': 12.0,
                    'maintainability_index': 45.0,
                    'halstead': {'effort': 50000.0}
                }
                metadata = extract_metadata(code, metrics)
                embedding = calculate_embedding_metrics(code, embedding_calculator, metadata)
                logger.info(f"Enhanced Embedding for {example_file_path}: {embedding}")
        except Exception as e:
            logger.error(f"Error calculating embedding for {example_file_path}: {e}")
    else:
        logger.warning(f"Example file '{example_file_path}' does not exist.")
    