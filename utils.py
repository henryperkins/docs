"""
utils.py

Utility functions for code processing, file handling, metrics calculations,
and token management. Provides core functionality for the documentation 
generation system.
"""

import os
import sys
import json
import logging
import asyncio
import tiktoken
import pathspec
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Set, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import tokenize
from io import StringIO

from code_chunk import CodeChunk
from token_utils import TokenManager
logger = logging.getLogger(__name__)

# Default configurations
DEFAULT_COMPLEXITY_THRESHOLDS = {"low": 10, "medium": 20, "high": 30}
DEFAULT_HALSTEAD_THRESHOLDS = {
    "volume": {"low": 100, "medium": 500, "high": 1000},
    "difficulty": {"low": 10, "medium": 20, "high": 30},
    "effort": {"low": 500, "medium": 1000, "high": 2000}
}
DEFAULT_MAINTAINABILITY_THRESHOLDS = {"low": 50, "medium": 70, "high": 85}

DEFAULT_EXCLUDED_DIRS = {
    '.git', '__pycache__', 'node_modules', '.venv', 
    '.idea', 'build', 'dist', '.bin', 'venv'
}
DEFAULT_EXCLUDED_FILES = {'.DS_Store', '.gitignore', '.env'}
DEFAULT_SKIP_TYPES = {
    '.json', '.md', '.txt', '.csv', '.lock', 
    '.pyc', '.pyo', '.pyd', '.git'
}

# Language mappings
LANGUAGE_MAPPING = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".go": "go",
    ".cpp": "cpp",
    ".c": "cpp",
    ".java": "java",
}

logger = logging.getLogger(__name__)


def is_valid_extension(ext: str, skip_types: Set[str]) -> bool:
    """
    Checks if a file extension is valid (not in skip list).
    
    Args:
        ext: File extension
        
    Returns:
        bool: True if valid, False otherwise
    """
    return ext not in skip_types

def chunk_code(
    file_path: Union[str, Path],
    language: str,
    max_chunk_size: int = 5000
) -> List[CodeChunk]:
    """
    Splits code into chunks based on top-level constructs (functions, classes) 
    without loading the entire file into memory. Integrates code summarization
    for Python code to reduce chunk size if a chunk exceeds max_chunk_size.

    Args:
        file_path: Path to the source file
        language: Programming language (supports 'python')
        max_chunk_size: Maximum number of tokens per chunk
    
    Returns:
        List[CodeChunk]: List of code chunks
    
    Raises:
        NotImplementedError: If the language is not Python.
        ChunkTooLargeError: If a chunk exceeds max_chunk_size even after summarization.
    """
    if language != 'python':
        raise NotImplementedError("Currently, chunk_code only supports Python language.")
    
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        return []
    
    chunks = []
    current_chunk_lines = []
    current_chunk_start_line = 1
    current_chunk_name = None  # Function or Class name
    current_chunk_type = None  # 'function' or 'class'

    try:
        with open(file_path, 'r', encoding='utf-8') as source_file:
            lines = []
            for line_number, line in enumerate(source_file, start=1):
                lines.append(line)
                if line.strip().startswith(('def ', 'async def ', 'class ')):
                    # Start of a new top-level definition
                    if current_chunk_lines:
                        # Create a code chunk for the previous definition
                        chunk_content = ''.join(current_chunk_lines)
                        tokens = TokenManager.count_tokens(chunk_content)

                        # Summarize ONLY if chunk exceeds max_chunk_size
                        if tokens.token_count > max_chunk_size:
                            chunk_content = summarize_code(chunk_content, language, max_chunk_size)
                            tokens = TokenManager.count_tokens(chunk_content) # Update tokens

                        chunk = CodeChunk(
                            file_path=str(file_path),
                            start_line=current_chunk_start_line,
                            end_line=line_number - 1,
                            function_name=current_chunk_name if current_chunk_type == 'function' else None,
                            class_name=current_chunk_name if current_chunk_type == 'class' else None,
                            chunk_content=chunk_content,
                            tokens=tokens.tokens,
                            token_count=tokens.token_count,
                            language=language,
                            is_async=current_chunk_type == 'async_function',
                            decorator_list=[],
                            docstring=None,
                            metadata={}
                        )

                        # Check chunk size after summarization (if it was summarized)
                        if chunk.token_count > max_chunk_size:
                            raise ChunkTooLargeError(
                                f"Chunk starting at line {current_chunk_start_line} in {file_path} "
                                f"exceeds {max_chunk_size} tokens even after summarization."
                            )

                        chunks.append(chunk)
                        current_chunk_lines = []
                        current_chunk_start_line = line_number
                        
                    # Update current chunk information
                    current_chunk_lines.append(line)
                    current_chunk_start_line = line_number
                    current_chunk_name = extract_definition_name(line)
                    current_chunk_type = ('async_function' if line.strip().startswith('async def ') else
                                          'function' if line.strip().startswith('def ') else 
                                          'class')
                else:
                    current_chunk_lines.append(line)
            
            # Add the last chunk
            if current_chunk_lines:
                chunk_content = ''.join(current_chunk_lines)
                tokens = TokenManager.count_tokens(chunk_content)

                # Summarize ONLY if chunk exceeds max_chunk_size
                if tokens.token_count > max_chunk_size:
                    chunk_content = summarize_code(chunk_content, language, max_chunk_size)
                    tokens = TokenManager.count_tokens(chunk_content) # Update tokens

                chunk = CodeChunk(
                    file_path=str(file_path),
                    start_line=current_chunk_start_line,
                    end_line=line_number,
                    function_name=current_chunk_name if current_chunk_type == 'function' else None,
                    class_name=current_chunk_name if current_chunk_type == 'class' else None,
                    chunk_content=chunk_content,
                    tokens=tokens.tokens,
                    token_count=tokens.token_count,
                    language=language,
                    is_async=current_chunk_type == 'async_function',
                    decorator_list=[],
                    docstring=None,
                    metadata={}
                )

                # Check chunk size after summarization (if it was summarized)
                if chunk.token_count > max_chunk_size:
                    raise ChunkTooLargeError(
                        f"Chunk starting at line {current_chunk_start_line} in {file_path} "
                        f"exceeds {max_chunk_size} tokens even after summarization."
                    )

                chunks.append(chunk)
    except Exception as e:
        logger.error(f"Error chunking file '{file_path}': {e}")
        return []

    return chunks

def extract_definition_name(line: str) -> Optional[str]:
    """
    Extracts the name of the function or class from its definition line.
    
    Args:
        line: Line containing the function or class definition.

    Returns:
        str: Name of the function or class, or None if not found.
    """
    tokens = tokenize.generate_tokens(StringIO(line).readline)
    for toknum, tokval, _, _, _ in tokens:
        if toknum == tokenize.NAME and tokval in ('def', 'class'):
            # Next token is the name
            toknum, tokval, _, _, _ = next(tokens)
            if toknum == tokenize.NAME:
                return tokval
    return None

def get_language(file_path: Union[str, Path]) -> Optional[str]:
    """
    Determines the programming language based on file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Optional[str]: Language name or None if unknown
    """
    ext = str(Path(file_path).suffix).lower()
    language = LANGUAGE_MAPPING.get(ext)
    logger.debug(f"Detected language for '{file_path}': {language}")
    return language

def is_valid_extension(ext: str, skip_types: Set[str]) -> bool:
    """
    Checks if a file extension is valid (not in skip list).
    
    Args:
        ext: File extension
        skip_types: Set of extensions to skip
        
    Returns:
        bool: True if valid, False otherwise
    """
    return ext.lower() not in skip_types

def get_threshold(metric: str, key: str, default: int) -> int:
    """
    Gets threshold value from environment or defaults.
    
    Args:
        metric: Metric name
        key: Threshold key (low/medium/high)
        default: Default value
        
    Returns:
        int: Threshold value
    """
    try:
        return int(os.getenv(f"{metric.upper()}_{key.upper()}_THRESHOLD", default))
    except ValueError:
        logger.error(
            f"Invalid environment variable for "
            f"{metric.upper()}_{key.upper()}_THRESHOLD"
        )
        return default

def is_binary(file_path: Union[str, Path]) -> bool:
    """
    Checks if a file is binary.
    
    Args:
        file_path: Path to check
        
    Returns:
        bool: True if file is binary
    """
    try:
        with open(file_path, "rb") as file:
            return b"\0" in file.read(1024)
    except Exception as e:
        logger.error(f"Error checking if file is binary '{file_path}': {e}")
        return True

def should_process_file(file_path: Union[str, Path], skip_types: Set[str]) -> bool:
    """
    Determines if a file should be processed.
    
    Args:
        file_path: Path to the file
        skip_types: File extensions to skip
        
    Returns:
        bool: True if file should be processed
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return False

    # Check if it's a symlink
    if file_path.is_symlink():
        return False
    
    if "scripts" in file_path.parts:
        return False

    # Check common excluded directories
    excluded_parts = {
        'node_modules', '.bin', '.git', '__pycache__', 
        'build', 'dist', 'venv', '.venv'
    }
    if any(part in excluded_parts for part in file_path.parts):
        return False

    # Check extension
    ext = file_path.suffix.lower()
    if (not ext or 
        ext in skip_types or 
        ext in {'.flake8', '.gitignore', '.env', '.pyc', '.pyo', '.pyd', '.git'} or
        ext.endswith('.d.ts')):
        return False

    # Check if binary
    if is_binary(file_path):
        return False

    return True

def load_gitignore(repo_path: Union[str, Path]) -> pathspec.PathSpec:
    """
    Loads .gitignore patterns.
    
    Args:
        repo_path: Repository root path
        
    Returns:
        pathspec.PathSpec: Compiled gitignore patterns
    """
    gitignore_path = Path(repo_path) / '.gitignore'
    patterns = []
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
    
    return pathspec.PathSpec.from_lines(
        pathspec.patterns.GitWildMatchPattern, 
        patterns
    )

def get_all_file_paths(
    repo_path: Union[str, Path],
    excluded_dirs: Set[str],
    excluded_files: Set[str],
    skip_types: Set[str]
) -> List[str]:
    """
    Gets all processable file paths in repository.
    
    Args:
        repo_path: Repository root
        excluded_dirs: Directories to exclude
        excluded_files: Files to exclude
        skip_types: Extensions to skip
        
    Returns:
        List[str]: List of file paths
    """
    repo_path = Path(repo_path)
    file_paths = []
    normalized_excluded_dirs = {
        os.path.normpath(repo_path / d) for d in excluded_dirs
    }

    # Add common node_modules patterns
    node_modules_patterns = {
        'node_modules',
        '.bin',
        'node_modules/.bin',
        '**/node_modules/**/.bin',
        '**/node_modules/**/node_modules'
    }
    normalized_excluded_dirs.update({
        os.path.normpath(repo_path / d) for d in node_modules_patterns
    })

    gitignore = load_gitignore(repo_path)

    for root, dirs, files in os.walk(repo_path, topdown=True):
        # Skip excluded directories
        if any(excluded in root for excluded in ['node_modules', '.bin']):
            dirs[:] = []
            continue

        # Filter directories
        dirs[:] = [
            d for d in dirs 
            if (os.path.normpath(Path(root) / d) not in normalized_excluded_dirs and
                not any(excluded in d for excluded in ['node_modules', '.bin']))
        ]

        for file in files:
            # Skip excluded files
            if file in excluded_files:
                continue
                
            # Get full path
            full_path = Path(root) / file
            
            # Check if file should be processed
            if should_process_file(full_path, skip_types):
                # Check gitignore
                relative_path = full_path.relative_to(repo_path)
                if not gitignore.match_file(str(relative_path)):
                    file_paths.append(str(full_path))

    logger.debug(f"Collected {len(file_paths)} files from '{repo_path}'.")
    return file_paths


async def clean_unused_imports_async(code: str, file_path: str) -> str:
    """
    Removes unused imports using autoflake.
    
    Args:
        code: Source code
        file_path: Path for display
        
    Returns:
        str: Code with unused imports removed
    """
    try:
        process = await asyncio.create_subprocess_exec(
            'autoflake',
            '--remove-all-unused-imports',
            '--remove-unused-variables',
            '--stdin-display-name', file_path,
            '-',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=code.encode())
        
        if process.returncode != 0:
            logger.error(f"Autoflake failed:\n{stderr.decode()}")
            return code
        
        return stdout.decode()
        
    except Exception as e:
        logger.error(f'Error running Autoflake: {e}')
        return code

async def format_with_black_async(code: str) -> str:
    """
    Formats code using Black.
    
    Args:
        code: Source code
        
    Returns:
        str: Formatted code
    """
    try:
        process = await asyncio.create_subprocess_exec(
            'black',
            '--quiet',
            '-',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=code.encode())
        
        if process.returncode != 0:
            logger.error(f"Black formatting failed: {stderr.decode()}")
            return code
            
        return stdout.decode()
        
    except Exception as e:
        logger.error(f'Error running Black: {e}')
        return code

async def run_flake8_async(file_path: Union[str, Path]) -> Optional[str]:
    """
    Runs Flake8 on a file.
    
    Args:
        file_path: Path to check
        
    Returns:
        Optional[str]: Flake8 output if errors found
    """
    try:
        process = await asyncio.create_subprocess_exec(
            'flake8',
            str(file_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            return stdout.decode() + stderr.decode()
            
        return None
        
    except Exception as e:
        logger.error(f'Error running Flake8: {e}')
        return None

def load_json_schema(schema_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Loads and validates a JSON schema file.
    
    Args:
        schema_path: Path to schema file
        
    Returns:
        Optional[Dict[str, Any]]: Loaded schema or None if invalid
    """
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        logger.debug(f"Loaded JSON schema from '{schema_path}'")
        return schema
    except FileNotFoundError:
        logger.error(f"Schema file not found: '{schema_path}'")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in schema file: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading schema: {e}")
        return None

def load_function_schema(schema_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads and validates function schema file.
    
    Args:
        schema_path: Path to schema file
        
    Returns:
        Dict[str, Any]: Validated schema
        
    Raises:
        ValueError: If schema is invalid
    """
    schema = load_json_schema(schema_path)
    if schema is None:
        raise ValueError("Failed to load schema file")
        
    if "functions" not in schema:
        raise ValueError("Schema missing 'functions' key")
        
    try:
        from jsonschema import Draft7Validator
        Draft7Validator.check_schema(schema)
    except Exception as e:
        raise ValueError(f"Invalid schema format: {e}")
        
    return schema
def get_metric_status(value: float, thresholds: Dict[str, int]) -> str:
    """Returns a status indicator based on metric value and thresholds."""
    if value <= thresholds["low"]:
        return "Low"
    elif value <= thresholds["medium"]:
        return "Medium"
    else:
        return "High"


def sanitize_filename(filename: str) -> str:
    """Sanitizes filename by removing invalid characters."""
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)

def load_config(
    config_path: Union[str, Path],
    excluded_dirs: Set[str],
    excluded_files: Set[str],
    skip_types: Set[str]
) -> Tuple[str, str]:
    """
    Loads configuration file and updates exclusion sets.
    
    Args:
        config_path: Path to config file
        excluded_dirs: Directories to exclude
        excluded_files: Files to exclude
        skip_types: Extensions to skip
        
    Returns:
        Tuple[str, str]: Project info and style guidelines
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        project_info = config.get("project_info", "")
        style_guidelines = config.get("style_guidelines", "")
        
        excluded_dirs.update(config.get("excluded_dirs", []))
        excluded_files.update(config.get("excluded_files", []))
        skip_types.update(config.get("skip_types", []))
        
        logger.debug(f"Loaded configuration from '{config_path}'")
        return project_info, style_guidelines
        
    except FileNotFoundError:
        logger.error(f"Config file not found: '{config_path}'")
        return "", ""
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return "", ""
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return "", ""

async def run_node_script_async(
    script_path: Union[str, Path], 
    input_json: str
) -> Optional[str]:
    """
    Runs a Node.js script asynchronously.
    
    Args:
        script_path: Path to Node.js script
        input_json: JSON input for script
        
    Returns:
        Optional[str]: Script output or None if failed
    """
    try:
        process = await asyncio.create_subprocess_exec(
            'node',
            str(script_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=input_json.encode())

        if process.returncode != 0:
            logger.error(
                f"Node.js script '{script_path}' failed: {stderr.decode()}"
            )
            return None

        return stdout.decode()
        
    except FileNotFoundError:
        logger.error("Node.js is not installed or not in PATH")
        return None
    except Exception as e:
        logger.error(f"Error running Node.js script: {e}")
        return None

async def run_node_insert_docstrings(
    script_name: str,
    input_data: Dict[str, Any],
    scripts_dir: Union[str, Path]
) -> Optional[str]:
    """
    Runs Node.js docstring insertion script.
    
    Args:
        script_name: Name of script file
        input_data: Data for script
        scripts_dir: Directory containing scripts
        
    Returns:
        Optional[str]: Modified code or None if failed
    """
    try:
        script_path = Path(scripts_dir) / script_name
        logger.debug(f"Running Node.js script: {script_path}")

        input_json = json.dumps(input_data)
        result = await run_node_script_async(script_path, input_json)
        
        if result:
            try:
                # Check if result is JSON
                parsed = json.loads(result)
                return parsed.get("code")
            except json.JSONDecodeError:
                # Return as plain text if not JSON
                return result
                
        return None

    except Exception as e:
        logger.error(f"Error running {script_name}: {e}")
        return None

@dataclass
class CodeMetrics:
    """
    Stores code metrics for a file or chunk.
    
    Attributes:
        complexity: Cyclomatic complexity
        maintainability: Maintainability index
        halstead: Halstead complexity metrics
        loc: Lines of code metrics
        documentation_coverage: Documentation coverage percentage
        test_coverage: Test coverage percentage if available
    """
    complexity: float
    maintainability: float
    halstead: Dict[str, float]
    loc: Dict[str, int]
    documentation_coverage: float
    test_coverage: Optional[float] = None

def calculate_metrics(
    code: str,
    file_path: Optional[Union[str, Path]] = None
) -> Optional[CodeMetrics]:
    """
    Calculates various code metrics.
    
    Args:
        code: Source code to analyze
        file_path: Optional file path for context
        
    Returns:
        Optional[CodeMetrics]: Calculated metrics or None if failed
    """
    try:
        from radon.complexity import cc_visit
        from radon.metrics import h_visit, mi_visit
        
        # Calculate complexity
        complexity = 0
        for block in cc_visit(code):
            complexity += block.complexity
            
        # Calculate maintainability
        maintainability = mi_visit(code, multi=False)
        
        # Calculate Halstead metrics
        h_visit_result = h_visit(code)
        if not h_visit_result:
            logger.warning("No Halstead metrics found")
            halstead = {
                "volume": 0,
                "difficulty": 0,
                "effort": 0,
                "time": 0,
                "bugs": 0
            }
        else:
            metrics = h_visit_result[0]
            halstead = {
                "volume": metrics.volume,
                "difficulty": metrics.difficulty,
                "effort": metrics.effort,
                "time": metrics.time,
                "bugs": metrics.bugs
            }
            
        # Calculate LOC metrics
        lines = code.splitlines()
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        doc_lines = [l for l in lines if l.strip().startswith('"""') or 
                    l.strip().startswith("'''")]
        
        loc = {
            "total": len(lines),
            "code": len(code_lines),
            "docs": len(doc_lines),
            "empty": len(lines) - len(code_lines) - len(doc_lines)
        }
        
        # Calculate documentation coverage
        doc_coverage = (len(doc_lines) / len(code_lines)) * 100 if code_lines else 0
        
        # Try to get test coverage if file path provided
        test_coverage = None
        if file_path:
            try:
                coverage_data = get_test_coverage(file_path)
                if coverage_data:
                    test_coverage = coverage_data.get("line_rate", 0) * 100
            except Exception as e:
                logger.debug(f"Could not get test coverage: {e}")
        
        return CodeMetrics(
            complexity=complexity,
            maintainability=maintainability,
            halstead=halstead,
            loc=loc,
            documentation_coverage=doc_coverage,
            test_coverage=test_coverage
        )
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return None

def get_test_coverage(file_path: Union[str, Path]) -> Optional[Dict[str, float]]:
    """
    Gets test coverage data for a file.
    
    Args:
        file_path: Path to source file
        
    Returns:
        Optional[Dict[str, float]]: Coverage metrics if available
    """
    try:
        # Look for coverage data in common locations
        coverage_paths = [
            Path.cwd() / '.coverage',
            Path.cwd() / 'coverage' / 'coverage.json',
            Path.cwd() / 'htmlcov' / 'coverage.json'
        ]
        
        for cov_path in coverage_paths:
            if cov_path.exists():
                with open(cov_path) as f:
                    coverage_data = json.load(f)
                    
                # Find data for this file
                rel_path = Path(file_path).relative_to(Path.cwd())
                file_data = coverage_data.get("files", {}).get(str(rel_path))
                if file_data:
                    return {
                        "line_rate": file_data.get("line_rate", 0),
                        "branch_rate": file_data.get("branch_rate", 0)
                    }
                    
        return None
        
    except Exception as e:
        logger.debug(f"Error getting coverage data: {e}")
        return None

def format_metrics(metrics: CodeMetrics) -> Dict[str, Any]:
    """
    Formats metrics for output.
    
    Args:
        metrics: CodeMetrics object
        
    Returns:
        Dict[str, Any]: Formatted metrics
    """
    return {
        "complexity": {
            "value": metrics.complexity,
            "severity": get_complexity_severity(metrics.complexity),
        },
        "maintainability": {
            "value": metrics.maintainability,
            "severity": get_maintainability_severity(metrics.maintainability),
        },
        "halstead": {
            metric: {
                "value": value,
                "severity": get_halstead_severity(metric, value)
            }
            for metric, value in metrics.halstead.items()
        },
        "lines_of_code": metrics.loc,
        "documentation": {
            "coverage": metrics.documentation_coverage,
            "severity": get_doc_coverage_severity(metrics.documentation_coverage)
        },
        "test_coverage": {
            "value": metrics.test_coverage,
            "severity": get_test_coverage_severity(metrics.test_coverage)
        } if metrics.test_coverage is not None else None
    }

def get_complexity_severity(value: float) -> str:
    """Gets severity level for complexity metric."""
    thresholds = DEFAULT_COMPLEXITY_THRESHOLDS
    if value <= thresholds["low"]:
        return "low"
    elif value <= thresholds["medium"]:
        return "medium"
    return "high"

def get_maintainability_severity(value: float) -> str:
    """Gets severity level for maintainability metric."""
    thresholds = DEFAULT_MAINTAINABILITY_THRESHOLDS
    if value >= thresholds["high"]:
        return "low"
    elif value >= thresholds["medium"]:
        return "medium"
    return "high"

def get_halstead_severity(metric: str, value: float) -> str:
    """Gets severity level for Halstead metrics."""
    thresholds = DEFAULT_HALSTEAD_THRESHOLDS.get(metric, {})
    if not thresholds:
        return "unknown"
    
    if value <= thresholds["low"]:
        return "low"
    elif value <= thresholds["medium"]:
        return "medium"
    return "high"

def get_doc_coverage_severity(value: float) -> str:
    """Gets severity level for documentation coverage."""
    if value >= 80:
        return "low"
    elif value >= 50:
        return "medium"
    return "high"

def get_test_coverage_severity(value: Optional[float]) -> str:
    """Gets severity level for test coverage."""
    if value is None:
        return "unknown"
    
    if value >= 80:
        return "low"
    elif value >= 60:
        return "medium"
    return "high"

def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    log_level: str = "INFO"
) -> None:
    """
    Sets up logging configuration.
    
    Args:
        log_file: Optional path to log file
        log_level: Logging level to use
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers
    )
