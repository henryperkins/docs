"""
utils.py

Core utility functions and classes for code processing, file handling,
metrics calculations, and token management. Provides the foundational
functionality for the documentation generation system.
"""

import os
import sys
import json
import logging
import asyncio
import tiktoken
import pathspec
import subprocess
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Set, Any, Optional, Union, Callable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
import tokenize
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import coverage
from coverage.files import PathAliases
from coverage.misc import CoverageException
import aiofiles
from aiofiles import os as aio_os
import re
from math import log2  # Add this for maintainability index calculation


# Try importing lizard first, fall back to radon if not available
try:
    import lizard
    USE_LIZARD = True
except ImportError:
    import radon.complexity as radon_cc
    import radon.metrics as radon_metrics
    USE_LIZARD = False

logger = logging.getLogger(__name__)

# Default configurations for metrics thresholds
DEFAULT_COMPLEXITY_THRESHOLDS = {"low": 10, "medium": 20, "high": 30}
DEFAULT_HALSTEAD_THRESHOLDS = {
    "volume": {"low": 100, "medium": 500, "high": 1000},
    "difficulty": {"low": 10, "medium": 20, "high": 30},
    "effort": {"low": 500, "medium": 1000, "high": 2000}
}
DEFAULT_MAINTAINABILITY_THRESHOLDS = {"low": 50, "medium": 70, "high": 85}

# Enhanced default patterns with more comprehensive coverage
DEFAULT_EXCLUDED_PATTERNS = {
    'dirs': {
        # Version Control
        '.git', '.svn', '.hg', '.bzr',
        # Python
        '__pycache__', '.pytest_cache', '.mypy_cache', '.coverage',
        'htmlcov', '.tox', '.nox',
        # Virtual Environments
        'venv', '.venv', 'env', '.env', 'virtualenv',
        # Node.js
        'node_modules',
        # Build/Distribution
        'build', 'dist', '.eggs', '*.egg-info',
        # IDE
        '.idea', '.vscode', '.vs', '.settings',
        # Other
        'tmp', 'temp', '.tmp', '.temp'
    },
    'files': {
        # System
        '.DS_Store', 'Thumbs.db', 'desktop.ini',
        # Python
        '*.pyc', '*.pyo', '*.pyd', '.python-version',
        '.coverage', 'coverage.xml', '.coverage.*',
        # Package Management
        'pip-log.txt', 'pip-delete-this-directory.txt',
        'poetry.lock', 'Pipfile.lock',
        # Environment
        '.env', '.env.*',
        # IDE
        '*.swp', '*.swo', '*~',
        # Build
        '*.spec', '*.manifest',
        # Documentation
        '*.pdf', '*.doc', '*.docx',
        # Other
        '*.log', '*.bak', '*.tmp'
    },
    'extensions': {
        # Python
        '.pyc', '.pyo', '.pyd', '.so',
        # Compilation
        '.o', '.obj', '.dll', '.dylib',
        # Package
        '.egg', '.whl',
        # Cache
        '.cache',
        # Documentation
        '.pdf', '.doc', '.docx',
        # Media
        '.jpg', '.jpeg', '.png', '.gif', '.ico',
        '.mov', '.mp4', '.avi',
        '.mp3', '.wav',
        # Archives
        '.zip', '.tar.gz', '.tgz', '.rar'
    }
}

# Language mappings with metadata
LANGUAGE_MAPPING = {
    ".py": {
        "name": "python",
        "comment_symbol": "#",
        "doc_strings": ['"""', "'''"],
        "supports_type_hints": True
    },
    ".js": {
        "name": "javascript",
        "comment_symbol": "//",
        "doc_strings": ["/**", "*/"],
        "supports_type_hints": False
    },
    ".ts": {
        "name": "typescript",
        "comment_symbol": "//",
        "doc_strings": ["/**", "*/"],
        "supports_type_hints": True
    },
    ".java": {
        "name": "java",
        "comment_symbol": "//",
        "doc_strings": ["/**", "*/"],
        "supports_type_hints": True
    },
    ".go": {
        "name": "go",
        "comment_symbol": "//",
        "doc_strings": ["/**", "*/"],
        "supports_type_hints": True
    }
}

# Custom Exceptions
class MetricsError(Exception):
    """Base exception for metrics-related errors."""
    pass

class FileOperationError(Exception):
    """Base exception for file operation errors."""
    pass

class CoverageFormatError(Exception):
    """Raised when coverage format is invalid or unrecognized."""
    pass

# Data Classes
@dataclass
class TokenResult:
    """Contains token analysis results."""
    tokens: List[str]
    token_count: int
    encoding_name: str
    special_tokens: Optional[Dict[str, int]] = None
    error: Optional[str] = None

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
class CoverageData:
    """Container for coverage metrics."""
    line_rate: float = 0.0
    branch_rate: float = 0.0
    complexity: float = 0.0
    timestamp: str = ""
    source_file: str = ""

# Core Utility Classes
class TokenManager:
    """Manages token counting and analysis with caching."""
    
    _encoders = {}
    _lock = threading.Lock()
    _cache = {}
    _max_cache_size = 1000
    
    @classmethod
    def get_encoder(cls, model_name: str = "gpt-4") -> Any:
        """Gets or creates a tiktoken encoder with thread safety."""
        with cls._lock:
            if model_name not in cls._encoders:
                try:
                    if model_name.startswith("gpt-4"):
                        encoding_name = "cl100k_base"
                    elif model_name.startswith("gpt-3"):
                        encoding_name = "p50k_base"
                    else:
                        encoding_name = "cl100k_base"  # default
                    
                    cls._encoders[model_name] = tiktoken.get_encoding(encoding_name)
                except Exception as e:
                    logger.error(f"Error creating encoder for {model_name}: {e}")
                    raise
            
            return cls._encoders[model_name]

    @classmethod
    def count_tokens(
        cls,
        text: str,
        model_name: str = "gpt-4",
        use_cache: bool = True
    ) -> TokenResult:
        """Counts tokens in text using tiktoken with caching."""
        if not text:
            return TokenResult([], 0, "", error="Empty text")
            
        if use_cache:
            cache_key = hash(text + model_name)
            if cache_key in cls._cache:
                return cls._cache[cache_key]
        
        try:
            encoder = cls.get_encoder(model_name)
            tokens = encoder.encode(text)
            special_tokens = cls._count_special_tokens(text)
            
            result = TokenResult(
                tokens=tokens,
                token_count=len(tokens),
                encoding_name=encoder.name,
                special_tokens=special_tokens
            )
            
            if use_cache:
                with cls._lock:
                    if len(cls._cache) >= cls._max_cache_size:
                        cls._cache.pop(next(iter(cls._cache)))
                    cls._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return TokenResult([], 0, "", error=str(e))

    @staticmethod
    def _count_special_tokens(text: str) -> Dict[str, int]:
        """Counts special tokens like newlines and code blocks."""
        return {
            "newlines": text.count("\n"),
            "code_blocks": text.count("```"),
            "inline_code": text.count("`") - (text.count("```") * 3)
        }

class FileHandler:
    """Handles file operations with caching and error handling."""
    
    _content_cache = {}
    _cache_lock = threading.Lock()
    _max_cache_size = 100
    _executor = ThreadPoolExecutor(max_workers=4)
    
    @classmethod
    async def read_file(
        cls,
        file_path: Union[str, Path],
        use_cache: bool = True,
        encoding: str = 'utf-8'
    ) -> Optional[str]:
        """Reads file content asynchronously with caching."""
        file_path = str(file_path)
        
        try:
            if use_cache:
                with cls._cache_lock:
                    cache_key = f"{file_path}:{encoding}"
                    if cache_key in cls._content_cache:
                        return cls._content_cache[cache_key]
        
            async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                content = await f.read()
        
            if use_cache:
                with cls._cache_lock:
                    if len(cls._content_cache) >= cls._max_cache_size:
                        cls._content_cache.pop(next(iter(cls._content_cache)))
                    cls._content_cache[cache_key] = content
        
            return content
            
        except UnicodeDecodeError:
            logger.warning(
                f"UnicodeDecodeError for {file_path} with {encoding}, "
                "trying with error handling"
            )
            try:
                async with aiofiles.open(
                    file_path,
                    'r',
                    encoding=encoding,
                    errors='replace'
                ) as f:
                    return await f.read()
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None

class EnhancedMetricsCalculator:
    """Enhanced metrics calculator with robust error handling."""

    def __init__(self):
        self.using_lizard = USE_LIZARD
        logger.info(f"Using {'lizard' if self.using_lizard else 'radon'} for metrics calculation")

    def calculate_metrics(
        self,
        code: str,
        file_path: Optional[Union[str, Path]] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculates comprehensive code metrics.
        
        Args:
            code: Source code to analyze
            file_path: Optional file path for coverage data
            language: Programming language
            
        Returns:
            Dict containing calculated metrics
        """
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
        """
        Calculates maintainability index using a modified version of the SEI formula.
        
        The formula considers:
        - Lines of code
        - Cyclomatic complexity
        - Halstead volume
        
        Args:
            code (str): Source code to analyze
            
        Returns:
            float: Maintainability index (0-100)
        """
        try:
            # Count lines of code (excluding empty lines and comments)
            lines = [
                line.strip()
                for line in code.splitlines()
                if line.strip() and not line.strip().startswith('#')
            ]
            loc = len(lines)
            
            if loc == 0:
                return 100.0
            
            # Calculate average line length as a complexity factor
            avg_line_length = sum(len(line) for line in lines) / loc
            
            # Count control structures as a basic complexity measure
            control_structures = len(re.findall(
                r'\b(if|else|elif|for|while|try|except|with)\b',
                code
            ))
            
            # Basic Halstead volume approximation
            operators = len(re.findall(
                r'[\+\-\*/=<>!&|%]+|and|or|not|in|is',
                code
            ))
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
            avg_complexity = (
                total_complexity / len(analysis.function_list)
                if analysis.function_list
                else 0.0
            )

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
                metrics.cyclomatic_complexity = (
                    total_cc / len(cc_blocks) if cc_blocks else 0.0
                )
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

    def _add_language_metrics(
        self,
        metrics: ComplexityMetrics,
        code: str,
        language: str
    ) -> None:
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

class CoverageHandler:
    """Handles multiple coverage report formats."""

    SUPPORTED_FORMATS = {'.coverage', '.xml', '.json', '.sqlite'}

    def __init__(self):
        self._coverage = None
        self._aliases = PathAliases()

    def get_test_coverage(
        self,
        file_path: Union[str, Path],
        coverage_path: Union[str, Path]
    ) -> Optional[CoverageData]:
        """Gets test coverage data from various report formats."""
        try:
            coverage_path = Path(coverage_path)
            file_path = Path(file_path)

            if not coverage_path.exists():
                logger.warning(f"Coverage file not found: {coverage_path}")
                return None

            if not file_path.exists():
                logger.warning(f"Source file not found: {file_path}")
                return None

            coverage_path = coverage_path.resolve()
            file_path = file_path.resolve()

            handler_map = {
                '.coverage': self._get_coverage_from_sqlite,
                '.xml': self._get_coverage_from_xml,
                '.json': self._get_coverage_from_json,
                '.sqlite': self._get_coverage_from_sqlite
            }

            if coverage_path.suffix not in handler_map:
                logger.warning(f"Unsupported coverage format: {coverage_path.suffix}")
                return None

            coverage_data = handler_map[coverage_path.suffix](coverage_path, file_path)
            if coverage_data:
                self._validate_coverage_data(coverage_data)
            return coverage_data

        except Exception as e:
            logger.error(f"Error getting test coverage: {str(e)}")
            return None

    def _get_coverage_from_sqlite(self, coverage_path: Path, file_path: Path) -> Optional[CoverageData]:
        """Gets coverage data from SQLite format."""
        try:
            conn = sqlite3.connect(str(coverage_path))
            cursor = conn.cursor()

            # Example query, adjust based on actual coverage DB schema
            cursor.execute("""
                SELECT line_rate, branch_rate, complexity, timestamp
                FROM coverage
                WHERE filename = ?
            """, (str(file_path),))
            row = cursor.fetchone()

            conn.close()

            if row:
                line_rate, branch_rate, complexity, timestamp = row
                return CoverageData(
                    line_rate=line_rate,
                    branch_rate=branch_rate,
                    complexity=complexity,
                    timestamp=datetime.fromtimestamp(timestamp).isoformat(),
                    source_file=str(file_path)
                )
            else:
                logger.warning(f"No coverage data found for {file_path} in {coverage_path}")
                return None

        except Exception as e:
            logger.error(f"Error reading SQLite coverage data: {e}")
            return None

    def _get_coverage_from_xml(self, coverage_path: Path, file_path: Path) -> Optional[CoverageData]:
        """Gets coverage data from XML format."""
        try:
            tree = ET.parse(coverage_path)
            root = tree.getroot()

            # Example parsing, adjust based on actual XML schema
            for file_elem in root.findall('.//file'):
                if file_elem.get('name') == str(file_path):
                    line_rate = float(file_elem.get('line-rate', 0.0))
                    branch_rate = float(file_elem.get('branch-rate', 0.0))
                    complexity = float(file_elem.get('complexity', 0.0))
                    timestamp = datetime.now().isoformat()  # XML might not have timestamp

                    return CoverageData(
                        line_rate=line_rate,
                        branch_rate=branch_rate,
                        complexity=complexity,
                        timestamp=timestamp,
                        source_file=str(file_path)
                    )

            logger.warning(f"No coverage data found for {file_path} in {coverage_path}")
            return None

        except Exception as e:
            logger.error(f"Error reading XML coverage data: {e}")
            return None

    def _get_coverage_from_json(self, coverage_path: Path, file_path: Path) -> Optional[CoverageData]:
        """Gets coverage data from JSON format."""
        try:
            with open(coverage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Example parsing, adjust based on actual JSON schema
            file_coverage = data.get('files', {}).get(str(file_path), {})
            if file_coverage:
                return CoverageData(
                    line_rate=file_coverage.get('line_rate', 0.0),
                    branch_rate=file_coverage.get('branch_rate', 0.0),
                    complexity=file_coverage.get('complexity', 0.0),
                    timestamp=datetime.now().isoformat(),  # JSON might not have timestamp
                    source_file=str(file_path)
                )
            else:
                logger.warning(f"No coverage data found for {file_path} in {coverage_path}")
                return None

        except Exception as e:
            logger.error(f"Error reading JSON coverage data: {e}")
            return None

    def _calculate_complexity(self, analysis: Any) -> float:
        """Calculates complexity from analysis data."""
        try:
            # Implement complexity calculation based on analysis
            return float(analysis.complexity)
        except Exception as e:
            logger.error(f"Error calculating complexity: {e}")
            return 0.0

    def _get_relative_path(self, file_path: Path) -> str:
        """Gets the relative path of a file."""
        try:
            return str(file_path.relative_to(self.repo_path))
        except ValueError:
            return str(file_path)

    def _validate_coverage_data(self, data: CoverageData) -> None:
        """Validates the coverage data."""
        try:
            if not (0.0 <= data.line_rate <= 1.0):
                raise CoverageFormatError("Line rate out of bounds")
            if not (0.0 <= data.branch_rate <= 1.0):
                raise CoverageFormatError("Branch rate out of bounds")
            # Add more validation as needed
        except CoverageFormatError as e:
            logger.error(f"Invalid coverage data: {e}")
            raise

class PathFilter:
    """Handles file path filtering based on various exclusion patterns."""
    
    def __init__(
        self,
        repo_path: Union[str, Path],
        excluded_dirs: Optional[Set[str]] = None,
        excluded_files: Optional[Set[str]] = None,
        skip_types: Optional[Set[str]] = None
    ):
        self.repo_path = Path(repo_path)
        self.excluded_dirs = (excluded_dirs or set()) | DEFAULT_EXCLUDED_PATTERNS['dirs']
        self.excluded_files = (excluded_files or set()) | DEFAULT_EXCLUDED_PATTERNS['files']
        self.skip_types = (skip_types or set()) | DEFAULT_EXCLUDED_PATTERNS['extensions']
        self.gitignore = load_gitignore(repo_path)
        
        self.file_patterns = pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern,
            self.excluded_files
        )

    def should_include_path(self, path: Path, relative_to: Optional<Path] = None) -> bool:
        """Determines if a path should be included based on exclusion rules."""
        try:
            check_path = path.relative_to(relative_to) if relative_to else path
            
            if any(part.startswith('.') for part in check_path.parts):
                return False
                
            if any(part in self.excluded_dirs for part in check_path.parts):
                return False
                
            if self.file_patterns.match_file(str(check_path)):
                return False
                
            if check_path.suffix.lower() in self.skip_types:
                return False
                
            if self.gitignore.match_file(str(check_path)):
                return False
                
            return True
                
        except Exception as e:
            logger.warning(f"Error checking path {path}: {str(e)}")
            return False

@lru_cache(maxsize=128)
def load_gitignore(repo_path: Union[str, Path]) -> pathspec.PathSpec:
    """Loads and caches .gitignore patterns."""
    patterns = []
    gitignore_path = Path(repo_path) / '.gitignore'
    
    try:
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                patterns = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith('#')
                ]
                logger.debug(f"Loaded {len(patterns)} patterns from .gitignore")
        else:
            logger.debug("No .gitignore file found")
            
    except Exception as e:
        logger.warning(f"Error reading .gitignore: {str(e)}")
        
    return pathspec.PathSpec.from_lines(
        pathspec.patterns.GitWildMatchPattern,
        patterns
    )

def get_all_file_paths(
    repo_path: Union[str, Path],
    excluded_dirs: Optional[Set[str]] = None,
    excluded_files: Optional[Set[str]] = None,
    skip_types: Optional[Set[str]] = None,
    follow_symlinks: bool = False
) -> List[str]:
    """Gets all file paths in a repository with improved filtering."""
    repo_path = Path(repo_path)
    if not repo_path.exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        return []
        
    path_filter = PathFilter(
        repo_path,
        excluded_dirs,
        excluded_files,
        skip_types
    )
    
    included_paths = []
    
    try:
        for root, dirs, files in os.walk(repo_path, followlinks=follow_symlinks):
            root_path = Path(root)
            
            # Filter directories in-place
            dirs[:] = [
                d for d in dirs
                if path_filter.should_include_path(root_path / d, repo_path)
            ]
            
            # Filter and add files
            for file in files:
                file_path = root_path / file
                if path_filter.should_include_path(file_path, repo_path):
                    included_paths.append(str(file_path))
                    
        logger.info(
            f"Found {len(included_paths)} files in {repo_path} "
            f"(excluded: dirs={len(path_filter.excluded_dirs)}, "
            f"files={len(path_filter.excluded_files)}, "
            f"types={len(path_filter.skip_types)})"
        )
        
        return included_paths
        
    except Exception as e:
        logger.error(f"Error walking repository: {str(e)}")
        return []

# Initialize global instances
metrics_calculator = EnhancedMetricsCalculator()
coverage_handler = CoverageHandler()

def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    log_level: str = "INFO",
    log_format: Optional[str] = None
) -> bool:
    """Sets up logging configuration."""
    try:
        if not log_format:
            log_format = (
                "%(asctime)s [%(levelname)s] "
                "%(name)s:%(lineno)d - %(message)s"
            )
        
        handlers = [logging.StreamHandler(sys.stdout)]
        
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            handlers.append(file_handler)
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=handlers
        )
        
        # Set lower level for external libraries
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        
        return True
        
    except Exception as e:
        print(f"Failed to set up logging: {e}")
        return False
