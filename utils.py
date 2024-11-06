# utils.py

"""
utils.py - Core utility functions and classes for file handling, logging setup,
and path filtering. Provides foundational functionality for the documentation
generation system.
"""

import os
import sys
import json
import logging
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict, Set, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import pathspec
from functools import lru_cache
import aiohttp

logger = logging.getLogger(__name__)

# Default configurations for excluded patterns
DEFAULT_EXCLUDED_PATTERNS = {
    'dirs': {
        '.git', '.svn', '.hg', '.bzr', '__pycache__', '.pytest_cache', '.mypy_cache',
        'htmlcov', '.tox', '.nox', 'venv', '.venv', 'env', '.env', 'virtualenv', 'node_modules',
        'build', 'dist', '.eggs', '*.egg-info', '.idea', '.vscode', '.vs', '.settings', 'tmp', 'temp'
    },
    'files': {
        '.DS_Store', 'Thumbs.db', 'desktop.ini', '*.pyc', '*.pyo', '*.pyd', '.python-version',
        '.coverage', 'coverage.xml', '.coverage.*', 'pip-log.txt', 'pip-delete-this-directory.txt',
        'poetry.lock', 'Pipfile.lock', '.env', '.env.*', '*.swp', '*.swo', '*~', '*.spec', '*.manifest',
        '*.pdf', '*.doc', '*.docx', '*.log', '*.bak', '*.tmp'
    },
    'extensions': {
        '.pyc', '.pyo', '.pyd', '.so', '.o', '.obj', '.dll', '.dylib', '.egg', '.whl', '.cache',
        '.pdf', '.doc', '.docx', '.jpg', '.jpeg', '.png', '.gif', '.ico', '.mov', '.mp4', '.avi',
        '.mp3', '.wav', '.zip', '.tar.gz', '.tgz', '.rar'
    }
}

def setup_logging(log_file: Optional[Union[str, Path]] = None, log_level: str = "INFO", log_format: Optional[str] = None) -> bool:
    """Sets up logging configuration."""
    try:
        if not log_format:
            log_format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
        
        handlers = [logging.StreamHandler(sys.stdout)]
        
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
            handlers.append(file_handler)
        
        logging.basicConfig(level=getattr(logging, log_level.upper()), format=log_format, handlers=handlers)
        
        # Set lower level for external libraries
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        
        return True
        
    except Exception as e:
        print(f"Failed to set up logging: {e}")
        return False

# FileHandler class for asynchronous file operations
class FileHandler:
    """Handles file operations with caching and error handling."""
    
    _content_cache = {}
    _cache_lock = threading.Lock()
    _max_cache_size = 100
    _executor = ThreadPoolExecutor(max_workers=4)
    
    @classmethod
    async def read_file(cls, file_path: Union[str, Path], use_cache: bool = True, encoding: str = 'utf-8') -> Optional[str]:
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
            logger.warning(f"UnicodeDecodeError for {file_path} with {encoding}, trying with error handling")
            try:
                async with aiofiles.open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    return await f.read()
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None

# PathFilter class for filtering file paths based on exclusion patterns
class PathFilter:
    """Handles file path filtering based on various exclusion patterns."""
    
    def __init__(self, repo_path: Union[str, Path], excluded_dirs: Optional[Set[str]] = None,
                 excluded_files: Optional[Set[str]] = None, skip_types: Optional[Set[str]] = None):
        self.repo_path = Path(repo_path)
        self.excluded_dirs = (excluded_dirs or set()) | DEFAULT_EXCLUDED_PATTERNS['dirs']
        self.excluded_files = (excluded_files or set()) | DEFAULT_EXCLUDED_PATTERNS['files']
        self.skip_types = (skip_types or set()) | DEFAULT_EXCLUDED_PATTERNS['extensions']
        self.gitignore = load_gitignore(repo_path)
        
        self.file_patterns = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, self.excluded_files)

    def should_include_path(self, path: Path, relative_to: Optional[Path] = None) -> bool:
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
                patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                logger.debug(f"Loaded {len(patterns)} patterns from .gitignore")
        else:
            logger.debug("No .gitignore file found")
            
    except Exception as e:
        logger.warning(f"Error reading .gitignore: {str(e)}")
        
    return pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, patterns)

def get_all_file_paths(repo_path: Union[str, Path], excluded_dirs: Optional[Set[str]] = None,
                       excluded_files: Optional[Set[str]] = None, skip_types: Optional[Set[str]] = None,
                       follow_symlinks: bool = False) -> List[str]:
    """Gets all file paths in a repository with improved filtering."""
    repo_path = Path(repo_path)
    if not repo_path.exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        return []
        
    path_filter = PathFilter(repo_path, excluded_dirs, excluded_files, skip_types)
    
    included_paths = []
    
    try:
        for root, dirs, files in os.walk(repo_path, followlinks=follow_symlinks):
            root_path = Path(root)
            
            # Filter directories in-place
            dirs[:] = [d for d in dirs if path_filter.should_include_path(root_path / d, repo_path)]
            
            # Filter and add files
            for file in files:
                file_path = root_path / file
                if path_filter.should_include_path(file_path, repo_path):
                    included_paths.append(str(file_path))
                    
        logger.info(f"Found {len(included_paths)} files in {repo_path} (excluded: dirs={len(path_filter.excluded_dirs)}, files={len(path_filter.excluded_files)}, types={len(path_filter.skip_types)})")
        
        return included_paths
        
    except Exception as e:
        logger.error(f"Error walking repository: {str(e)}")
        return []

def handle_api_error(e: Exception, attempt: int, max_retries: int) -> bool:
    """Handles API errors and determines if a retry should occur."""
    error_type = type(e).__name__
    should_retry = False
    if isinstance(e, aiohttp.ClientError):
        error_type = "NetworkError"
        should_retry = True
    elif isinstance(e, asyncio.TimeoutError):
        error_type = "TimeoutError"
        should_retry = True
    else:
        # Add custom logic to determine if other errors should be retried
        pass

    logger.error(f"API request failed: {error_type}")
    if should_retry and attempt < max_retries:
        logger.warning(f"Retry {attempt}/{max_retries} after error: {error_type}")
    return should_retry
