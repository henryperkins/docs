"""
combined_module.py

Combines code chunking, management, and summarization operations with context awareness,
utilizing AST analysis to create meaningful and syntactically valid code segments.
"""

import ast
import logging
import os
import uuid
import hashlib
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import List, Optional, Dict, Set, Any, Tuple, Union
from pathlib import Path
import itertools
from collections import defaultdict
from metrics_combined import EnhancedMetricsCalculator, ComplexityMetrics
from token_utils import TokenManager, TokenizationError, TokenizationResult

# Configure logger
logger = logging.getLogger(__name__)

class ChunkType(Enum):
    """Enumeration of possible chunk types."""
    MODULE = "module"
    CLASS = "class"
    METHOD = "method"
    FUNCTION = "function"
    NESTED_FUNCTION = "nested_function"
    CLASS_METHOD = "class_method"
    STATIC_METHOD = "static_method"
    PROPERTY = "property"
    ASYNC_FUNCTION = "async_function"
    ASYNC_METHOD = "async_method"
    DECORATOR = "decorator"

@dataclass(frozen=True)
class ChunkMetadata:
    """Stores metadata about a code chunk, including complexity."""
    start_line: int
    end_line: int
    chunk_type: ChunkType
    token_count: int = 0
    dependencies: Set[int] = field(default_factory=set)
    used_by: Set[int] = field(default_factory=set)
    complexity: Optional[float] = None

@dataclass(frozen=True)
class CodeChunk:
    """Immutable representation of a code chunk with metadata."""
    file_path: str
    start_line: int
    end_line: int
    function_name: Optional[str]
    class_name: Optional[str]
    chunk_content: str
    language: str
    is_async: bool = False
    decorator_list: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    parent_chunk_id: Optional[int] = None
    chunk_id: int = field(init=False)
    _chunk_counter = itertools.count()
    metadata: ChunkMetadata = field(init=False)

    @property
    def tokens(self) -> Optional[List[int]]:
        """Returns cached tokens."""
        return self._tokens

    @property
    def token_count(self) -> int:
        """Returns the number of tokens."""
        return len(self._tokens) if self._tokens else 0

    def __post_init__(self):
        """Initializes CodeChunk with tokens and complexity."""
        object.__setattr__(self, "chunk_id", next(self._chunk_counter))

        # Use EnhancedMetricsCalculator to calculate complexity
        metrics_calculator = EnhancedMetricsCalculator()
        metrics = metrics_calculator.calculate_metrics(self.chunk_content, self.file_path, self.language)

        complexity = metrics.get('complexity', None)

        metadata = ChunkMetadata(
            start_line=self.start_line,
            end_line=self.end_line,
            chunk_type=self._determine_chunk_type(),
            token_count=self.token_count,
            complexity=complexity
        )
        object.__setattr__(self, "metadata", metadata)

    def _determine_chunk_type(self) -> ChunkType:
        """Determines the type of this chunk based on its properties."""
        if self.class_name and self.function_name:
            if 'staticmethod' in self.decorator_list:
                return ChunkType.STATIC_METHOD
            elif 'classmethod' in self.decorator_list:
                return ChunkType.CLASS_METHOD
            elif 'property' in self.decorator_list:
                return ChunkType.PROPERTY
            elif self.is_async:
                return ChunkType.ASYNC_METHOD
            return ChunkType.METHOD
        elif self.class_name:
            return ChunkType.CLASS
        elif self.function_name:
            if self.is_async:
                return ChunkType.ASYNC_FUNCTION
            return ChunkType.NESTED_FUNCTION if self.parent_chunk_id else ChunkType.FUNCTION
        elif self.decorator_list:
            return ChunkType.DECORATOR
        return ChunkType.MODULE

    def _calculate_hash(self) -> str:
        """Calculates a SHA256 hash of the chunk content."""
        return hashlib.sha256(self.chunk_content.encode('utf-8')).hexdigest()

    def get_context_string(self) -> str:
        """Returns a concise string representation of the chunk's context."""
        parts = [
            f"File: {self.file_path}",
            f"Lines: {self.start_line}-{self.end_line}"
        ]

        if self.class_name:
            parts.append(f"Class: {self.class_name}")
        if self.function_name:
            prefix = "Async " if self.is_async else ""
            parts.append(f"{prefix}Function: {self.function_name}")
        if self.decorator_list:
            parts.append(f"Decorators: {', '.join(self.decorator_list)}")

        return ", ".join(parts)

    def get_hierarchy_path(self) -> str:
        """Returns the full hierarchy path of the chunk."""
        parts = [Path(self.file_path).stem]
        if self.class_name:
            parts.append(self.class_name)
        if self.function_name:
            parts.append(self.function_name)
        return ".".join(parts)

    def can_merge_with(self, other: 'CodeChunk') -> bool:
        """Determines if this chunk can be merged with another using AST analysis."""
        if not (
            self.file_path == other.file_path and
            self.language == other.language and
            self.end_line + 1 == other.start_line
        ):
            return False

        combined_content = self.chunk_content + '\n' + other.chunk_content
        try:
            ast.parse(combined_content)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def merge(chunk1: 'CodeChunk', chunk2: 'CodeChunk') -> 'CodeChunk':
        """Creates a new chunk by merging two chunks using AST analysis."""
        if not chunk1.can_merge_with(chunk2):
            raise ValueError("Chunks cannot be merged, AST validation failed.")

        combined_content = chunk1.chunk_content + '\n' + chunk2.chunk_content
        tokens = TokenManager.count_tokens(combined_content)

        combined_dependencies = chunk1.metadata.dependencies.union(chunk2.metadata.dependencies)
        combined_used_by = chunk1.metadata.used_by.union(chunk2.metadata.used_by)

        chunk_type = chunk1.metadata.chunk_type

        complexity = chunk1.metadata.complexity
        if chunk2.metadata.complexity is not None:
            complexity += chunk2.metadata.complexity

        new_metadata = ChunkMetadata(
            start_line=chunk1.start_line,
            end_line=chunk2.end_line,
            chunk_type=chunk_type,
            token_count=chunk1.token_count + chunk2.token_count,
            dependencies=combined_dependencies,
            used_by=combined_used_by,
            complexity=complexity
        )

        return CodeChunk(
            file_path=chunk1.file_path,
            start_line=chunk1.start_line,
            end_line=chunk2.end_line,
            function_name=chunk1.function_name or chunk2.function_name,
            class_name=chunk1.class_name or chunk2.class_name,
            chunk_content=combined_content,
            token_count=tokens.token_count,
            language=chunk1.language,
            is_async=chunk1.is_async or chunk2.is_async,
            decorator_list=list(set(chunk1.decorator_list + chunk2.decorator_list)),
            docstring=chunk1.docstring or chunk2.docstring,
            parent_chunk_id=chunk1.parent_chunk_id,
            metadata=new_metadata
        )

    def get_possible_split_points(self) -> List[int]:
        """Returns a list of line numbers where the chunk can be split without breaking syntax."""
        try:
            tree = ast.parse(self.chunk_content)
        except SyntaxError:
            return []

        possible_split_points = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.stmt, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if hasattr(node, 'lineno'):
                    split_line = self.start_line + node.lineno - 1
                    if split_line < self.end_line:
                        possible_split_points.append(split_line)

        return sorted(set(possible_split_points))

    def split(self, split_point: int) -> List['CodeChunk']:
        """Splits chunk at specified line number using AST analysis."""
        valid_split_points = self.get_possible_split_points()

        if split_point == self.start_line:
            raise ValueError(f"Cannot split at the very beginning of the chunk (line {split_point}).")
        if split_point == self.end_line:
            raise ValueError(f"Cannot split at the very end of the chunk (line {split_point}).")

        if split_point not in valid_split_points:
            raise ValueError(
                f"Invalid split point at line {split_point}. "
                f"Valid split points are: {valid_split_points}"
            )

        lines = self.chunk_content.splitlines(keepends=True)
        split_idx = split_point - self.start_line

        if split_idx <= 0 or split_idx >= len(lines):
            raise ValueError(
                f"Split index {split_idx} derived from split point {split_point} is out of bounds."
            )

        chunk1_content = ''.join(lines[:split_idx])
        tokens1 = TokenManager.count_tokens(chunk1_content)

        chunk1 = CodeChunk(
            file_path=self.file_path,
            start_line=self.start_line,
            end_line=split_point - 1,
            function_name=self.function_name,
            class_name=self.class_name,
            chunk_content=chunk1_content,
            token_count=tokens1.token_count,
            language=self.language,
            is_async=self.is_async,
            decorator_list=self.decorator_list,
            docstring=self.docstring,
            parent_chunk_id=self.parent_chunk_id,
            metadata=replace(self.metadata, end_line=split_point - 1)
        )

        chunk2_content = ''.join(lines[split_idx:])
        tokens2 = TokenManager.count_tokens(chunk2_content)

        chunk2 = CodeChunk(
            file_path=self.file_path,
            start_line=split_point,
            end_line=self.end_line,
            function_name=self.function_name,
            class_name=self.class_name,
            chunk_content=chunk2_content,
            token_count=tokens2.token_count,
            language=self.language,
            is_async=self.is_async,
            decorator_list=self.decorator_list,
            docstring=self.docstring,
            parent_chunk_id=self.parent_chunk_id,
            metadata=replace(self.metadata, start_line=split_point)
        )

        return [chunk1, chunk2]

    def get_metrics(self) -> Dict[str, Any]:
        return {
            'complexity': self.metadata.complexity,
            'token_count': self.token_count,
            'start_line': self.metadata.start_line,
            'end_line': self.metadata.end_line,
            'type': self.metadata.chunk_type.value,
            'has_docstring': self.docstring is not None,
            'is_async': self.is_async,
            'decorator_count': len(self.decorator_list),
            'has_parent': self.parent_chunk_id is not None
        }

    def __repr__(self) -> str:
        """Returns a detailed string representation of the chunk."""
        content_preview = (
            f"{self.chunk_content[:50]}..."
            if len(self.chunk_content) > 50
            else self.chunk_content
        ).replace('\n', '\\n')

        return (
            f"CodeChunk(file='{self.file_path}', "
            f"lines={self.start_line}-{self.end_line}, "
            f"type={self.metadata.chunk_type.value}, "
            f"content='{content_preview}', "
            f"tokens={self.token_count})"
        )

class ChunkManager:
    """Manages code chunking operations."""

    def __init__(self, max_tokens: int = 4096, overlap: int = 200, repo_path: str = "."):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.token_manager = TokenManager()
        self.project_structure = self.analyze_project_structure(repo_path)

    def analyze_project_structure(self, repo_path: str) -> Dict:
        """Analyzes the project's directory structure."""
        project_structure = {"modules": {}}
        for root, dirs, files in os.walk(repo_path):
            if "__init__.py" in files:
                module_name = os.path.basename(root)
                project_structure["modules"][module_name] = {
                    "files": [os.path.join(root, f) for f in files if f.endswith(".py")],
                    "dependencies": []
                }

        for module_name, module_data in project_structure["modules"].items():
            for file_path in module_data["files"]:
                with open(file_path, "r") as f:
                    code = f.read()
                    try:
                        tree = ast.parse(code)
                        analyzer = DependencyAnalyzer(project_structure)
                        analyzer.visit(tree)
                        module_data["dependencies"] = list(analyzer.dependencies)
                    except SyntaxError:
                        logger.warning(f"Syntax error in {file_path}, skipping dependency analysis.")

        return project_structure

    def create_chunks(self, code: str, file_path: str, language: str) -> List[CodeChunk]:
        """Creates code chunks with context awareness."""
        if language.lower() == "python":
            return self._create_python_chunks(code, file_path)
        else:
            return self._create_simple_chunks(code, file_path, language)

    def _create_python_chunks(self, code: str, file_path: str) -> List[CodeChunk]:
        """Creates chunks for Python code using AST analysis."""
        try:
            tree = ast.parse(code)
            chunks = []
            current_chunk_lines = []
            current_chunk_start = 1
            current_token_count = 0

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if current_chunk_lines:
                        chunks.append(self._create_chunk_from_lines(
                            current_chunk_lines,
                            current_chunk_start,
                            file_path,
                            "python"
                        ))
                        current_chunk_lines = []
                        current_token_count = 0

                    current_chunk_start = node.lineno
                    current_chunk_lines.extend(code.splitlines()[node.lineno - 1:node.end_lineno])
                    current_token_count += self.token_manager.count_tokens("\n".join(current_chunk_lines)).token_count

                    while current_token_count >= self.max_tokens - self.overlap:
                        split_point = self._find_split_point(node, current_chunk_lines)
                        if split_point is None:
                            logger.warning(f"Chunk too large to split: {node.name} in {file_path}")
                            break

                        chunk_lines = current_chunk_lines[:split_point]
                        chunks.append(self._create_chunk_from_lines(
                            chunk_lines,
                            current_chunk_start,
                            file_path,
                            "python"
                        ))

                        current_chunk_start += len(chunk_lines)
                        current_chunk_lines = current_chunk_lines[split_point:]
                        current_token_count = self.token_manager.count_tokens("\n".join(current_chunk_lines)).token_count

                elif current_chunk_lines:
                    current_chunk_lines.append(code.splitlines()[node.lineno - 1])
                    current_token_count += self.token_manager.count_tokens(code.splitlines()[node.lineno - 1]).token_count

                    if current_token_count >= self.max_tokens - self.overlap:
                        chunks.append(self._create_chunk_from_lines(
                            current_chunk_lines,
                            current_chunk_start,
                            file_path,
                            "python"
                        ))
                        current_chunk_lines = []
                        current_token_count = 0
                        current_chunk_start = node.lineno

            if current_chunk_lines:
                chunks.append(self._create_chunk_from_lines(
                    current_chunk_lines,
                    current_chunk_start,
                    file_path,
                    "python"
                ))

            return chunks

        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error creating Python chunks: {e}")
            return []

    def _create_simple_chunks(self, code: str, file_path: str, language: str) -> List[CodeChunk]:
        """Creates chunks based on lines of code with a simple token count limit."""
        chunks = []
        current_chunk_lines = []
        current_chunk_start = 1
        current_token_count = 0

        for i, line in enumerate(code.splitlines(), 1):
            current_chunk_lines.append(line)
            current_token_count += self.token_manager.count_tokens(line).token_count

            if current_token_count >= self.max_tokens - self.overlap:
                chunks.append(self._create_chunk_from_lines(
                    current_chunk_lines,
                    current_chunk_start,
                    file_path,
                    language
                ))
                current_chunk_lines = []
                current_token_count = 0
                current_chunk_start = i + 1

        if current_chunk_lines:
            chunks.append(self._create_chunk_from_lines(
                current_chunk_lines,
                current_chunk_start,
                file_path,
                language
            ))

        return chunks

    def _create_chunk_from_lines(
        self,
        lines: List[str],
        start_line: int,
        file_path: str,
        language: str
    ) -> CodeChunk:
        """Creates a CodeChunk from a list of lines."""
        chunk_content = "\n".join(lines)
        end_line = start_line + len(lines) - 1
        chunk_id = f"{file_path}:{start_line}-{end_line}"
        return CodeChunk(
            chunk_id=chunk_id,
            content=chunk_content,
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            language=language,
        )

    async def get_contextual_chunks(
        self,
        chunk: CodeChunk,
        all_chunks: List[CodeChunk],
        max_tokens: int = 4096
    ) -> List[CodeChunk]:
        """Retrieves contextually relevant chunks for a given chunk."""
        context_chunks = [chunk]
        current_tokens = chunk.token_count
        visited_chunks = {chunk.chunk_id}

        module_name = self._get_module_name(chunk.file_path)
        if module_name:
            module_dependencies = self.project_structure["modules"][module_name].get("dependencies", [])
            for dep_module in module_dependencies:
                for c in all_chunks:
                    if (c.file_path.startswith(dep_module.replace(".", "/")) and
                        c.chunk_id not in visited_chunks and
                        current_tokens + c.token_count <= max_tokens):
                        context_chunks.append(c)
                        current_tokens += c.token_count
                        visited_chunks.add(c.chunk_id)

        return context_chunks

    def _get_module_name(self, file_path: str) -> Optional[str]:
        """Gets the module name from a file path."""
        for module_name, module_data in self.project_structure["modules"].items():
            if file_path in module_data["files"]:
                return module_name
        return None

    def _find_split_point(self, node: ast.AST, lines: List[str]) -> Optional[int]:
        """Finds a suitable split point within a function or class."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for i, line in enumerate(lines):
                if line.strip().startswith("return") or line.strip().startswith("yield"):
                    return i + 1
        elif isinstance(node, ast.ClassDef):
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    return i
        return None

class DependencyAnalyzer(ast.NodeVisitor):
    """Analyzes dependencies within a Python file."""

    def __init__(self, project_structure: Dict):
        self.project_structure = project_structure
        self.dependencies: Set[str] = set()
        self.scope_stack: List[ast.AST] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.scope_stack.append(node)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.scope_stack.append(node)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        self.scope_stack.append(node)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_Call(self, node: ast.Call):
        called_func_name = self._get_called_function_name(node)
        if called_func_name and not self._is_in_current_scope(called_func_name):
            self._add_dependency_if_exists(called_func_name)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name):
            object_name = node.value.id
            attribute_name = node.attr
            full_name = f"{object_name}.{attribute_name}"
            if not self._is_in_current_scope(full_name):
                self._add_dependency_if_exists(full_name)
        self.generic_visit(node)

    def _get_called_function_name(self, node: ast.Call) -> Optional[str]:
        """Gets the name of the called function."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _is_in_current_scope(self, name: str) -> bool:
        """Checks if a name is defined in the current scope."""
        return any(name in [n.name for n in self.scope_stack if hasattr(n, 'name')])

    def _add_dependency_if_exists(self, name: str):
        """Adds a dependency if the name exists in another module."""
        for other_module_name, other_module_data in self.project_structure["modules"].items():
            if any(name in f for f in other_module_data["files"]):
                self.dependencies.add(other_module_name)

class SummarizationStrategy(Enum):
    """Available summarization strategies."""
    PRESERVE_STRUCTURE = auto()
    PRESERVE_INTERFACES = auto()
    PRESERVE_DEPENDENCIES = auto()
    PRESERVE_COMPLEXITY = auto()
    AGGRESSIVE = auto()

@dataclass
class SummarizationConfig:
    """Configuration for code summarization."""
    strategy: SummarizationStrategy
    max_tokens: int
    preserve_docstrings: bool = True
    preserve_types: bool = True
    preserve_decorators: bool = True
    min_block_tokens: int = 50
    importance_threshold: float = 0.5

@dataclass
class CodeSegment:
    """Represents a segment of code with metadata."""
    content: str
    start_line: int
    end_line: int
    segment_type: str
    importance: float = 0.0
    token_count: int = 0
    dependencies: Set[str] = field(default_factory=set)

class CodeSummarizer:
    """Handles intelligent code summarization with AST analysis."""

    def __init__(self, token_manager: TokenManager):
        self.token_manager = token_manager
        self._ast_cache: Dict[str, ast.AST] = {}
        self._importance_cache: Dict[str, float] = {}

    async def summarize_chunk(
        self,
        chunk: CodeChunk,
        config: SummarizationConfig
    ) -> Optional[CodeChunk]:
        """Summarizes a code chunk according to the specified strategy."""
        try:
            tree = await self._parse_code(chunk.chunk_content)
            if not tree:
                return None

            if config.strategy == SummarizationStrategy.PRESERVE_STRUCTURE:
                summarized = await self._summarize_preserve_structure(tree, chunk, config)
            elif config.strategy == SummarizationStrategy.PRESERVE_INTERFACES:
                summarized = await self._summarize_preserve_interfaces(tree, chunk, config)
            elif config.strategy == SummarizationStrategy.PRESERVE_DEPENDENCIES:
                summarized = await self._summarize_preserve_dependencies(tree, chunk, config)
            elif config.strategy == SummarizationStrategy.PRESERVE_COMPLEXITY:
                summarized = await self._summarize_preserve_complexity(tree, chunk, config)
            else:
                summarized = await self._summarize_aggressive(tree, chunk, config)

            if not summarized:
                return None

            return CodeChunk(
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                function_name=chunk.function_name,
                class_name=chunk.class_name,
                chunk_content=summarized,
                language=chunk.language,
                is_async=chunk.is_async,
                decorator_list=chunk.decorator_list,
                docstring=chunk.docstring,
                parent_chunk_id=chunk.parent_chunk_id
            )

        except Exception as e:
            logger.error(f"Error summarizing chunk: {str(e)}")
            return None

    async def _summarize_preserve_structure(
        self,
        tree: ast.AST,
        chunk: CodeChunk,
        config: SummarizationConfig
    ) -> Optional[str]:
        """Summarizes code while preserving structural elements."""
        class StructurePreservingVisitor(ast.NodeTransformer):
            def __init__(self, config: SummarizationConfig):
                self.config = config

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                new_decorators = node.decorator_list if self.config.preserve_decorators else []
                docstring = ast.get_docstring(node)
                new_body = [ast.Expr(ast.Str(s=docstring))] if docstring and self.config.preserve_docstrings else []
                for stmt in node.body:
                    if isinstance(stmt, ast.Return):
                        new_body.append(stmt)
                    elif isinstance(stmt, (ast.Assert, ast.Raise)):
                        new_body.append(stmt)
                    elif len(new_body) < 2:
                        new_body.append(stmt)
                return ast.FunctionDef(
                    name=node.name,
                    args=node.args,
                    body=new_body,
                    decorator_list=new_decorators,
                    returns=node.returns if self.config.preserve_types else None
                )

            def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
                new_body = []
                docstring = ast.get_docstring(node)
                if docstring and self.config.preserve_docstrings:
                    new_body.append(ast.Expr(ast.Str(s=docstring)))
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        new_body.append(self.visit_FunctionDef(item))
                    elif isinstance(item, ast.ClassDef):
                        new_body.append(self.visit_ClassDef(item))
                    elif isinstance(item, (ast.AnnAssign, ast.Assign)):
                        new_body.append(item)
                return ast.ClassDef(
                    name=node.name,
                    bases=node.bases,
                    keywords=node.keywords,
                    body=new_body,
                    decorator_list=node.decorator_list if self.config.preserve_decorators else []
                )

        transformer = StructurePreservingVisitor(config)
        transformed_tree = transformer.visit(tree)
        try:
            return ast.unparse(transformed_tree)
        except Exception as e:
            logger.error(f"Error unparsing transformed AST: {str(e)}")
            return None

    async def _summarize_preserve_interfaces(
        self,
        tree: ast.AST,
        chunk: CodeChunk,
        config: SummarizationConfig
    ) -> Optional[str]:
        """Preserves interface definitions while simplifying implementations."""
        class InterfacePreservingVisitor(ast.NodeTransformer):
            def __init__(self, config: SummarizationConfig):
                self.config = config

            def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
                new_body = []
                docstring = ast.get_docstring(node)
                if docstring and self.config.preserve_docstrings:
                    new_body.append(ast.Expr(ast.Str(s=docstring)))
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        new_body.append(self.visit_FunctionDef(item))
                    elif isinstance(item, (ast.AnnAssign, ast.Assign)):
                        new_body.append(item)
                return ast.ClassDef(
                    name=node.name,
                    bases=node.bases,
                    keywords=node.keywords,
                    body=new_body,
                    decorator_list=node.decorator_list if self.config.preserve_decorators else []
                )

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                new_body = []
                docstring = ast.get_docstring(node)
                if docstring and self.config.preserve_docstrings:
                    new_body.append(ast.Expr(ast.Str(s=docstring)))
                if not new_body:
                    new_body.append(ast.Pass())
                return ast.FunctionDef(
                    name=node.name,
                    args=node.args,
                    body=new_body,
                    decorator_list=node.decorator_list if self.config.preserve_decorators else [],
                    returns=node.returns if self.config.preserve_types else None
                )

        transformer = InterfacePreservingVisitor(config)
        transformed_tree = transformer.visit(tree)
        return ast.unparse(transformed_tree)

    async def _summarize_preserve_dependencies(
        self,
        tree: ast.AST,
        chunk: CodeChunk,
        config: SummarizationConfig
    ) -> Optional[str]:
        """Preserves code segments essential for dependency relationships."""
        class DependencyPreservingVisitor(ast.NodeTransformer):
            def __init__(self, chunk: CodeChunk, config: SummarizationConfig):
                self.chunk = chunk
                self.config = config
                self.dependencies = chunk.metadata.dependencies
                self.used_names = set()

            def visit_Name(self, node: ast.Name) -> ast.AST:
                if isinstance(node, ast.Name):
                    self.used_names.add(node.id)
                return node

            def visit_FunctionDef(self, node: ast.FunctionDef) -> Optional[ast.AST]:
                self.used_names.clear()
                self.generic_visit(node)
                if self.used_names.intersection(self.dependencies):
                    return self._preserve_function(node)
                return self._simplify_function(node)

            def _preserve_function(self, node: ast.FunctionDef) -> ast.FunctionDef:
                new_body = []
                docstring = ast.get_docstring(node)
                if docstring and self.config.preserve_docstrings:
                    new_body.append(ast.Expr(ast.Str(s=docstring)))
                for stmt in node.body:
                    if self._uses_dependencies(stmt):
                        new_body.append(stmt)
                if not new_body:
                    new_body.append(ast.Pass())
                return ast.FunctionDef(
                    name=node.name,
                    args=node.args,
                    body=new_body,
                    decorator_list=node.decorator_list if self.config.preserve_decorators else [],
                    returns=node.returns if self.config.preserve_types else None
                )

            def _simplify_function(self, node: ast.FunctionDef) -> ast.FunctionDef:
                new_body = []
                docstring = ast.get_docstring(node)
                if docstring and self.config.preserve_docstrings:
                    new_body.append(ast.Expr(ast.Str(s=docstring)))
                new_body.append(ast.Pass())
                return ast.FunctionDef(
                    name=node.name,
                    args=node.args,
                    body=new_body,
                    decorator_list=[],
                    returns=node.returns if self.config.preserve_types else None
                )

            def _uses_dependencies(self, node: ast.AST) -> bool:
                self.used_names.clear()
                self.visit(node)
                return bool(self.used_names.intersection(self.dependencies))

        transformer = DependencyPreservingVisitor(chunk, config)
        transformed_tree = transformer.visit(tree)
        return ast.unparse(transformed_tree)

    async def _summarize_preserve_complexity(
        self,
        tree: ast.AST,
        chunk: CodeChunk,
        config: SummarizationConfig
    ) -> Optional[str]:
        """Preserves complex code segments while simplifying others."""
        class ComplexityPreservingVisitor(ast.NodeTransformer):
            def __init__(self, config: SummarizationConfig):
                self.config = config
                self.complexity_threshold = 5

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                complexity = self._calculate_complexity(node)
                if complexity >= self.complexity_threshold:
                    return self._preserve_complex_function(node)
                return self._simplify_function(node)

            def _calculate_complexity(self, node: ast.AST) -> int:
                complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.With, ast.AsyncWith, ast.AsyncFor)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1
                    elif isinstance(child, ast.Compare):
                        complexity += len(child.ops)
                return complexity

            def _preserve_complex_function(self, node: ast.FunctionDef) -> ast.FunctionDef:
                new_body = []
                docstring = ast.get_docstring(node)
                if docstring and self.config.preserve_docstrings:
                    new_body.append(ast.Expr(ast.Str(s=docstring)))
                for stmt in node.body:
                    if self._is_complex_statement(stmt):
                        new_body.append(stmt)
                    elif isinstance(stmt, (ast.Return, ast.Raise, ast.Assert)):
                        new_body.append(stmt)
                if not new_body:
                    new_body.append(ast.Pass())
                return ast.FunctionDef(
                    name=node.name,
                    args=node.args,
                    body=new_body,
                    decorator_list=node.decorator_list if self.config.preserve_decorators else [],
                    returns=node.returns if self.config.preserve_types else None
                )

            def _is_complex_statement(self, node: ast.AST) -> bool:
                return (
                    isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With, ast.AsyncWith, ast.AsyncFor)) or
                    self._calculate_complexity(node) > 2
                )

        transformer = ComplexityPreservingVisitor(config)
        transformed_tree = transformer.visit(tree)
        return ast.unparse(transformed_tree)

    async def _summarize_aggressive(
        self,
        tree: ast.AST,
        chunk: CodeChunk,
        config: SummarizationConfig
    ) -> Optional[str]:
        """Performs aggressive summarization while maintaining validity."""
        class AggressiveTransformer(ast.NodeTransformer):
            def __init__(self, config: SummarizationConfig):
                self.config = config

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                new_body = []
                docstring = ast.get_docstring(node)
                if docstring and self.config.preserve_docstrings:
                    new_body.append(ast.Expr(ast.Str(s=docstring)))
                new_body.append(ast.Pass())
                return ast.FunctionDef(
                    name=node.name,
                    args=self._simplify_arguments(node.args),
                    body=new_body,
                    decorator_list=[],
                    returns=node.returns if self.config.preserve_types else None
                )

            def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
                new_body = []
                docstring = ast.get_docstring(node)
                if docstring and self.config.preserve_docstrings:
                    new_body.append(ast.Expr(ast.Str(s=docstring)))
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        new_body.append(self.visit_FunctionDef(item))
                if not new_body:
                    new_body.append(ast.Pass())
                return ast.ClassDef(
                    name=node.name,
                    bases=[],
                    keywords=[],
                    body=new_body,
                    decorator_list=[]
                )

            def _simplify_arguments(self, args: ast.arguments) -> ast.arguments:
                return ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=arg.arg, annotation=None) for arg in args.args],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[]
                )

        transformer = AggressiveTransformer(config)
        transformed_tree = transformer.visit(tree)
        return ast.unparse(transformed_tree)

    async def _parse_code(self, content: str) -> Optional[ast.AST]:
        """Parses code into AST with error handling."""
        try:
            return ast.parse(content)
        except SyntaxError as e:
            logger.error(f"Syntax error parsing code: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error parsing code: {str(e)}")
            return None

    async def get_summary_metrics(self, chunk: CodeChunk) -> Dict[str, Any]:
        """Gets metrics about the summarization process."""
        try:
            tree = await self._parse_code(chunk.chunk_content)
            if not tree:
                return {}

            metrics = {
                'original_lines': len(chunk.chunk_content.splitlines()),
                'original_tokens': chunk.token_count,
                'complexity_score': self._get_node_complexity(tree),
                'dependency_count': len(self._get_node_dependencies(tree)),
                'critical_score': self._get_critical_score(tree),
                'maintenance_score': self._get_maintenance_score(tree),
                'usage_patterns': self._get_usage_count(tree)
            }

            return metrics

        except Exception as e:
            logger.error(f"Error getting summary metrics: {str(e)}")
            return {}

    def clear_caches(self) -> None:
        """Clears internal caches."""
        self._ast_cache.clear()
        self._importance_cache.clear()
        logger.debug("Cleared summarizer caches")

    # Additional utility methods for summarization metrics
    def _get_node_complexity(self, node: ast.AST) -> float:
        """Calculates complexity score for a node."""
        complexity = 1.0
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1.0
            elif isinstance(child, ast.Try):
                complexity += 0.5
            elif isinstance(child, ast.BoolOp):
                complexity += 0.3 * len(child.values)
            elif isinstance(child, ast.Compare):
                complexity += 0.2 * len(child.ops)
        return min(complexity, 10.0)

    def _get_node_dependencies(self, node: ast.AST) -> Set[str]:
        """Gets dependencies for a node."""
        dependencies = set()
        class DependencyVisitor(ast.NodeVisitor):
            def visit_Name(self, node: ast.Name):
                if isinstance(node.ctx, ast.Load):
                    dependencies.add(node.id)
            def visit_Attribute(self, node: ast.Attribute):
                if isinstance(node.ctx, ast.Load):
                    dependencies.add(node.attr)
        DependencyVisitor().visit(node)
        return dependencies

    def _get_usage_count(self, node: ast.AST) -> int:
        """Gets usage count for variables in node."""
        usage_count = 0
        class UsageVisitor(ast.NodeVisitor):
            def visit_Name(self, node: ast.Name):
                nonlocal usage_count
                if isinstance(node.ctx, ast.Load):
                    usage_count += 1
        UsageVisitor().visit(node)
        return usage_count

    def _get_critical_score(self, node: ast.AST) -> float:
        """Calculates criticality score for a node."""
        score = 0.0
        critical_patterns = {
            'error': 0.8,
            'exception': 0.8,
            'validate': 0.6,
            'check': 0.5,
            'verify': 0.5,
            'assert': 0.7,
            'security': 0.9,
            'auth': 0.9
        }
        class CriticalityVisitor(ast.NodeVisitor):
            def visit_Name(self, node: ast.Name):
                nonlocal score
                for pattern, value in critical_patterns.items():
                    if pattern in node.id.lower():
                        score = max(score, value)
            def visit_Str(self, node: ast.Str):
                nonlocal score
                for pattern, value in critical_patterns.items():
                    if pattern in node.s.lower():
                        score = max(score, value * 0.5)
        CriticalityVisitor().visit(node)
        return score

    def _get_maintenance_score(self, node: ast.AST) -> float:
        """Calculates maintenance score based on code quality indicators."""
        score = 0.0
        class MaintenanceVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef):
                nonlocal score
                if ast.get_docstring(node):
                    score += 0.3
                if len(node.args.args) > 5:
                    score -= 0.2
            def visit_ClassDef(self, node: ast.ClassDef):
                nonlocal score
                if ast.get_docstring(node):
                    score += 0.3
                if len(node.bases) > 2:
                    score -= 0.2
            def visit_Try(self, node: ast.Try):
                nonlocal score
                score += 0.2
        MaintenanceVisitor().visit(node)
        return max(0.0, min(1.0, score))
