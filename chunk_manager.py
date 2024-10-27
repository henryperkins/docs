"""
chunk_manager.py

Manages code chunking operations with context awareness, utilizing AST analysis
to create meaningful and syntactically valid code segments.
"""

import ast
import logging
import os
from typing import List, Optional, Dict, Set
from code_chunk import CodeChunk
from token_utils import TokenManager

logger = logging.getLogger(__name__)

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