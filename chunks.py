import ast
import logging
from typing import List, Optional
from pathlib import Path

# Assuming these are imported from another module
from token_utils import TokenManager, TokenizationError
from code_chunk import CodeChunk, ChunkType, ChunkMetadata

logger = logging.getLogger(__name__)

class ChunkManager:
    """Manages code chunking operations."""

    def __init__(self, max_tokens: int = 4096, overlap: int = 200):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.token_manager = TokenManager()

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

        # Use TokenManager to count tokens
        token_result = self.token_manager.count_tokens(chunk_content)

        return CodeChunk(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            function_name=None,  # Set appropriately
            class_name=None,  # Set appropriately
            chunk_content=chunk_content,
            language=language,
            metadata=ChunkMetadata(
                start_line=start_line,
                end_line=end_line,
                chunk_type=ChunkType.MODULE,  # Set appropriately
                token_count=token_result.token_count
            )
        )

    def split_chunk(self, chunk: CodeChunk, split_point: int) -> List[CodeChunk]:
        """Splits a chunk at the specified line number."""
        lines = chunk.chunk_content.splitlines()
        if split_point <= 0 or split_point >= len(lines):
            raise ValueError("Invalid split point")

        chunk1_content = "\n".join(lines[:split_point])
        chunk2_content = "\n".join(lines[split_point:])

        token_result1 = self.token_manager.count_tokens(chunk1_content)
        token_result2 = self.token_manager.count_tokens(chunk2_content)

        chunk1 = CodeChunk(
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.start_line + split_point - 1,
            function_name=chunk.function_name,
            class_name=chunk.class_name,
            chunk_content=chunk1_content,
            language=chunk.language,
            metadata=ChunkMetadata(
                start_line=chunk.start_line,
                end_line=chunk.start_line + split_point - 1,
                chunk_type=chunk.metadata.chunk_type,
                token_count=token_result1.token_count
            )
        )

        chunk2 = CodeChunk(
            file_path=chunk.file_path,
            start_line=chunk.start_line + split_point,
            end_line=chunk.end_line,
            function_name=chunk.function_name,
            class_name=chunk.class_name,
            chunk_content=chunk2_content,
            language=chunk.language,
            metadata=ChunkMetadata(
                start_line=chunk.start_line + split_point,
                end_line=chunk.end_line,
                chunk_type=chunk.metadata.chunk_type,
                token_count=token_result2.token_count
            )
        )

        return [chunk1, chunk2]

    def merge_chunks(self, chunk1: CodeChunk, chunk2: CodeChunk) -> CodeChunk:
        """Merges two chunks into a single chunk."""
        if chunk1.file_path != chunk2.file_path or chunk1.end_line + 1 != chunk2.start_line:
            raise ValueError("Chunks cannot be merged")

        merged_content = chunk1.chunk_content + "\n" + chunk2.chunk_content
        token_result = self.token_manager.count_tokens(merged_content)

        return CodeChunk(
            file_path=chunk1.file_path,
            start_line=chunk1.start_line,
            end_line=chunk2.end_line,
            function_name=chunk1.function_name or chunk2.function_name,
            class_name=chunk1.class_name or chunk2.class_name,
            chunk_content=merged_content,
            language=chunk1.language,
            metadata=ChunkMetadata(
                start_line=chunk1.start_line,
                end_line=chunk2.end_line,
                chunk_type=chunk1.metadata.chunk_type,
                token_count=token_result.token_count
            )
        )

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
