"""
code_summarization.py

Implements intelligent code summarization and truncation strategies
with AST-based analysis and semantic preservation.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Union
from enum import Enum, auto
import logging
from collections import defaultdict

from code_chunk import CodeChunk, ChunkType
from token_utils import TokenManager, TokenizationResult

logger = logging.getLogger(__name__)

class SummarizationStrategy(Enum):
    """Available summarization strategies."""
    PRESERVE_STRUCTURE = auto()  # Keep code structure, remove details
    PRESERVE_INTERFACES = auto()  # Keep interfaces, remove implementations
    PRESERVE_DEPENDENCIES = auto()  # Keep dependency-related code
    PRESERVE_COMPLEXITY = auto()  # Keep complex logic
    AGGRESSIVE = auto()  # Maximum reduction

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
        """
        Summarizes a code chunk according to the specified strategy.
        
        Args:
            chunk: Code chunk to summarize
            config: Summarization configuration
            
        Returns:
            Summarized code chunk or None if summarization fails
        """
        try:
            # Parse and analyze code
            tree = await self._parse_code(chunk.content)
            if not tree:
                return None

            # Select appropriate summarization method
            if config.strategy == SummarizationStrategy.PRESERVE_STRUCTURE:
                summarized = await self._summarize_preserve_structure(
                    tree, chunk, config
                )
            elif config.strategy == SummarizationStrategy.PRESERVE_INTERFACES:
                summarized = await self._summarize_preserve_interfaces(
                    tree, chunk, config
                )
            elif config.strategy == SummarizationStrategy.PRESERVE_DEPENDENCIES:
                summarized = await self._summarize_preserve_dependencies(
                    tree, chunk, config
                )
            elif config.strategy == SummarizationStrategy.PRESERVE_COMPLEXITY:
                summarized = await self._summarize_preserve_complexity(
                    tree, chunk, config
                )
            else:  # AGGRESSIVE
                summarized = await self._summarize_aggressive(
                    tree, chunk, config
                )

            if not summarized:
                return None

            # Create new chunk with summarized content
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
        """
        Summarizes code while preserving structural elements.
        
        Maintains class/function definitions, signature information,
        and key structural elements while reducing implementation details.
        """
        class StructurePreservingVisitor(ast.NodeTransformer):
            def __init__(self, config: SummarizationConfig):
                self.config = config
                self.in_function_body = False
                self.current_tokens = 0

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                # Always keep function signature
                self.in_function_body = True
                
                # Preserve decorators if configured
                if self.config.preserve_decorators:
                    new_decorators = node.decorator_list
                else:
                    new_decorators = []

                # Keep docstring if present and configured
                docstring = ast.get_docstring(node)
                if docstring and self.config.preserve_docstrings:
                    new_body = [ast.Expr(ast.Str(s=docstring))]
                else:
                    new_body = []

                # Simplify function body
                for stmt in node.body:
                    if isinstance(stmt, ast.Return):
                        new_body.append(stmt)  # Keep return statements
                    elif isinstance(stmt, (ast.Assert, ast.Raise)):
                        new_body.append(stmt)  # Keep assertions and raises
                    elif len(new_body) < 2:  # Keep minimal representative body
                        new_body.append(stmt)

                self.in_function_body = False
                
                return ast.FunctionDef(
                    name=node.name,
                    args=node.args,
                    body=new_body,
                    decorator_list=new_decorators,
                    returns=node.returns if self.config.preserve_types else None
                )

            def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
                # Preserve class structure with minimal method bodies
                new_body = []
                
                # Keep docstring if present and configured
                docstring = ast.get_docstring(node)
                if docstring and self.config.preserve_docstrings:
                    new_body.append(ast.Expr(ast.Str(s=docstring)))

                # Process class body
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        new_body.append(self.visit_FunctionDef(item))
                    elif isinstance(item, ast.ClassDef):
                        new_body.append(self.visit_ClassDef(item))
                    elif isinstance(item, (ast.AnnAssign, ast.Assign)):
                        new_body.append(item)  # Keep class attributes

                return ast.ClassDef(
                    name=node.name,
                    bases=node.bases,
                    keywords=node.keywords,
                    body=new_body,
                    decorator_list=node.decorator_list if self.config.preserve_decorators else []
                )

        # Apply transformation
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
        """
        Preserves interface definitions while simplifying implementations.
        """
        class InterfacePreservingVisitor(ast.NodeTransformer):
            def __init__(self, config: SummarizationConfig):
                self.config = config
                self.interface_methods = set()
                self.current_class = None

            def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
                self.current_class = node.name
                new_body = []
                
                # Keep docstring
                docstring = ast.get_docstring(node)
                if docstring and self.config.preserve_docstrings:
                    new_body.append(ast.Expr(ast.Str(s=docstring)))

                # Process class body
                for item in node.body:
                    if self._is_interface_element(item):
                        if isinstance(item, ast.FunctionDef):
                            new_body.append(self._simplify_method(item))
                        else:
                            new_body.append(item)

                self.current_class = None
                return ast.ClassDef(
                    name=node.name,
                    bases=node.bases,
                    keywords=node.keywords,
                    body=new_body,
                    decorator_list=node.decorator_list if self.config.preserve_decorators else []
                )

            def _is_interface_element(self, node: ast.AST) -> bool:
                """Determines if a node is part of the interface."""
                if isinstance(node, ast.FunctionDef):
                    # Check for abstract methods or property decorators
                    return (
                        any(self._is_interface_decorator(d) for d in node.decorator_list) or
                        self._is_empty_implementation(node)
                    )
                elif isinstance(node, (ast.AnnAssign, ast.Assign)):
                    # Keep type annotations and constants
                    return True
                return False

            def _is_interface_decorator(self, node: ast.AST) -> bool:
                """Checks if a decorator marks an interface method."""
                if isinstance(node, ast.Name):
                    return node.id in {'abstractmethod', 'property', 'abstractproperty'}
                elif isinstance(node, ast.Attribute):
                    return node.attr in {'abstractmethod', 'property', 'abstractproperty'}
                return False

            def _is_empty_implementation(self, node: ast.FunctionDef) -> bool:
                """Checks if a method has an empty implementation."""
                body = node.body
                return (
                    len(body) == 1 and
                    (isinstance(body[0], ast.Pass) or
                     isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Str))
                )

            def _simplify_method(self, node: ast.FunctionDef) -> ast.FunctionDef:
                """Simplifies a method while preserving interface information."""
                new_body = []
                
                # Keep docstring
                docstring = ast.get_docstring(node)
                if docstring and self.config.preserve_docstrings:
                    new_body.append(ast.Expr(ast.Str(s=docstring)))
                
                # Add pass statement if needed
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
        """
        Preserves code segments essential for dependency relationships.
        """
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
                # Check if function uses or provides dependencies
                self.used_names.clear()
                self.generic_visit(node)
                
                if self.used_names.intersection(self.dependencies):
                    return self._preserve_function(node)
                return self._simplify_function(node)

            def _preserve_function(self, node: ast.FunctionDef) -> ast.FunctionDef:
                """Preserves function with minimal modifications."""
                new_body = []
                
                # Keep docstring
                docstring = ast.get_docstring(node)
                if docstring and self.config.preserve_docstrings:
                    new_body.append(ast.Expr(ast.Str(s=docstring)))
                
                # Keep dependency-related statements
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
                """Creates a simplified version of the function."""
                new_body = []
                
                # Keep docstring
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
                """Checks if an AST node uses any dependencies."""
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
        """
        Preserves complex code segments while simplifying others.
        """
        class ComplexityPreservingVisitor(ast.NodeTransformer):
            def __init__(self, config: SummarizationConfig):
                self.config = config
                self.complexity_threshold = 5  # Adjustable threshold

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                complexity = self._calculate_complexity(node)
                if complexity >= self.complexity_threshold:
                    return self._preserve_complex_function(node)
                return self._simplify_function(node)

            def _calculate_complexity(self, node: ast.AST) -> int:
                """Calculates cyclomatic complexity."""
                complexity = 1  # Base complexity
                
                for child in ast.walk(node):
                    # Increment complexity for control flow statements
                    if isinstance(child, (
                        ast.If, ast.While, ast.For,
                        ast.ExceptHandler, ast.With,
                        ast.AsyncWith, ast.AsyncFor
                    )):
                        complexity += 1
                    # Increment for each logical operator
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1
                    # Increment for each conditional operator
                    elif isinstance(child, ast.Compare):
                        complexity += len(child.ops)
                
                return complexity

            def _preserve_complex_function(self, node: ast.FunctionDef) -> ast.FunctionDef:
                """Preserves complex function with minimal modifications."""
                new_body = []
                
                # Keep docstring
                docstring = ast.get_docstring(node)
                if docstring and self.config.preserve_docstrings:
                    new_body.append(ast.Expr(ast.Str(s=docstring)))
                
                # Preserve complex logic blocks
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
                """Determines if a statement is complex."""
                return (
                    isinstance(node, (
                        ast.If, ast.While, ast.For,
                        ast.Try, ast.With, ast.AsyncWith,
                        ast.AsyncFor
                    )) or
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
        """
        Performs aggressive summarization while maintaining validity.
        """
        class AggressiveTransformer(ast.NodeTransformer):
            def __init__(self, config: SummarizationConfig):
                self.config = config
                self.essential_elements = set()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                """Aggressively simplifies function definitions."""
                new_body = []
                
                # Keep only docstring if configured
                docstring = ast.get_docstring(node)
                if docstring and self.config.preserve_docstrings:
                    new_body.append(ast.Expr(ast.Str(s=docstring)))
                
                # Add minimal implementation
                new_body.append(ast.Pass())

                return ast.FunctionDef(
                    name=node.name,
                    args=self._simplify_arguments(node.args),
                    body=new_body,
                    decorator_list=[],
                    returns=node.returns if self.config.preserve_types else None
                )

            def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
                """Aggressively simplifies class definitions."""
                new_body = []
                
                # Keep only docstring if configured
                docstring = ast.get_docstring(node)
                if docstring and self.config.preserve_docstrings:
                    new_body.append(ast.Expr(ast.Str(s=docstring)))
                
                # Keep only essential methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if self._is_essential_method(item):
                            new_body.append(self.visit_FunctionDef(item))
                
                if not new_body:
                    new_body.append(ast.Pass())

                return ast.ClassDef(
                    name=node.name,
                    bases=[],  # Remove inheritance
                    keywords=[],
                    body=new_body,
                    decorator_list=[]
                )

            def _simplify_arguments(self, args: ast.arguments) -> ast.arguments:
                """Simplifies function arguments."""
                return ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=arg.arg, annotation=None) for arg in args.args],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[]
                )

            def _is_essential_method(self, node: ast.FunctionDef) -> bool:
                """Determines if a method is essential."""
                return (
                    node.name in {'__init__', '__str__', '__repr__'} or
                    any(d.id == 'property' for d in node.decorator_list
                        if isinstance(d, ast.Name))
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

    def _preserve_critical_elements(self, node: ast.AST) -> bool:
        """Enhanced critical element detection."""
        return any([
            self._is_error_handling(node),
            self._is_state_modification(node),
            self._is_external_interaction(node),
            self._contains_critical_variables(node),
            self._is_conditional_logic(node)
        ])

    def _is_error_handling(self, node: ast.AST) -> bool:
        """Checks if node contains error handling logic."""
        return isinstance(node, (ast.Try, ast.Raise, ast.Assert))

    def _is_state_modification(self, node: ast.AST) -> bool:
        """Checks if node modifies important state."""
        if isinstance(node, ast.Assign):
            return any(
                isinstance(target, ast.Attribute) 
                for target in node.targets
            )
        return False

    def _is_external_interaction(self, node: ast.AST) -> bool:
        """Checks if node interacts with external systems."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id in {
                    'open', 'read', 'write', 'print',
                    'input', 'requests', 'socket'
                }
        return False

    def _contains_critical_variables(self, node: ast.AST) -> bool:
        """Checks if node contains critical variable usage."""
        critical_patterns = {'password', 'secret', 'key', 'token', 'config'}
        
        if isinstance(node, ast.Name):
            return any(
                pattern in node.id.lower()
                for pattern in critical_patterns
            )
        return False

    def _is_conditional_logic(self, node: ast.AST) -> bool:
        """Checks if node contains important conditional logic."""
        return isinstance(node, (ast.If, ast.While, ast.For))

    def _calculate_semantic_importance(self, node: ast.AST) -> float:
        """Enhanced semantic importance calculation."""
        factors = {
            'complexity': self._get_node_complexity(node),
            'dependencies': len(self._get_node_dependencies(node)),
            'usage': self._get_usage_count(node),
            'criticality': self._get_critical_score(node),
            'maintenance': self._get_maintenance_score(node)
        }
        
        weights = {
            'complexity': 0.25,
            'dependencies': 0.25,
            'usage': 0.2,
            'criticality': 0.2,
            'maintenance': 0.1
        }
        
        return sum(score * weights[factor] for factor, score in factors.items())

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
                
        return min(complexity, 10.0)  # Cap complexity score

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
        
        # Check for critical patterns
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
                        score = max(score, value * 0.5)  # Lower weight for strings
                        
        CriticalityVisitor().visit(node)
        return score

    def _get_maintenance_score(self, node: ast.AST) -> float:
        """Calculates maintenance score based on code quality indicators."""
        score = 0.0
        
        class MaintenanceVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef):
                nonlocal score
                # Check for documentation
                if ast.get_docstring(node):
                    score += 0.3
                # Check argument count
                if len(node.args.args) > 5:
                    score -= 0.2
                    
            def visit_ClassDef(self, node: ast.ClassDef):
                nonlocal score
                # Check for documentation
                if ast.get_docstring(node):
                    score += 0.3
                # Check inheritance depth
                if len(node.bases) > 2:
                    score -= 0.2
                    
            def visit_Try(self, node: ast.Try):
                nonlocal score
                # Reward error handling
                score += 0.2
                
        MaintenanceVisitor().visit(node)
        return max(0.0, min(1.0, score))  # Normalize score

    async def get_summary_metrics(self, chunk: CodeChunk) -> Dict[str, Any]:
        """Gets metrics about the summarization process."""
        try:
            tree = await self._parse_code(chunk.content)
            if not tree:
                return {}

            metrics = {
                'original_lines': len(chunk.content.splitlines()),
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