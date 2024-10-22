import ast
import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from collections import defaultdict
import re

@dataclass
class DependencyInfo:
    """Stores information about code dependencies."""
    name: str
    type: str
    version: Optional[str]
    references: List[str]
    locations: List[Dict[str, int]]  # file and line numbers
    is_internal: bool

class CodeMetricsAnalyzer:
    """Analyzes code for various metrics and dependencies."""
    
    def __init__(self, code: str, file_path: str):
        self.code = code
        self.file_path = file_path
        self.ast_tree = ast.parse(code)
        self.dependency_graph = nx.DiGraph()
        self.cognitive_complexity = 0
        self.dependencies: Dict[str, DependencyInfo] = {}
        
    def analyze(self) -> Dict:
        """Performs comprehensive code analysis."""
        return {
            "cognitive_complexity": self.calculate_cognitive_complexity(),
            "dependencies": self.extract_dependencies(),
            "coverage_metrics": self.calculate_coverage_metrics(),
            "code_quality": self.analyze_code_quality(),
            "dependency_graph": self.generate_dependency_graph()
        }

    def calculate_cognitive_complexity(self) -> int:
        """
        Calculates cognitive complexity based on:
        - Nesting levels
        - Logical operations
        - Control flow breaks
        - Recursion
        """
        class CognitiveComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                self.nesting_level = 0

            def visit_If(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1

            def visit_While(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1

            def visit_For(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1

            def visit_Try(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1

            def visit_BoolOp(self, node):
                self.complexity += len(node.values) - 1
                self.generic_visit(node)

            def visit_Break(self, node):
                self.complexity += 1

            def visit_Continue(self, node):
                self.complexity += 1

        visitor = CognitiveComplexityVisitor()
        visitor.visit(self.ast_tree)
        return visitor.complexity

    def extract_dependencies(self) -> Dict[str, DependencyInfo]:
        """
        Extracts both internal and external dependencies with detailed information.
        """
        class DependencyVisitor(ast.NodeVisitor):
            def __init__(self):
                self.dependencies = defaultdict(lambda: DependencyInfo(
                    name="",
                    type="",
                    version=None,
                    references=[],
                    locations=[],
                    is_internal=False
                ))
                self.current_function = None

            def visit_Import(self, node):
                for name in node.names:
                    self.add_dependency(name.name, "import", node.lineno)

            def visit_ImportFrom(self, node):
                module = node.module or ""
                for name in node.names:
                    full_name = f"{module}.{name.name}" if module else name.name
                    self.add_dependency(full_name, "import_from", node.lineno)

            def add_dependency(self, name: str, dep_type: str, lineno: int):
                dep = self.dependencies[name]
                dep.name = name
                dep.type = dep_type
                dep.is_internal = self.is_internal_dependency(name)
                dep.locations.append({
                    "line": lineno,
                    "type": dep_type
                })

            @staticmethod
            def is_internal_dependency(name: str) -> bool:
                internal_patterns = [
                    r'^\..*',  # Relative imports
                    r'^(?!django|flask|requests|numpy|pandas).*'  # Non-standard library
                ]
                return any(re.match(pattern, name) for pattern in internal_patterns)

        visitor = DependencyVisitor()
        visitor.visit(self.ast_tree)
        return visitor.dependencies

    def calculate_coverage_metrics(self) -> Dict:
        """
        Calculates various coverage metrics including:
        - Line coverage
        - Branch coverage
        - Function coverage
        """
        return {
            "line_coverage": self._calculate_line_coverage(),
            "branch_coverage": self._calculate_branch_coverage(),
            "function_coverage": self._calculate_function_coverage()
        }

    def _calculate_line_coverage(self) -> float:
        """Calculates line coverage percentage."""
        total_lines = len(self.code.splitlines())
        covered_lines = total_lines - len(self._get_unreachable_lines())
        return (covered_lines / total_lines) * 100 if total_lines > 0 else 0

    def _calculate_branch_coverage(self) -> float:
        """Calculates branch coverage percentage."""
        branches = self._get_branches()
        covered_branches = self._get_covered_branches(branches)
        return (covered_branches / len(branches)) * 100 if branches else 0

    def _calculate_function_coverage(self) -> float:
        """Calculates function coverage percentage."""
        functions = self._get_functions()
        covered_functions = self._get_covered_functions(functions)
        return (covered_functions / len(functions)) * 100 if functions else 0

    def analyze_code_quality(self) -> Dict:
        """
        Analyzes code quality metrics including:
        - Method length
        - Argument count
        - Cognitive complexity
        - Docstring coverage
        - Import complexity
        """
        return {
            "method_length_score": self._analyze_method_lengths(),
            "argument_count_score": self._analyze_argument_counts(),
            "cognitive_complexity_score": self._analyze_cognitive_complexity(),
            "docstring_coverage_score": self._analyze_docstring_coverage(),
            "import_complexity_score": self._analyze_import_complexity()
        }

    def generate_dependency_graph(self) -> nx.DiGraph:
        """
        Generates a dependency graph showing relationships between modules.
        """
        for dep_name, dep_info in self.dependencies.items():
            self.dependency_graph.add_node(
                dep_name,
                type=dep_info.type,
                is_internal=dep_info.is_internal
            )
            
            # Add edges for related dependencies
            for ref in dep_info.references:
                if ref in self.dependencies:
                    self.dependency_graph.add_edge(dep_name, ref)

        return self.dependency_graph

    def _get_unreachable_lines(self) -> Set[int]:
        """Identifies unreachable lines of code."""
        unreachable = set()
        
        class UnreachableCodeVisitor(ast.NodeVisitor):
            def visit_Return(self, node):
                # Check for code after return statements
                parent = node
                while hasattr(parent, 'parent'):
                    parent = parent.parent
                    if isinstance(parent, ast.FunctionDef):
                        break
                
                if isinstance(parent, ast.FunctionDef):
                    for child in ast.iter_child_nodes(parent):
                        if hasattr(child, 'lineno') and child.lineno > node.lineno:
                            unreachable.add(child.lineno)

        visitor = UnreachableCodeVisitor()
        visitor.visit(self.ast_tree)
        return unreachable

    def _analyze_method_lengths(self) -> float:
        """Analyzes method lengths and returns a score."""
        lengths = []
        
        class MethodLengthVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                lengths.append(node.end_lineno - node.lineno)

        visitor = MethodLengthVisitor()
        visitor.visit(self.ast_tree)
        
        if not lengths:
            return 100.0
            
        avg_length = sum(lengths) / len(lengths)
        return max(0, 100 - (avg_length - 15) * 2)  # Penalize methods longer than 15 lines

    def _analyze_argument_counts(self) -> float:
        """Analyzes function argument counts and returns a score."""
        arg_counts = []
        
        class ArgumentCountVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                arg_counts.append(len(node.args.args))

        visitor = ArgumentCountVisitor()
        visitor.visit(self.ast_tree)
        
        if not arg_counts:
            return 100.0
            
        avg_args = sum(arg_counts) / len(arg_counts)
        return max(0, 100 - (avg_args - 4) * 10)  # Penalize functions with more than 4 arguments

    def _get_branches(self) -> List[ast.AST]:
        """Gets all branch nodes in the code."""
        branches = []
        
        class BranchVisitor(ast.NodeVisitor):
            def visit_If(self, node):
                branches.append(node)
                self.generic_visit(node)
                
            def visit_While(self, node):
                branches.append(node)
                self.generic_visit(node)
                
            def visit_For(self, node):
                branches.append(node)
                self.generic_visit(node)

        visitor = BranchVisitor()
        visitor.visit(self.ast_tree)
        return branches

    def _get_covered_branches(self, branches: List[ast.AST]) -> int:
        """Estimates number of covered branches based on complexity."""
        # This is a placeholder - in a real implementation, you'd use actual coverage data
        return len(branches) // 2

    def _get_functions(self) -> List[ast.FunctionDef]:
        """Gets all function definitions in the code."""
        functions = []
        
        class FunctionVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                functions.append(node)
                self.generic_visit(node)

        visitor = FunctionVisitor()
        visitor.visit(self.ast_tree)
        return functions

    def _get_covered_functions(self, functions: List[ast.FunctionDef]) -> int:
        """Estimates number of covered functions based on complexity."""
        # This is a placeholder - in a real implementation, you'd use actual coverage data
        return len(functions) - len([f for f in functions if self._is_complex_function(f)])

    def _is_complex_function(self, function: ast.FunctionDef) -> bool:
        """Determines if a function is complex based on various metrics."""
        return (
            len(function.body) > 20 or  # Long function
            len(function.args.args) > 5 or  # Many arguments
            len(list(ast.walk(function))) > 50  # Many AST nodes
        )
