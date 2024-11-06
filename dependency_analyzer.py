import ast
from typing import Set
import logging

logger = logging.getLogger(__name__)

class DependencyAnalyzer(ast.NodeVisitor):
    """Analyzes dependencies within a Python file using AST."""

    def __init__(self):
        self.dependencies: Set[str] = set()
        self.imported_names: Set[str] = set()

    def analyze(self, code: str) -> Set[str]:
        """
        Analyzes the given code and returns a set of dependencies.

        Args:
            code (str): The source code to analyze.

        Returns:
            Set[str]: A set of dependencies found in the code.
        """
        self.dependencies.clear()
        self.imported_names.clear()
        try:
            tree = ast.parse(code)
            self.visit(tree)
        except SyntaxError as e:
            logger.error(f"Syntax error during dependency analysis: {e}")
        return self.dependencies

    def visit_Import(self, node: ast.Import):
        """
        Visits import statements and records imported module names.

        Args:
            node (ast.Import): The import node to visit.
        """
        for alias in node.names:
            self.imported_names.add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """
        Visits import-from statements and records imported module and object names.

        Args:
            node (ast.ImportFrom): The import-from node to visit.
        """
        module = node.module or ''
        for alias in node.names:
            full_name = f"{module}.{alias.name}"
            self.imported_names.add(full_name)

    def visit_Name(self, node: ast.Name):
        """
        Visits name nodes and adds them to dependencies if they are used and imported.

        Args:
            node (ast.Name): The name node to visit.
        """
        if isinstance(node.ctx, ast.Load) and node.id in self.imported_names:
            self.dependencies.add(node.id)

    def visit_Call(self, node: ast.Call):
        """
        Visits call nodes and adds function names to dependencies if they are imported.

        Args:
            node (ast.Call): The call node to visit.
        """
        if isinstance(node.func, ast.Name) and node.func.id in self.imported_names:
            self.dependencies.add(node.func.id)
        self.generic_visit(node)
