import os
import ast
import networkx as nx


def find_imports(file_path):
    """Analyzes the given file to find import statements and records them for further processing.

    Args:
        file_path (str): The path to the Python file to analyze.

    Returns:
        list: A list of import statements found in the file."""
    with open(file_path, "r", encoding="utf-8") as f:
        root = ast.parse(f.read(), filename=file_path)
    imports = set()
    for node in ast.walk(root):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imports.add(node.module)
    return imports


def build_dependency_graph(directory):
    """Builds a dependency graph for Python files within the specified directory by analyzing import statements.

    Args:
        directory (str): The directory containing Python files to analyze.

    Returns:
        dict: A dictionary representing the dependency graph of modules."""
    graph = nx.DiGraph()
    py_files = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                abs_path = os.path.abspath(os.path.join(root, file))
                module_name = os.path.relpath(abs_path, directory).replace(os.sep, ".").rsplit(".py", 1)[0]
                py_files[module_name] = abs_path

    for module_name, file_path in py_files.items():
        imports = find_imports(file_path)
        for imp in imports:
            if imp in py_files:
                graph.add_edge(module_name, imp)
    return graph


def find_cycles(directory):
    """Detects cycles in the dependency graph within the provided directory.

    Args:
        directory (str): The directory to analyze for circular dependencies.

    Returns:
        list: A list of identified cycles in the dependency graph."""
    graph = build_dependency_graph(directory)
    cycles = list(nx.simple_cycles(graph))
    return cycles
