import os
import ast
import networkx as nx

def find_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
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
    graph = nx.DiGraph()
    py_files = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                abs_path = os.path.abspath(os.path.join(root, file))
                module_name = os.path.relpath(abs_path, directory).replace(os.sep, '.').rsplit('.py', 1)[0]
                py_files[module_name] = abs_path

    for module_name, file_path in py_files.items():
        imports = find_imports(file_path)
        for imp in imports:
            if imp in py_files:
                graph.add_edge(module_name, imp)
    return graph

def find_cycles(directory):
    graph = build_dependency_graph(directory)
    cycles = list(nx.simple_cycles(graph))
    return cycles