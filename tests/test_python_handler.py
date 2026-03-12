import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from language_functions.python_handler import PythonHandler, DocstringTransformer


class TestDocstringTransformer:
    def test_insert_function_docstring(self):
        code = '''
def hello(name):
    return f"Hello, {name}"
'''
        documentation = {
            "docstring_format": "Google",
            "summary": "",
            "functions": [
                {"name": "hello", "docstring": "Greets a person by name.", "args": [{"name": "name", "type": "str"}]}
            ],
            "classes": [],
        }
        transformer = DocstringTransformer(documentation, "Google", preserve_existing=False)
        tree = ast.parse(code)
        modified = transformer.visit(tree)
        result = ast.unparse(modified)
        assert "Greets a person by name" in result

    def test_insert_class_docstring(self):
        code = '''
class Foo:
    pass
'''
        documentation = {
            "docstring_format": "Google",
            "summary": "",
            "functions": [],
            "classes": [
                {"name": "Foo", "docstring": "A foo class."}
            ],
        }
        transformer = DocstringTransformer(documentation, "Google", preserve_existing=False)
        tree = ast.parse(code)
        modified = transformer.visit(tree)
        result = ast.unparse(modified)
        assert "A foo class" in result

    def test_replace_existing_docstring(self):
        code = '''
def hello():
    """Old docstring."""
    pass
'''
        documentation = {
            "docstring_format": "Google",
            "summary": "",
            "functions": [
                {"name": "hello", "docstring": "New docstring.", "args": []}
            ],
            "classes": [],
        }
        transformer = DocstringTransformer(documentation, "Google", preserve_existing=False)
        tree = ast.parse(code)
        modified = transformer.visit(tree)
        result = ast.unparse(modified)
        assert "New docstring" in result
        assert "Old docstring" not in result

    def test_preserve_existing(self):
        code = '''
def hello():
    """Old docstring."""
    pass
'''
        documentation = {
            "docstring_format": "Google",
            "summary": "",
            "functions": [
                {"name": "hello", "docstring": "New docstring.", "args": []}
            ],
            "classes": [],
        }
        transformer = DocstringTransformer(documentation, "Google", preserve_existing=True)
        tree = ast.parse(code)
        modified = transformer.visit(tree)
        result = ast.unparse(modified)
        assert "Old docstring" in result


class TestPythonHandlerValidation:
    def test_validate_valid_code(self):
        schema = {"functions": [{"name": "generate_documentation", "parameters": {}}]}
        handler = PythonHandler(schema)
        assert handler.validate_code("x = 1\n") is True

    def test_validate_invalid_syntax(self):
        schema = {"functions": [{"name": "generate_documentation", "parameters": {}}]}
        handler = PythonHandler(schema)
        assert handler.validate_code("def (broken::\n") is False


class TestPythonHandlerInsertDocstrings:
    def test_round_trip(self):
        schema = {"functions": [{"name": "generate_documentation", "parameters": {}}]}
        handler = PythonHandler(schema)
        code = "def add(a, b):\n    return a + b\n"
        docs = {
            "docstring_format": "Google",
            "summary": "",
            "functions": [{"name": "add", "docstring": "Adds two numbers.", "args": []}],
            "classes": [],
        }
        result = handler.insert_docstrings(code, docs)
        assert "Adds two numbers" in result
        # Should still be valid Python
        compile(result, "<test>", "exec")
