import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from write_documentation_report import generate_markdown_report, MarkdownFormatter


class TestGenerateMarkdownReport:
    def test_basic_report(self):
        response = {
            "summary": "A utility module.",
            "functions": [
                {"name": "add", "docstring": "Adds numbers.", "args": ["a", "b"], "complexity": 1}
            ],
            "classes": [],
        }
        result = generate_markdown_report(response, "utils.py")
        assert "# utils.py" in result
        assert "## Summary" in result
        assert "A utility module" in result
        assert "## Functions" in result
        assert "add" in result

    def test_with_metrics_badges(self):
        response = {"summary": "Test.", "functions": [], "classes": []}
        metrics = {"complexity": 5, "halstead": {"volume": 100, "difficulty": 5, "effort": 500}, "maintainability_index": 80}
        result = generate_markdown_report(response, "test.py", metrics=metrics)
        assert "# test.py" in result
        assert "img.shields.io" in result

    def test_classes_section(self):
        response = {
            "summary": "",
            "functions": [],
            "classes": [
                {"name": "Foo", "docstring": "A foo class.", "methods": [
                    {"name": "bar", "docstring": "Bar method.", "args": [], "async": False, "type": "None"}
                ]}
            ],
        }
        result = generate_markdown_report(response, "foo.py")
        assert "## Classes" in result
        assert "Foo" in result

    def test_empty_response(self):
        result = generate_markdown_report({}, "empty.py")
        assert "# empty.py" in result


class TestMarkdownFormatterTable:
    def test_format_table(self):
        fmt = MarkdownFormatter()
        result = fmt.format_table(["Name", "Value"], [["a", "1"], ["b", "2"]])
        assert "| Name | Value |" in result
        assert "| a | 1 |" in result
