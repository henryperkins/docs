"""Smoke tests that verify the full import chain works without errors."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImportChain:
    """Verify that all modules import cleanly after the pipeline changes."""

    def test_import_metrics(self):
        from metrics import (
            MetricsManager, MetricsAnalyzer, MetricsThresholds, MetricsResult,
            calculate_code_metrics, DEFAULT_EMPTY_METRICS, validate_metrics,
            calculate_quality_score, normalize_score, get_default_halstead_metrics,
        )

    def test_import_handlers(self):
        from language_functions import get_handler
        from language_functions.base_handler import BaseHandler
        from language_functions.python_handler import PythonHandler
        from language_functions.js_ts_handler import JSTsHandler
        from language_functions.go_handler import GoHandler
        from language_functions.cpp_handler import CppHandler
        from language_functions.java_handler import JavaHandler
        from language_functions.html_handler import HTMLHandler
        from language_functions.css_handler import CSSHandler

    def test_import_report(self):
        from write_documentation_report import (
            generate_markdown_report, write_documentation_report,
            BadgeGenerator, MarkdownFormatter,
        )

    def test_import_process_manager(self):
        from process_manager import DocumentationProcessManager, DocumentationRequest

    def test_handler_factory(self):
        from language_functions import get_handler

        schema = {"functions": [{"name": "generate_documentation", "parameters": {"type": "object", "properties": {}}}]}

        assert get_handler("python", schema) is not None
        assert get_handler("javascript", schema) is not None
        assert get_handler("typescript", schema) is not None
        assert get_handler("go", schema) is not None
        assert get_handler("cpp", schema) is not None
        assert get_handler("java", schema) is not None
        assert get_handler("html", schema) is not None
        assert get_handler("css", schema) is not None
        assert get_handler("brainfuck", schema) is None

    def test_handler_factory_with_metrics_analyzer(self):
        from language_functions import get_handler
        from metrics import MetricsAnalyzer

        schema = {"functions": [{"name": "generate_documentation", "parameters": {"type": "object", "properties": {}}}]}
        analyzer = MetricsAnalyzer()
        handler = get_handler("python", schema, analyzer)
        assert handler is not None
        assert handler.metrics_analyzer is analyzer

    def test_markdown_report_generation(self):
        from write_documentation_report import generate_markdown_report

        response = {
            "summary": "Test file.",
            "functions": [{"name": "main", "docstring": "Entry point.", "args": [], "complexity": 2}],
            "classes": [{"name": "App", "docstring": "Main app.", "methods": []}],
            "variables": [{"name": "x", "type": "int", "description": "A number."}],
            "constants": [{"name": "PI", "type": "float", "description": "Pi constant."}],
        }
        metrics = {
            "complexity": 5,
            "halstead": {"volume": 200, "difficulty": 8, "effort": 1600},
            "maintainability_index": 75.0,
        }
        md = generate_markdown_report(response, "app.py", metrics)
        assert "# app.py" in md
        assert "## Summary" in md
        assert "## Functions" in md
        assert "## Classes" in md
        assert "## Variables" in md
        assert "## Constants" in md
        assert "## Metrics" in md
        assert "img.shields.io" in md
