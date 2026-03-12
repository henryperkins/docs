import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics import (
    MetricsThresholds,
    MetricsAnalyzer,
    MetricsResult,
    DEFAULT_EMPTY_METRICS,
    get_default_halstead_metrics,
    validate_metrics,
    calculate_quality_score,
    normalize_score,
    calculate_code_metrics,
)


class TestMetricsThresholds:
    def test_defaults(self):
        t = MetricsThresholds()
        assert t.complexity_high == 15
        assert t.complexity_warning == 10
        assert t.maintainability_low == 20.0
        assert t.halstead_effort_high == 1_000_000.0

    def test_custom(self):
        t = MetricsThresholds(complexity_high=20)
        assert t.complexity_high == 20


class TestDefaultEmptyMetrics:
    def test_shape(self):
        assert DEFAULT_EMPTY_METRICS["complexity"] == 0
        assert DEFAULT_EMPTY_METRICS["halstead"]["volume"] == 0
        assert DEFAULT_EMPTY_METRICS["halstead"]["difficulty"] == 0
        assert DEFAULT_EMPTY_METRICS["halstead"]["effort"] == 0
        assert DEFAULT_EMPTY_METRICS["maintainability_index"] == 100.0


class TestGetDefaultHalsteadMetrics:
    def test_returns_zeros(self):
        h = get_default_halstead_metrics()
        assert h == {"volume": 0, "difficulty": 0, "effort": 0}


class TestNormalizeScore:
    def test_middle(self):
        assert normalize_score(50, 0, 100) == 50.0

    def test_clamp_low(self):
        assert normalize_score(-10, 0, 100) == 0.0

    def test_clamp_high(self):
        assert normalize_score(200, 0, 100) == 100.0

    def test_narrow_range(self):
        assert normalize_score(5, 5, 5) == 100.0


class TestValidateMetrics:
    def test_valid(self):
        m = {
            "complexity": 5,
            "halstead": {"volume": 100, "difficulty": 10, "effort": 1000},
            "maintainability_index": 80.0,
        }
        assert validate_metrics(m) is True

    def test_missing_key(self):
        assert validate_metrics({"complexity": 5}) is False

    def test_non_numeric(self):
        m = {
            "complexity": "high",
            "halstead": {"volume": 0, "difficulty": 0, "effort": 0},
            "maintainability_index": 80.0,
        }
        assert validate_metrics(m) is False


class TestCalculateQualityScore:
    def test_perfect(self):
        m = {
            "complexity": 1,
            "halstead": {"volume": 0, "difficulty": 0, "effort": 0},
            "maintainability_index": 100.0,
        }
        score = calculate_quality_score(m)
        assert 80 <= score <= 100

    def test_bad(self):
        m = {
            "complexity": 50,
            "halstead": {"volume": 5000, "difficulty": 50, "effort": 2000000},
            "maintainability_index": 5.0,
        }
        score = calculate_quality_score(m)
        assert 0 <= score <= 30


class TestMetricsAnalyzer:
    def test_add_and_summary(self):
        analyzer = MetricsAnalyzer()
        r1 = MetricsResult(success=True, metrics={"complexity": 5}, error=None)
        r2 = MetricsResult(success=False, metrics=None, error="parse error")
        analyzer.add_result(r1)
        analyzer.add_result(r2)
        summary = analyzer.get_summary()
        assert summary["total"] == 2
        assert summary["successful"] == 1
        assert summary["failed"] == 1


class TestMetricsResult:
    def test_creation(self):
        r = MetricsResult(success=True, metrics={"complexity": 3}, error=None)
        assert r.success is True
        assert r.metrics["complexity"] == 3


class TestCalculateCodeMetrics:
    def test_python_simple(self):
        code = "def add(a, b):\n    return a + b\n"
        result = asyncio.get_event_loop().run_until_complete(
            calculate_code_metrics(code, "test.py", "python")
        )
        assert result.success is True
        assert "complexity" in result.metrics
        assert "halstead" in result.metrics
        assert "maintainability_index" in result.metrics

    def test_non_python_returns_defaults(self):
        result = asyncio.get_event_loop().run_until_complete(
            calculate_code_metrics("var x = 1;", "test.js", "javascript")
        )
        assert result.success is True
        assert result.metrics["complexity"] == 0

    def test_invalid_python(self):
        result = asyncio.get_event_loop().run_until_complete(
            calculate_code_metrics("def (broken::", "bad.py", "python")
        )
        assert result.success is False
        assert result.error is not None
