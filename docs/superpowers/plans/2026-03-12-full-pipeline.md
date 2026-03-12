# Full Documentation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up the existing handler/schema/inserter architecture so the pipeline extracts code structure via AST handlers, gets structured LLM responses via function calling, inserts docstrings, validates code, and generates Markdown reports with badges.

**Architecture:** Add missing MetricsAnalyzer/metrics functions to `metrics.py`. Fix the handler class hierarchy (rename `BaseLanguageHandler` → `BaseHandler`, standardize constructors). Fix the broken `DocstringTransformer` AST manipulation. Replace the broken Jinja2 `DocumentationGenerator` with direct Markdown generation. Rewrite `_process_single_file` in `process_manager.py` to use handlers + function-calling structured output.

**Tech Stack:** Python 3, radon (code metrics), ast (Python parsing), aiohttp (API calls), jsonschema (validation), pytest (testing)

---

## Chunk 1: Foundation — Test Setup, MetricsAnalyzer, Base Handler

### Task 1: Set Up Test Infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Install pytest**

```bash
pip install pytest pytest-asyncio
```

- [ ] **Step 2: Create test directory and init file**

Create `tests/__init__.py` (empty file).

Create `tests/conftest.py`:

```python
import sys
from pathlib import Path

# Add project root to path so tests can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 3: Verify pytest runs**

Run: `cd /home/azureuser/docs && python3 -m pytest tests/ -v --co`
Expected: "no tests ran" (collected 0 items)

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "chore: set up pytest test infrastructure"
```

---

### Task 2: Build MetricsAnalyzer and Helpers in `metrics.py`

**Files:**
- Modify: `metrics.py` (add new classes/functions after existing code, around line 225)
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write tests for metrics helpers**

Create `tests/test_metrics.py`:

```python
import pytest
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


import asyncio
from metrics import calculate_code_metrics


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/azureuser/docs && python3 -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ImportError: cannot import name 'MetricsThresholds' from 'metrics'`

- [ ] **Step 3: Implement metrics additions**

Add the following to the **end** of `metrics.py` (after the existing `MetricsManager` class, line 225):

```python
# --- Code metrics analysis (used by language handlers) ---

@dataclass
class MetricsThresholds:
    """Thresholds for code quality metrics."""
    complexity_high: int = 15
    complexity_warning: int = 10
    maintainability_low: float = 20.0
    halstead_effort_high: float = 1_000_000.0


@dataclass
class MetricsResult:
    """Result from code metrics calculation."""
    success: bool
    metrics: Optional[Dict[str, Any]]
    error: Optional[str]


def _make_empty_metrics() -> Dict[str, Any]:
    """Returns a fresh copy of the empty metrics template."""
    return {
        "complexity": 0,
        "halstead": {"volume": 0, "difficulty": 0, "effort": 0},
        "maintainability_index": 100.0,
    }

DEFAULT_EMPTY_METRICS: Dict[str, Any] = _make_empty_metrics()


def get_default_halstead_metrics() -> Dict[str, Any]:
    """Returns default zero-valued Halstead metrics."""
    return {"volume": 0, "difficulty": 0, "effort": 0}


def normalize_score(value: float, min_val: float, max_val: float) -> float:
    """Clamps and scales a value to the 0-100 range."""
    if max_val <= min_val:
        return 100.0
    clamped = max(min_val, min(value, max_val))
    return ((clamped - min_val) / (max_val - min_val)) * 100.0


def validate_metrics(metrics: Dict[str, Any]) -> bool:
    """Checks that required keys exist and values are numeric."""
    required_keys = {"complexity", "halstead", "maintainability_index"}
    if not required_keys.issubset(metrics.keys()):
        return False
    if not isinstance(metrics.get("halstead"), dict):
        return False
    for key in ("volume", "difficulty", "effort"):
        if key not in metrics["halstead"]:
            return False
        if not isinstance(metrics["halstead"][key], (int, float)):
            return False
    if not isinstance(metrics.get("complexity"), (int, float)):
        return False
    if not isinstance(metrics.get("maintainability_index"), (int, float)):
        return False
    return True


def calculate_quality_score(metrics: Dict[str, Any]) -> float:
    """Weighted composite quality score from 0-100."""
    # Weights: complexity 30%, maintainability 40%, halstead effort 30%
    complexity = metrics.get("complexity", 0)
    mi = metrics.get("maintainability_index", 0)
    effort = metrics.get("halstead", {}).get("effort", 0)

    # Lower complexity is better (0-50 range typical)
    complexity_score = normalize_score(50 - complexity, 0, 50)
    # Higher MI is better (0-100 range)
    mi_score = normalize_score(mi, 0, 100)
    # Lower effort is better (0-2_000_000 range typical)
    effort_score = normalize_score(2_000_000 - effort, 0, 2_000_000)

    return 0.30 * complexity_score + 0.40 * mi_score + 0.30 * effort_score


async def calculate_code_metrics(
    code: str, file_path: str, language: str = "python"
) -> 'MetricsResult':
    """Calculates code metrics using radon (Python only)."""
    if language != "python":
        return MetricsResult(success=True, metrics=_make_empty_metrics(), error=None)
    try:
        from radon.complexity import cc_visit
        from radon.metrics import h_visit, mi_visit

        # Cyclomatic complexity: average across all blocks
        blocks = cc_visit(code)
        total_cc = sum(b.complexity for b in blocks) if blocks else 0

        # Halstead metrics
        h = h_visit(code)
        if hasattr(h, "total") and h.total:
            halstead = {
                "volume": h.total.volume or 0,
                "difficulty": h.total.difficulty or 0,
                "effort": h.total.effort or 0,
            }
        else:
            halstead = get_default_halstead_metrics()

        # Maintainability index
        mi = mi_visit(code, True)

        metrics = {
            "complexity": total_cc,
            "halstead": halstead,
            "maintainability_index": mi,
        }
        return MetricsResult(success=True, metrics=metrics, error=None)

    except Exception as e:
        logger.error(f"Error calculating metrics for {file_path}: {e}")
        return MetricsResult(success=False, metrics=_make_empty_metrics(), error=str(e))


class MetricsAnalyzer:
    """Accumulates per-file metrics results and produces a summary."""

    def __init__(self, thresholds: Optional[MetricsThresholds] = None):
        self.thresholds = thresholds or MetricsThresholds()
        self._results: List['MetricsResult'] = []

    def add_result(self, result: 'MetricsResult') -> None:
        """Appends a metrics result to the internal list."""
        self._results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Returns aggregate stats across all analyzed files."""
        successful = [r for r in self._results if r.success]
        failed = [r for r in self._results if not r.success]
        return {
            "total": len(self._results),
            "successful": len(successful),
            "failed": len(failed),
        }
```

Note: the existing `ProcessingResult` dataclass at line 32-38 is different from `MetricsResult` — `ProcessingResult` is for chunk processing, `MetricsResult` is for code metrics. Both stay.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/azureuser/docs && python3 -m pytest tests/test_metrics.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add metrics.py tests/test_metrics.py
git commit -m "feat: add MetricsAnalyzer, MetricsThresholds, and code metrics helpers to metrics.py"
```

---

### Task 3: Fix `base_handler.py` — Rename Class, Fix ABC Import

**Files:**
- Modify: `language_functions/base_handler.py`

There are two bugs:
1. Line 22: `class BaseLanguageHandler(ABC)` but only `import abc` exists (no `from abc import ABC`). Must use `abc.ABC` or fix import.
2. Handlers import `BaseHandler` (e.g., `python_handler.py:11`) but the class is named `BaseLanguageHandler`.
3. Constructor has a hard type dependency on `MetricsAnalyzer` which doesn't need to be enforced at this level.

- [ ] **Step 1: Fix base_handler.py**

Replace the entire file content of `language_functions/base_handler.py`:

```python
"""
base_handler.py

Abstract base class for language-specific handlers.
"""

from __future__ import annotations

import abc
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseHandler(abc.ABC):
    """
    Abstract base class for language-specific code handlers.

    Provides a common interface for extracting structure, inserting docstrings,
    and validating code.
    """

    def __init__(self, function_schema: Dict[str, Any], metrics_analyzer=None):
        self.function_schema = function_schema
        self.metrics_analyzer = metrics_analyzer

    @abc.abstractmethod
    async def extract_structure(self, code: str, file_path: str) -> Dict[str, Any]:
        raise NotImplementedError

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        logger.info("Inserting docstrings (base — no-op)...")
        return code

    @abc.abstractmethod
    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        raise NotImplementedError

    def _calculate_complexity(self, code: str) -> Optional[float]:
        return None
```

Key changes:
- `from abc import ABC` style removed — use `abc.ABC` with `import abc`
- Class renamed `BaseLanguageHandler` → `BaseHandler` (matches all handler imports)
- Constructor accepts `metrics_analyzer=None` (no hard type import)
- Removed the `MetricsAnalyzer` import at module level (was causing circular import issues)
- Stripped docstrings from abstract methods to keep it minimal

- [ ] **Step 2: Verify import works**

Run: `cd /home/azureuser/docs && python3 -c "from language_functions.base_handler import BaseHandler; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add language_functions/base_handler.py
git commit -m "fix: rename BaseLanguageHandler to BaseHandler, fix ABC import"
```

---

### Task 4: Fix All Non-Python Handler Constructors

**Files:**
- Modify: `language_functions/js_ts_handler.py:33`
- Modify: `language_functions/go_handler.py:25`
- Modify: `language_functions/cpp_handler.py:25`
- Modify: `language_functions/java_handler.py:25`
- Modify: `language_functions/html_handler.py:25`
- Modify: `language_functions/css_handler.py:25`

Every non-Python handler has `__init__(self, function_schema)` but needs `__init__(self, function_schema, metrics_analyzer=None)` to match `BaseHandler` and the factory in `__init__.py`.

- [ ] **Step 1: Fix js_ts_handler.py constructor**

In `language_functions/js_ts_handler.py`, change line 33:

From:
```python
    def __init__(self, function_schema: Dict[str, Any]):
        self.function_schema = function_schema
        self.script_dir = os.path.join(
            os.path.dirname(__file__), "..", "scripts")
```

To:
```python
    def __init__(self, function_schema: Dict[str, Any], metrics_analyzer=None):
        self.function_schema = function_schema
        self.metrics_analyzer = metrics_analyzer
        self.script_dir = os.path.join(
            os.path.dirname(__file__), "..", "scripts")
```

- [ ] **Step 2: Fix go_handler.py constructor**

In `language_functions/go_handler.py`, change line 25:

From:
```python
    def __init__(self, function_schema: Dict[str, Any]):
        self.function_schema = function_schema
```

To:
```python
    def __init__(self, function_schema: Dict[str, Any], metrics_analyzer=None):
        self.function_schema = function_schema
        self.metrics_analyzer = metrics_analyzer
```

- [ ] **Step 3: Fix cpp_handler.py constructor**

In `language_functions/cpp_handler.py`, change line 25:

From:
```python
    def __init__(self, function_schema: Dict[str, Any]):
        self.function_schema = function_schema
```

To:
```python
    def __init__(self, function_schema: Dict[str, Any], metrics_analyzer=None):
        self.function_schema = function_schema
        self.metrics_analyzer = metrics_analyzer
```

- [ ] **Step 4: Fix java_handler.py constructor**

In `language_functions/java_handler.py`, change line 25:

From:
```python
    def __init__(self, function_schema: Dict[str, Any]):
        self.function_schema = function_schema
```

To:
```python
    def __init__(self, function_schema: Dict[str, Any], metrics_analyzer=None):
        self.function_schema = function_schema
        self.metrics_analyzer = metrics_analyzer
```

- [ ] **Step 5: Fix html_handler.py constructor**

In `language_functions/html_handler.py`, change line 25:

From:
```python
    def __init__(self, function_schema: Dict[str, Any]):
        self.function_schema = function_schema
```

To:
```python
    def __init__(self, function_schema: Dict[str, Any], metrics_analyzer=None):
        self.function_schema = function_schema
        self.metrics_analyzer = metrics_analyzer
```

- [ ] **Step 6: Fix css_handler.py constructor**

In `language_functions/css_handler.py`, change line 25:

From:
```python
    def __init__(self, function_schema: Dict[str, Any]):
        self.function_schema = function_schema
```

To:
```python
    def __init__(self, function_schema: Dict[str, Any], metrics_analyzer=None):
        self.function_schema = function_schema
        self.metrics_analyzer = metrics_analyzer
```

- [ ] **Step 7: Verify all handlers import cleanly**

Run: `cd /home/azureuser/docs && python3 -c "from language_functions.base_handler import BaseHandler; from language_functions.go_handler import GoHandler; from language_functions.cpp_handler import CppHandler; from language_functions.java_handler import JavaHandler; from language_functions.html_handler import HTMLHandler; from language_functions.css_handler import CSSHandler; print('All handlers OK')"`
Expected: `All handlers OK`

- [ ] **Step 8: Commit**

```bash
git add language_functions/js_ts_handler.py language_functions/go_handler.py language_functions/cpp_handler.py language_functions/java_handler.py language_functions/html_handler.py language_functions/css_handler.py
git commit -m "fix: add metrics_analyzer parameter to all handler constructors"
```

---

## Chunk 2: Python Handler, Factory, Markdown Report

### Task 5: Fix `python_handler.py` — Imports, DocstringTransformer, Validation

**Files:**
- Modify: `language_functions/python_handler.py`
- Create: `tests/test_python_handler.py`

Three bugs to fix:
1. Lines 12-21: Imports `calculate_code_metrics`, `DEFAULT_EMPTY_METRICS`, etc. from `metrics` — these now exist after Task 2
2. Lines 275-313: `DocstringTransformer` sets `node.docstring = docstring` which is not valid AST. Must insert `ast.Expr(ast.Constant(value=docstring))` as first body statement
3. Line 253: `os.unlink(temp_file_path)` but `os` is not imported

- [ ] **Step 1: Write test for DocstringTransformer fix**

Create `tests/test_python_handler.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/azureuser/docs && python3 -m pytest tests/test_python_handler.py -v`
Expected: FAIL — `DocstringTransformer` inserts fail because `node.docstring = ...` is not valid AST

- [ ] **Step 3: Add `os` import to python_handler.py**

In `language_functions/python_handler.py`, add `import os` after line 4 (`import logging`):

```python
import ast
import logging
import os
import subprocess
```

- [ ] **Step 4: Fix DocstringTransformer to use proper AST insertion**

Replace the `_insert_docstring` helper approach. In `language_functions/python_handler.py`, add a helper method and fix all visit methods in the `DocstringTransformer` class.

Replace lines 267-338 (the entire `DocstringTransformer` class) with:

```python
class DocstringTransformer(ast.NodeTransformer):
    """Transformer for inserting docstrings into AST nodes."""

    def __init__(self, documentation: Dict[str, Any], docstring_format: str, preserve_existing=False):
        self.documentation = documentation
        self.docstring_format = docstring_format
        self.preserve_existing = preserve_existing

    def _insert_docstring(self, node, docstring):
        """Insert a docstring as the first statement of a node body."""
        if not docstring or not hasattr(node, "body"):
            return node
        docstring_node = ast.Expr(value=ast.Constant(value=docstring))
        # Remove existing docstring if present
        if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
            node.body[0] = docstring_node
        else:
            node.body.insert(0, docstring_node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Adds or updates docstring to function definitions."""
        for func in self.documentation.get("functions", []):
            if func["name"] == node.name:
                if not self.preserve_existing or not ast.get_docstring(node):
                    docstring = self._format_docstring(
                        func.get("docstring", ""), self.docstring_format,
                        func.get("args", []), func.get("returns"))
                    self._insert_docstring(node, docstring)
                break
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Adds or updates docstring to async function definitions."""
        for func in self.documentation.get("functions", []):
            if func["name"] == node.name:
                if not self.preserve_existing or not ast.get_docstring(node):
                    docstring = self._format_docstring(
                        func.get("docstring", ""), self.docstring_format,
                        func.get("args", []), func.get("returns"))
                    self._insert_docstring(node, docstring)
                break
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Adds or updates docstring to class definitions."""
        for cls in self.documentation.get("classes", []):
            if cls["name"] == node.name:
                if not self.preserve_existing or not ast.get_docstring(node):
                    docstring = self._format_docstring(
                        cls.get("docstring", ""), self.docstring_format)
                    self._insert_docstring(node, docstring)
                break
        self.generic_visit(node)
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Adds or updates docstring to the module."""
        summary = self.documentation.get("summary", "")
        if summary and (not self.preserve_existing or not ast.get_docstring(node)):
            docstring = self._format_docstring(summary, self.docstring_format)
            self._insert_docstring(node, docstring)
        self.generic_visit(node)
        return node

    def _format_docstring(self, docstring: str, format: str = "Google",
                          args: List[Dict] = None, returns: Optional[str] = None) -> str:
        """Formats the docstring according to the specified format."""
        if not docstring:
            return ""
        if format == "Google":
            formatted_docstring = docstring.strip() + "\n\n"
            if args:
                formatted_docstring += "Args:\n"
                for arg in args:
                    arg_name = arg if isinstance(arg, str) else arg.get("name", "")
                    arg_type = "" if isinstance(arg, str) else arg.get("type", "Any")
                    arg_desc = "" if isinstance(arg, str) else arg.get("description", "")
                    formatted_docstring += f"    {arg_name} ({arg_type}): {arg_desc}\n"
            if returns:
                formatted_docstring += f"\nReturns:\n    {returns}\n"
            return formatted_docstring
        elif format == "NumPy":
            formatted_docstring = docstring.strip() + "\n\n"
            if args:
                formatted_docstring += "Parameters\n----------\n"
                for arg in args:
                    arg_name = arg if isinstance(arg, str) else arg.get("name", "")
                    arg_type = "" if isinstance(arg, str) else arg.get("type", "Any")
                    arg_desc = "" if isinstance(arg, str) else arg.get("description", "")
                    formatted_docstring += f"{arg_name} : {arg_type}\n    {arg_desc}\n"
            if returns:
                formatted_docstring += "\nReturns\n-------\n"
                formatted_docstring += f"{returns}\n"
            return formatted_docstring
        return docstring
```

- [ ] **Step 5: Fix validate_code to use py_compile as fallback**

Replace `PythonHandler.validate_code` (lines 232-264) with:

```python
    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """Validates Python code. Uses py_compile (always available), falls back gracefully."""
        logger.info("Validating code...")
        try:
            compile(code, file_path or "<string>", "exec")
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validating code: {e}", exc_info=True)
            return False
```

This replaces the pylint-based validation with Python's built-in `compile()` which is always available and catches syntax errors.

- [ ] **Step 6: Fix PythonHandler constructor to handle metrics_analyzer=None**

Change line 26:

From:
```python
    def __init__(self, function_schema: Dict[str, Any], metrics_analyzer: MetricsAnalyzer):
```

To:
```python
    def __init__(self, function_schema: Dict[str, Any], metrics_analyzer=None):
```

And in `extract_structure` around line 58, guard the `add_result` call:

From:
```python
                self.metrics_analyzer.add_result(metrics_result)
```

To:
```python
                if self.metrics_analyzer:
                    self.metrics_analyzer.add_result(metrics_result)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /home/azureuser/docs && python3 -m pytest tests/test_python_handler.py -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add language_functions/python_handler.py tests/test_python_handler.py
git commit -m "fix: DocstringTransformer AST insertion, add os import, py_compile validation fallback"
```

---

### Task 6: Fix Handler Factory (`__init__.py` and `language_functions.py`)

**Files:**
- Modify: `language_functions/__init__.py`
- Modify: `language_functions/language_functions.py`

Two problems:
1. `__init__.py` line 46: `from metrics import MetricsAnalyzer` — this now works after Task 2, but the import should be lazy/optional to avoid import-order issues
2. `language_functions.py` has a duplicate `get_handler()` that only takes 2 args and doesn't pass `metrics_analyzer`

- [ ] **Step 1: Fix `language_functions/__init__.py`**

Replace the entire file:

```python
"""
language_functions Package

Provides language-specific handlers for extracting code structures,
inserting documentation, and validating code.
"""

import logging
from typing import Dict, Any, Optional

from .python_handler import PythonHandler
from .java_handler import JavaHandler
from .js_ts_handler import JSTsHandler
from .go_handler import GoHandler
from .cpp_handler import CppHandler
from .html_handler import HTMLHandler
from .css_handler import CSSHandler
from .base_handler import BaseHandler

logger = logging.getLogger(__name__)

__all__ = ["get_handler", "BaseHandler"]

_HANDLER_MAP = {
    "python": PythonHandler,
    "java": JavaHandler,
    "javascript": JSTsHandler,
    "js": JSTsHandler,
    "typescript": JSTsHandler,
    "ts": JSTsHandler,
    "go": GoHandler,
    "cpp": CppHandler,
    "c++": CppHandler,
    "cxx": CppHandler,
    "html": HTMLHandler,
    "htm": HTMLHandler,
    "css": CSSHandler,
}


def get_handler(
    language: str,
    function_schema: Dict[str, Any],
    metrics_analyzer=None,
) -> Optional[BaseHandler]:
    """Factory function to retrieve the appropriate language handler."""
    if function_schema is None:
        logger.error("Function schema is None. Cannot retrieve handler.")
        return None

    handler_class = _HANDLER_MAP.get(language.lower())
    if handler_class:
        return handler_class(function_schema, metrics_analyzer)

    logger.debug(f"No handler available for language: {language}")
    return None
```

Key changes:
- Removed `from metrics import MetricsAnalyzer` (no need for type import)
- Removed `from .language_functions import insert_docstrings` (will fix that module next)
- Used a dict map instead of if/elif chain
- `metrics_analyzer` defaults to `None`

- [ ] **Step 2: Fix `language_functions/language_functions.py`**

Replace the entire file — remove the duplicate `get_handler()` and update `insert_docstrings()` to use the factory from `__init__.py`:

```python
"""
language_functions.py

Utility for inserting docstrings into source code using language handlers.
"""

import json
import logging
from typing import Dict, Any, Optional

from utils import load_function_schema

logger = logging.getLogger(__name__)


def insert_docstrings(
    original_code: str,
    documentation: Dict[str, Any],
    language: str,
    schema_path: str,
) -> str:
    """Inserts docstrings/comments into code using the appropriate language handler."""
    logger.debug(f"Processing docstrings for language: {language}")

    try:
        function_schema = load_function_schema(schema_path)
    except (ValueError, FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.error(f"Error loading function schema: {e}")
        return original_code
    except Exception as e:
        logger.error(f"Unexpected error during schema loading: {e}", exc_info=True)
        return original_code

    # Import get_handler from the package __init__
    from language_functions import get_handler

    handler = get_handler(language, function_schema)
    if not handler:
        logger.warning(f"Unsupported language '{language}'. Skipping docstring insertion.")
        return original_code

    if documentation is None:
        logger.error("Documentation is None. Skipping docstring insertion.")
        return original_code

    try:
        updated_code = handler.insert_docstrings(original_code, documentation)
        logger.debug("Docstring insertion completed successfully.")
        return updated_code
    except Exception as e:
        logger.error(f"Error inserting docstrings: {e}", exc_info=True)
        return original_code
```

Key changes:
- Removed duplicate `get_handler()` function entirely
- Removed all handler class imports (uses the factory)
- Uses `from language_functions import get_handler` (deferred import to avoid circular)

- [ ] **Step 3: Verify imports work end-to-end**

Run: `cd /home/azureuser/docs && python3 -c "from language_functions import get_handler; h = get_handler('python', {'functions': []}); print(type(h).__name__)"`
Expected: `PythonHandler`

- [ ] **Step 4: Commit**

```bash
git add language_functions/__init__.py language_functions/language_functions.py
git commit -m "fix: consolidate get_handler factory, remove duplicate in language_functions.py"
```

---

### Task 7: Fix Markdown Report Generation (`write_documentation_report.py`)

**Files:**
- Modify: `write_documentation_report.py`
- Create: `tests/test_markdown_report.py`

The `MarkdownFormatter.__init__` uses `PackageLoader('documentation', 'templates')` — the `documentation` package/templates don't exist. The `DocumentationGenerator` class depends on these templates. Replace with direct Markdown generation.

- [ ] **Step 1: Write test for markdown report generation**

Create `tests/test_markdown_report.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/azureuser/docs && python3 -m pytest tests/test_markdown_report.py -v`
Expected: FAIL — `ImportError: cannot import name 'generate_markdown_report'` (function doesn't exist yet), and `MarkdownFormatter()` will fail because of broken Jinja2 `PackageLoader`

- [ ] **Step 3: Rewrite `write_documentation_report.py`**

Replace the file. Keep: `BadgeConfig`, `BadgeGenerator`, `MarkdownFormatter` (fix its `__init__`), `write_documentation_report()`. Remove: `DocumentationGenerator` (broken Jinja2). Add: `generate_markdown_report()`.

Full replacement for `write_documentation_report.py`:

```python
import re
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import aiofiles
import aiofiles.os
from utils import sanitize_filename
from shared_functions import (
    DEFAULT_COMPLEXITY_THRESHOLDS,
    DEFAULT_HALSTEAD_THRESHOLDS,
    DEFAULT_MAINTAINABILITY_THRESHOLDS,
)

logger = logging.getLogger(__name__)

# Global write lock for thread safety
write_lock = asyncio.Lock()


class DocumentationError(Exception):
    pass


class FileWriteError(DocumentationError):
    pass


@dataclass
class BadgeConfig:
    """Configuration for badge generation."""
    metric_name: str
    value: Union[int, float]
    thresholds: Dict[str, int]
    logo: Optional[str] = None
    style: str = "flat-square"
    label_color: Optional[str] = None

    def get_color(self) -> str:
        low, medium, high = (
            self.thresholds["low"],
            self.thresholds["medium"],
            self.thresholds["high"],
        )
        if self.value <= low:
            return "success"
        elif self.value <= medium:
            return "yellow"
        else:
            return "critical"


class BadgeGenerator:
    """Badge generation for metrics."""

    _badge_template = (
        "![{label}](https://img.shields.io/badge/"
        "{encoded_label}-{value}-{color}"
        "?style={style}{logo_part}{label_color_part})"
    )

    @classmethod
    def generate_badge(cls, config: BadgeConfig) -> str:
        try:
            label = config.metric_name.replace("_", " ").title()
            encoded_label = label.replace(" ", "%20")
            color = config.get_color()
            value = f"{config.value:.2f}" if isinstance(config.value, float) else str(config.value)
            logo_part = f"&logo={config.logo}" if config.logo else ""
            label_color_part = f"&labelColor={config.label_color}" if config.label_color else ""
            return cls._badge_template.format(
                label=label, encoded_label=encoded_label, value=value,
                color=color, style=config.style, logo_part=logo_part,
                label_color_part=label_color_part,
            )
        except Exception as e:
            logger.error(f"Error generating badge: {e}")
            return ""

    @classmethod
    def generate_all_badges(cls, metrics: Dict[str, Any]) -> str:
        badges = []
        try:
            if (complexity := metrics.get("complexity")) is not None:
                badges.append(cls.generate_badge(BadgeConfig(
                    metric_name="Complexity", value=complexity,
                    thresholds=DEFAULT_COMPLEXITY_THRESHOLDS, logo="codeClimate",
                )))
            if halstead := metrics.get("halstead"):
                logo_map = {"volume": "stackOverflow", "difficulty": "codewars", "effort": "atlassian"}
                for name, key in [("Volume", "volume"), ("Difficulty", "difficulty"), ("Effort", "effort")]:
                    if (val := halstead.get(key)) is not None:
                        badges.append(cls.generate_badge(BadgeConfig(
                            metric_name=name, value=val,
                            thresholds=DEFAULT_HALSTEAD_THRESHOLDS[key],
                            logo=logo_map[key],
                        )))
            if (mi := metrics.get("maintainability_index")) is not None:
                badges.append(cls.generate_badge(BadgeConfig(
                    metric_name="Maintainability", value=mi,
                    thresholds=DEFAULT_MAINTAINABILITY_THRESHOLDS, logo="codeclimate",
                )))
            return " ".join(badges)
        except Exception as e:
            logger.error(f"Error generating badges: {e}")
            return ""


class MarkdownFormatter:
    """Markdown formatting utilities."""

    @staticmethod
    def truncate_description(description: str, max_length: int = 100, ellipsis: str = "...") -> str:
        if not description or len(description) <= max_length:
            return description
        truncated = description[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated + ellipsis

    @staticmethod
    def sanitize_text(text: str) -> str:
        special_chars = r'[`*_{}[$()#+\-.!|]'
        text = re.sub(special_chars, lambda m: '\\' + m.group(0), str(text))
        text = text.replace('\n', ' ').replace('\r', '')
        return ' '.join(text.split())

    def format_table(self, headers: List[str], rows: List[List[Any]], alignment: Optional[List[str]] = None) -> str:
        if not headers or not rows:
            return ""
        try:
            headers = [self.sanitize_text(str(h)) for h in headers]
            if not alignment:
                alignment = ['left'] * len(headers)
            align_map = {'left': ':---', 'center': ':---:', 'right': '---:'}
            separators = [align_map.get(a, ':---') for a in alignment]
            table_lines = [
                f"| {' | '.join(headers)} |",
                f"| {' | '.join(separators)} |",
            ]
            for row in rows:
                row = (row + [''] * len(headers))[:len(headers)]
                sanitized = [self.sanitize_text(str(c)) for c in row]
                table_lines.append(f"| {' | '.join(sanitized)} |")
            return '\n'.join(table_lines)
        except Exception as e:
            logger.error(f"Error formatting table: {e}")
            return ""


def generate_markdown_report(
    structured_response: Dict[str, Any],
    file_path: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate Markdown documentation from structured LLM response."""
    sections = []
    formatter = MarkdownFormatter()

    # Header with badges
    sections.append(f"# {Path(file_path).name}")
    if metrics:
        badge_str = BadgeGenerator.generate_all_badges(metrics)
        if badge_str:
            sections.append(badge_str)

    # Summary
    if summary := structured_response.get("summary"):
        sections.append(f"## Summary\n\n{summary}")

    # Functions table
    if functions := structured_response.get("functions"):
        headers = ["Function", "Args", "Description", "Complexity"]
        rows = []
        for f in functions:
            args_list = f.get("args", [])
            args_str = ", ".join(
                a if isinstance(a, str) else a.get("name", "") for a in args_list
            )
            desc = formatter.truncate_description(f.get("docstring", ""), 100)
            rows.append([f.get("name", ""), args_str, desc, str(f.get("complexity", ""))])
        sections.append(f"## Functions\n\n{formatter.format_table(headers, rows)}")

    # Classes table
    if classes := structured_response.get("classes"):
        headers = ["Class", "Description", "Methods"]
        rows = []
        for cls in classes:
            methods = cls.get("methods", [])
            method_names = ", ".join(m.get("name", "") for m in methods)
            desc = formatter.truncate_description(cls.get("docstring", ""), 100)
            rows.append([cls.get("name", ""), desc, method_names])
        sections.append(f"## Classes\n\n{formatter.format_table(headers, rows)}")

    # Variables table
    if variables := structured_response.get("variables"):
        headers = ["Variable", "Type", "Description"]
        rows = [[v.get("name", ""), v.get("type", ""), v.get("description", "")] for v in variables]
        sections.append(f"## Variables\n\n{formatter.format_table(headers, rows)}")

    # Constants table
    if constants := structured_response.get("constants"):
        headers = ["Constant", "Type", "Description"]
        rows = [[c.get("name", ""), c.get("type", ""), c.get("description", "")] for c in constants]
        sections.append(f"## Constants\n\n{formatter.format_table(headers, rows)}")

    # Metrics section
    if metrics:
        metrics_lines = ["## Metrics\n"]
        if c := metrics.get("complexity"):
            metrics_lines.append(f"- **Cyclomatic Complexity:** {c}")
        if mi := metrics.get("maintainability_index"):
            metrics_lines.append(f"- **Maintainability Index:** {mi:.1f}")
        if h := metrics.get("halstead"):
            metrics_lines.append(f"- **Halstead Volume:** {h.get('volume', 0):.1f}")
            metrics_lines.append(f"- **Halstead Difficulty:** {h.get('difficulty', 0):.1f}")
            metrics_lines.append(f"- **Halstead Effort:** {h.get('effort', 0):.1f}")
        sections.append("\n".join(metrics_lines))

    return "\n\n".join(sections)


async def write_documentation_report(
    documentation: Optional[Dict[str, Any]],
    language: str,
    file_path: str,
    repo_root: str,
    output_dir: str,
    project_id: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Writes documentation to JSON and Markdown files."""
    if not documentation:
        logger.warning(f"No documentation to write for '{file_path}'")
        return None

    try:
        async with write_lock:
            project_output_dir = Path(output_dir) / project_id
            await aiofiles.os.makedirs(project_output_dir, exist_ok=True)

            relative_path = Path(file_path).relative_to(repo_root)
            safe_parts = [sanitize_filename(part) for part in relative_path.parts]
            output_subdir = project_output_dir / Path(*safe_parts[:-1]) if len(safe_parts) > 1 else project_output_dir
            await aiofiles.os.makedirs(output_subdir, exist_ok=True)
            safe_filename = sanitize_filename(relative_path.name)
            base_path = output_subdir / safe_filename

            # Write JSON
            json_path = base_path.with_suffix(".json")
            try:
                async with aiofiles.open(json_path, "w") as f:
                    await f.write(json.dumps(documentation, indent=2, sort_keys=True))
            except Exception as e:
                logger.error(f"Error writing JSON to {json_path}: {e}")
                raise FileWriteError(f"Failed to write JSON: {e}")

            # Write Markdown
            if documentation.get("generate_markdown", True):
                try:
                    markdown_content = generate_markdown_report(
                        documentation, file_path, metrics
                    )
                    md_path = base_path.with_suffix(".md")
                    async with aiofiles.open(md_path, "w") as f:
                        await f.write(markdown_content)
                except Exception as e:
                    logger.error(f"Error writing Markdown: {e}")
                    raise FileWriteError(f"Failed to write Markdown: {e}")

            logger.info(f"Documentation written to {json_path}")
            return documentation

    except FileWriteError:
        raise
    except Exception as e:
        logger.error(f"Error writing documentation report: {e}")
        raise DocumentationError(f"Documentation write failed: {e}")
```

Key changes:
- Removed `MarkdownFormatter.__init__` (was using broken Jinja2 `PackageLoader`)
- Removed entire `DocumentationGenerator` class (used missing templates)
- Removed `TemplateError` exception (no longer needed)
- Removed `from jinja2 import ...` import
- Added `generate_markdown_report()` function with tables for functions, classes, variables, constants, and metrics
- `write_documentation_report()` now calls `generate_markdown_report()` instead of `DocumentationGenerator`
- `MarkdownFormatter` is now stateless (static methods + instance method for table)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/azureuser/docs && python3 -m pytest tests/test_markdown_report.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run all tests**

Run: `cd /home/azureuser/docs && python3 -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add write_documentation_report.py tests/test_markdown_report.py
git commit -m "feat: replace broken Jinja2 DocumentationGenerator with direct Markdown generation"
```

---

## Chunk 3: Pipeline Rewrite, Main.py, Integration

### Task 8: Update `main.py` to Pass Schema Path

**Files:**
- Modify: `main.py:126-131`

- [ ] **Step 1: Pass schema_path to DocumentationProcessManager**

In `main.py`, change lines 126-131:

From:
```python
        manager = DocumentationProcessManager(
            repo_root=repo_path,
            output_dir=output_dir,
            provider_configs=provider_configs,
            max_concurrency=args.concurrency
        )
```

To:
```python
        manager = DocumentationProcessManager(
            repo_root=repo_path,
            output_dir=output_dir,
            provider_configs=provider_configs,
            max_concurrency=args.concurrency,
            schema_path=args.schema,
        )
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "feat: pass --schema path to DocumentationProcessManager"
```

---

### Task 9: Rewrite `process_manager.py` Pipeline

**Files:**
- Modify: `process_manager.py`
- Create: `tests/test_pipeline.py`

This is the core change: rewrite `_process_single_file` to use handlers, function-calling structured output, docstring insertion, and markdown report generation.

**Note:** The old `_process_single_file` called `ChunkManager.create_chunks()`, `context_manager.add_code_chunk()`, `DependencyAnalyzer().analyze()`, and `TokenManager.count_tokens()`. These are intentionally removed — the new pipeline extracts structure via language handlers and sends code directly with the function schema. The chunking/context/dependency logic was not producing useful output (it fed `code[:3000]` to a generic prompt). The `ChunkManager` and `HierarchicalContextManager` instances remain on `self` for potential future use but are no longer called in the per-file flow.

- [ ] **Step 1: Write test for response parsing logic**

Create `tests/test_pipeline.py`:

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_structured_response(api_response):
    """Extract structured response from API response (function-calling format)."""
    if not api_response or "choices" not in api_response:
        return None
    choice = api_response["choices"][0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        try:
            return json.loads(tool_calls[0]["function"]["arguments"])
        except (json.JSONDecodeError, KeyError, IndexError):
            return None
    # Fallback: no tool_calls, return None (freeform response)
    return None


class TestParseStructuredResponse:
    def test_with_tool_calls(self):
        api_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "generate_documentation",
                            "arguments": json.dumps({
                                "summary": "A test module.",
                                "functions": [{"name": "foo", "docstring": "Does foo.", "args": [], "async": False}],
                                "classes": [],
                                "docstring_format": "Google",
                            })
                        }
                    }]
                }
            }]
        }
        result = parse_structured_response(api_response)
        assert result is not None
        assert result["summary"] == "A test module."
        assert result["functions"][0]["name"] == "foo"

    def test_without_tool_calls(self):
        api_response = {
            "choices": [{
                "message": {
                    "content": "Here is the documentation..."
                }
            }]
        }
        result = parse_structured_response(api_response)
        assert result is None

    def test_empty_response(self):
        assert parse_structured_response(None) is None
        assert parse_structured_response({}) is None

    def test_malformed_arguments(self):
        api_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "generate_documentation",
                            "arguments": "not valid json{{"
                        }
                    }]
                }
            }]
        }
        result = parse_structured_response(api_response)
        assert result is None
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd /home/azureuser/docs && python3 -m pytest tests/test_pipeline.py -v`
Expected: All PASS (the parse function is self-contained in the test file for now)

- [ ] **Step 3: Rewrite `process_manager.py`**

Modify `process_manager.py`. Changes:

**A) Add schema_path to constructor (line 126-148):**

Add `schema_path` parameter and initialize function schema + MetricsAnalyzer. Add new imports.

Replace the imports block (lines 1-27) with:

```python
"""
process_manager.py

Documentation generation process manager with integrated pipeline.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, validator
import threading
import uuid
import aiohttp

from provider_config import load_provider_configs, ProviderConfig
from tokens import TokenManager
from chunks import ChunkManager
from dependency_analyzer import DependencyAnalyzer
from context import HierarchicalContextManager
from utils import setup_logging, load_function_schema
from metrics import MetricsManager

logger = logging.getLogger(__name__)
```

Notes:
- Added `import json` (needed for structured response parsing).
- Kept `import os` (used by `_get_already_processed`, `get_manager_instance`, `__main__` block).
- Removed unused `load_json_schema` and `handle_api_error` imports (not used anywhere in this file).

Replace the `__init__` method of `DocumentationProcessManager` (lines 126-148):

```python
    def __init__(
        self,
        repo_root: str,
        output_dir: str,
        provider_configs: Dict[str, ProviderConfig],
        max_concurrency: int = 5,
        cache_dir: Optional[str] = None,
        metrics_manager: MetricsManager = None,
        schema_path: Optional[str] = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.provider_configs = provider_configs
        self.max_concurrency = max_concurrency
        self.metrics_manager = metrics_manager or MetricsManager()

        # Initialize managers
        self.chunk_manager = ChunkManager(max_tokens=4096, overlap=200)
        self.context_manager = HierarchicalContextManager(cache_dir=cache_dir)

        # Load function schema for structured output
        self.function_schema = None
        if schema_path:
            try:
                self.function_schema = load_function_schema(schema_path)
                logger.info(f"Loaded function schema from {schema_path}")
            except Exception as e:
                logger.warning(f"Could not load function schema: {e}. Falling back to generic prompts.")

        # Code metrics analyzer
        from metrics import MetricsAnalyzer
        self.metrics_analyzer = MetricsAnalyzer()

        # Task tracking
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._task_status: Dict[str, Dict[str, Any]] = {}
```

**B) Rewrite `_process_single_file` (lines 220-327):**

Replace the entire method:

```python
    async def _process_single_file(
        self,
        file_path: str,
        request: 'DocumentationRequest',
        session: aiohttp.ClientSession,
        api_handler: 'APIHandler'
    ) -> Dict[str, Any]:
        """Processes a single file: extract structure, call API with function schema, insert docstrings, write docs."""
        start_time = datetime.now()
        try:
            # Read file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except UnicodeDecodeError:
                logger.debug(f"Skipping binary file: {file_path}")
                return {"file_path": file_path, "success": True, "skipped": "binary"}

            language = self._detect_language(file_path)

            # Skip non-code files
            if self._is_non_code_file(file_path):
                logger.debug(f"Skipping non-code file: {file_path}")
                return {"file_path": file_path, "success": True, "skipped": "non-code"}

            # Try to get a language handler for structured pipeline
            handler = None
            extracted_structure = None
            if self.function_schema:
                from language_functions import get_handler
                handler = get_handler(language, self.function_schema, self.metrics_analyzer)

            if handler:
                # === STRUCTURED PIPELINE ===
                # Note: some handlers (Cpp, Java, Go) define extract_structure as
                # a regular def, not async def. Use inspect to handle both.
                import inspect
                try:
                    result = handler.extract_structure(code, file_path)
                    if inspect.isawaitable(result):
                        extracted_structure = await result
                    else:
                        extracted_structure = result
                except Exception as e:
                    logger.warning(f"Structure extraction failed for {file_path}: {e}")
                    extracted_structure = None

            # Build API request
            config = self.provider_configs[request.provider]
            api_url = (
                f"{config.endpoint}/openai/deployments/"
                f"{config.deployment_name}/chat/completions"
                f"?api-version={config.api_version}"
            )

            if handler and extracted_structure and self.function_schema:
                # Structured request with function calling
                payload = {
                    "messages": [
                        {"role": "system", "content": (
                            "You are a code documentation generator. Analyze the provided code structure "
                            "and generate comprehensive documentation. Fill in docstrings for all functions "
                            "and classes. Use Google-style docstrings for Python."
                        )},
                        {"role": "user", "content": (
                            f"File: {file_path}\nLanguage: {language}\n\n"
                            f"Extracted structure:\n{json.dumps(extracted_structure, indent=2, default=str)}\n\n"
                            f"Source code:\n{code[:8000]}"
                        )},
                    ],
                    "tools": [{
                        "type": "function",
                        "function": self.function_schema["functions"][0],
                    }],
                    "tool_choice": {"type": "function", "function": {"name": "generate_documentation"}},
                    "max_completion_tokens": 4096,
                    "temperature": config.temperature,
                }
            else:
                # Generic request (unsupported language or no schema)
                payload = {
                    "messages": [
                        {"role": "system", "content": "You are a documentation generator. Generate comprehensive docstrings and documentation for the given code."},
                        {"role": "user", "content": f"Generate documentation for this code:\n\n{code[:8000]}"},
                    ],
                    "max_completion_tokens": config.max_tokens,
                    "temperature": config.temperature,
                }

            # Call API
            api_response = await api_handler.call_provider_api(
                endpoint=api_url, payload=payload
            )

            # Parse response
            structured = None
            doc_content = None
            if api_response and "choices" in api_response:
                choice = api_response["choices"][0]
                message = choice.get("message", {})
                tool_calls = message.get("tool_calls", [])
                if tool_calls:
                    try:
                        structured = json.loads(tool_calls[0]["function"]["arguments"])
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        logger.warning(f"Failed to parse structured response for {file_path}: {e}")
                if not structured:
                    doc_content = message.get("content", "")

            # Insert docstrings (if structured response and not safe mode)
            if structured and handler and not request.safe_mode:
                try:
                    modified_code = handler.insert_docstrings(code, structured)
                    if handler.validate_code(modified_code, file_path):
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(modified_code)
                        logger.info(f"Docstrings inserted into {file_path}")
                    else:
                        logger.warning(f"Validation failed after docstring insertion for {file_path}, skipping source write")
                except Exception as e:
                    logger.warning(f"Docstring insertion failed for {file_path}: {e}")

            # Build documentation dict for output
            documentation = structured or {}
            if not structured and doc_content:
                documentation = {
                    "content": doc_content,
                    "file_path": file_path,
                    "language": language,
                }

            documentation["file_path"] = file_path
            documentation["language"] = language
            documentation["generate_markdown"] = bool(structured)

            # Calculate metrics for the documentation
            file_metrics = None
            if structured and extracted_structure:
                file_metrics = extracted_structure.get("metrics")

            # Write documentation output
            from write_documentation_report import write_documentation_report
            await write_documentation_report(
                documentation=documentation,
                language=language,
                file_path=file_path,
                repo_root=str(self.repo_root),
                output_dir=str(self.output_dir),
                project_id=request.project_id,
                metrics=file_metrics,
            )
            logger.info(f"Documentation written for {file_path}")

            processing_time = (datetime.now() - start_time).total_seconds()
            try:
                self.metrics_manager.record_file_processing(
                    success=True, processing_time=processing_time)
            except Exception as me:
                logger.warning(f"Metrics recording failed: {me}")

            return {
                "file_path": file_path,
                "success": True,
                "structured": structured is not None,
                "language": language,
            }

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            logger.error(f"Error processing file {file_path}: {error_msg}", exc_info=True)
            processing_time = (datetime.now() - start_time).total_seconds()
            try:
                self.metrics_manager.record_file_processing(
                    success=False, processing_time=processing_time, error_type=error_msg)
            except Exception as me:
                logger.warning(f"Metrics recording failed: {me}")
            return {"file_path": file_path, "success": False, "error": error_msg}
```

- [ ] **Step 4: Run all tests**

Run: `cd /home/azureuser/docs && python3 -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Verify module imports**

Run: `cd /home/azureuser/docs && python3 -c "from process_manager import DocumentationProcessManager; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add process_manager.py tests/test_pipeline.py
git commit -m "feat: rewrite _process_single_file with structured output, handler pipeline, docstring insertion"
```

---

### Task 10: Integration Smoke Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration smoke test**

Create `tests/test_integration.py`:

```python
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
```

- [ ] **Step 2: Run all tests**

Run: `cd /home/azureuser/docs && python3 -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration smoke tests for full pipeline import chain"
```

- [ ] **Step 4: Final commit with all changes**

```bash
git add -A
git status
```

Verify no untracked or unstaged files remain. If clean, the implementation is complete.

---

## File Change Summary

| File | Action | Task |
|------|--------|------|
| `tests/__init__.py` | Create | 1 |
| `tests/conftest.py` | Create | 1 |
| `tests/test_metrics.py` | Create | 2 |
| `metrics.py` | Modify (append ~100 lines) | 2 |
| `language_functions/base_handler.py` | Rewrite | 3 |
| `language_functions/js_ts_handler.py` | Modify (constructor) | 4 |
| `language_functions/go_handler.py` | Modify (constructor) | 4 |
| `language_functions/cpp_handler.py` | Modify (constructor) | 4 |
| `language_functions/java_handler.py` | Modify (constructor) | 4 |
| `language_functions/html_handler.py` | Modify (constructor) | 4 |
| `language_functions/css_handler.py` | Modify (constructor) | 4 |
| `tests/test_python_handler.py` | Create | 5 |
| `language_functions/python_handler.py` | Modify (3 fixes) | 5 |
| `language_functions/__init__.py` | Rewrite | 6 |
| `language_functions/language_functions.py` | Rewrite | 6 |
| `tests/test_markdown_report.py` | Create | 7 |
| `write_documentation_report.py` | Rewrite | 7 |
| `main.py` | Modify (1 line) | 8 |
| `process_manager.py` | Modify (constructor + method) | 9 |
| `tests/test_pipeline.py` | Create | 9 |
| `tests/test_integration.py` | Create | 10 |
