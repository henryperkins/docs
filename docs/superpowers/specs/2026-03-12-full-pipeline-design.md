# Full Documentation Pipeline Design

**Goal:** Wire up the existing handler/schema/inserter architecture so the app actually extracts code structure, gets structured LLM responses via function calling, inserts docstrings into source files, validates modified code, and generates Markdown reports with metrics badges.

**Current state:** The pipeline sends `code[:3000]` with a generic prompt and saves freeform LLM text as JSON. None of the existing language handlers, function schema, inserters, or report generators are called.

**Target state:** A per-file pipeline that:
1. Extracts code structure via language-specific AST handlers
2. Sends structure + code to Azure OpenAI with function schema as structured output format
3. Parses the structured JSON response (functions with docstrings, args, metrics)
4. Inserts docstrings back into source files (when not in safe-mode)
5. Validates modified code before writing
6. Generates JSON + Markdown documentation with badges and metrics

---

## Component 1: MetricsAnalyzer (`metrics.py`)

Build the missing functions that `python_handler.py` imports.

### New classes/functions to add to `metrics.py`:

**`MetricsThresholds`** (dataclass):
- `complexity_high: int = 15`, `complexity_warning: int = 10`
- `maintainability_low: float = 20.0`
- `halstead_effort_high: float = 1000000.0`

**`MetricsAnalyzer`** (class):
- `__init__(self, thresholds=None)` — stores `MetricsThresholds`, accumulates results
- `add_result(self, result)` — appends a metrics result to internal list
- `get_summary(self)` — returns aggregate stats across all analyzed files

**`calculate_code_metrics(code, file_path, language="python")`** (async function):
- Uses radon: `cc_visit(code)` for cyclomatic complexity, `h_visit(code)` for Halstead, `mi_visit(code)` for maintainability
- Returns dataclass `MetricsResult(success: bool, metrics: dict, error: str)`
- `metrics` dict matches the function schema shape: `{complexity, halstead: {volume, difficulty, effort}, maintainability_index}`

**`DEFAULT_EMPTY_METRICS`** (dict constant):
```python
DEFAULT_EMPTY_METRICS = {
    "complexity": 0,
    "halstead": {"volume": 0, "difficulty": 0, "effort": 0},
    "maintainability_index": 100.0,
}
```

**`get_default_halstead_metrics()`**: Returns `{"volume": 0, "difficulty": 0, "effort": 0}`

**`validate_metrics(metrics)`**: Checks required keys exist and values are numeric. Returns bool.

**`calculate_quality_score(metrics)`**: Weighted composite of complexity, maintainability, halstead. Returns 0-100.

**`normalize_score(value, min_val, max_val)`**: Clamps and scales a value to 0-100 range.

All use radon (already installed). No new dependencies.

---

## Component 2: Pipeline Rewrite (`process_manager.py`)

### New `_process_single_file` flow:

```
read file (utf-8, skip binary)
  -> detect language
  -> skip non-code files (.md, .csv, etc.)
  -> load function schema (once, cached on self)
  -> get language handler via get_handler(language, schema, metrics_analyzer)
  -> IF handler exists (supported language):
       handler.extract_structure(code, file_path)
       build LLM request with tools=[function_schema]
       parse tool_calls[0].function.arguments as structured JSON
     ELSE (unsupported language):
       send code with generic prompt, get freeform text response
  -> IF not safe_mode AND handler exists AND structured response valid:
       handler.insert_docstrings(code, structured_response)
       handler.validate_code(modified_code)
       IF valid: write modified source file
       ELSE: log warning, skip source write
  -> write documentation output (JSON + Markdown)
  -> record metrics
```

### LLM request format (structured):

```python
{
    "messages": [
        {"role": "system", "content": "You are a code documentation generator. Analyze the provided code structure and generate comprehensive documentation. Fill in docstrings for all functions and classes. Use Google-style docstrings for Python."},
        {"role": "user", "content": f"File: {file_path}\nLanguage: {language}\n\nExtracted structure:\n{json.dumps(extracted_structure, indent=2)}\n\nSource code:\n{code[:8000]}"}
    ],
    "tools": [{
        "type": "function",
        "function": function_schema["functions"][0]
    }],
    "tool_choice": {"type": "function", "function": {"name": "generate_documentation"}},
    "max_completion_tokens": 4096,
    "temperature": config.temperature
}
```

### Response parsing:

```python
if api_response and "choices" in api_response:
    choice = api_response["choices"][0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        structured = json.loads(tool_calls[0]["function"]["arguments"])
    else:
        # Fallback: try content as freeform
        structured = None
```

### Changes to `DocumentationProcessManager.__init__`:

- Load function schema once: `self.function_schema = load_function_schema(schema_path)` — passed from `main.py`
- Create `MetricsAnalyzer` instance: `self.metrics_analyzer = MetricsAnalyzer()`

### Changes to `main.py`:

- Pass `function_schema` path to `DocumentationProcessManager` constructor (re-add the `--schema` arg usage that was removed as dead code)

---

## Component 3: Handler Fixes (`language_functions/`)

### `base_handler.py`:
- Rename class `BaseLanguageHandler` to `BaseHandler` (all handlers import `BaseHandler`)
- Constructor: `__init__(self, function_schema, metrics_analyzer=None)`

### `language_functions/__init__.py`:
- `get_handler(language, function_schema, metrics_analyzer=None)` — the single factory
- Pass `metrics_analyzer` to all handler constructors
- Remove broken `from metrics import MetricsAnalyzer` (will import from the fixed metrics.py)

### `language_functions/language_functions.py`:
- Delete duplicate `get_handler()` function
- `insert_docstrings()` calls `get_handler` from `__init__.py`

### Individual handlers:
- All constructors: `__init__(self, function_schema, metrics_analyzer=None)`
- `PythonHandler`: uses `metrics_analyzer.add_result()` if available
- All others: accept and store `metrics_analyzer` but don't require it

---

## Component 4: Docstring Insertion

### `PythonHandler.insert_docstrings` fix:

The `DocstringTransformer` sets `node.docstring = ...` which is not valid AST manipulation. Fix to insert `ast.Expr(ast.Constant(value=docstring))` as the first body statement:

```python
def _insert_docstring(self, node, docstring):
    """Insert a docstring as the first statement of a node body."""
    docstring_node = ast.Expr(value=ast.Constant(value=docstring))
    # Remove existing docstring if present
    if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
        node.body[0] = docstring_node
    else:
        node.body.insert(0, docstring_node)
    return node
```

### Non-Python handlers:

JS/TS/Go/C++/Java/HTML/CSS handlers shell out to Node.js scripts in `scripts/`. These work if `npm install` has been run in `scripts/`. The pipeline should:
- Check if `scripts/node_modules` exists
- If not, log a warning and skip insertion for non-Python files
- Never fail the whole pipeline because Node.js isn't available

### Validation:

- `PythonHandler.validate_code()` uses pylint — this works but pylint may not be installed. Use `py_compile.compile()` as fallback (syntax check only, no linting).
- Non-Python validation: skip if tools not available, log warning.

---

## Component 5: Markdown Report Generation (`write_documentation_report.py`)

### Replace broken Jinja2 pipeline:

Delete the `DocumentationGenerator` class that uses `PackageLoader('documentation', 'templates')` (templates don't exist).

Add `generate_markdown_report(structured_response, file_path, metrics)` function:

```python
def generate_markdown_report(structured_response, file_path, metrics=None):
    """Generate Markdown documentation from structured LLM response."""
    sections = []

    # Header with badges
    sections.append(f"# {Path(file_path).name}")
    if metrics:
        sections.append(BadgeGenerator.generate_all_badges(metrics))

    # Summary
    if summary := structured_response.get("summary"):
        sections.append(f"## Summary\n\n{summary}")

    # Functions table
    if functions := structured_response.get("functions"):
        formatter = MarkdownFormatter()
        headers = ["Function", "Args", "Description", "Complexity"]
        rows = [[f["name"], ", ".join(a if isinstance(a,str) else a.get("name","") for a in f.get("args",[])),
                 f.get("docstring","")[:100], str(f.get("complexity",""))] for f in functions]
        sections.append(f"## Functions\n\n{formatter.format_table(headers, rows)}")

    # Classes table (similar)
    # Variables table (similar)
    # Metrics section

    return "\n\n".join(sections)
```

### `write_documentation_report()` changes:

Replace the `if documentation.get("generate_markdown", True)` block:
- Instead of instantiating `DocumentationGenerator` (Jinja2), call `generate_markdown_report()`
- This uses the existing `BadgeGenerator` and `MarkdownFormatter` which work

---

## File Change Map

| File | Changes |
|------|---------|
| `metrics.py` | Add MetricsAnalyzer, MetricsThresholds, calculate_code_metrics, helper functions |
| `process_manager.py` | Rewrite _process_single_file: extract_structure, function calling, docstring insertion |
| `main.py` | Re-add schema_path to DocumentationProcessManager constructor |
| `language_functions/base_handler.py` | Rename BaseLanguageHandler -> BaseHandler, fix constructor |
| `language_functions/__init__.py` | Fix get_handler signature, fix MetricsAnalyzer import |
| `language_functions/language_functions.py` | Remove duplicate get_handler, use factory from __init__ |
| `language_functions/python_handler.py` | Fix imports, fix DocstringTransformer AST manipulation |
| `language_functions/js_ts_handler.py` | Fix constructor to accept metrics_analyzer |
| `language_functions/go_handler.py` | Fix constructor |
| `language_functions/cpp_handler.py` | Fix constructor |
| `language_functions/java_handler.py` | Fix constructor |
| `language_functions/html_handler.py` | Fix constructor |
| `language_functions/css_handler.py` | Fix constructor |
| `write_documentation_report.py` | Replace Jinja2 pipeline with direct Markdown generation |

---

## Not in scope

- Embedding/similarity features (metrics_utils.py) — optional ML, already gracefully degraded
- FastAPI server endpoints — work as-is, not part of CLI flow
- documentation-system/ React app — separate frontend, not touched
- Adding new language handlers — existing 8 languages are sufficient
