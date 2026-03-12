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
