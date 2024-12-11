# Module: docstring_utils

## Overview
**File:** `simple2/docstring_utils.py`
**Description:** """
Docstring Utilities Module

Provides comprehensive validation for Python docstrings with detailed type checking
and content validation.
"""

from typing import Dict, List, Any, Optional, Tuple
from jsonschema import validate, ValidationError, Draft7Validator
import ast
import re
import logging
import inspect
from typing_extensions import get_origin, get_args

# Configure logging
logger = logging.getLogger(__name__)

# Comprehensive schema for docstring validation
DOCSTRING_SCHEMA = {
    "type": "object",
    "properties": {
        "docstring": {
            "type": "string",
            "minLength": 10,
            "description": "The complete generated docstring."
        },
        "summary": {
            "type": "string",
            "minLength": 10,
            "description": "A concise summary of the function/class/module."
        },
        "parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$"},
                    "type": {"type": "string"},
                    "description": {"type": "string", "minLength": 10},
                    "optional": {"type": "boolean"},
                    "default": {"type": ["string", "number", "boolean", "null"]}
                },
                "required": ["name", "type", "description"]
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "description": {"type": "string", "minLength": 10}
            },
            "required": ["type", "description"]
        },
        "raises": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "exception": {"type": "string"},
                    "description": {"type": "string", "minLength": 10}
                },
                "required": ["exception", "description"]
            }
        },
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "minLength": 1},
                    "description": {"type": "string"}
                },
                "required": ["code"]
            }
        }
    },
    "required": ["docstring", "summary", "parameters", "returns"],
    "additionalProperties": False
}


class DocstringValidator:
    """Validates docstrings with comprehensive type checking and content validation."""

    def __init__(self):
        """Initialize the validator with the comprehensive schema."""
        self.validator = Draft7Validator(DOCSTRING_SCHEMA)
        logger.debug("DocstringValidator initialized with schema.")

    def _validate_type_string(self, type_str: str) -> List[str]:
        """
        Simplified and practical type string validation that actually works.

        Args:
            type_str: The type string to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        logger.debug(f"Validating type string: {type_str}")

        if type_str.lower() in ('none', 'nonetype'):
            return errors

        if not type_str:
            return ["Empty type string"]

        basic_type_pattern = r'^[A-Za-z_][A-Za-z0-9_]*'
        valid_patterns = [
            r'^[A-Za-z_][A-Za-z0-9_]*$',  # Basic types
            r'^[A-Za-z_][A-Za-z0-9_]*$.*$$',  # Generic types
            r'^Union$.*$$',  # Union types
            r'^Optional$.*$$',  # Optional types
            r'^Callable($.*$)?$',  # Callable types
            # Fully qualified names
            r'^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$',
        ]

        if not any(re.match(pattern, type_str) for pattern in valid_patterns):
            errors.append(f"Invalid type format: {type_str}")
            logger.warning(f"Invalid type format detected: {type_str}")

        return errors

    def validate_docstring(self, docstring_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Simplified docstring validation that's actually practical.
        """
        errors = []
        logger.debug("Validating docstring data.")

        required_fields = ["summary", "parameters", "returns"]
        for field in required_fields:
            if field not in docstring_data:
                errors.append(f"Missing required field: {field}")
                logger.error(f"Missing required field in docstring: {field}")

        if errors:
            return False, errors

        if "parameters" in docstring_data:
            for param in docstring_data["parameters"]:
                if "name" not in param:
                    errors.append("Parameter missing name")
                    logger.error("Parameter missing name.")
                    continue

                if "type" not in param:
                    errors.append(f"Parameter {param['name']} missing type")
                    logger.error(f"Parameter {param['name']} missing type.")
                else:
                    type_errors = self._validate_type_string(param["type"])
                    if type_errors:
                        errors.extend(
                            f"Parameter {param['name']}: {err}" for err in type_errors)

                if "description" not in param:
                    errors.append(
                        f"Parameter {param['name']} missing description")
                    logger.error(
                        f"Parameter {param['name']} missing description.")
                elif len(param["description"].strip()) < 10:
                    errors.append(
                        f"Parameter {param['name']} description too short")
                    logger.warning(
                        f"Parameter {param['name']} description too short.")

        if "returns" in docstring_data:
            returns = docstring_data["returns"]
            if "type" not in returns:
                errors.append("Return missing type")
                logger.error("Return missing type.")
            else:
                type_errors = self._validate_type_string(returns["type"])
                errors.extend(f"Return type: {err}" for err in type_errors)

            if "description" not in returns or len(returns["description"].strip()) < 10:
                errors.append("Return missing or too short description")
                logger.warning("Return description missing or too short.")

        if "summary" in docstring_data:
            summary = docstring_data["summary"].strip()
            if len(summary) < 10:
                errors.append("Summary too short (minimum 10 characters)")
                logger.warning("Summary too short.")

        return len(errors) == 0, errors

    def _validate_generic_type(self, type_str: str) -> List[str]:
        """Validate a generic type annotation."""
        errors = []
        logger.debug(f"Validating generic type: {type_str}")
        try:
            base_type = type_str[:type_str.index('[')]
            params = type_str[type_str.index('[') + 1:type_str.rindex(']')]

            if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', base_type):
                errors.append(f"Invalid base type: {base_type}")
                logger.warning(f"Invalid base type detected: {base_type}")

            for param in params.split(','):
                param = param.strip()
                param_errors = self._validate_type_string(param)
                errors.extend(param_errors)

        except Exception as e:
            errors.append(f"Invalid generic type format: {type_str}")
            logger.exception(f"Exception during generic type validation: {e}")

        return errors

    def _validate_union_type(self, type_str: str) -> List[str]:
        """Validate a Union type annotation."""
        errors = []
        logger.debug(f"Validating Union type: {type_str}")
        try:
            types = type_str[type_str.index('[') + 1:type_str.rindex(']')]
            for type_part in types.split(','):
                type_part = type_part.strip()
                errors.extend(self._validate_type_string(type_part))
        except Exception as e:
            errors.append(f"Invalid Union type format: {type_str}")
            logger.exception(f"Exception during Union type validation: {e}")
        return errors

    def _validate_optional_type(self, type_str: str) -> List[str]:
        """Validate an Optional type annotation."""
        errors = []
        logger.debug(f"Validating Optional type: {type_str}")
        try:
            inner_type = type_str[type_str.index('[') + 1:type_str.rindex(']')]
            errors.extend(self._validate_type_string(inner_type))
        except Exception as e:
            errors.append(f"Invalid Optional type format: {type_str}")
            logger.exception(f"Exception during Optional type validation: {e}")
        return errors

    def _validate_exception_name(self, exception_name: str) -> bool:
        """Validate an exception name."""
        logger.debug(f"Validating exception name: {exception_name}")
        if exception_name in dir(__builtins__):
            return True
        return bool(re.match(r'^[A-Z][a-zA-Z0-9]*(?:Error|Exception|Warning)$', exception_name))

    def _validate_content_quality(self, docstring_data: Dict[str, Any]) -> List[str]:
        """Validate the quality of docstring content."""
        errors = []
        logger.debug("Validating content quality of docstring.")

        if 'summary' in docstring_data:
            summary = docstring_data['summary']
            if len(summary.split()) < 3:
                errors.append("Summary is too brief (minimum 3 words)")
                logger.warning("Summary is too brief.")
            if not summary[0].isupper():
                errors.append("Summary should start with a capital letter")
                logger.warning("Summary should start with a capital letter.")
            if not summary.rstrip().endswith(('.', '?', '!')):
                errors.append("Summary should end with proper punctuation")
                logger.warning("Summary should end with proper punctuation.")

        if 'parameters' in docstring_data:
            for param in docstring_data['parameters']:
                desc = param.get('description', '')
                if len(desc.split()) < 3:
                    errors.append(
                        f"Description for parameter '{param['name']}' is too brief")
                    logger.warning(
                        f"Description for parameter '{param['name']}' is too brief.")
                if not desc[0].isupper():
                    errors.append(
                        f"Description for parameter '{param['name']}' should start with a capital letter")
                    logger.warning(
                        f"Description for parameter '{param['name']}' should start with a capital letter.")

        if 'returns' in docstring_data:
            ret_desc = docstring_data['returns'].get('description', '')
            if len(ret_desc.split()) < 3:
                errors.append("Return description is too brief")
                logger.warning("Return description is too brief.")
            if not ret_desc[0].isupper():
                errors.append(
                    "Return description should start with a capital letter")
                logger.warning(
                    "Return description should start with a capital letter.")

        return errors

    def _evaluate_type_expression(self, type_str: str) -> Any:
        """Safely evaluate a type expression."""
        namespace = {
            'List': List,
            'Dict': Dict,
            'Set': set,
            'Tuple': Tuple,
            'Optional': Optional,
            'Union': Union,
            'Any': Any
        }
        try:
            return eval(type_str, namespace)
        except Exception as e:
            logger.exception(f"Invalid type expression: {str(e)}")
            raise ValueError(f"Invalid type expression: {str(e)}")


def parse_docstring(docstring: str) -> Dict[str, Any]:
    """Parse a docstring into structured sections."""
    logger.debug("Parsing docstring.")
    if not docstring:
        return {"docstring": ""}

    sections = {"docstring": docstring.strip()}
    current_section = 'summary'
    current_content = []

    lines = docstring.split('\n')
    for line in lines:
        line = line.strip()

        if line.endswith(':') and line.rstrip(':').lower() in ['args', 'arguments', 'parameters',
                                                               'returns', 'raises', 'examples']:
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()

            current_section = line.rstrip(':').lower()
            if current_section == 'args' or current_section == 'arguments':
                current_section = 'parameters'
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections[current_section] = '\n'.join(current_content).strip()

    return _process_sections(sections)


def _process_sections(sections: Dict[str, str]) -> Dict[str, Any]:
    """Process parsed sections into schema format."""
    logger.debug("Processing parsed sections into schema format.")
    processed = {
        "docstring": sections.get("docstring", ""),
        "summary": sections.get("summary", ""),
        "parameters": [],
        "returns": {"type": "None", "description": "No return value."},
        "raises": [],
        "examples": []
    }

    if "parameters" in sections:
        params = _parse_parameters(sections["parameters"])
        processed["parameters"] = params

    if "returns" in sections:
        returns = _parse_returns(sections["returns"])
        processed["returns"] = returns

    if "raises" in sections:
        raises = _parse_raises(sections["raises"])
        processed["raises"] = raises

    if "examples" in sections:
        examples = _parse_examples(sections["examples"])
        processed["examples"] = examples

    return processed


def _parse_parameters(params_str: str) -> List[Dict[str, Any]]:
    """Parse parameter section into structured format."""
    logger.debug("Parsing parameters section.")
    params = []
    current_param = None

    for line in params_str.split('\n'):
        line = line.strip()
        if not line:
            continue

        if ':' in line and not line.startswith(' '):
            if current_param:
                params.append(current_param)

            name, rest = line.split(':', 1)
            name = name.strip()
            type_desc = rest.strip()

            if '  ' in type_desc:
                type_str, desc = type_desc.split('  ', 1)
            else:
                type_str, desc = type_desc, ""

            current_param = {
                "name": name,
                "type": type_str.strip(),
                "description": desc.strip()
            }
        elif current_param:
            current_param["description"] = current_param["description"] + " " + line

    if current_param:
        params.append(current_param)

    return params


def _parse_returns(returns_str: str) -> Dict[str, str]:
    """Parse return section into structured format."""
    logger.debug("Parsing returns section.")
    if ':' in returns_str:
        type_str, desc = returns_str.split(':', 1)
        return {
            "type": type_str.strip(),
            "description": desc.strip()
        }
    return {
        "type": "None",
        "description": returns_str.strip() or "No return value."
    }


def _parse_raises(raises_str: str) -> List[Dict[str, str]]:
    """Parse raises section into structured format."""
    logger.debug("Parsing raises section.")
    raises = []
    current_exception = None

    for line in raises_str.split('\n'):
        line = line.strip()
        if not line:
            continue

        if ':' in line and not line.startswith(' '):
            if current_exception:
                raises.append(current_exception)

            exc, desc = line.split(':', 1)
            current_exception = {
                "exception": exc.strip(),
                "description": desc.strip()
            }
        elif current_exception:
            current_exception["description"] = current_exception["description"] + " " + line

    if current_exception:
        raises.append(current_exception)

    return raises


def _parse_examples(examples_str: str) -> List[Dict[str, str]]:
    """Parse examples section into structured format."""
    logger.debug("Parsing examples section.")
    examples = []
    current_example = None
    in_code_block = False
    code_lines = []

    for line in examples_str.split('\n'):
        if line.strip().startswith('```'):
            if in_code_block:
                if current_example:
                    current_example["code"] = '\n'.join(code_lines)
                    examples.append(current_example)
                    code_lines = []
                    current_example = None
            else:
                current_example = {"code": "", "description": ""}
                code_lines = []
            in_code_block = not in_code_block
        elif in_code_block:
            code_lines.append(line)
        elif line.strip() and current_example is None:
            current_example = {"code": "", "description": line.strip()}
        elif line.strip() and current_example:
            current_example["description"] += " " + line.strip()

    if current_example and code_lines:
        current_example["code"] = '\n'.join(code_lines)
        examples.append(current_example)

    return examples


def analyze_code_element_docstring(node: ast.AST) -> List[str]:
    """
    Analyze the docstring of a code element for completeness.

    Args:
        node (ast.AST): The AST node representing the code element

    Returns:
        List[str]: A list of issues found in the docstring
    """
    logger.debug(
        f"Analyzing docstring for node: {getattr(node, 'name', '<unknown>')}")
    issues = []
    docstring = ast.get_docstring(node)
    if not docstring:
        issues.append("Missing docstring.")
        logger.warning("Missing docstring detected.")
        return issues

    parsed_docstring = parse_docstring(docstring)
    validator = DocstringValidator()
    is_valid, validation_errors = validator.validate_docstring(
        parsed_docstring)

    if not is_valid:
        issues.extend(validation_errors)
        logger.warning(f"Validation errors found: {validation_errors}")

    return issues


def validate_and_fix_docstring(docstring: str) -> Tuple[str, List[str]]:
    """
    Validate and attempt to fix common docstring issues.

    Args:
        docstring (str): The docstring to validate and fix

    Returns:
        Tuple[str, List[str]]: (fixed_docstring, list_of_changes)
    """
    logger.debug("Validating and fixing docstring.")
    changes = []
    fixed_docstring = docstring

    fixes = [
        (r'\n\s*\n\s*\n', '\n\n'),  # Fix multiple blank lines
        (r'^(\s*[A-Za-z]+:)\s*([A-Z])', r'\1\n    \2'),  # Fix section headers
        (r'(?<=\n)(\s{4,})', '    '),  # Normalize indentation
        (r'"""[\s\n]*', '"""'),  # Fix opening quotes
        (r'[\s\n]*"""$', '"""'),  # Fix closing quotes
    ]

    for pattern, replacement in fixes:
        new_docstring = re.sub(pattern, replacement, fixed_docstring)
        if new_docstring != fixed_docstring:
            changes.append(f"Fixed pattern: {pattern}")
            fixed_docstring = new_docstring

    logger.info(f"Docstring validation and fixes applied: {changes}")
    return fixed_docstring, changes


def parse_and_validate_docstring(docstring: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Parse and validate a docstring comprehensively.

    Args:
        docstring (str): The docstring to parse and validate

    Returns:
        Tuple[Optional[Dict[str, Any]], List[str]]: (parsed_docstring, validation_errors)
    """
    logger.debug("Parsing and validating docstring.")
    if not docstring:
        return None, ["Empty docstring"]

    try:
        fixed_docstring, fix_changes = validate_and_fix_docstring(docstring)

        parsed_data = parse_docstring(fixed_docstring)
        if not parsed_data:
            return None, ["Failed to parse docstring"]

        validator = DocstringValidator()
        is_valid, validation_errors = validator.validate_docstring(parsed_data)

        if fix_changes:
            validation_errors.extend(
                [f"Applied fix: {change}" for change in fix_changes])

        return parsed_data if is_valid else None, validation_errors

    except Exception as e:
        logger.exception(f"Error processing docstring: {str(e)}")
        return None, [f"Error processing docstring: {str(e)}"]


## AI-Generated Documentation


**Summary:** No summary provided.


**Description:** No description provided.



## Classes

| Class | Inherits From | Complexity Score* |
|-------|---------------|-------------------|
| `DocstringValidator` | `` | 0 |

### Class Methods

| Class | Method | Parameters | Returns | Complexity Score* |
|-------|--------|------------|---------|-------------------|
| `DocstringValidator` | `validate_docstring` | `(self: Any, docstring_data: Dict[(str, Any)])` | `Tuple[(bool, List[str])]` | 0 |

## Source Code
```python
"""
Docstring Utilities Module

Provides comprehensive validation for Python docstrings with detailed type checking
and content validation.
"""

from typing import Dict, List, Any, Optional, Tuple
from jsonschema import validate, ValidationError, Draft7Validator
import ast
import re
import logging
import inspect
from typing_extensions import get_origin, get_args

# Configure logging
logger = logging.getLogger(__name__)

# Comprehensive schema for docstring validation
DOCSTRING_SCHEMA = {
    "type": "object",
    "properties": {
        "docstring": {
            "type": "string",
            "minLength": 10,
            "description": "The complete generated docstring."
        },
        "summary": {
            "type": "string",
            "minLength": 10,
            "description": "A concise summary of the function/class/module."
        },
        "parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$"},
                    "type": {"type": "string"},
                    "description": {"type": "string", "minLength": 10},
                    "optional": {"type": "boolean"},
                    "default": {"type": ["string", "number", "boolean", "null"]}
                },
                "required": ["name", "type", "description"]
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "description": {"type": "string", "minLength": 10}
            },
            "required": ["type", "description"]
        },
        "raises": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "exception": {"type": "string"},
                    "description": {"type": "string", "minLength": 10}
                },
                "required": ["exception", "description"]
            }
        },
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "minLength": 1},
                    "description": {"type": "string"}
                },
                "required": ["code"]
            }
        }
    },
    "required": ["docstring", "summary", "parameters", "returns"],
    "additionalProperties": False
}


class DocstringValidator:
    """Validates docstrings with comprehensive type checking and content validation."""

    def __init__(self):
        """Initialize the validator with the comprehensive schema."""
        self.validator = Draft7Validator(DOCSTRING_SCHEMA)
        logger.debug("DocstringValidator initialized with schema.")

    def _validate_type_string(self, type_str: str) -> List[str]:
        """
        Simplified and practical type string validation that actually works.

        Args:
            type_str: The type string to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        logger.debug(f"Validating type string: {type_str}")

        if type_str.lower() in ('none', 'nonetype'):
            return errors

        if not type_str:
            return ["Empty type string"]

        basic_type_pattern = r'^[A-Za-z_][A-Za-z0-9_]*'
        valid_patterns = [
            r'^[A-Za-z_][A-Za-z0-9_]*$',  # Basic types
            r'^[A-Za-z_][A-Za-z0-9_]*$.*$$',  # Generic types
            r'^Union$.*$$',  # Union types
            r'^Optional$.*$$',  # Optional types
            r'^Callable($.*$)?$',  # Callable types
            # Fully qualified names
            r'^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$',
        ]

        if not any(re.match(pattern, type_str) for pattern in valid_patterns):
            errors.append(f"Invalid type format: {type_str}")
            logger.warning(f"Invalid type format detected: {type_str}")

        return errors

    def validate_docstring(self, docstring_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Simplified docstring validation that's actually practical.
        """
        errors = []
        logger.debug("Validating docstring data.")

        required_fields = ["summary", "parameters", "returns"]
        for field in required_fields:
            if field not in docstring_data:
                errors.append(f"Missing required field: {field}")
                logger.error(f"Missing required field in docstring: {field}")

        if errors:
            return False, errors

        if "parameters" in docstring_data:
            for param in docstring_data["parameters"]:
                if "name" not in param:
                    errors.append("Parameter missing name")
                    logger.error("Parameter missing name.")
                    continue

                if "type" not in param:
                    errors.append(f"Parameter {param['name']} missing type")
                    logger.error(f"Parameter {param['name']} missing type.")
                else:
                    type_errors = self._validate_type_string(param["type"])
                    if type_errors:
                        errors.extend(
                            f"Parameter {param['name']}: {err}" for err in type_errors)

                if "description" not in param:
                    errors.append(
                        f"Parameter {param['name']} missing description")
                    logger.error(
                        f"Parameter {param['name']} missing description.")
                elif len(param["description"].strip()) < 10:
                    errors.append(
                        f"Parameter {param['name']} description too short")
                    logger.warning(
                        f"Parameter {param['name']} description too short.")

        if "returns" in docstring_data:
            returns = docstring_data["returns"]
            if "type" not in returns:
                errors.append("Return missing type")
                logger.error("Return missing type.")
            else:
                type_errors = self._validate_type_string(returns["type"])
                errors.extend(f"Return type: {err}" for err in type_errors)

            if "description" not in returns or len(returns["description"].strip()) < 10:
                errors.append("Return missing or too short description")
                logger.warning("Return description missing or too short.")

        if "summary" in docstring_data:
            summary = docstring_data["summary"].strip()
            if len(summary) < 10:
                errors.append("Summary too short (minimum 10 characters)")
                logger.warning("Summary too short.")

        return len(errors) == 0, errors

    def _validate_generic_type(self, type_str: str) -> List[str]:
        """Validate a generic type annotation."""
        errors = []
        logger.debug(f"Validating generic type: {type_str}")
        try:
            base_type = type_str[:type_str.index('[')]
            params = type_str[type_str.index('[') + 1:type_str.rindex(']')]

            if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', base_type):
                errors.append(f"Invalid base type: {base_type}")
                logger.warning(f"Invalid base type detected: {base_type}")

            for param in params.split(','):
                param = param.strip()
                param_errors = self._validate_type_string(param)
                errors.extend(param_errors)

        except Exception as e:
            errors.append(f"Invalid generic type format: {type_str}")
            logger.exception(f"Exception during generic type validation: {e}")

        return errors

    def _validate_union_type(self, type_str: str) -> List[str]:
        """Validate a Union type annotation."""
        errors = []
        logger.debug(f"Validating Union type: {type_str}")
        try:
            types = type_str[type_str.index('[') + 1:type_str.rindex(']')]
            for type_part in types.split(','):
                type_part = type_part.strip()
                errors.extend(self._validate_type_string(type_part))
        except Exception as e:
            errors.append(f"Invalid Union type format: {type_str}")
            logger.exception(f"Exception during Union type validation: {e}")
        return errors

    def _validate_optional_type(self, type_str: str) -> List[str]:
        """Validate an Optional type annotation."""
        errors = []
        logger.debug(f"Validating Optional type: {type_str}")
        try:
            inner_type = type_str[type_str.index('[') + 1:type_str.rindex(']')]
            errors.extend(self._validate_type_string(inner_type))
        except Exception as e:
            errors.append(f"Invalid Optional type format: {type_str}")
            logger.exception(f"Exception during Optional type validation: {e}")
        return errors

    def _validate_exception_name(self, exception_name: str) -> bool:
        """Validate an exception name."""
        logger.debug(f"Validating exception name: {exception_name}")
        if exception_name in dir(__builtins__):
            return True
        return bool(re.match(r'^[A-Z][a-zA-Z0-9]*(?:Error|Exception|Warning)$', exception_name))

    def _validate_content_quality(self, docstring_data: Dict[str, Any]) -> List[str]:
        """Validate the quality of docstring content."""
        errors = []
        logger.debug("Validating content quality of docstring.")

        if 'summary' in docstring_data:
            summary = docstring_data['summary']
            if len(summary.split()) < 3:
                errors.append("Summary is too brief (minimum 3 words)")
                logger.warning("Summary is too brief.")
            if not summary[0].isupper():
                errors.append("Summary should start with a capital letter")
                logger.warning("Summary should start with a capital letter.")
            if not summary.rstrip().endswith(('.', '?', '!')):
                errors.append("Summary should end with proper punctuation")
                logger.warning("Summary should end with proper punctuation.")

        if 'parameters' in docstring_data:
            for param in docstring_data['parameters']:
                desc = param.get('description', '')
                if len(desc.split()) < 3:
                    errors.append(
                        f"Description for parameter '{param['name']}' is too brief")
                    logger.warning(
                        f"Description for parameter '{param['name']}' is too brief.")
                if not desc[0].isupper():
                    errors.append(
                        f"Description for parameter '{param['name']}' should start with a capital letter")
                    logger.warning(
                        f"Description for parameter '{param['name']}' should start with a capital letter.")

        if 'returns' in docstring_data:
            ret_desc = docstring_data['returns'].get('description', '')
            if len(ret_desc.split()) < 3:
                errors.append("Return description is too brief")
                logger.warning("Return description is too brief.")
            if not ret_desc[0].isupper():
                errors.append(
                    "Return description should start with a capital letter")
                logger.warning(
                    "Return description should start with a capital letter.")

        return errors

    def _evaluate_type_expression(self, type_str: str) -> Any:
        """Safely evaluate a type expression."""
        namespace = {
            'List': List,
            'Dict': Dict,
            'Set': set,
            'Tuple': Tuple,
            'Optional': Optional,
            'Union': Union,
            'Any': Any
        }
        try:
            return eval(type_str, namespace)
        except Exception as e:
            logger.exception(f"Invalid type expression: {str(e)}")
            raise ValueError(f"Invalid type expression: {str(e)}")


def parse_docstring(docstring: str) -> Dict[str, Any]:
    """Parse a docstring into structured sections."""
    logger.debug("Parsing docstring.")
    if not docstring:
        return {"docstring": ""}

    sections = {"docstring": docstring.strip()}
    current_section = 'summary'
    current_content = []

    lines = docstring.split('\n')
    for line in lines:
        line = line.strip()

        if line.endswith(':') and line.rstrip(':').lower() in ['args', 'arguments', 'parameters',
                                                               'returns', 'raises', 'examples']:
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()

            current_section = line.rstrip(':').lower()
            if current_section == 'args' or current_section == 'arguments':
                current_section = 'parameters'
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections[current_section] = '\n'.join(current_content).strip()

    return _process_sections(sections)


def _process_sections(sections: Dict[str, str]) -> Dict[str, Any]:
    """Process parsed sections into schema format."""
    logger.debug("Processing parsed sections into schema format.")
    processed = {
        "docstring": sections.get("docstring", ""),
        "summary": sections.get("summary", ""),
        "parameters": [],
        "returns": {"type": "None", "description": "No return value."},
        "raises": [],
        "examples": []
    }

    if "parameters" in sections:
        params = _parse_parameters(sections["parameters"])
        processed["parameters"] = params

    if "returns" in sections:
        returns = _parse_returns(sections["returns"])
        processed["returns"] = returns

    if "raises" in sections:
        raises = _parse_raises(sections["raises"])
        processed["raises"] = raises

    if "examples" in sections:
        examples = _parse_examples(sections["examples"])
        processed["examples"] = examples

    return processed


def _parse_parameters(params_str: str) -> List[Dict[str, Any]]:
    """Parse parameter section into structured format."""
    logger.debug("Parsing parameters section.")
    params = []
    current_param = None

    for line in params_str.split('\n'):
        line = line.strip()
        if not line:
            continue

        if ':' in line and not line.startswith(' '):
            if current_param:
                params.append(current_param)

            name, rest = line.split(':', 1)
            name = name.strip()
            type_desc = rest.strip()

            if '  ' in type_desc:
                type_str, desc = type_desc.split('  ', 1)
            else:
                type_str, desc = type_desc, ""

            current_param = {
                "name": name,
                "type": type_str.strip(),
                "description": desc.strip()
            }
        elif current_param:
            current_param["description"] = current_param["description"] + " " + line

    if current_param:
        params.append(current_param)

    return params


def _parse_returns(returns_str: str) -> Dict[str, str]:
    """Parse return section into structured format."""
    logger.debug("Parsing returns section.")
    if ':' in returns_str:
        type_str, desc = returns_str.split(':', 1)
        return {
            "type": type_str.strip(),
            "description": desc.strip()
        }
    return {
        "type": "None",
        "description": returns_str.strip() or "No return value."
    }


def _parse_raises(raises_str: str) -> List[Dict[str, str]]:
    """Parse raises section into structured format."""
    logger.debug("Parsing raises section.")
    raises = []
    current_exception = None

    for line in raises_str.split('\n'):
        line = line.strip()
        if not line:
            continue

        if ':' in line and not line.startswith(' '):
            if current_exception:
                raises.append(current_exception)

            exc, desc = line.split(':', 1)
            current_exception = {
                "exception": exc.strip(),
                "description": desc.strip()
            }
        elif current_exception:
            current_exception["description"] = current_exception["description"] + " " + line

    if current_exception:
        raises.append(current_exception)

    return raises


def _parse_examples(examples_str: str) -> List[Dict[str, str]]:
    """Parse examples section into structured format."""
    logger.debug("Parsing examples section.")
    examples = []
    current_example = None
    in_code_block = False
    code_lines = []

    for line in examples_str.split('\n'):
        if line.strip().startswith('```'):
            if in_code_block:
                if current_example:
                    current_example["code"] = '\n'.join(code_lines)
                    examples.append(current_example)
                    code_lines = []
                    current_example = None
            else:
                current_example = {"code": "", "description": ""}
                code_lines = []
            in_code_block = not in_code_block
        elif in_code_block:
            code_lines.append(line)
        elif line.strip() and current_example is None:
            current_example = {"code": "", "description": line.strip()}
        elif line.strip() and current_example:
            current_example["description"] += " " + line.strip()

    if current_example and code_lines:
        current_example["code"] = '\n'.join(code_lines)
        examples.append(current_example)

    return examples


def analyze_code_element_docstring(node: ast.AST) -> List[str]:
    """
    Analyze the docstring of a code element for completeness.

    Args:
        node (ast.AST): The AST node representing the code element

    Returns:
        List[str]: A list of issues found in the docstring
    """
    logger.debug(
        f"Analyzing docstring for node: {getattr(node, 'name', '<unknown>')}")
    issues = []
    docstring = ast.get_docstring(node)
    if not docstring:
        issues.append("Missing docstring.")
        logger.warning("Missing docstring detected.")
        return issues

    parsed_docstring = parse_docstring(docstring)
    validator = DocstringValidator()
    is_valid, validation_errors = validator.validate_docstring(
        parsed_docstring)

    if not is_valid:
        issues.extend(validation_errors)
        logger.warning(f"Validation errors found: {validation_errors}")

    return issues


def validate_and_fix_docstring(docstring: str) -> Tuple[str, List[str]]:
    """
    Validate and attempt to fix common docstring issues.

    Args:
        docstring (str): The docstring to validate and fix

    Returns:
        Tuple[str, List[str]]: (fixed_docstring, list_of_changes)
    """
    logger.debug("Validating and fixing docstring.")
    changes = []
    fixed_docstring = docstring

    fixes = [
        (r'\n\s*\n\s*\n', '\n\n'),  # Fix multiple blank lines
        (r'^(\s*[A-Za-z]+:)\s*([A-Z])', r'\1\n    \2'),  # Fix section headers
        (r'(?<=\n)(\s{4,})', '    '),  # Normalize indentation
        (r'"""[\s\n]*', '"""'),  # Fix opening quotes
        (r'[\s\n]*"""$', '"""'),  # Fix closing quotes
    ]

    for pattern, replacement in fixes:
        new_docstring = re.sub(pattern, replacement, fixed_docstring)
        if new_docstring != fixed_docstring:
            changes.append(f"Fixed pattern: {pattern}")
            fixed_docstring = new_docstring

    logger.info(f"Docstring validation and fixes applied: {changes}")
    return fixed_docstring, changes


def parse_and_validate_docstring(docstring: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Parse and validate a docstring comprehensively.

    Args:
        docstring (str): The docstring to parse and validate

    Returns:
        Tuple[Optional[Dict[str, Any]], List[str]]: (parsed_docstring, validation_errors)
    """
    logger.debug("Parsing and validating docstring.")
    if not docstring:
        return None, ["Empty docstring"]

    try:
        fixed_docstring, fix_changes = validate_and_fix_docstring(docstring)

        parsed_data = parse_docstring(fixed_docstring)
        if not parsed_data:
            return None, ["Failed to parse docstring"]

        validator = DocstringValidator()
        is_valid, validation_errors = validator.validate_docstring(parsed_data)

        if fix_changes:
            validation_errors.extend(
                [f"Applied fix: {change}" for change in fix_changes])

        return parsed_data if is_valid else None, validation_errors

    except Exception as e:
        logger.exception(f"Error processing docstring: {str(e)}")
        return None, [f"Error processing docstring: {str(e)}"]

```