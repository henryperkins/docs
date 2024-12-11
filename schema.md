# Module: schema

## Overview
**File:** `simple2/schema.py`
**Description:** """
Schema Definitions for Docstring Components

This module defines data structures for representing various components of a docstring,
such as parameters, return values, exceptions, notes, and examples. It uses Python's
TypedDict and Enum to enforce structure and type safety.

Version: 1.0.0
Author: Development Team
"""

import time
from typing import TypedDict, List, Optional, Union
from enum import Enum
import json
from pathlib import Path


class DocstringParameter(TypedDict):
    """
    Represents a parameter in a function's docstring.

    Attributes:
        name (str): The name of the parameter.
        type (str): The type of the parameter.
        description (str): A description of the parameter.
        optional (bool): Indicates if the parameter is optional.
        default_value (Optional[str]): The default value of the parameter, if any.
    """
    name: str
    type: str
    description: str
    optional: bool
    default_value: Optional[str]


class DocstringReturns(TypedDict):
    """
    Represents the return value section of a function's docstring.

    Attributes:
        type (str): The type of the return value.
        description (str): A description of the return value.
    """
    type: str
    description: str


class DocstringException(TypedDict):
    """
    Represents an exception that a function may raise, as documented in its docstring.

    Attributes:
        exception (str): The name of the exception.
        description (str): A description of the circumstances under which the exception is raised.
    """
    exception: str
    description: str


class NoteType(Enum):
    """
    Enum for categorizing types of notes in a docstring.

    Attributes:
        NOTE (str): A general note.
        WARNING (str): A warning note.
        TIP (str): A tip or suggestion.
        IMPORTANT (str): An important note.
    """
    NOTE = "note"
    WARNING = "warning"
    TIP = "tip"
    IMPORTANT = "important"


class DocstringNote(TypedDict):
    """
    Represents a note in a docstring.

    Attributes:
        type (NoteType): The type of note (e.g., NOTE, WARNING).
        content (str): The content of the note.
    """
    type: NoteType
    content: str


class DocstringExample(TypedDict):
    """
    Represents an example in a docstring.

    Attributes:
        code (str): The example code snippet.
        description (Optional[str]): A description of what the example demonstrates.
    """
    code: str
    description: Optional[str]


class DocstringMetadata(TypedDict):
    """
    Represents metadata associated with a docstring.

    Attributes:
        author (Optional[str]): The author of the code.
        since_version (Optional[str]): The version since the code is available.
        deprecated (Optional[dict]): Information about deprecation, if applicable.
        complexity (Optional[dict]): Information about the complexity of the code.
    """
    author: Optional[str]
    since_version: Optional[str]
    deprecated: Optional[dict]
    complexity: Optional[dict]


class DocstringSchema(TypedDict):
    """
    Represents the overall schema of a docstring, including various components.

    Attributes:
        description (str): A description of the function or class.
        parameters (List[DocstringParameter]): A list of parameters.
        returns (DocstringReturns): Information about the return value.
        raises (Optional[List[DocstringException]]): A list of exceptions that may be raised.
        examples (Optional[List[DocstringExample]]): A list of examples demonstrating usage.
        notes (Optional[List[DocstringNote]]): A list of notes.
        metadata (Optional[DocstringMetadata]): Additional metadata about the docstring.
    """
    description: str
    parameters: List[DocstringParameter]
    returns: DocstringReturns
    raises: Optional[List[DocstringException]]
    examples: Optional[List[DocstringExample]]
    notes: Optional[List[DocstringNote]]
    metadata: Optional[DocstringMetadata]


def load_schema() -> dict:
    """
    Load the JSON schema for docstrings from a file.

    Returns:
        dict: The loaded JSON schema.

    Raises:
        FileNotFoundError: If the schema file does not exist.
        json.JSONDecodeError: If the schema file contains invalid JSON.
    """
    schema_path = Path(__file__).parent / 'docstring_schema.json'
    with open(schema_path) as f:
        return json.load(f)


try:
    JSON_SCHEMA = load_schema()
except Exception as e:
    print(f"Warning: Could not load JSON schema: {e}")
    JSON_SCHEMA = {}


## AI-Generated Documentation


**Summary:** No summary provided.


**Description:** No description provided.



## Classes

| Class | Inherits From | Complexity Score* |
|-------|---------------|-------------------|
| `DocstringParameter` | `TypedDict` | 0 |
| `DocstringReturns` | `TypedDict` | 0 |
| `DocstringException` | `TypedDict` | 0 |
| `NoteType` | `Enum` | 0 |
| `DocstringNote` | `TypedDict` | 0 |
| `DocstringExample` | `TypedDict` | 0 |
| `DocstringMetadata` | `TypedDict` | 0 |
| `DocstringSchema` | `TypedDict` | 0 |

### Class Methods

| Class | Method | Parameters | Returns | Complexity Score* |
|-------|--------|------------|---------|-------------------|

## Source Code
```python
"""
Schema Definitions for Docstring Components

This module defines data structures for representing various components of a docstring,
such as parameters, return values, exceptions, notes, and examples. It uses Python's
TypedDict and Enum to enforce structure and type safety.

Version: 1.0.0
Author: Development Team
"""

import time
from typing import TypedDict, List, Optional, Union
from enum import Enum
import json
from pathlib import Path


class DocstringParameter(TypedDict):
    """
    Represents a parameter in a function's docstring.

    Attributes:
        name (str): The name of the parameter.
        type (str): The type of the parameter.
        description (str): A description of the parameter.
        optional (bool): Indicates if the parameter is optional.
        default_value (Optional[str]): The default value of the parameter, if any.
    """
    name: str
    type: str
    description: str
    optional: bool
    default_value: Optional[str]


class DocstringReturns(TypedDict):
    """
    Represents the return value section of a function's docstring.

    Attributes:
        type (str): The type of the return value.
        description (str): A description of the return value.
    """
    type: str
    description: str


class DocstringException(TypedDict):
    """
    Represents an exception that a function may raise, as documented in its docstring.

    Attributes:
        exception (str): The name of the exception.
        description (str): A description of the circumstances under which the exception is raised.
    """
    exception: str
    description: str


class NoteType(Enum):
    """
    Enum for categorizing types of notes in a docstring.

    Attributes:
        NOTE (str): A general note.
        WARNING (str): A warning note.
        TIP (str): A tip or suggestion.
        IMPORTANT (str): An important note.
    """
    NOTE = "note"
    WARNING = "warning"
    TIP = "tip"
    IMPORTANT = "important"


class DocstringNote(TypedDict):
    """
    Represents a note in a docstring.

    Attributes:
        type (NoteType): The type of note (e.g., NOTE, WARNING).
        content (str): The content of the note.
    """
    type: NoteType
    content: str


class DocstringExample(TypedDict):
    """
    Represents an example in a docstring.

    Attributes:
        code (str): The example code snippet.
        description (Optional[str]): A description of what the example demonstrates.
    """
    code: str
    description: Optional[str]


class DocstringMetadata(TypedDict):
    """
    Represents metadata associated with a docstring.

    Attributes:
        author (Optional[str]): The author of the code.
        since_version (Optional[str]): The version since the code is available.
        deprecated (Optional[dict]): Information about deprecation, if applicable.
        complexity (Optional[dict]): Information about the complexity of the code.
    """
    author: Optional[str]
    since_version: Optional[str]
    deprecated: Optional[dict]
    complexity: Optional[dict]


class DocstringSchema(TypedDict):
    """
    Represents the overall schema of a docstring, including various components.

    Attributes:
        description (str): A description of the function or class.
        parameters (List[DocstringParameter]): A list of parameters.
        returns (DocstringReturns): Information about the return value.
        raises (Optional[List[DocstringException]]): A list of exceptions that may be raised.
        examples (Optional[List[DocstringExample]]): A list of examples demonstrating usage.
        notes (Optional[List[DocstringNote]]): A list of notes.
        metadata (Optional[DocstringMetadata]): Additional metadata about the docstring.
    """
    description: str
    parameters: List[DocstringParameter]
    returns: DocstringReturns
    raises: Optional[List[DocstringException]]
    examples: Optional[List[DocstringExample]]
    notes: Optional[List[DocstringNote]]
    metadata: Optional[DocstringMetadata]


def load_schema() -> dict:
    """
    Load the JSON schema for docstrings from a file.

    Returns:
        dict: The loaded JSON schema.

    Raises:
        FileNotFoundError: If the schema file does not exist.
        json.JSONDecodeError: If the schema file contains invalid JSON.
    """
    schema_path = Path(__file__).parent / 'docstring_schema.json'
    with open(schema_path) as f:
        return json.load(f)


try:
    JSON_SCHEMA = load_schema()
except Exception as e:
    print(f"Warning: Could not load JSON schema: {e}")
    JSON_SCHEMA = {}

```