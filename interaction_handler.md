# Module: interaction_handler

## Overview
**File:** `simple2/interaction_handler.py`
**Description:** import asyncio
import hashlib
import os
import time
import json
from typing import Dict, Tuple, Optional, List
import ast
from dotenv import load_dotenv
from api.models.api_client import AzureOpenAIClient
from core.config import AzureOpenAIConfig
from docs import DocStringManager
from core.cache import Cache
from core.logger import log_info, log_error, log_debug, log_warning, log_exception
from core.monitoring import SystemMonitor
from extract.extraction_manager import ExtractionManager
from docstring_utils import (
    analyze_code_element_docstring,
    parse_docstring,
    DocstringValidator,
    parse_and_validate_docstring
)
from api.response_parser import ResponseParser
from core.utils import handle_exceptions
from core.metrics import Metrics

# Load environment variables from .env file
load_dotenv()


class InteractionHandler:
    """
    Manages the orchestration of docstring generation with integrated caching,
    monitoring, and error handling. This class processes functions and classes
    in batches and interacts with the Azure OpenAI API to generate documentation.
    """

    def __init__(
        self,
        client: Optional[AzureOpenAIClient] = None,
        config: Optional[AzureOpenAIConfig] = None,
        cache_config: Optional[Dict] = None,
        batch_size: int = 5,
    ):
        """
        Initialize the InteractionHandler with necessary components.

        Args:
            client (Optional[AzureOpenAIClient]): The Azure OpenAI client instance.
            config (Optional[AzureOpenAIConfig]): Configuration for Azure OpenAI.
            cache_config (Optional[Dict]): Configuration for the cache.
            batch_size (int): Number of functions to process concurrently.
        """
        if client is None:
            if not config:
                raise ValueError(
                    "Azure OpenAI configuration must be provided if client is not supplied."
                )
            self.client = AzureOpenAIClient(config=config)
        else:
            self.client = client

        self.config = config
        self.cache = Cache(**(cache_config or {}))
        self.monitor = SystemMonitor()
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(batch_size)
        self.extraction_manager = ExtractionManager()
        self.response_parser = ResponseParser()
        self.validator = DocstringValidator()
        self.metrics = Metrics()
        log_info("Interaction Handler initialized with batch processing capability")

    @handle_exceptions(log_error)
    async def process_all_functions(
        self, source_code: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Process all functions and classes in the source code with batch processing and rate limiting.

        Args:
            source_code (str): The complete source code to process.

        Returns:
            Tuple[Optional[str], Optional[str]]: Updated source code and documentation.
        """
        log_debug("Starting batch processing of all functions and classes.")

        if not source_code:
            log_error("Source code is missing.")
            return None, None

        self.docstring_manager = DocStringManager(source_code)

        metadata = self.extraction_manager.extract_metadata(source_code)
        functions = metadata["functions"]
        classes = metadata.get("classes", [])

        log_info(
            f"Extracted {len(functions)} functions and {len(classes)} classes from source code."
        )

        function_results = []
        for i in range(0, len(functions), self.batch_size):
            batch = functions[i: i + self.batch_size]
            batch_tasks = [
                self.process_function(source_code, func_info) for func_info in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            function_results.extend(batch_results)

        class_results = []
        for class_info in classes:
            class_result = await self.process_class(source_code, class_info)
            if class_result:
                class_results.append(class_result)

        documentation_entries = []

        for function_info, (docstring, metadata) in zip(functions, function_results):
            if docstring:
                self.docstring_manager.insert_docstring(
                    function_info["node"], docstring)
                if metadata:
                    documentation_entries.append(
                        {
                            "function_name": function_info["name"],
                            "complexity_score": metadata.get("complexity_score", 0),
                            "docstring": docstring,
                            "summary": metadata.get("summary", ""),
                            "changelog": metadata.get("changelog", ""),
                        }
                    )

        for class_info, (docstring, metadata) in zip(classes, class_results):
            if docstring:
                self.docstring_manager.insert_docstring(
                    class_info["node"], docstring)
                if metadata:
                    documentation_entries.append(
                        {
                            "class_name": class_info["name"],
                            "docstring": docstring,
                            "summary": metadata.get("summary", ""),
                            "methods": class_info.get("methods", []),
                        }
                    )

        updated_code = self.docstring_manager.update_source_code(
            documentation_entries)
        documentation = self.docstring_manager.generate_markdown_documentation(
            documentation_entries)

        total_items = len(functions) + len(classes)
        self.monitor.log_batch_completion(total_items)
        log_info("Batch processing completed successfully.")

        return updated_code, documentation

    @handle_exceptions(log_error)
    async def process_function(
        self, source_code: str, function_info: Dict
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Process a single function with enhanced error handling and monitoring.

        Args:
            source_code (str): The source code containing the function.
            function_info (Dict): Metadata about the function to process.

        Returns:
            Tuple[Optional[str], Optional[Dict]]: The generated docstring and metadata, or None if failed.
        """
        func_name = function_info.get("name", "unknown")
        start_time = time.time()

        try:
            if not function_info or 'name' not in function_info:
                log_error("Invalid function info provided")
                return None, None
            if 'node' not in function_info:
                log_error(f"Missing AST node for function: {func_name}")
                return None, None
            if 'returns' not in function_info:
                function_info['returns'] = 'None'

            cache_key = self._generate_cache_key(function_info['node'])
            cached_response = await self.cache.get_cached_docstring(cache_key)
            if cached_response:
                is_valid, validation_errors = self.validator.validate_docstring(
                    cached_response)
                if is_valid:
                    log_info(f"Using valid cached docstring for {func_name}")
                    return cached_response["docstring"], cached_response
                else:
                    log_warning(
                        f"Invalid cached docstring for {func_name}, errors: {validation_errors}"
                    )
                    await self.cache.invalidate_by_tags([cache_key])

            max_retries = self.config.max_retries if self.config else 3
            for attempt in range(max_retries):
                try:
                    response = await self.client.generate_docstring(
                        func_name=func_name,
                        params=function_info["args"],
                        return_type=function_info["returns"],
                        complexity_score=function_info.get(
                            "complexity_score", 0),
                        existing_docstring=function_info["docstring"],
                        decorators=function_info["decorators"],
                        exceptions=function_info.get("exceptions", []),
                    )

                    if response and response.get("content"):
                        is_valid, validation_errors = self.validator.validate_docstring(
                            response["content"]
                        )

                        if is_valid:
                            await self.cache.save_docstring(
                                cache_key,
                                response["content"],
                                tags=[f"func:{func_name}"]
                            )
                            return response["content"]["docstring"], response["content"]
                        else:
                            log_warning(
                                f"Generated docstring validation failed for {func_name}: "
                                f"{validation_errors}"
                            )

                except Exception as e:
                    log_exception(
                        f"Error generating docstring (attempt {attempt + 1}): {e}")

            log_error(f"Failed to generate valid docstring for {func_name}")
            return None, None

        except Exception as e:
            log_exception(f"Error processing function {func_name}: {e}")
            return None, None

    @handle_exceptions(log_error)
    async def process_class(
        self, source_code: str, class_info: Dict
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Process a single class with enhanced error handling and monitoring.

        Args:
            source_code (str): The source code containing the class.
            class_info (Dict): Metadata about the class to process.

        Returns:
            Tuple[Optional[str], Optional[Dict]]: The generated docstring and metadata, or None if failed.
        """
        class_name = class_info.get("name", "unknown")
        start_time = time.time()

        try:
            response = await self.client.generate_docstring(
                func_name=class_name,
                params=[],
                return_type="None",
                complexity_score=0,
                existing_docstring=class_info.get("docstring", ""),
                decorators=class_info.get("decorators", []),
                exceptions=[],
            )

            if response and response.get("content"):
                return response["content"].get("docstring"), response["content"]
            return None, None

        except Exception as e:
            log_exception(f"Error processing class {class_name}: {e}")
            return None, None

    def _generate_cache_key(self, function_node: Optional[ast.FunctionDef]) -> str:
        """
        Generate a unique cache key for a function.

        Args:
            function_node (Optional[ast.FunctionDef]): The function node to generate a key for.

        Returns:
            str: The generated cache key.
        """
        if not function_node or not isinstance(function_node, ast.FunctionDef):
            return hashlib.md5(str(time.time()).encode()).hexdigest()

        try:
            func_signature = f"{function_node.name}({', '.join(arg.arg for arg in function_node.args.args)})"
            return hashlib.md5(func_signature.encode()).hexdigest()
        except Exception as e:
            log_exception(f"Error generating cache key: {e}")
            return hashlib.md5(str(time.time()).encode()).hexdigest()


## AI-Generated Documentation


**Summary:** No summary provided.


**Description:** No description provided.



## Classes

| Class | Inherits From | Complexity Score* |
|-------|---------------|-------------------|
| `InteractionHandler` | `` | 0 |

### Class Methods

| Class | Method | Parameters | Returns | Complexity Score* |
|-------|--------|------------|---------|-------------------|
| `InteractionHandler` | `process_all_functions` | `(self: Any, source_code: str)` | `Tuple[(Optional[str], Optional[str])]` | 0 |
| `InteractionHandler` | `process_function` | `(self: Any, source_code: str, function_info: Dict)` | `Tuple[(Optional[str], Optional[Dict])]` | 0 |
| `InteractionHandler` | `process_class` | `(self: Any, source_code: str, class_info: Dict)` | `Tuple[(Optional[str], Optional[Dict])]` | 0 |

## Source Code
```python
import asyncio
import hashlib
import os
import time
import json
from typing import Dict, Tuple, Optional, List
import ast
from dotenv import load_dotenv
from api.models.api_client import AzureOpenAIClient
from core.config import AzureOpenAIConfig
from docs import DocStringManager
from core.cache import Cache
from core.logger import log_info, log_error, log_debug, log_warning, log_exception
from core.monitoring import SystemMonitor
from extract.extraction_manager import ExtractionManager
from docstring_utils import (
    analyze_code_element_docstring,
    parse_docstring,
    DocstringValidator,
    parse_and_validate_docstring
)
from api.response_parser import ResponseParser
from core.utils import handle_exceptions
from core.metrics import Metrics

# Load environment variables from .env file
load_dotenv()


class InteractionHandler:
    """
    Manages the orchestration of docstring generation with integrated caching,
    monitoring, and error handling. This class processes functions and classes
    in batches and interacts with the Azure OpenAI API to generate documentation.
    """

    def __init__(
        self,
        client: Optional[AzureOpenAIClient] = None,
        config: Optional[AzureOpenAIConfig] = None,
        cache_config: Optional[Dict] = None,
        batch_size: int = 5,
    ):
        """
        Initialize the InteractionHandler with necessary components.

        Args:
            client (Optional[AzureOpenAIClient]): The Azure OpenAI client instance.
            config (Optional[AzureOpenAIConfig]): Configuration for Azure OpenAI.
            cache_config (Optional[Dict]): Configuration for the cache.
            batch_size (int): Number of functions to process concurrently.
        """
        if client is None:
            if not config:
                raise ValueError(
                    "Azure OpenAI configuration must be provided if client is not supplied."
                )
            self.client = AzureOpenAIClient(config=config)
        else:
            self.client = client

        self.config = config
        self.cache = Cache(**(cache_config or {}))
        self.monitor = SystemMonitor()
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(batch_size)
        self.extraction_manager = ExtractionManager()
        self.response_parser = ResponseParser()
        self.validator = DocstringValidator()
        self.metrics = Metrics()
        log_info("Interaction Handler initialized with batch processing capability")

    @handle_exceptions(log_error)
    async def process_all_functions(
        self, source_code: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Process all functions and classes in the source code with batch processing and rate limiting.

        Args:
            source_code (str): The complete source code to process.

        Returns:
            Tuple[Optional[str], Optional[str]]: Updated source code and documentation.
        """
        log_debug("Starting batch processing of all functions and classes.")

        if not source_code:
            log_error("Source code is missing.")
            return None, None

        self.docstring_manager = DocStringManager(source_code)

        metadata = self.extraction_manager.extract_metadata(source_code)
        functions = metadata["functions"]
        classes = metadata.get("classes", [])

        log_info(
            f"Extracted {len(functions)} functions and {len(classes)} classes from source code."
        )

        function_results = []
        for i in range(0, len(functions), self.batch_size):
            batch = functions[i: i + self.batch_size]
            batch_tasks = [
                self.process_function(source_code, func_info) for func_info in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            function_results.extend(batch_results)

        class_results = []
        for class_info in classes:
            class_result = await self.process_class(source_code, class_info)
            if class_result:
                class_results.append(class_result)

        documentation_entries = []

        for function_info, (docstring, metadata) in zip(functions, function_results):
            if docstring:
                self.docstring_manager.insert_docstring(
                    function_info["node"], docstring)
                if metadata:
                    documentation_entries.append(
                        {
                            "function_name": function_info["name"],
                            "complexity_score": metadata.get("complexity_score", 0),
                            "docstring": docstring,
                            "summary": metadata.get("summary", ""),
                            "changelog": metadata.get("changelog", ""),
                        }
                    )

        for class_info, (docstring, metadata) in zip(classes, class_results):
            if docstring:
                self.docstring_manager.insert_docstring(
                    class_info["node"], docstring)
                if metadata:
                    documentation_entries.append(
                        {
                            "class_name": class_info["name"],
                            "docstring": docstring,
                            "summary": metadata.get("summary", ""),
                            "methods": class_info.get("methods", []),
                        }
                    )

        updated_code = self.docstring_manager.update_source_code(
            documentation_entries)
        documentation = self.docstring_manager.generate_markdown_documentation(
            documentation_entries)

        total_items = len(functions) + len(classes)
        self.monitor.log_batch_completion(total_items)
        log_info("Batch processing completed successfully.")

        return updated_code, documentation

    @handle_exceptions(log_error)
    async def process_function(
        self, source_code: str, function_info: Dict
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Process a single function with enhanced error handling and monitoring.

        Args:
            source_code (str): The source code containing the function.
            function_info (Dict): Metadata about the function to process.

        Returns:
            Tuple[Optional[str], Optional[Dict]]: The generated docstring and metadata, or None if failed.
        """
        func_name = function_info.get("name", "unknown")
        start_time = time.time()

        try:
            if not function_info or 'name' not in function_info:
                log_error("Invalid function info provided")
                return None, None
            if 'node' not in function_info:
                log_error(f"Missing AST node for function: {func_name}")
                return None, None
            if 'returns' not in function_info:
                function_info['returns'] = 'None'

            cache_key = self._generate_cache_key(function_info['node'])
            cached_response = await self.cache.get_cached_docstring(cache_key)
            if cached_response:
                is_valid, validation_errors = self.validator.validate_docstring(
                    cached_response)
                if is_valid:
                    log_info(f"Using valid cached docstring for {func_name}")
                    return cached_response["docstring"], cached_response
                else:
                    log_warning(
                        f"Invalid cached docstring for {func_name}, errors: {validation_errors}"
                    )
                    await self.cache.invalidate_by_tags([cache_key])

            max_retries = self.config.max_retries if self.config else 3
            for attempt in range(max_retries):
                try:
                    response = await self.client.generate_docstring(
                        func_name=func_name,
                        params=function_info["args"],
                        return_type=function_info["returns"],
                        complexity_score=function_info.get(
                            "complexity_score", 0),
                        existing_docstring=function_info["docstring"],
                        decorators=function_info["decorators"],
                        exceptions=function_info.get("exceptions", []),
                    )

                    if response and response.get("content"):
                        is_valid, validation_errors = self.validator.validate_docstring(
                            response["content"]
                        )

                        if is_valid:
                            await self.cache.save_docstring(
                                cache_key,
                                response["content"],
                                tags=[f"func:{func_name}"]
                            )
                            return response["content"]["docstring"], response["content"]
                        else:
                            log_warning(
                                f"Generated docstring validation failed for {func_name}: "
                                f"{validation_errors}"
                            )

                except Exception as e:
                    log_exception(
                        f"Error generating docstring (attempt {attempt + 1}): {e}")

            log_error(f"Failed to generate valid docstring for {func_name}")
            return None, None

        except Exception as e:
            log_exception(f"Error processing function {func_name}: {e}")
            return None, None

    @handle_exceptions(log_error)
    async def process_class(
        self, source_code: str, class_info: Dict
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Process a single class with enhanced error handling and monitoring.

        Args:
            source_code (str): The source code containing the class.
            class_info (Dict): Metadata about the class to process.

        Returns:
            Tuple[Optional[str], Optional[Dict]]: The generated docstring and metadata, or None if failed.
        """
        class_name = class_info.get("name", "unknown")
        start_time = time.time()

        try:
            response = await self.client.generate_docstring(
                func_name=class_name,
                params=[],
                return_type="None",
                complexity_score=0,
                existing_docstring=class_info.get("docstring", ""),
                decorators=class_info.get("decorators", []),
                exceptions=[],
            )

            if response and response.get("content"):
                return response["content"].get("docstring"), response["content"]
            return None, None

        except Exception as e:
            log_exception(f"Error processing class {class_name}: {e}")
            return None, None

    def _generate_cache_key(self, function_node: Optional[ast.FunctionDef]) -> str:
        """
        Generate a unique cache key for a function.

        Args:
            function_node (Optional[ast.FunctionDef]): The function node to generate a key for.

        Returns:
            str: The generated cache key.
        """
        if not function_node or not isinstance(function_node, ast.FunctionDef):
            return hashlib.md5(str(time.time()).encode()).hexdigest()

        try:
            func_signature = f"{function_node.name}({', '.join(arg.arg for arg in function_node.args.args)})"
            return hashlib.md5(func_signature.encode()).hexdigest()
        except Exception as e:
            log_exception(f"Error generating cache key: {e}")
            return hashlib.md5(str(time.time()).encode()).hexdigest()

```