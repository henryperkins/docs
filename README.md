Certainly! Below is the **fully updated README** for your **Automated Documentation Generator** script. This version incorporates the latest enhancements, corrections, and structural changes based on your revised script and the suggestions provided.

---

# Automated Documentation Generator

## 📄 Overview

The **Automated Documentation Generator** is a Python script designed to streamline the process of generating and inserting documentation into your codebase. Leveraging OpenAI's GPT-4 API, the script automatically generates detailed docstrings and comments for various programming languages, enhancing code readability and maintainability.

## 🚀 Key Features

- **Multi-Language Support**: Supports Python, JavaScript, TypeScript, HTML, and CSS.
- **Asynchronous Processing**: Utilizes asynchronous programming for efficient handling of multiple files concurrently.
- **Robust Structure Extraction**: Employs AST (Abstract Syntax Tree) and Tree-sitter for accurate code structure analysis.
- **Enhanced Prompt Generation**: Considers existing docstrings/comments to preserve and enhance existing documentation.
- **Comprehensive Error Handling**: Implements retry mechanisms and detailed logging to handle API rate limits and other potential issues gracefully.
- **Backup Mechanism**: Creates backups of original files before making modifications to prevent data loss.
- **Configurable Exclusions**: Allows customization of excluded directories, files, and file types through a configuration file.
- **Output Compilation**: Generates a consolidated Markdown file (`output.md`) containing all the inserted documentation for easy reference.

---

## 📦 Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
    - [Command-Line Arguments](#command-line-arguments)
    - [Example Usage](#example-usage)
3. [Script Functionality](#script-functionality)
    - [1. Imports and Dependencies](#1-imports-and-dependencies)
    - [2. Configuration and Initialization](#2-configuration-and-initialization)
    - [3. Function Definitions](#3-function-definitions)
        - [a. Language Determination](#a-language-determination)
        - [b. File Type and Binary Checks](#b-file-type-and-binary-checks)
        - [c. Configuration Loading](#c-configuration-loading)
        - [d. File Collection](#d-file-collection)
        - [e. Python Code Validation](#e-python-code-validation)
        - [f. Structure Extraction](#f-structure-extraction)
            - [i. Python Structure Extraction](#i-python-structure-extraction)
            - [ii. JavaScript/TypeScript Structure Extraction](#ii-javascripttypescript-structure-extraction)
            - [iii. HTML Structure Extraction](#iii-html-structure-extraction)
            - [iv. CSS Structure Extraction](#iv-css-structure-extraction)
        - [g. Prompt Generation](#g-prompt-generation)
        - [h. Documentation Fetching](#h-documentation-fetching)
        - [i. Docstring and Comment Insertion](#i-docstring-and-comment-insertion)
            - [i. Python Docstrings](#i-python-docstrings)
            - [ii. JavaScript/TypeScript Docstrings](#ii-javascripttypescript-docstrings)
            - [iii. HTML Comments](#iii-html-comments)
            - [iv. CSS Comments](#iv-css-comments)
        - [j. File Processing](#j-file-processing)
        - [k. Batch Processing](#k-batch-processing)
    - [4. Main Execution Flow](#4-main-execution-flow)
4. [Environment Setup and Dependencies](#environment-setup-and-dependencies)
5. [Deployment and Execution Steps](#deployment-and-execution-steps)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)
8. [Functionality Overview](#functionality-overview)
9. [Final Recommendations](#final-recommendations)
10. [License](#license)

---

## Installation

### Prerequisites

- **Python Version**: Ensure you're using Python 3.9 or later.
- **Git**: Required for cloning Tree-sitter language repositories.

### Clone the Repository

```bash
git clone https://github.com/yourusername/docs-generator.git
cd docs-generator
```

### Set Up Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**

```plaintext
aiohttp
aiofiles
tqdm
python-dotenv
beautifulsoup4
tinycss2
tree-sitter
astor
```

### Build Tree-sitter Language Libraries

The script relies on Tree-sitter for parsing JavaScript and TypeScript. You need to build the language libraries before running the script.

1. **Clone Tree-sitter Repositories:**

    ```bash
    mkdir vendor
    cd vendor
    git clone https://github.com/tree-sitter/tree-sitter-javascript.git
    git clone https://github.com/tree-sitter/tree-sitter-typescript.git
    cd ..
    ```

2. **Build the Language Library:**

    ```bash
    python3 -c "
    from tree_sitter import Language
    Language.build_library(
        'build/my-languages.so',
        [
            'vendor/tree-sitter-javascript',
            'vendor/tree-sitter-typescript/typescript',
        ]
    )
    "
    ```

    This command compiles the Tree-sitter parsers for JavaScript and TypeScript into a shared library (`my-languages.so`) located in the `build` directory.

---

## Usage

### Command-Line Arguments

- **Essential Arguments**:
  - `repo_path`: Path to the local repository containing source code.
  - `-c` or `--config`: Path to a configuration JSON file for additional exclusions (default: `config.json`).
  - `--concurrency`: Number of concurrent API requests (default: `5`).
  - `-o` or `--output`: Path to the output Markdown file where documentation will be written (default: `output.md`).
  - `--model`: OpenAI model to use (default: `gpt-4`).
  - `--skip-types`: Comma-separated list of file extensions to skip (overrides default skip types).

### Example Usage

```bash
python3 docs.py /path/to/your/source/code -c config.json -o output.md --concurrency 10 --model gpt-4 --skip-types .json,.md
```

---

## Script Functionality

### 1. Imports and Dependencies

The script imports necessary modules for asynchronous operations, parsing, logging, and interacting with the OpenAI API.

### 2. Configuration and Initialization

- **Environment Variables**: Utilizes `python-dotenv` to load environment variables from a `.env` file, specifically the `OPENAI_API_KEY`.
- **Logging**: Configured to log detailed information and errors to `docs_generation.log`.
- **Tree-sitter Initialization**: Builds and loads Tree-sitter language parsers for JavaScript and TypeScript.

### 3. Function Definitions

#### a. Language Determination

```python
def get_language(ext: str) -> str:
    language_mapping = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.html': 'html',
        '.htm': 'html',
        '.css': 'css',
    }
    return language_mapping.get(ext.lower(), 'plaintext')
```

- **Purpose**: Determines the programming language based on the file extension.
- **Description**: Maps common file extensions to their corresponding programming languages.

#### b. File Type and Binary Checks

```python
def is_binary(file_path: str) -> bool:
    try:
        with open(file_path, 'rb') as file:
            chunk = file.read(1024)
            return b'\0' in chunk
    except Exception as e:
        logger.error(f"Error checking if file is binary '{file_path}': {e}")
        return True
```

- **Purpose**: Checks if a file is binary.
- **Description**: Reads a portion of the file in binary mode and checks for null bytes to determine if it's a binary file.

#### c. Configuration Loading

```python
def load_config(config_path: str, excluded_dirs: Set[str], excluded_files: Set[str], skip_types: Set[str]) -> None:
    try:
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
            additional_dirs = config.get('excluded_dirs', [])
            additional_files = config.get('excluded_files', [])
            additional_skip_types = config.get('skip_types', [])
            excluded_dirs.update(additional_dirs)
            excluded_files.update(additional_files)
            skip_types.update(additional_skip_types)
            logger.info(f"Loaded config from {config_path}.")
    except FileNotFoundError:
        logger.warning(f"Configuration file '{config_path}' not found. Using default exclusions.")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing configuration file '{config_path}': {e}")
    except Exception as e:
        logger.error(f"Error loading configuration file '{config_path}': {e}")
```

- **Purpose**: Loads additional exclusions from a configuration JSON file.
- **Description**: Updates the sets of excluded directories and files based on the provided configuration file.

#### d. File Collection

```python
def get_all_file_paths(repo_path: str, excluded_dirs: Set[str], excluded_files: Set[str]) -> List[str]:
    file_paths = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [
            d for d in dirs if d not in excluded_dirs and not d.startswith('.')]
        for file in files:
            if file in excluded_files or file.startswith('.'):
                continue
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    logger.info(f"Collected {len(file_paths)} files from '{repo_path}'.")
    return file_paths
```

- **Purpose**: Collects all file paths in the repository, excluding specified directories and files.
- **Description**: Walks through the directory tree and gathers all file paths while respecting the exclusions.

#### e. Python Code Validation

```python
def is_valid_python_code(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in Python code: {e}")
        return False
```

- **Purpose**: Validates the syntax of modified Python code.
- **Description**: Parses the Python code to check for syntax errors, ensuring that inserted docstrings do not break the code.

#### f. Structure Extraction

##### i. Python Structure Extraction

```python
def extract_python_structure(file_content: str) -> Optional[Dict]:
    try:
        tree = ast.parse(file_content)
        parent_map = {}

        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent_map[child] = node

        functions = []
        classes = []

        def get_node_source(node):
            try:
                return ast.unparse(node)
            except AttributeError:
                return astor.to_source(node).strip()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "args": [],
                    "returns": {"type": "Any"},
                    "decorators": [],
                    "docstring": ast.get_docstring(node) or ""
                }

                for arg in node.args.args:
                    arg_type = "Any"
                    if arg.annotation:
                        arg_type = get_node_source(arg.annotation)
                    func_info["args"].append({
                        "name": arg.arg,
                        "type": arg_type
                    })

                if node.returns:
                    func_info["returns"]["type"] = get_node_source(
                        node.returns)

                for decorator in node.decorator_list:
                    func_info["decorators"].append(get_node_source(decorator))

                parent = parent_map.get(node)
                if isinstance(parent, ast.ClassDef):
                    class_name = parent.name
                    class_obj = next(
                        (cls for cls in classes if cls["name"] == class_name), None)
                    if not class_obj:
                        class_obj = {
                            "name": class_name,
                            "bases": [get_node_source(base) for base in parent.bases],
                            "methods": [],
                            "decorators": [],
                            "docstring": ast.get_docstring(parent) or ""
                        }
                        for decorator in parent.decorator_list:
                            class_obj["decorators"].append(
                                get_node_source(decorator))
                        classes.append(class_obj)
                    class_obj["methods"].append(func_info)
                else:
                    functions.append(func_info)

            elif isinstance(node, ast.ClassDef):
                class_exists = any(cls["name"] == node.name for cls in classes)
                if not class_exists:
                    class_info = {
                        "name": node.name,
                        "bases": [get_node_source(base) for base in node.bases],
                        "methods": [],
                        "decorators": [],
                        "docstring": ast.get_docstring(node) or ""
                    }
                    for decorator in node.decorator_list:
                        class_info["decorators"].append(
                            get_node_source(decorator))
                    classes.append(class_info)

        return {
            "language": "python",
            "functions": functions,
            "classes": classes
        }
    except Exception as e:
        logger.error(f"Error parsing Python code: {e}")
        return None
```

- **Purpose**: Extracts the structure of Python code, including classes and functions, along with their docstrings and annotations.
- **Description**: Parses the Python file to extract classes, functions, their names, docstrings, arguments, return types, and decorators.

##### ii. JavaScript/TypeScript Structure Extraction

```python
def extract_js_ts_structure(file_content: str, language: str) -> Optional[Dict]:
    try:
        parser = Parser()
        if language == 'javascript':
            parser.set_language(JAVASCRIPT_LANGUAGE)
        elif language == 'typescript':
            parser.set_language(TYPESCRIPT_LANGUAGE)
        else:
            logger.error(f"Unsupported language: {language}")
            return None

        tree = parser.parse(bytes(file_content, "utf8"))
        root_node = tree.root_node

        functions = []
        classes = []

        code_bytes = file_content.encode('utf8')

        def extract_function_info(node, existing_docstring):
            func_name_node = node.child_by_field_name('name') or node.child_by_field_name('property')
            func_name = func_name_node.text.decode('utf8') if func_name_node else 'anonymousFunction'
            params = []
            params_node = node.child_by_field_name('parameters')
            if params_node:
                for param in params_node.named_children:
                    param_info = extract_parameter_info(param)
                    params.append(param_info)
            return_type = "Any"
            return_type_node = None
            for child in node.children:
                if child.type == 'type_annotation':
                    return_type_node = child
                    break
            if return_type_node:
                return_type = code_bytes[return_type_node.start_byte:return_type_node.end_byte].decode('utf8').lstrip(':').strip()
            func_info = {
                "name": func_name,
                "args": params,
                "returns": {"type": return_type},
                "docstring": "",
                "existing_docstring": existing_docstring,
                "start_byte": node.start_byte,
                "end_byte": node.end_byte
            }
            return func_info

        def extract_parameter_info(param_node):
            param_name = code_bytes[param_node.start_byte:param_node.end_byte].decode('utf8')
            param_type = "Any"
            type_annotation_node = None
            for child in param_node.children:
                if child.type == 'type_annotation':
                    type_annotation_node = child
                    break
            if type_annotation_node:
                param_type = code_bytes[type_annotation_node.start_byte:type_annotation_node.end_byte].decode('utf8').lstrip(':').strip()
            return {"name": param_name, "type": param_type}

        def traverse(node, parent_class=None):
            existing_docstring = extract_preceding_jsdoc(node)

            if node.type in ['function_declaration', 'method_definition']:
                func_info = extract_function_info(node, existing_docstring)
                if parent_class:
                    parent_class['methods'].append(func_info)
                else:
                    functions.append(func_info)
            elif node.type == 'class_declaration':
                class_name_node = node.child_by_field_name('name')
                class_name = class_name_node.text.decode('utf8') if class_name_node else 'AnonymousClass'
                class_info = {
                    "name": class_name,
                    "methods": [],
                    "docstring": "",
                    "existing_docstring": existing_docstring,
                    "start_byte": node.start_byte,
                    "end_byte": node.end_byte
                }
                classes.append(class_info)
                body = node.child_by_field_name('body')
                if body:
                    for child in body.named_children:
                        traverse(child, parent_class=class_info)
            else:
                for child in node.named_children:
                    traverse(child, parent_class=parent_class)

        def extract_preceding_jsdoc(node):
            preceding_comments = []
            cursor = node.walk()
            while cursor.goto_prev_sibling():
                sibling = cursor.node
                if sibling.type == 'comment':
                    comment_text = code_bytes[sibling.start_byte:sibling.end_byte].decode('utf8')
                    if comment_text.strip().startswith('/**'):
                        preceding_comments.insert(0, comment_text.strip())
                    else:
                        break
                elif sibling.is_named:
                    break
                else:
                    continue
            existing_docstring = '\n'.join(preceding_comments)
            return existing_docstring

        traverse(root_node)

        return {
            'language': language,
            'functions': functions,
            'classes': classes,
            'tree': tree,
            'source_code': file_content
        }

    except Exception as e:
        logger.error(f"Error parsing {language} code: {e}")
        return None
```

- **Purpose**: Extracts the structure of JavaScript or TypeScript code, including classes and functions, along with their docstrings and annotations.
- **Description**: Parses the JS/TS file using Tree-sitter to extract classes, functions, their names, docstrings, arguments, return types, and decorators.

##### iii. HTML Structure Extraction

```python
def extract_html_structure(file_content: str) -> Optional[Dict]:
    try:
        soup = BeautifulSoup(file_content, 'html.parser')
        elements = []

        def traverse(node):
            for child in node.children:
                if isinstance(child, str):
                    continue
                if child.name:
                    element_info = {
                        'tag': child.name,
                        'attributes': dict(child.attrs),
                        'text': child.get_text(strip=True),
                        'docstring': ''
                    }
                    elements.append(element_info)
                    traverse(child)

        traverse(soup)

        return {
            'language': 'html',
            'elements': elements
        }

    except Exception as e:
        logger.error(f"Error parsing HTML code: {e}")
        return None
```

- **Purpose**: Extracts the structure of HTML code, including tags, attributes, and text content.
- **Description**: Uses BeautifulSoup to parse the HTML file and gather information about each element for documentation purposes.

##### iv. CSS Structure Extraction

```python
def extract_css_structure(file_content: str) -> Optional[Dict]:
    try:
        rules = tinycss2.parse_stylesheet(file_content)
        style_rules = []

        for rule in rules:
            if rule.type == 'qualified-rule':
                prelude = tinycss2.serialize(rule.prelude).strip()
                content = tinycss2.serialize(rule.content).strip()
                rule_info = {
                    'selector': prelude,
                    'declarations': content,
                    'docstring': ''
                }
                style_rules.append(rule_info)

        return {
            'language': 'css',
            'rules': style_rules
        }

    except Exception as e:
        logger.error(f"Error parsing CSS code: {e}")
        return None
```

- **Purpose**: Extracts the structure of CSS code, including selectors and declarations.
- **Description**: Uses TinyCSS2 to parse the CSS file and gather information about each rule for documentation purposes.

#### g. Prompt Generation

```python
def generate_documentation_prompt(
    code_structure: Dict,
    project_info: Optional[str] = None,
    style_guidelines: Optional[str] = None
) -> str:
    """
    Generates a prompt for the OpenAI API to create documentation based on the code structure.

    Args:
        code_structure (Dict): The structured representation of the code.
        project_info (Optional[str]): Additional information about the project.
        style_guidelines (Optional[str]): Specific documentation style guidelines to follow.

    Returns:
        str: The prompt to send to the OpenAI API.
    """
    language = code_structure.get('language', 'code')
    json_structure = json.dumps(code_structure, indent=2)

    prompt_parts = [
        f"You are an expert {language} developer and technical writer.",
    ]

    if project_info:
        prompt_parts.append(f"The code belongs to a project that {project_info}.")

    if style_guidelines:
        prompt_parts.append(f"Please follow these documentation style guidelines: {style_guidelines}")

    # Include existing docstrings/comments to allow the AI to enhance them
    prompt_parts.append(
        f"""
Given the following {language} code structure in JSON format, generate detailed docstrings or comments for each function, method, class, element, or rule. Include descriptions of all parameters, return types, and any relevant details. Preserve and enhance existing documentation where applicable.

Code Structure:
{json_structure}

Please provide the updated docstrings or comments in the same JSON format, with the 'docstring' fields filled in.
"""
    )

    prompt = '\n'.join(prompt_parts)
    return prompt
```

- **Purpose**: Constructs a tailored prompt for the OpenAI API to generate or enhance documentation based on the extracted code structure. It optionally includes project-specific information and style guidelines to align the documentation with the project's needs.
- **Description**: Provides the code structure in JSON format, along with instructions to preserve and enhance existing documentation.

#### h. Documentation Fetching

```python
async def fetch_documentation(session: aiohttp.ClientSession, prompt: str, retry: int = 3) -> Optional[str]:
    """
    Fetches generated documentation from the OpenAI API based on the prompt.

    Args:
        session (aiohttp.ClientSession): The aiohttp session for making requests.
        prompt (str): The prompt to send to the API.
        retry (int): Number of retry attempts.

    Returns:
        Optional[str]: Generated documentation in JSON format if successful, None otherwise.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.2
    }

    for attempt in range(1, retry + 1):
        try:
            async with SEMAPHORE:
                async with session.post(
                    OPENAI_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=120
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Enhanced response validation
                        if 'choices' in data and len(data['choices']) > 0 and 'message' in data['choices'][0]:
                            documentation = data['choices'][0]['message']['content'].strip()
                            logger.info(f"Generated documentation:\n{documentation}")
                            return documentation
                        else:
                            logger.error("Unexpected API response structure.")
                            return None
                    elif response.status in {429, 500, 502, 503, 504}:
                        # Handle rate limiting and server errors with exponential backoff
                        error_text = await response.text()
                        logger.warning(
                            f"API rate limit or server error (status {response.status}). "
                            f"Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds. "
                            f"Response: {error_text}"
                        )
                        await asyncio.sleep(2 ** attempt)
                    else:
                        error_text = await response.text()
                        logger.error(f"API request failed with status {response.status}: {error_text}")
                        return None
        except asyncio.TimeoutError:
            logger.error(
                f"Request timed out during attempt {attempt}/{retry}. "
                f"Retrying in {2 ** attempt} seconds."
            )
            await asyncio.sleep(2 ** attempt)
        except aiohttp.ClientError as e:
            logger.error(
                f"Client error during API request: {e}. "
                f"Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds."
            )
            await asyncio.sleep(2 ** attempt)

    logger.error("Failed to generate documentation after multiple attempts.")
    return None
```

- **Purpose**: Fetches generated documentation from the OpenAI API based on the provided prompt.
- **Description**: Sends a prompt to the OpenAI API to generate documentation and implements retry logic with exponential backoff to handle transient issues like rate limits or server errors. Includes enhanced response validation to ensure the expected structure is received.

#### i. Docstring and Comment Insertion

##### i. Python Docstrings

```python
def insert_python_docstrings(file_content: str, docstrings: Dict) -> str:
    """
    Inserts docstrings into Python code using the AST.

    Args:
        file_content (str): Original Python source code.
        docstrings (Dict): Generated docstrings to insert.

    Returns:
        str: Modified Python source code with docstrings inserted.
    """
    try:
        tree = ast.parse(file_content)
        parent_map = {}

        # Implement parent tracking
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent_map[child] = node

        # Map function and class names to docstrings
        func_doc_map = {func['name']: func['docstring'] for func in docstrings.get('functions', [])}
        class_doc_map = {cls['name']: cls['docstring'] for cls in docstrings.get('classes', [])}
        method_doc_map = {}
        for cls in docstrings.get('classes', []):
            for method in cls.get('methods', []):
                method_doc_map[(cls['name'], method['name'])] = method['docstring']

        class DocstringInserter(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                parent = parent_map.get(node)
                if isinstance(parent, ast.ClassDef):
                    # It's a method
                    key = (parent.name, node.name)
                    docstring = method_doc_map.get(key)
                    if docstring:
                        if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                            # Insert docstring if not present
                            node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
                        else:
                            # Replace existing docstring
                            node.body[0] = ast.Expr(value=ast.Constant(value=docstring))
                else:
                    # It's a top-level function
                    docstring = func_doc_map.get(node.name)
                    if docstring:
                        if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                            node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
                        else:
                            node.body[0] = ast.Expr(value=ast.Constant(value=docstring))
                return node

            def visit_ClassDef(self, node):
                self.generic_visit(node)
                docstring = class_doc_map.get(node.name)
                if docstring:
                    if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                        node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
                    else:
                        node.body[0] = ast.Expr(value=ast.Constant(value=docstring))
                return node

        inserter = DocstringInserter()
        new_tree = inserter.visit(tree)
        new_code = astor.to_source(new_tree)

        # Validate the modified code
        try:
            ast.parse(new_code)
        except SyntaxError as e:
            logger.error(f"Syntax error in modified Python code: {e}")
            return file_content  # Return original content if there's an error

        return new_code
    except Exception as e:
        logger.error(f"Error inserting docstrings into Python code: {e}")
        return file_content  # Return original content if there's an error
```

- **Purpose**: Inserts or updates docstrings in Python code based on the generated documentation.
- **Description**: Parses the Python AST, maps existing functions and classes to their respective docstrings, and inserts or replaces docstrings accordingly. Ensures that the modified code remains syntactically correct.

##### ii. JavaScript/TypeScript Docstrings

```python
def insert_js_ts_docstrings(docstrings: Dict) -> str:
    """
    Inserts JSDoc comments into JavaScript/TypeScript code using tree-sitter.

    Args:
        docstrings (Dict): Generated docstrings to insert, including the AST and source code.

    Returns:
        str: Modified source code with docstrings inserted.
    """
    try:
        source_code = docstrings['source_code']
        tree = docstrings['tree']
        code_bytes = source_code.encode('utf8')

        inserts = []

        for func in docstrings.get('functions', []):
            start_byte = func['start_byte']
            docstring = func['docstring']
            if docstring:
                formatted_comment = format_jsdoc_comment(docstring)
                inserts.append((start_byte, formatted_comment))

        for cls in docstrings.get('classes', []):
            start_byte = cls['start_byte']
            docstring = cls['docstring']
            if docstring:
                formatted_comment = format_jsdoc_comment(docstring)
                inserts.append((start_byte, formatted_comment))
            for method in cls.get('methods', []):
                start_byte = method['start_byte']
                docstring = method['docstring']
                if docstring:
                    formatted_comment = format_jsdoc_comment(docstring)
                    inserts.append((start_byte, formatted_comment))

        # Sort inserts by position in descending order to avoid offset issues during insertion
        inserts.sort(key=lambda x: x[0], reverse=True)

        new_code = source_code

        for start_byte, comment in inserts:
            # Convert byte position to character position
            # Note: This assumes single-byte characters; for multi-byte, consider using a more robust method
            char_pos = start_byte
            # Insert the comment before the function or class declaration
            new_code = new_code[:char_pos] + comment + '\n' + new_code[char_pos:]

        return new_code

    except Exception as e:
        logger.error(f"Error inserting docstrings into JS/TS code: {e}")
        return docstrings.get('source_code', '')  # Return original content if there's an error
```

- **Purpose**: Inserts or updates JSDoc comments in JavaScript/TypeScript code based on the generated documentation.
- **Description**: Parses the JS/TS AST, maps existing functions and classes to their respective docstrings, and inserts or replaces JSDoc comments accordingly. Ensures that the modified code maintains correct formatting and syntax.

**Helper Function: `format_jsdoc_comment`**

```python
def format_jsdoc_comment(docstring: str) -> str:
    """
    Formats a docstring into a JSDoc comment block.

    Args:
        docstring (str): The docstring to format.

    Returns:
        str: The formatted JSDoc comment.
    """
    comment_lines = ['/**']
    for line in docstring.strip().split('\n'):
        comment_lines.append(f' * {line}')
    comment_lines.append(' */')
    return '\n'.join(comment_lines)
```

- **Purpose**: Formats a plain docstring into a JSDoc-compatible comment block.
- **Description**: Structures the docstring with the appropriate JSDoc syntax for better integration and readability.

##### iii. HTML Comments

```python
def insert_html_comments(file_content: str, docstrings: Dict) -> str:
    """
    Inserts comments into HTML code using BeautifulSoup.

    Args:
        file_content (str): Original HTML source code.
        docstrings (Dict): Generated docstrings to insert.

    Returns:
        str: Modified HTML source code with comments inserted.
    """
    try:
        soup = BeautifulSoup(file_content, 'html.parser')
        elements = docstrings.get('elements', [])

        element_map = {}
        for element in elements:
            key = (element['tag'], tuple(sorted(element['attributes'].items())), element['text'])
            element_map[key] = element['docstring']

        def traverse_and_insert(node):
            for child in node.children:
                if isinstance(child, str):
                    continue
                if child.name:
                    key = (child.name, tuple(sorted(child.attrs.items())), child.get_text(strip=True))
                    docstring = element_map.get(key)
                    if docstring:
                        # Check for existing comments to avoid duplication
                        existing_comments = child.find_all(string=lambda text: isinstance(text, Comment))
                        if not existing_comments:
                            comment = Comment(f" {docstring} ")
                            if child.name in ['head', 'body', 'html']:
                                # Insert comment inside the element
                                child.insert(0, comment)
                            else:
                                # Insert comment before the element
                                child.insert_before(comment)
                    traverse_and_insert(child)

        traverse_and_insert(soup)
        return str(soup)

    except Exception as e:
        logger.error(f"Error inserting comments into HTML code: {e}")
        return file_content
```

- **Purpose**: Inserts or updates HTML comments based on the generated documentation.
- **Description**: Parses the HTML using BeautifulSoup, maps elements to their respective docstrings, and inserts comments either before or inside the relevant HTML tags. Includes checks to prevent duplicate comments.

##### iv. CSS Comments

```python
def insert_css_comments(file_content: str, docstrings: Dict) -> str:
    """
    Inserts comments into CSS code.

    Args:
        file_content (str): Original CSS source code.
        docstrings (Dict): Generated docstrings to insert.

    Returns:
        str: Modified CSS source code with comments inserted.
    """
    try:
        rules = tinycss2.parse_stylesheet(file_content)
        style_rules = docstrings.get('rules', [])
        rule_map = {}
        for rule in style_rules:
            key = rule['selector']
            docstring = rule.get('docstring', '')
            if key in rule_map:
                rule_map[key] += f"\n{docstring}"
            else:
                rule_map[key] = docstring

        modified_content = ''
        inserted_selectors = set()

        for rule in rules:
            if rule.type == 'qualified-rule':
                selector = tinycss2.serialize(rule.prelude).strip()
                if selector not in inserted_selectors:
                    docstring = rule_map.get(selector)
                    if docstring:
                        modified_content += f"/* {docstring} */\n"
                    inserted_selectors.add(selector)
                modified_content += tinycss2.serialize(rule).strip() + '\n'
            else:
                modified_content += tinycss2.serialize(rule).strip() + '\n'

        return modified_content

    except Exception as e:
        logger.error(f"Error inserting comments into CSS code: {e}")
        return file_content
```

- **Purpose**: Inserts or updates CSS comments based on the generated documentation.
- **Description**: Parses the CSS using TinyCSS2, maps selectors to their respective docstrings, and inserts comments above the relevant CSS rules. Ensures that comments are not duplicated for selectors that appear multiple times.

#### j. File Processing

```python
async def process_file(session: aiohttp.ClientSession, file_path: str, skip_types: Set[str], output_file: str) -> None:
    """
    Processes a single file: reads content, generates documentation, inserts it, and writes outputs.

    Args:
        session (aiohttp.ClientSession): HTTP session.
        file_path (str): Path to the source file.
        skip_types (Set[str]): Set of file extensions to skip.
        output_file (str): Output Markdown file.
    """
    try:
        _, ext = os.path.splitext(file_path)
        logger.debug(f"Processing file: {file_path} with extension: {ext}")

        # Validate file extension
        if not is_valid_extension(ext, skip_types):
            logger.info(f"Skipping file '{file_path}' due to extension '{ext}'")
            return

        # Check if the file is binary
        if is_binary(file_path):
            logger.info(f"Skipping binary file '{file_path}'")
            return

        language = get_language(ext)

        if language == 'plaintext':
            logger.info(f"Skipping file '{file_path}' with unrecognized language.")
            return

        # Read file content directly
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
        except Exception as e:
            logger.error(f"Failed to read '{file_path}': {e}")
            return

        # Extract code structure
        if language == 'python':
            code_structure = extract_python_structure(content)
        elif language in ['javascript', 'typescript']:
            code_structure = extract_js_ts_structure(content, language)
        elif language == 'html':
            code_structure = extract_html_structure(content)
        elif language == 'css':
            code_structure = extract_css_structure(content)
        else:
            logger.warning(f"Language '{language}' not supported for structured extraction.")
            return

        if code_structure is None:
            logger.error(f"Failed to extract structure from '{file_path}'")
            return

        # Generate prompt with existing docstrings/comments considered
        prompt = generate_documentation_prompt(code_structure)

        # Generate documentation using OpenAI API
        documentation = await fetch_documentation(session, prompt)
        if not documentation:
            logger.error(f"Failed to generate documentation for '{file_path}'")
            return

        # Parse the generated documentation JSON
        try:
            updated_code_structure = json.loads(documentation)
            if language in ['javascript', 'typescript']:
                updated_code_structure['tree'] = code_structure['tree']
                updated_code_structure['source_code'] = code_structure['source_code']
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse documentation JSON for '{file_path}': {e}")
            return

        # Insert documentation into code
        if language == 'python':
            new_content = insert_python_docstrings(content, updated_code_structure)
        elif language in ['javascript', 'typescript']:
            new_content = insert_js_ts_docstrings(updated_code_structure)
        elif language == 'html':
            new_content = insert_html_comments(content, updated_code_structure)
        elif language == 'css':
            new_content = insert_css_comments(content, updated_code_structure)
        else:
            new_content = content  # Fallback to original content

        # Validate modified code if possible
        if language == 'python':
            if not is_valid_python_code(new_content):
                logger.error(f"Modified Python code is invalid. Aborting insertion for '{file_path}'")
                return

        # Backup and write the modified content back to the file
        try:
            backup_path = file_path + '.bak'
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.rename(file_path, backup_path)
            logger.info(f"Backup created at '{backup_path}'")

            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(new_content)
            logger.info(f"Inserted comments into '{file_path}'")
        except Exception as e:
            logger.error(f"Error writing to '{file_path}': {e}")
            # Restore from backup
            if os.path.exists(backup_path):
                os.remove(file_path)
                os.rename(backup_path, file_path)
                logger.info(f"Restored original file from backup for '{file_path}'")
            return

        # Append the documented code to the output Markdown file
        try:
            async with OUTPUT_LOCK:
                async with aiofiles.open(output_file, 'a', encoding='utf-8') as f:
                    header = f"# File: {file_path}\n\n"
                    code_block = f"```{language}\n{new_content}\n```\n\n"
                    await f.write(header)
                    await f.write(code_block)
            logger.info(f"Successfully processed and documented '{file_path}'")
        except Exception as e:
            logger.error(f"Error writing documentation for '{file_path}': {e}")

    except Exception as e:
        logger.error(f"Unexpected error processing '{file_path}': {e}")
        return
```

- **Purpose**: Processes a single file by generating enhanced documentation and writing the output to the specified Markdown file.
- **Description**:
  - **Validation**: Checks if the file should be skipped based on its extension or if it's binary.
  - **Reading**: Asynchronously reads the file content.
  - **Structure Extraction**: Extracts the code structure based on the programming language.
  - **Prompt Generation**: Creates a prompt for the OpenAI API, considering existing documentation.
  - **Documentation Fetching**: Retrieves generated documentation from the OpenAI API.
  - **Insertion**: Inserts the generated documentation into the source code.
  - **Validation**: Ensures the modified code is syntactically correct (for Python).
  - **Backup and Writing**: Creates a backup of the original file and writes the modified content.
  - **Output Compilation**: Appends the documented code to the output Markdown file.
  - **Error Handling**: Comprehensive error logging and handling to ensure robustness.

#### k. Batch Processing

```python
async def process_all_files(file_paths: List[str], skip_types: Set[str], output_file: str) -> None:
    """
    Processes all files asynchronously.

    Args:
        file_paths (List[str]): List of file paths to process.
        skip_types (Set[str]): Set of file extensions to skip.
        output_file (str): Output Markdown file.
    """
    global OUTPUT_LOCK
    OUTPUT_LOCK = asyncio.Lock()

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(process_file(session, file_path, skip_types, output_file))
            for file_path in file_paths
        ]

        # Use tqdm to display progress
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Files"):
            try:
                await f
            except Exception as e:
                logger.error(f"Error processing a file: {e}")
```

- **Purpose**: Manages the asynchronous processing of all collected files by creating and handling concurrent tasks.
- **Description**:
  - **Concurrency Control**: Utilizes asyncio's event loop and tasks to process multiple files simultaneously, improving efficiency.
  - **Progress Tracking**: Integrates `tqdm` to provide a real-time progress bar, offering visibility into the processing status.
  - **Resource Management**: Uses a single `aiohttp.ClientSession` for all HTTP requests, which is more efficient than creating multiple sessions.

### 4. Main Execution Flow

```python
def main() -> None:
    """
    The main function that orchestrates the documentation generation process.
    It parses command-line arguments, loads configurations, collects files, and initiates asynchronous processing.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Automatically generate and insert comments/docstrings into source files using OpenAI's GPT-4 API."
    )
    parser.add_argument(
        "repo_path",
        help="Path to the code repository"
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to config.json",
        default="config.json"
    )
    parser.add_argument(
        "--concurrency",
        help="Number of concurrent requests",
        type=int,
        default=5
    )
    parser.add_argument(
        "-o", "--output",
        help="Output Markdown file",
        default="output.md"
    )
    parser.add_argument(
        "--model",
        help="OpenAI model to use (default: gpt-4)",
        default="gpt-4"
    )
    parser.add_argument(
        "--skip-types",
        help="Comma-separated list of file extensions to skip",
        default=""
    )

    args = parser.parse_args()

    repo_path = args.repo_path
    config_path = args.config
    concurrency = args.concurrency
    output_file = args.output

    if not os.path.isdir(repo_path):
        logger.error(f"Invalid repository path: '{repo_path}' is not a directory.")
        sys.exit(1)

    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    excluded_files = set(DEFAULT_EXCLUDED_FILES)
    skip_types = set(DEFAULT_SKIP_TYPES)
    if args.skip_types:
        skip_types.update(args.skip_types.split(','))

    load_config(config_path, excluded_dirs, excluded_files, skip_types)

    file_paths = get_all_file_paths(repo_path, excluded_dirs, excluded_files)
    if not file_paths:
        logger.error("No files found to process.")
        sys.exit(1)

    logger.info(f"Starting documentation generation for {len(file_paths)} files.")

    # Ensure output file is empty or create it
    open(output_file, 'w').close()

    global SEMAPHORE
    SEMAPHORE = asyncio.Semaphore(concurrency)

    global MODEL_NAME
    MODEL_NAME = args.model  # Get the model name from arguments

    # Run the asynchronous processing
    try:
        asyncio.run(process_all_files(file_paths, skip_types, output_file))
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)

    logger.info("Documentation generation completed successfully.")
```

- **Purpose**: Serves as the entry point of the script, handling the orchestration of the documentation generation process.
- **Description**:
  - **Argument Parsing**: Utilizes `argparse` to handle command-line arguments for repository path, configuration file, concurrency level, output file, model selection, and file type exclusions.
  - **Validation**: Checks the validity of the repository path and ensures that there are files to process.
  - **Configuration Loading**: Loads additional exclusion rules from the specified configuration file.
  - **File Collection**: Gathers all relevant file paths to process.
  - **Output Preparation**: Clears or creates the output Markdown file to store the generated documentation.
  - **Concurrency Setup**: Initializes an asyncio semaphore based on the specified concurrency level to manage the number of simultaneous API requests.
  - **Processing Initiation**: Starts the asynchronous processing of all collected files.
  - **Error Handling**: Catches and logs any exceptions during the processing loop, ensuring that the script exits gracefully in case of critical failures.

---

## Environment Setup and Dependencies

### Python Version

Ensure you're using a compatible Python version (e.g., Python 3.9 or later).

### Dependencies

Install required packages using `pip`:

```bash
pip install -r requirements.txt
```

**Dependencies Breakdown**:

- `aiohttp`: For asynchronous HTTP requests to the OpenAI API.
- `aiofiles`: For asynchronous file operations.
- `tqdm`: For displaying progress bars during processing.
- `python-dotenv`: For loading environment variables from a `.env` file.
- `beautifulsoup4`: For parsing and manipulating HTML content.
- `tinycss2`: For parsing CSS files.
- `tree-sitter`: For parsing JavaScript and TypeScript code.
- `astor`: For converting Python AST back into source code.

### Environment Variables

#### OpenAI API Key

Store your OpenAI API key securely.

1. **Create a `.env` file in your project directory**:

    ```bash
    touch .env
    ```

2. **Add your API key to `.env`**:

    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```

3. **Ensure `.env` is included in `.gitignore`** to prevent accidental commits.

    **`.gitignore`:**

    ```plaintext
    .env
    build/
    *.bak
    *.log
    ```

---

## Deployment and Execution Steps

### Local Deployment

1. **Set Up Virtual Environment**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Build Tree-sitter Language Libraries**:

    Follow the instructions in the [Installation](#installation) section to clone and build the Tree-sitter language libraries.

4. **Run the Script**:

    ```bash
    python3 docs.py /path/to/your/source/code -c config.json -o output.md
    ```

### Cloud Server Deployment

1. **Provision a VM**: Use services like AWS EC2, Azure Virtual Machines, or DigitalOcean Droplets.

2. **SSH into the Server**:

    ```bash
    ssh username@your-server-ip
    ```

3. **Set Up Environment**: Repeat the local deployment steps on the server.

4. **Run the Script**: Execute the script as you would locally.

### Docker Containerization

1. **Create a `Dockerfile`**: Define the environment and dependencies.

    **`Dockerfile`:**

    ```dockerfile
    FROM python:3.9-slim

    # Set environment variables
    ENV PYTHONDONTWRITEBYTECODE 1
    ENV PYTHONUNBUFFERED 1

    # Set work directory
    WORKDIR /app

    # Install
