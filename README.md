I don't have the capability to directly commit changes to the repository. You will need to manually update the `README.md` file with the provided content.

Here is the updated content for your `README.md` file:

```markdown
# Code Documentation Generator

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Scripts Overview](#scripts-overview)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)

## Overview

The **Code Documentation Generator** is a comprehensive tool designed to automate the creation of detailed documentation for various codebases. Leveraging OpenAI's GPT-4 API, this tool analyzes code structures—including functions, classes, API endpoints, HTML elements, and CSS rules—and generates well-structured documentation based on predefined guidelines. It supports multiple programming languages such as Python, JavaScript, TypeScript, HTML, and CSS.

## Features

- **Automated Documentation Generation:** Automatically generates summaries, docstrings, and comments for codebases.
- **Multi-Language Support:** Supports Python, JavaScript, TypeScript, HTML, and CSS.
- **Configurable Exclusions:** Easily exclude directories, files, or specific file types from documentation generation.
- **Concurrency Control:** Manage the number of concurrent API requests to optimize performance.
- **Safe Mode:** Generate documentation without modifying original files.
- **Comprehensive Logging:** Detailed logs for monitoring and troubleshooting.
- **Backup Mechanism:** Automatically creates backups before modifying files.

## Architecture

The tool consists of several interconnected scripts and modules:

- **`main.py`**: The entry point for executing the documentation generation process.
- **`language_handlers.py`**: Handles extraction and insertion of documentation for different programming languages.
- **`utils.py`**: Provides utility functions for file operations, configuration loading, and prompt generation.
- **JavaScript Utilities**:
  - **`extract_structure.js`**: Extracts structural information from JavaScript/TypeScript files.
  - **`insert_docstrings.js`**: Inserts JSDoc comments into JavaScript/TypeScript files.

## Installation

### Prerequisites

- **Python 3.8+**
- **Node.js** (required for processing JavaScript and TypeScript files)
- **OpenAI API Key**

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/henryperkins/docs.git
   cd docs
   ```

2. **Set Up Python Environment**

   It's recommended to use a virtual environment.

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Node.js Dependencies**

   Navigate to the `docs` directory and install necessary Node.js packages.

   ```bash
   cd docs
   npm install typescript @types/node
   cd ..
   ```

5. **Configure Environment Variables**

   Create a `.env` file in the root directory and add your OpenAI API key:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

The primary script to run is `main.py`. It orchestrates the entire documentation generation process.

### Command-Line Arguments

```bash
python main.py <repo_path> [options]
```

### Options

- `-c`, `--config`: Path to `config.json` (default: `config.json`)
- `--concurrency`: Number of concurrent API requests (default: `5`)
- `-o`, `--output`: Output Markdown file (default: `output.md`)
- `--model`: OpenAI model to use (default: `gpt-4`)
- `--skip-types`: Comma-separated list of file extensions to skip (default: `''`)
- `--project-info`: Information about the project
- `--style-guidelines`: Documentation style guidelines to follow
- `--safe-mode`: Run in safe mode (no files will be modified)

### Example

```bash
python main.py /path/to/your/codebase \
    --config config.json \
    --concurrency 10 \
    --output documentation.md \
    --model gpt-4o-mini \
    --skip-types .json,.md \
    --project-info "handles user authentication and data processing" \
    --style-guidelines "Follow Google Python Style Guide" \
    --safe-mode
```

## Configuration

The `config.json` file allows you to specify various settings for the documentation generation process.

### Sample `config.json`

```json
{
  "excluded_dirs": [".git", "__pycache__", "node_modules", ".venv", ".idea"],
  "excluded_files": [".DS_Store"],
  "skip_types": [".json", ".md", ".txt", ".csv", ".lock"],
  "project_info": "handles user authentication and data processing",
  "style_guidelines": "Follow Google Python Style Guide"
}
```

## Scripts Overview

### 1. `main.py`

**Summary:**

Orchestrates the documentation generation process by analyzing code components and generating detailed documentation using OpenAI's GPT-4 API.

**Key Features:**

- Parses command-line arguments.
- Loads configuration settings.
- Collects all relevant file paths.
- Initiates asynchronous processing of files.
- Generates an output Markdown file with summaries, changes, and code blocks.

### 2. `language_handlers.py`

**Summary:**

Provides functionality for extracting and inserting documentation into various code formats, including Python, JavaScript, TypeScript, HTML, and CSS.

**Key Features:**

- **Python**: Extracts structure using AST and inserts docstrings.
- **JavaScript/TypeScript**: Utilizes Node.js scripts to extract structure and insert JSDoc comments.
- **HTML**: Parses HTML to extract elements and insert comments.
- **CSS**: Extracts CSS rules and inserts comments accordingly.

### 3. `utils.py`

**Summary:**

Contains utility functions for handling file operations, loading configurations, and generating documentation prompts.

**Key Features:**

- Determines programming language based on file extension.
- Checks if a file is binary.
- Loads and updates configuration settings.
- Retrieves all file paths while respecting exclusions.
- Generates prompts for the AI model based on code structure.

### 4. `extract_structure.js`

**Summary:**

Analyzes and extracts structural information from JavaScript/TypeScript code files.

**Key Features:**

- Traverses AST nodes to extract details about functions and classes.
- Outputs the structure in JSON format.

### 5. `insert_docstrings.js`

**Summary:**

Processes documentation data to insert JSDoc comments into JavaScript/TypeScript code.

**Key Features:**

- Formats JSDoc comments based on provided documentation.
- Inserts comments at appropriate positions in the code.

### Recent Changes

**utils.py:**
- Added new imports: `subprocess`, `tempfile`, `astor`, `BeautifulSoup`, `tinycss2`.
- Updated functions to utilize these imports.
- Enhanced functionality for extracting and inserting Python docstrings.

**language_functions.py:**
- Added `import os # Added import`
- Added imports: `fnmatch`, `json`, `subprocess`, `logging`, `traceback`, `from typing import Optional`
- Added functions for extracting and inserting Python docstrings.

**file_handlers.py:**
- Modified functions for handling JS/TS file processing and inserting docstrings.
- Added `async def insert_docstrings_for_file(js_ts_file, documentation_file):` for processing JS/TS files.
- Updated `fetch_documentation` and `fetch_summary` functions to improve error handling and logging.

## Logging

All operations are logged to `docs_generation.log` with detailed information about each step, including errors and processing statuses. This log file is crucial for monitoring the documentation generation process and troubleshooting any issues that arise.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add some feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Open a Pull Request**

Please ensure your code follows the project's coding standards and includes relevant tests.

## License

This project is licensed under the [MIT License](LICENSE).
