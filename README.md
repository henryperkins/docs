# DocScribe: AI-Powered Multi-Language Code Documentation Generator

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Supported Languages](#supported-languages)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Configuration File](#configuration-file)
  - [Examples](#examples)
- [Configuration and Customization](#configuration-and-customization)
- [Documentation Report](#documentation-report)
- [Error Handling and Logging](#error-handling-and-logging)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Introduction

**DocScribe** is an advanced, cross-language code documentation tool designed to automate the generation of high-quality, consistent, and comprehensive documentation for codebases written in multiple programming languages. Leveraging the power of OpenAI's GPT-4 API, DocScribe analyzes your code's structure and generates human-readable, informative documentation that adheres to best practices and style guidelines.

By automating the documentation process, DocScribe helps developers maintain up-to-date documentation, improve code readability, facilitate onboarding, and enhance overall code quality.
DocScribe now supports both OpenAI and Azure OpenAI, giving you flexibility in choosing your AI provider.

## Features

- **Multi-Language Support**: Automatically document Python, JavaScript, TypeScript, HTML, and CSS codebases.
- **AI-Powered Generation**: Utilizes OpenAI's GPT-4 for generating detailed and context-aware documentation.
- **Language-Specific Parsers**: Uses appropriate parsing tools for accurate code analysis.
- **Direct Code Modification**: Inserts generated documentation directly into source files, with backups.
- **Customizable Configuration**: Flexible settings via `config.json` and command-line arguments.
- **Comprehensive Reporting**: Generates a Markdown report summarizing all documentation added.
- **Robust Logging and Error Handling**: Detailed logs and exception handling for smooth operation.
- **Azure OpenAI Support**: Option to use Azure OpenAI services instead of regular OpenAI API.

## Supported Languages

- **Python**
- **JavaScript / TypeScript**
- **HTML**
- **CSS**

Support for additional languages can be added via custom parsers and extensions.

## Installation

### Prerequisites

- **Python 3.9** or higher
- **Node.js** (v12 or higher)
- **OpenAI API key** or **Azure OpenAI credentials**

### Clone the Repository

```bash
git clone https://github.com/yourusername/docscribe.git
cd docscribe
```

### Install Python Dependencies

Create and activate a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Install Node.js Dependencies

Navigate to the `scripts` directory and install Node.js dependencies:

```bash
cd scripts
npm install
cd ..
```

### Set Up Environment Variables

Create a `.env` file in the root directory and add your OpenAI API key or Azure OpenAI credentials:

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
echo "AZURE_OPENAI_KEY=your_azure_key_here" >> .env
echo "AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here" >> .env
```

Alternatively, set the `OPENAI_API_KEY` environment variable in your system.

## Usage

### Command-Line Arguments

Run the script using:

```bash
python main.py /path/to/your/repository [options]
```

**Options:**

- `-c`, `--config`: Path to `config.json` file (default: `config.json`).
- `--concurrency`: Number of concurrent API requests (default: 5).
- `-o`, `--output`: Output Markdown file for the documentation report (default: `output.md`).
- `--model`: OpenAI model to use (default: `gpt-4`).
- `--skip-types`: Comma-separated list of file extensions to skip.
- `--project-info`: Project-specific information to include (overrides `config.json`).
- `--style-guidelines`: Documentation style guidelines to follow (overrides `config.json`).
- `--safe-mode`: Run in safe mode without modifying files.
- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
- `--schema`: Path to the function schema JSON file (default: `function_schema.json`).
- `--use-azure`: Use Azure OpenAI instead of regular OpenAI API.

### Configuration File

DocScribe can be configured using a `config.json` file. This allows you to set default parameters and options without specifying them every time via command-line arguments.

**Example `config.json`:**

```json
{
  "project_info": "This project is designed to...",
  "style_guidelines": "Please adhere to the PEP 8 style guide.",
  "excluded_dirs": ["tests", "docs"],
  "excluded_files": ["setup.py"],
  "skip_types": [".json", ".md"],
  "function_schema_path": "function_schema.json"
}
```

### Examples

**Basic Usage:**

```bash
python main.py /path/to/your/repository
```

**Using Custom Configuration:**

```bash
python main.py /path/to/your/repository -c custom_config.json
```

**Running in Safe Mode (No Files Modified):**

```bash
python main.py /path/to/your/repository --safe-mode
```

**Specifying OpenAI Model and Concurrency:**

```bash
python main.py /path/to/your/repository --model gpt-4 --concurrency 10
```

**Using Azure OpenAI with a specific model deployment:**

This example uses Azure OpenAI with the `gpt4o` model deployment and a custom configuration file:

```bash
python main.py /path/to/your/repository --use-azure --model gpt4o -c config.json
```

## Configuration and Customization

- **API Credentials**: Set via `.env` file or environment variables (`OPENAI_API_KEY` or `AZURE_OPENAI_KEY` and `AZURE_OPENAI_ENDPOINT`).
- **Project Information**: Provide context to the AI for better documentation.
- **Style Guidelines**: Customize the style and format of the generated documentation.
- **Excluded Directories and Files**: Specify paths to ignore during traversal.
- **Supported File Extensions**: Extend or modify the file types to include.
- **Concurrency Level**: Adjust based on your network and API rate limits.
- **Output File**: Define the name and location of the documentation report.
**Azure OpenAI Configuration:**
- Set `AZURE_OPENAI_KEY` and `AZURE_OPENAI_ENDPOINT` in your `.env` file or as environment variables.
- Use the `--use-azure` flag to enable Azure OpenAI.
- Specify the Azure OpenAI model deployment name using the `--model` argument (e.g., `gpt4o`).


## Documentation Report

After execution, DocScribe generates a Markdown report summarizing the documentation added to each file. The report includes:

- **Table of Contents**: Auto-generated for easy navigation.
- **File Summaries**: High-level overviews of each file's purpose.
- **Changes Made**: Lists any modifications or additions.
- **Detailed Documentation**: Function and class docstrings, parameter explanations, etc.
- **Code Blocks**: Snippets showing the updated code with inserted documentation.

**Example Output:**

```markdown
# Documentation Generation Report

## Table of Contents

- [File: src/main.py](#file-srcmainpy)
  - [Summary](#summary)
  - [Changes Made](#changes-made)
  - [Functions](#functions)
  - [Classes](#classes)

...

## File: src/main.py

### Summary

This file contains the main execution logic of the application, handling user input and orchestrating the core functions.

### Changes Made

- Added docstrings to all functions and classes.
- Updated function `process_data` to include parameter descriptions.

### Functions

| Function     | Arguments       | Description                         | Async |
|--------------|-----------------|-------------------------------------|-------|
| `process_data` | `data, options` | Processes input data according to specified options. | No    |

### Classes

#### Class: `DataProcessor`

A class responsible for processing data and generating output.

| Method       | Arguments     | Description                         | Async | Type     |
|--------------|---------------|-------------------------------------|-------|----------|
| `run`        | `self`        | Executes the data processing routine.| No    | function |

...
```

## Error Handling and Logging

- **Logging**: All activities are logged to `docs_generation.log` and console output.
- **Log Levels**: Can be adjusted via the `--log-level` argument.
- **Error Handling**: The script gracefully handles errors, logging them without stopping the entire process.
- **Retries**: Implements retry logic for transient errors, such as network timeouts.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue on GitHub.

**Steps to Contribute:**

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **OpenAI** for providing the GPT-4 model.
- **Microsoft Azure** for providing Azure OpenAI services.
- **Contributors** who have helped improve this project.
- **Community** for feedback and support.

---

**Contact Information:**

For questions or support, please contact [hperkin4@asu.edu](mailto:hperkin4@asu.edu).
