# Documentation Generation Tool

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT%204-blue.svg)

![DocuScribe Logo](https://github.com/henryperkins/docs/blob/26fc21625be86c63726fd3608a13b9687520214a/DALL%C2%B7E%202024-10-05%2001.00.45%20-%20A%20high-resolution%2C%20modern%2C%20abstract%20logo%20representing%20an%20AI%20documentation%20generation%20tool%20called%20'DocuScribe'.%20Focus%20on%20depicting%20complex%20synapses%2C%20ne.webp?raw=true)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Example Commands](#example-commands)
- [Function Schema](#function-schema)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

The **Documentation Generation Tool** is a powerful Python-based utility designed to automatically generate comprehensive documentation for your code repositories. Leveraging the capabilities of OpenAI's GPT-4 or Azure OpenAI services, this tool analyzes your codebase and produces detailed documentation, including summaries, function docstrings, class descriptions, and more.

## Features

- **Automated Documentation**: Generate overviews, summaries, and detailed docstrings for functions and classes.
- **Language Support**: Supports multiple programming languages including Python, JavaScript, TypeScript, HTML, CSS, and more.
- **Customizable Configuration**: Exclude specific directories, files, or file types from processing.
- **Code Formatting and Cleanup**: Integrates tools like Black, Flake8, and Autoflake to ensure code quality.
- **Concurrency Control**: Manage the number of concurrent API requests for efficient processing.
- **Azure OpenAI Integration**: Seamlessly switch between OpenAI and Azure OpenAI services.
- **Comprehensive Logging**: Detailed logs for monitoring and troubleshooting.

## Prerequisites

- **Python**: Version 3.8 or higher.
- **Node.js**: Required for running auxiliary Node.js scripts.
- **API Keys**:
  - **OpenAI API Key**: For accessing OpenAI's GPT-4 services.
  - **Azure OpenAI Key and Endpoint**: If using Azure OpenAI services.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/documentation-generation-tool.git
   cd documentation-generation-tool
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Node.js Dependencies**

   Ensure you have Node.js installed. Then, install any required Node.js packages.

   ```bash
   cd scripts
   npm install
   cd ..
   ```

5. **Install External Tools**

   The script relies on external tools like `black`, `flake8`, and `autoflake`. Install them using `pip`:

   ```bash
   pip install black flake8 autoflake
   ```

   Ensure that these tools are accessible in your system's PATH.

## Configuration

1. **Environment Variables**

   Create a `.env` file in the project's root directory to store your API keys and other configurations securely.

   ```bash
   touch .env
   ```

   **Example `.env` File:**

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   AZURE_OPENAI_KEY=your_azure_openai_key_here
   AZURE_OPENAI_ENDPOINT=https://your-azure-openai-endpoint.azure.com/
   DEPLOYMENT_NAME=gpt4o
   ```

   **Notes:**
   - **`OPENAI_API_KEY`**: Required if using OpenAI's GPT-4.
   - **`AZURE_OPENAI_KEY`** and **`AZURE_OPENAI_ENDPOINT`**: Required if using Azure OpenAI services.
   - **`DEPLOYMENT_NAME`**: The name of your Azure OpenAI deployment.

2. **Configuration File (`config.json`)**

   Optionally, you can create a `config.json` file to specify additional configurations such as excluded directories, files, and style guidelines.

   **Example `config.json`:**

   ```json
   {
     "project_info": "This project automates the generation of documentation for code repositories.",
     "style_guidelines": "Follow PEP8 for Python code and Airbnb style guide for JavaScript.",
     "excluded_dirs": [".git", "__pycache__", "node_modules", ".venv", ".idea"],
     "excluded_files": [".DS_Store"],
     "skip_types": [".json", ".md", ".txt", ".csv", ".lock"]
   }
   ```

## Usage

The tool is executed via the command line with various arguments to customize its behavior.

### Command-Line Arguments

| Argument            | Description                                                   | Default                                             |
|---------------------|---------------------------------------------------------------|-----------------------------------------------------|
| `repo_path`         | **(Required)** Path to the code repository.                   | N/A                                                 |
| `-c`, `--config`    | Path to `config.json` for additional configurations.          | `config.json`                                       |
| `--concurrency`     | Number of concurrent API requests.                           | `5`                                                 |
| `-o`, `--output`    | Output Markdown file for the generated documentation.         | `output.md`                                         |
| `--model`           | OpenAI model to use (e.g., `gpt-4`, `gpt-4o`).                | `gpt-4`                                             |
| `--skip-types`      | Comma-separated list of file extensions to skip (e.g., `.exe,.bin`). | `""`                                         |
| `--project-info`    | Additional information about the project.                     | `""`                                                |
| `--style-guidelines`| Documentation style guidelines to follow.                      | `""`                                                |
| `--safe-mode`       | Run in safe mode (no files will be modified).                  | `False`                                             |
| `--log-level`       | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). | `INFO`                                          |
| `--schema`          | Path to `function_schema.json`.                                | `schemas/function_schema.json`                      |
| `--use-azure`       | Use Azure OpenAI instead of regular OpenAI API.                | `False`                                             |

### Example Commands

1. **Basic Usage with OpenAI GPT-4**

   ```bash
   python main.py /path/to/repo --model gpt-4
   ```

2. **Using Azure OpenAI**

   ```bash
   python main.py /path/to/repo --model gpt-4o --use-azure
   ```

3. **Specifying Custom Configuration and Schema Paths**

   ```bash
   python main.py /path/to/repo --config /path/to/config.json --schema /path/to/schemas/custom_function_schema.json
   ```

4. **Running in Safe Mode with Increased Concurrency**

   ```bash
   python main.py /path/to/repo --concurrency 10 --safe-mode
   ```

5. **Excluding Specific File Types and Setting Log Level to DEBUG**

   ```bash
   python main.py /path/to/repo --skip-types .exe,.bin --log-level DEBUG
   ```

## Function Schema

The `function_schema.json` file defines the structure of the functions that the OpenAI API will use for generating documentation. Ensure that this schema aligns with OpenAI's expectations, primarily that it is a **list** of function definitions.

**Example `function_schema.json`:**

```json
[
  {
    "name": "generate_documentation",
    "description": "Generates comprehensive documentation for the provided code structure.",
    "parameters": {
      "type": "object",
      "properties": {
        "summary": {
          "type": "string",
          "description": "A detailed summary of the file."
        },
        "changes_made": {
          "type": "array",
          "items": { "type": "string" },
          "description": "List of changes made to the file."
        },
        "functions": {
          "type": "array",
          "items": { "$ref": "#/definitions/function" },
          "description": "List of documented functions."
        },
        "classes": {
          "type": "array",
          "items": { "$ref": "#/definitions/class" },
          "description": "List of documented classes."
        }
      },
      "required": ["summary"],
      "definitions": {
        "function": {
          "type": "object",
          "properties": {
            "name": { "type": "string" },
            "docstring": { "type": "string" },
            "args": {
              "type": "array",
              "items": { "type": "string" }
            },
            "async": { "type": "boolean" }
          },
          "required": ["name", "docstring", "args", "async"]
        },
        "class": {
          "type": "object",
          "properties": {
            "name": { "type": "string" },
            "docstring": { "type": "string" },
            "methods": {
              "type": "array",
              "items": { "$ref": "#/definitions/method" }
            }
          },
          "required": ["name", "docstring", "methods"]
        },
        "method": {
          "type": "object",
          "properties": {
            "name": { "type": "string" },
            "docstring": { "type": "string" },
            "args": {
              "type": "array",
              "items": { "type": "string" }
            },
            "async": { "type": "boolean" },
            "type": { "type": "string" }
          },
          "required": ["name", "docstring", "args", "async", "type"]
        }
      }
    }
  }
]
```

**Key Points:**

- **List Format**: The schema is a list containing function definitions.
- **Function Definition**: Each function has a `name`, `description`, and `parameters`.
- **Parameters**: Define the expected input for the function, including properties and required fields.
- **Definitions**: Reusable components (`function`, `class`, `method`) defined to structure the documentation output.

## Logging

The tool provides comprehensive logging to help monitor its operation and troubleshoot issues.

- **Log File**: `documentation_generation.log` located in the project's root directory.
- **Log Levels**:
  - **DEBUG**: Detailed information, typically of interest only when diagnosing problems.
  - **INFO**: Confirmation that things are working as expected.
  - **WARNING**: An indication that something unexpected happened.
  - **ERROR**: Due to a more serious problem, the software has not been able to perform some function.
  - **CRITICAL**: A serious error, indicating that the program itself may be unable to continue running.

**Configuring Log Level:**

Use the `--log-level` argument to set the desired logging level.

```bash
python main.py /path/to/repo --log-level DEBUG
```

## Troubleshooting

### Common Issues

1. **`function_schema.json` Not Found**

   - **Error Message:**
     ```
     [CRITICAL] __main__: Failed to load function schema from '/path/to/schemas/function_schema.json'. Exiting.
     ```
   - **Solution:**
     - Ensure that `function_schema.json` exists in the specified `schemas` directory.
     - Verify the path using the `--schema` argument.
     - Check file permissions to ensure the script can read the file.

2. **AzureOpenAI Initialization Error**

   - **Error Message:**
     ```
     [CRITICAL] __main__: An unexpected error occurred: AzureOpenAI.__init__() got an unexpected keyword argument 'api_base'
     ```
   - **Solution:**
     - Verify that you are using the correct version of the OpenAI Python library that supports the `AzureOpenAI` class with the `api_base` parameter.
     - Update the OpenAI library:
       ```bash
       pip install --upgrade openai
       ```
     - Check the `AzureOpenAI` class documentation to ensure correct usage.

3. **Missing External Tools (`autoflake`, `flake8`, `black`)**

   - **Error Messages:**
     ```
     [ERROR] __main__: Autoflake is not installed. Please install it using 'pip install autoflake'.
     ```
   - **Solution:**
     - Install the missing tools:
       ```bash
       pip install autoflake flake8 black
       ```
     - Ensure they are accessible in your system's PATH.

4. **Node.js Scripts Not Found**

   - **Error Message:**
     ```
     [ERROR] __main__: Node.js script /path/to/script.js not found.
     ```
   - **Solution:**
     - Ensure Node.js is installed:
       ```bash
       node -v
       ```
     - Install required Node.js packages:
       ```bash
       cd scripts
       npm install
       cd ..
       ```
     - Verify the path to the Node.js scripts.

### General Tips

- **Check Logs**: Review the `documentation_generation.log` file for detailed error messages and stack traces.
- **Environment Variables**: Ensure all required environment variables are set correctly in the `.env` file.
- **Dependencies**: Confirm that all Python and Node.js dependencies are installed.
- **Permissions**: Verify that the script has the necessary permissions to read and write files in the specified directories.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your feature description"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Open a Pull Request**

   Describe your changes and provide context for reviewers.

## License

This project is licensed under the [MIT License](LICENSE).

---

*Developed with ❤️ by [Your Name](https://github.com/yourusername)*
