# AI-Driven Code Documentation Generator

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT%204-blue.svg)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/yourusername/yourrepo/ci.yml?branch=main)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Languages](#supported-languages)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

AI-Driven Code Documentation Generator is a powerful tool that automates the creation of comprehensive, structured documentation for your codebase. Leveraging OpenAI's GPT-4 (or Azure OpenAI) and advanced code parsing techniques, this tool extracts code structures, generates detailed docstrings following Google-style conventions, and inserts them back into your code. It supports multiple programming languages, ensuring your projects are well-documented with minimal effort.

![Demo](https://your-repo-url/demo.gif)

## Features

- **Multi-Language Support**: Automatically generates documentation for Python, JavaScript, TypeScript, Go, C++, Java, HTML, and CSS.
- **Structured Documentation**: Produces JSON-based documentation conforming to a predefined schema for consistency and ease of use.
- **Google-Style Docstrings**: Inserts detailed, Google-style docstrings into your codebase.
- **Asynchronous Processing**: Efficiently handles large codebases with asynchronous file processing.
- **Customizable Prompts**: Tailor AI prompts to fit your project's specific documentation needs.
- **Comprehensive Logging**: Detailed logs with log rotation to monitor the documentation process.
- **Configuration Flexibility**: Customize settings via `config.json` and environment variables.
- **Documentation Report**: Generates a Markdown report with all generated documentation and a dynamically generated Table of Contents.

## Supported Languages

- **Python**
- **JavaScript**
- **TypeScript**
- **Go**
- **C++**
- **Java**
- **HTML**
- **CSS**

## Installation

### Prerequisites

- **Python 3.9+**
- **Node.js (for JavaScript/TypeScript handlers)**
- **OpenAI API Key** or **Azure OpenAI Credentials**
- **Git** (for cloning the repository)

### Clone the Repository

```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
```

### Set Up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Install Node.js Dependencies

Navigate to the `scripts` directory and install dependencies:

```bash
cd scripts
npm install
cd ..
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory and add the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
ENDPOINT_URL=your_azure_endpoint_url
DEPLOYMENT_NAME=your_azure_deployment_name

# Other Configurations
MODEL_NAME=gpt-4  # or your specific deployment ID for Azure
```

### `config.json`

Create a `config.json` file to specify additional settings:

```json
{
  "project_info": "This project is an AI-driven documentation generator that automates the creation of comprehensive docstrings.",
  "style_guidelines": "Follow Google Python Style Guide for docstrings.",
  "excluded_dirs": ["tests", "docs"],
  "excluded_files": ["setup.py", "manage.py"],
  "skip_types": [".json", ".md", ".txt", ".csv", ".lock"]
}
```

## Usage

### Running the Documentation Generator

Use the following command to run the documentation generator:

```bash
python3 main.py /path/to/your/project -c config.json --use-azure --concurrency 3 -o documentation_report.md
```

### Command-Line Arguments

- `/path/to/your/project`: Path to the root directory of your codebase.
- `-c config.json`: Path to the configuration file.
- `--use-azure`: (Optional) Flag to use Azure OpenAI instead of OpenAI.
- `--concurrency 3`: (Optional) Number of concurrent API requests.
- `-o documentation_report.md`: (Optional) Output file for the documentation report.

### Example

```bash
python3 main.py ./my_project -c config.json --use-azure --concurrency 5 -o docs_output.md
```

## How It Works

1. **Configuration**: The tool reads configurations from `config.json` and environment variables.
2. **File Collection**: It traverses the specified project directory, excluding directories and files as configured.
3. **Code Structure Extraction**: For each supported file, it extracts the code structure (modules, classes, functions, variables, constants) using language-specific handlers.
4. **Documentation Generation**: Sends the extracted structure to the AI model with a tailored prompt to generate structured documentation conforming to `function_schema.json`.
5. **Docstring Insertion**: Inserts the generated Google-style docstrings back into the source code.
6. **Validation**: Validates the modified code using syntax checks and tools like `flake8`.
7. **Reporting**: Compiles a Markdown report with all generated documentation and a Table of Contents.

## How to Customize

### Updating the Function Schema

The `function_schema.json` defines the structure of the documentation. You can modify this schema to include additional fields or adjust existing ones to fit your project's needs.

### Changing Docstring Styles

Currently, the tool inserts Google-style docstrings. To switch to another style (e.g., NumPy or reStructuredText), modify the `format_function_docstring` method in `utils.py` accordingly.

### Adding Support for More Languages

To add support for additional programming languages:

1. **Create a Language Handler**: Implement a new handler class in `language_functions/` that inherits from `BaseHandler`.
2. **Implement `extract_structure` and `insert_docstrings`**: Ensure these methods conform to the `function_schema.json`.
3. **Update `LANGUAGE_MAPPING`**: Add the new language's file extensions to the `LANGUAGE_MAPPING` dictionary in `utils.py`.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add YourFeature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions, suggestions, or support, please contact [your.email@example.com](mailto:your.email@example.com).

---

*Enhance your codebase documentation effortlessly with AI-driven precision.*

