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
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The AI-Driven Code Documentation Generator automates the creation of structured documentation for codebases. Utilizing OpenAI's GPT-4 (or Azure OpenAI) and advanced parsing techniques, it extracts code structures and generates insightful docstrings. This tool enhances maintainability and collaboration for projects of any size.

![Demo](https://github.com/yourusername/yourrepo/raw/main/demo.gif)

## Features

- **Multi-Language Support**: Automatically generates documentation for Python, JavaScript, TypeScript, Go, C++, Java, HTML, and CSS.
    
- **Structured Documentation**: Produces JSON-based documentation conforming to a predefined schema for consistency and ease of use.
    
- **Google-Style Docstrings**: Inserts detailed, Google-style docstrings into your codebase.
    
- **Asynchronous Processing**: Efficiently handles large codebases with asynchronous file processing.
    
- **Customizable Prompts**: Tailor AI prompts to fit your project's specific documentation needs.
    
- **Comprehensive Logging**: Detailed logs with log rotation to monitor the documentation process.
    
- **Configuration Flexibility**: Customize settings via `config.json` and environment variables.
    
- **Documentation Report**: Generates a Markdown report with all generated documentation and a dynamically generated Table of Contents.
    
- **Unified Badge Function**: Generates badges for Cyclomatic Complexity, Halstead Metrics (Volume, Difficulty, Effort), and Maintainability Index.
    
- **Compact Badge Style**: Uses `flat-square` style for badges.
    
- **Dynamic Thresholds**: Customizable thresholds for different metrics.
    
- **Enhanced `generate_documentation_report`**: Integrates badge generation into the report.
    
- **Markdown Structure**: Uses tables for readability.
    
- **Environment-Based Thresholds**: Fetches thresholds from environment variables.
    
- **Asynchronous File Writing**: Uses `aiofiles` for non-blocking operations.
    
- **Comprehensive Scoring**: Calculates maximum complexity across functions and methods.
    
- **Robust Error Management**: Handles missing Halstead metrics and logs errors.
    
- **Flexibility and Customization**: Dynamic thresholds and compact badge styles.

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
- **Node.js** (for JavaScript/TypeScript handlers)
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

### Install Dependencies

#### Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Install Node.js Dependencies

Navigate to the `scripts` directory and install dependencies:

```bash
cd scripts
npm install
cd ..
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
ENDPOINT_URL=your_azure_endpoint_url
DEPLOYMENT_NAME=your_azure_deployment_name

# Other Configurations
MODEL_NAME=gpt-4
```

### `config.json`

Create a `config.json` file:

```json
{
  "project_info": "This project is an AI-driven documentation generator.",
  "style_guidelines": "Follow Google Python Style Guide.",
  "excluded_dirs": ["tests", "docs"],
  "excluded_files": ["setup.py", "manage.py"],
  "skip_types": [".json", ".md", ".txt", ".csv", ".lock"]
}
```

## Usage

### Running the Documentation Generator

```bash
python3 main.py /path/to/your/project -c config.json --use-azure --concurrency 3 -o documentation_report.md
```

### Command-Line Arguments

- `/path/to/your/project`: Path to the codebase.
- `-c config.json`: Path to the configuration file.
- `--use-azure`: Use Azure OpenAI.
- `--concurrency 3`: Number of concurrent API requests.
- `-o documentation_report.md`: Output file for the report.

### Example

```bash
python3 main.py ./my_project -c config.json --use-azure --concurrency 5 -o docs_output.md
```

## How It Works

1. **Configuration**: Reads from `config.json` and environment variables.
2. **File Collection**: Traverses the project directory.
3. **Code Structure Extraction**: Extracts structures using handlers.
4. **Documentation Generation**: Uses AI to generate documentation.
5. **Docstring Insertion**: Inserts docstrings into source code.
6. **Validation**: Validates code with syntax checks.
7. **Reporting**: Compiles Markdown report with badges.

## Customization

### Updating the Function Schema

Modify `function_schema.json` to fit your needs.

### Changing Docstring Styles

Modify `format_function_docstring` in `utils.py`.

### Adding Support for More Languages

1. **Create a Language Handler**: Inherit from `BaseHandler`.
2. **Implement Methods**: `extract_structure` and `insert_docstrings`.
3. **Update `LANGUAGE_MAPPING`**: Add file extensions in `utils.py`.

## Contributing

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

For questions or support, contact [hperkin4@asu.edu](mailto:hperkin4@asu.edu).

You can also reach out through:

- [GitHub Issues](https://github.com/yourusername/yourrepo/issues)
- [Discussion Forum](https://github.com/yourusername/yourrepo/discussions)

*Enhance your codebase documentation effortlessly with AI-driven precision.*
