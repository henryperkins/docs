# Documentation Generation Tool

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT%204-blue.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Clone the Repository](#2-clone-the-repository)
  - [3. Setting Up the Python Environment](#3-setting-up-the-python-environment)
    - [3.1. Using Virtual Environments](#31-using-virtual-environments)
    - [3.2. Installing Python Dependencies](#32-installing-python-dependencies)
  - [4. Setting Up Node.js Environment (Optional)](#4-setting-up-nodejs-environment-optional)
    - [4.1. Installing Node.js](#41-installing-nodejs)
    - [4.2. Installing Node.js Dependencies](#42-installing-nodejs-dependencies)
- [Configuration](#configuration)
- [Usage](#usage)
- [Command-Line Arguments](#command-line-arguments)
- [Examples](#examples)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The **Documentation Generation Tool** is a powerful utility designed to automatically generate and insert comprehensive documentation—such as comments and docstrings—into your codebase. Leveraging OpenAI's GPT-4 model, this tool analyzes your source code across multiple programming languages and enhances it with detailed documentation, improving code readability and maintainability.

## Features

- **Multi-Language Support:** Automatically handles Python, JavaScript/TypeScript, HTML, and CSS files.
- **Asynchronous Processing:** Utilizes asynchronous I/O for efficient handling of large codebases.
- **Customizable Configuration:** Easily specify excluded directories, files, and file types.
- **Function Calling with OpenAI:** Uses OpenAI's GPT-4 function calling for structured and accurate documentation generation.
- **Safe Mode:** Option to run without modifying any files, allowing for dry runs.
- **Comprehensive Logging:** Detailed logs for monitoring and debugging, with log rotation to manage log file sizes.
- **Documentation Reports:** Generates Markdown reports summarizing the documentation process and changes made.
- **Backup Mechanism:** Creates backups of original files before making any modifications to ensure data safety.

## Installation

Setting up the **Documentation Generation Tool** involves configuring both Python and Node.js environments. Follow the steps below to ensure a smooth setup process.

### 1. Prerequisites

Before installing the tool, ensure that your system meets the following prerequisites:

- **Python 3.8 or Higher:** The tool is built using Python 3.8+. You can download the latest version from [python.org](https://www.python.org/downloads/).
- **Node.js (Optional):** Required for processing JavaScript/TypeScript files if you intend to use Node.js scripts.
- **Git:** To clone the repository. Install from [git-scm.com](https://git-scm.com/downloads) if not already installed.

### 2. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/yourusername/documentation-generation-tool.git
cd documentation-generation-tool
```

**Note:** Replace `yourusername` with your actual GitHub username.

### 3. Setting Up the Python Environment

#### 3.1. Using Virtual Environments

It's highly recommended to use a virtual environment to manage Python dependencies. This ensures that the tool's dependencies do not interfere with other Python projects on your system.

##### **Option 1: Using `venv` (Built-in)**

1. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment:**

   - **On macOS and Linux:**

     ```bash
     source venv/bin/activate
     ```

   - **On Windows:**

     ```bash
     venv\Scripts\activate
     ```

##### **Option 2: Using `pipenv`**

`pipenv` is a popular tool that combines `pip` and `virtualenv` for better dependency management.

1. **Install `pipenv`:**

   ```bash
   pip install pipenv
   ```

2. **Create and Activate the Virtual Environment:**

   ```bash
   pipenv shell
   ```

##### **Option 3: Using `poetry`**

`poetry` is another robust tool for dependency management and packaging in Python.

1. **Install `poetry`:**

   Follow the installation instructions from the [official website](https://python-poetry.org/docs/#installation).

2. **Initialize the Project:**

   ```bash
   poetry init
   ```

3. **Activate the Virtual Environment:**

   ```bash
   poetry shell
   ```

**Choose the option that best fits your workflow.**

#### 3.2. Installing Python Dependencies

Once the virtual environment is activated, install the required Python packages.

1. **Ensure You Are in the Project Directory:**

   ```bash
   cd documentation-generation-tool
   ```

2. **Install Dependencies Using `pip`:**

   ```bash
   pip install -r requirements.txt
   ```

   **Note:** Ensure that a `requirements.txt` file exists in the repository root. If not, create one with the necessary dependencies.

   **Sample `requirements.txt`:**

   ```txt
   aiohttp
   aiofiles
   astor
   beautifulsoup4
   tinycss2
   tqdm
   python-dotenv
   ```

3. **Verify Installation:**

   After installation, verify that all packages are correctly installed.

   ```bash
   pip list
   ```

### 4. Setting Up Node.js Environment (Optional)

If your project involves processing JavaScript or TypeScript files, you'll need to set up the Node.js environment.

#### 4.1. Installing Node.js

1. **Download Node.js:**

   Visit the [official Node.js website](https://nodejs.org/) and download the latest LTS (Long Term Support) version suitable for your operating system.

2. **Install Node.js:**

   Follow the installation prompts specific to your OS.

3. **Verify Installation:**

   ```bash
   node -v
   npm -v
   ```

   Both commands should return version numbers, confirming successful installation.

#### 4.2. Installing Node.js Dependencies

1. **Navigate to the `scripts/` Directory:**

   ```bash
   cd scripts
   ```

2. **Initialize `package.json` (If Not Present):**

   If a `package.json` file does not exist, initialize it:

   ```bash
   npm init -y
   ```

3. **Install Required Packages:**

   Install necessary Node.js packages as specified in your project.

   ```bash
   npm install
   ```

   **Note:** Ensure that a `package.json` file exists with the required dependencies. If not, you may need to add scripts and dependencies manually.

4. **Verify Installation:**

   ```bash
   npm list
   ```

   This command should display the installed packages.

5. **Return to the Project Root:**

   ```bash
   cd ..
   ```

**Optional:** If your project utilizes TypeScript, ensure that TypeScript is installed globally or as a dev dependency.

```bash
npm install typescript --save-dev
```

And initialize a `tsconfig.json` if necessary:

```bash
npx tsc --init
```

## Configuration

Proper configuration is essential for the tool to function as expected. Below are the steps to configure the tool according to your project's requirements.

### 1. OpenAI API Key

The tool interacts with OpenAI's GPT-4 API, requiring an API key for authentication.

1. **Obtain an API Key:**

   Sign up or log in to your OpenAI account and navigate to the [API keys section](https://platform.openai.com/account/api-keys) to generate a new API key.

2. **Set the API Key as an Environment Variable:**

   It's crucial to keep your API key secure. Set it as an environment variable rather than hardcoding it into your scripts.

   - **On macOS and Linux:**

     ```bash
     export OPENAI_API_KEY='your-openai-api-key'
     ```

   - **On Windows (Command Prompt):**

     ```cmd
     set OPENAI_API_KEY=your-openai-api-key
     ```

   - **On Windows (PowerShell):**

     ```powershell
     $env:OPENAI_API_KEY="your-openai-api-key"
     ```

3. **Using a `.env` File (Optional):**

   For convenience, especially during development, you can store environment variables in a `.env` file.

   - **Create a `.env` File in the Project Root:**

     ```bash
     touch .env
     ```

   - **Add the API Key to `.env`:**

     ```env
     OPENAI_API_KEY=your-openai-api-key
     ```

   - **Ensure `.env` is Ignored by Git:**

     Add `.env` to your `.gitignore` to prevent accidental commits of sensitive information.

     ```bash
     echo ".env" >> .gitignore
     ```

4. **Loading Environment Variables:**

   Ensure that your Python scripts load environment variables from the `.env` file. The `python-dotenv` package facilitates this.

   - **Example Usage in Python:**

     ```python
     from dotenv import load_dotenv
     load_dotenv()
     ```

### 2. Configuration File (`config.json`)

The tool supports additional configurations through a `config.json` file. This file allows you to specify excluded directories, files, file types, project information, and style guidelines.

1. **Create `config.json` in the Project Root:**

   ```bash
   touch config.json
   ```

2. **Populate `config.json` with Configuration Settings:**

   ```json
   {
     "excluded_dirs": ["venv", "__pycache__", "node_modules", ".git"],
     "excluded_files": [".gitignore", "README.md", "LICENSE"],
     "skip_types": [".txt", ".md"],
     "project_info": "This project is a web application built with Django and React.",
     "style_guidelines": "Follow PEP 8 for Python and Airbnb style guide for JavaScript."
   }
   ```

   - **excluded_dirs:** Directories to exclude from processing.
   - **excluded_files:** Specific files to exclude.
   - **skip_types:** File extensions to skip.
   - **project_info:** Information about the project to provide context for documentation.
   - **style_guidelines:** Documentation style guidelines to follow.

3. **Customizing `config.json`:**

   Modify the values as per your project's requirements. For instance, if your project uses a different set of directories or follows different style guidelines, update the corresponding fields.

### 3. Function Schema (`function_schema.json`)

The `function_schema.json` file defines the structure required for OpenAI's function calling, enabling structured and accurate documentation generation.

1. **Create `function_schema.json` in the Project Root:**

   ```bash
   touch function_schema.json
   ```

2. **Define the Schema:**

   Populate `function_schema.json` with the necessary schema definitions. Below is a sample schema:

   ```json
   {
     "name": "generate_documentation",
     "description": "Generates comprehensive documentation for the provided code structure.",
     "parameters": {
       "type": "object",
       "properties": {
         "summary": {
           "type": "string",
           "description": "A brief summary of the code structure."
         },
         "changes_made": {
           "type": "array",
           "items": {
             "type": "string"
           },
           "description": "List of changes made during documentation insertion."
         },
         "code_snippet": {
           "type": "string",
           "description": "The snippet of code to be documented."
         }
       },
       "required": ["summary", "changes_made"]
     }
   }
   ```

   - **name:** The name of the function to be called.
   - **description:** A brief description of what the function does.
   - **parameters:** Defines the structure of the parameters the function accepts.

3. **Customize the Schema:**

   Adjust the schema to match the specific requirements of your documentation generation process. Ensure that all required fields are accurately defined to facilitate seamless integration with OpenAI's API.

## Usage

Once the environment is set up and configurations are in place, you can start generating documentation for your codebase. Below are the steps and examples to guide you through using the tool effectively.

### 1. Basic Command

Generate documentation for a repository with default settings.

```bash
python main.py /path/to/your/repository
```

### 2. Using a Custom Configuration File

Specify a different `config.json` file to override default settings.

```bash
python main.py /path/to/your/repository -c custom_config.json
```

### 3. Setting Concurrency Level

Adjust the number of concurrent API requests to optimize performance.

```bash
python main.py /path/to/your/repository --concurrency 10
```

### 4. Specifying Output Markdown File

Define a custom name for the documentation report.

```bash
python main.py /path/to/your/repository -o docs_report.md
```

### 5. Selecting a Different OpenAI Model

Choose a specific OpenAI model for documentation generation.

```bash
python main.py /path/to/your/repository --model gpt-4-32k
```

### 6. Skipping Specific File Types

Exclude certain file extensions from being processed.

```bash
python main.py /path/to/your/repository --skip-types .txt,.md
```

### 7. Providing Project Information and Style Guidelines

Enhance documentation generation by providing project context and adhering to specific style guidelines.

```bash
python main.py /path/to/your/repository --project-info "Backend service built with Flask." --style-guidelines "Google style guide."
```

### 8. Running in Safe Mode

Execute the tool without making any changes to the files. This is useful for testing and previewing potential documentation insertions.

```bash
python main.py /path/to/your/repository --safe-mode
```

### 9. Setting Log Level

Adjust the verbosity of logs for monitoring and debugging purposes.

```bash
python main.py /path/to/your/repository --log-level DEBUG
```

### 10. Specifying a Custom Function Schema

Use a different `function_schema.json` to tailor function calling parameters.

```bash
python main.py /path/to/your/repository --schema custom_schema.json
```

## Command-Line Arguments

The tool offers a variety of command-line options to customize its behavior. Below is a detailed description of each argument:

| Argument                | Description                                                  | Default                 |
|-------------------------|--------------------------------------------------------------|-------------------------|
| `repo_path`             | **(Positional)** Path to the code repository.               | _None_                  |
| `-c`, `--config`        | Path to `config.json`.                                       | `config.json`           |
| `--concurrency`         | Number of concurrent API requests.                          | `5`                     |
| `-o`, `--output`        | Output Markdown file for the documentation report.           | `output.md`             |
| `--model`               | OpenAI model to use (e.g., `gpt-4`).                         | `gpt-4`                 |
| `--skip-types`          | Comma-separated list of file extensions to skip.             | _Empty_                 |
| `--project-info`        | Information about the project.                               | _Empty_                 |
| `--style-guidelines`    | Documentation style guidelines to follow.                   | _Empty_                 |
| `--safe-mode`           | Run in safe mode without modifying any files.                | `False`                 |
| `--log-level`           | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). | `INFO`              |
| `--schema`              | Path to `function_schema.json`.                              | `function_schema.json`  |

**Example Usage:**

```bash
python main.py ./my_project -c custom_config.json --concurrency 10 -o docs_report.md --model gpt-4-32k --skip-types .txt,.md --project-info "Backend service built with Flask." --style-guidelines "Google style guide." --safe-mode --log-level DEBUG --schema custom_schema.json
```

## Examples

### 1. Generating Documentation for a Python Project

```bash
python main.py ./python_project
```

### 2. Running in Safe Mode

Execute the tool without making any changes to the files. This allows you to preview the documentation that would be generated.

```bash
python main.py ./python_project --safe-mode
```

### 3. Specifying Custom Configuration and Output File

Use a custom `config.json` and specify a different output file name.

```bash
python main.py ./javascript_project -c custom_config.json -o js_docs.md
```

### 4. Skipping Specific File Types

Exclude `.txt` and `.md` files from being processed.

```bash
python main.py ./web_project --skip-types .txt,.md
```

### 5. Providing Project Information and Style Guidelines

Enhance the generated documentation by providing context about the project and adhering to specific style guidelines.

```bash
python main.py ./flask_project --project-info "Backend service built with Flask." --style-guidelines "Google style guide."
```

### 6. Setting a Custom Function Schema

Use a different function schema to tailor the function calling parameters.

```bash
python main.py ./typescript_project --schema custom_schema.json
```

## Logging

The tool provides comprehensive logging to help monitor its operations and troubleshoot issues.

### Log File

- **Filename:** `docs_generation.log`
- **Location:** Root directory of the project.
- **Features:** 
  - **Rotating Logs:** The log file rotates after reaching 5 MB, keeping up to 5 backup logs to prevent excessive disk usage.
  - **Content:** Includes detailed logs with timestamps, log levels, module names, function names, and line numbers.

### Console Output

- **Log Levels:** Displays logs based on the specified `--log-level`.
  - **Levels Available:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
  - **Usage:** Higher log levels (`INFO` and above) are displayed by default, while `DEBUG` provides more verbose output for troubleshooting.

### Changing Log Levels

Adjust the verbosity of the logs using the `--log-level` argument.

```bash
python main.py ./my_project --log-level DEBUG
```

**Common Log Levels:**

- **DEBUG:** Detailed information, typically of interest only when diagnosing problems.
- **INFO:** Confirmation that things are working as expected.
- **WARNING:** An indication that something unexpected happened.
- **ERROR:** Due to a more serious problem, the software has not been able to perform some function.
- **CRITICAL:** A serious error, indicating that the program itself may be unable to continue running.

## Contributing

Contributions are welcome! To contribute to this project, please follow these steps:

1. **Fork the Repository:**

   Click the "Fork" button at the top-right corner of the repository page.

2. **Clone Your Fork:**

   ```bash
   git clone https://github.com/yourusername/documentation-generation-tool.git
   cd documentation-generation-tool
   ```

3. **Create a New Branch:**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

4. **Make Your Changes:**

   Implement your feature or fix a bug.

5. **Commit Your Changes:**

   ```bash
   git add .
   git commit -m "Add feature: YourFeatureName"
   ```

6. **Push to Your Fork:**

   ```bash
   git push origin feature/YourFeatureName
   ```

7. **Create a Pull Request:**

   Navigate to the original repository and click "Compare & pull request."

### Guidelines

- **Code Style:** Follow PEP 8 for Python code and Airbnb style guide for JavaScript/TypeScript.
- **Testing:** Ensure that your changes pass all existing tests and add new tests where necessary.
- **Documentation:** Update the README and other documentation as needed to reflect your changes.
- **Commit Messages:** Use clear and descriptive commit messages.
- **Issue Tracking:** Before submitting a pull request, consider opening an issue to discuss your proposed changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions, suggestions, or support, please open an issue on the [GitHub repository](https://github.com/yourusername/documentation-generation-tool) or contact the maintainer at [youremail@example.com](mailto:youremail@example.com).

---

*Enhance your codebase's readability and maintainability effortlessly with the Documentation Generation Tool!*

## Appendix: Troubleshooting

### Common Issues and Solutions

1. **OpenAI API Key Not Set:**

   - **Symptom:** The tool logs a critical error stating that the `OPENAI_API_KEY` is not set.
   - **Solution:** Ensure that the `OPENAI_API_KEY` environment variable is correctly set. Refer to the [Configuration](#configuration) section.

2. **Invalid Model Name:**

   - **Symptom:** The tool logs an error about an invalid OpenAI model name.
   - **Solution:** Verify that the model name specified with `--model` is correct and supported. Check the [Command-Line Arguments](#command-line-arguments) section.

3. **Missing `config.json`:**

   - **Symptom:** The tool logs a warning about a missing `config.json` file.
   - **Solution:** Create a `config.json` file in the project root or use default settings. Refer to the [Configuration](#configuration) section.

4. **Node.js Scripts Not Executable:**

   - **Symptom:** Errors related to Node.js scripts failing to execute.
   - **Solution:** Ensure that Node.js is correctly installed and that all Node.js dependencies are installed by following the [Setting Up Node.js Environment](#4-setting-up-nodejs-environment-optional) instructions.

5. **Permission Denied Errors:**

   - **Symptom:** The tool fails to read or write certain files due to permission issues.
   - **Solution:** Ensure that you have the necessary read/write permissions for the repository directory and its contents.

6. **Syntax Errors After Documentation Insertion:**

   - **Symptom:** Modified files contain syntax errors.
   - **Solution:** 
     - Check the logs for detailed error messages.
     - Run the tool in `--safe-mode` to preview changes without modifying files.
     - Ensure that the original code is syntactically correct before running the tool.

## Future Enhancements

- **Support for Additional Languages:** Expand support to include more programming languages and file types.
- **Interactive Mode:** Implement an interactive mode where users can approve changes before they are applied.
- **Integration with CI/CD Pipelines:** Allow the tool to be integrated into continuous integration and deployment workflows for automated documentation generation.
- **Enhanced Configuration Options:** Provide more granular configuration settings, such as specifying documentation sections or formatting preferences.
- **Localization Support:** Enable the generation of documentation in multiple languages.

---

*Happy Documenting! 🚀*
