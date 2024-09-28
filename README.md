# Automated Code Documentation Generator

This project is a Python-based tool that automates the generation of documentation for code repositories. It scans through the files in a given repository, extracts the structure of the code (functions, classes, methods, etc.), and uses OpenAI's GPT-4 API to generate docstrings or comments. The script supports multiple programming languages, including Python, JavaScript/TypeScript, HTML, and CSS. The generated documentation is then inserted back into the original code files and can also be compiled into a Markdown file.

---

## Table of Contents

- [Features](#features)
- [Supported Languages](#supported-languages)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Automated Documentation**: Generates detailed docstrings or comments for functions, classes, methods, HTML elements, and CSS rules.
- **Multi-Language Support**: Supports Python, JavaScript/TypeScript, HTML, and CSS.
- **Asynchronous Processing**: Uses `asyncio` for concurrent file processing and API calls.
- **Customizable**: Allows exclusion of specific directories, files, and file types via configuration.
- **Backup Creation**: Automatically creates backups of original files before modifying them.
- **Error Handling**: Robust error handling and logging throughout the process.

---

## Supported Languages

- **Python**
- **JavaScript**
- **TypeScript**
- **HTML**
- **CSS**

---

## Prerequisites

- **Python 3.7 or higher**
- **OpenAI API Key**: Required to access the GPT-4 API.
- **Node.js and NPM**: Needed for installing `esprima` for JavaScript/TypeScript parsing.
- **Git**: If cloning the repository.

---

## Installation

1. **Clone the Repository** (or download the source code):

   ```bash
   git clone https://github.com/yourusername/automated-doc-generator.git
   cd automated-doc-generator
   ```

2. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Esprima** (for JavaScript/TypeScript parsing):

   Ensure you have Node.js and NPM installed. Then run:

   ```bash
   npm install -g esprima
   ```

5. **Set Up OpenAI API Key**:

   Create a `.env` file in the project root directory and add your OpenAI API key:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

---

## Configuration

You can customize the behavior of the script using a `config.json` file. Create a `config.json` in the project root directory with the following structure:

```json
{
  "excluded_dirs": ["dir_to_exclude1", "dir_to_exclude2"],
  "excluded_files": ["file_to_exclude1", "file_to_exclude2"],
  "skip_types": [".ext_to_skip1", ".ext_to_skip2"]
}
```

- **excluded_dirs**: Directories to exclude from processing.
- **excluded_files**: Specific files to exclude.
- **skip_types**: File extensions to skip.

If no `config.json` is provided, default values will be used.

---

## Usage

Run the script using the following command:

```bash
python main.py <repo_path> [options]
```

### Positional Arguments:

- `<repo_path>`: Path to the code repository you want to process.

### Optional Arguments:

- `-c`, `--config`: Path to `config.json` (default: `config.json`).
- `--concurrency`: Number of concurrent requests (default: 5).
- `-o`, `--output`: Output Markdown file (default: `output.md`).
- `--model`: OpenAI model to use (default: `gpt-4`).
- `--skip-types`: Comma-separated list of file extensions to skip.

### Example:

```bash
python main.py ./my_project -c ./config.json --concurrency 10 -o documentation.md --model gpt-4 --skip-types .env,.log
```

---

## Examples

### Basic Usage:

```bash
python main.py ./my_project
```

### Custom Configuration:

```bash
python main.py ./my_project -c ./config.json
```

### Specifying Concurrency and Output File:

```bash
python main.py ./my_project --concurrency 8 -o docs.md
```

---

## Logs

The script generates a log file named `docs_generation.log` in the project root directory. It contains detailed logs for the execution, which is helpful for debugging.

---

## Dependencies

The script relies on several Python libraries and external tools:

### Python Libraries:

- `aiohttp`
- `aiofiles`
- `argparse`
- `asyncio`
- `ast`
- `astor`
- `beautifulsoup4`
- `dotenv`
- `json`
- `logging`
- `tqdm`
- `tinycss2`
- `typing`

Install them using:

```bash
pip install -r requirements.txt
```

### External Tools:

- **Esprima**: JavaScript/TypeScript parser used via subprocess.

  Install globally using NPM:

  ```bash
  npm install -g esprima
  ```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your message here"
   ```

4. **Push to Your Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **OpenAI**: For providing the GPT-4 API.
- **Contributors**: Thanks to all who have contributed to this project.
- **Community**: For the support and feedback.

---

Feel free to customize and enhance this README as per your project's requirements.