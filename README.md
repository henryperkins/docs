# Autodocumentation to Markdown

## Overview

`autodocumentation to markdown` is a Python script designed to automate the generation of comprehensive documentation for a multi-language codebase. The script leverages OpenAI's GPT-4 API to generate detailed, Google-style docstrings/comments for various code elements, including functions, classes, API endpoints, HTML elements, and CSS rules. The generated documentation is compiled into a single Markdown file, enhancing code maintainability and collaboration.

## Features

- **Multi-Language Support:** Supports Python, JavaScript, TypeScript, Java, C++, and Go.
- **Comprehensive Documentation:** Generates detailed docstrings/comments for functions, classes, methods, and more.
- **Asynchronous Processing:** Efficiently handles multiple files and API requests concurrently.
- **Robust Error Handling:** Gracefully handles various errors, including file encoding issues, binary file detection, API failures, and network errors.
- **Configuration and Customization:** Allows users to specify exclusions and customize script behavior via a configuration file (`config.json`).
- **Deployment Flexibility:** Can be deployed locally, on cloud servers, in Docker containers, and integrated with CI/CD pipelines.

## Prerequisites

- Python 3.7+
- Node.js (for JavaScript and TypeScript processing)
- OpenAI API Key

## Installation

1. **Clone the Repository:**

    ```sh
    git clone https://github.com/yourusername/autodocumentation-to-markdown.git
    cd autodocumentation-to-markdown
    ```

2. **Set Up a Virtual Environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables:**

    Create a `.env` file in the project root and add your OpenAI API key:

    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    ```

## Usage

### Command-Line Arguments

```sh
python main.py <repo_path> [options]
```

- `<repo_path>`: Path to the code repository.
- `-c, --config`: Path to `config.json` (default: `config.json`).
- `--concurrency`: Number of concurrent requests (default: 5).
- `-o, --output`: Output Markdown file (default: `output.md`).
- `--model`: OpenAI model to use (default: `gpt-4`).
- `--skip-types`: Comma-separated list of file extensions to skip.
- `--project-info`: Information about the project.
- `--style-guidelines`: Documentation style guidelines to follow.
- `--safe-mode`: Run in safe mode (no files will be modified).

### Example

```sh
python main.py /path/to/repo --concurrency 10 --output docs.md --safe-mode
```

## Configuration

### `config.json`

You can customize the script's behavior by editing the `config.json` file:

```json
{
    "excluded_dirs": ["node_modules", ".venv"],
    "excluded_files": [".DS_Store"],
    "skip_types": [".json", ".md", ".txt", ".csv", ".lock"],
    "project_info": "Your project information here",
    "style_guidelines": "Your style guidelines here"
}
```

## Deployment

### Local Deployment

1. **Set Up the Environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

2. **Run the Script:**

    ```sh
    python main.py /path/to/repo
    ```

### Cloud Server Deployment

1. **Set Up the Server:**

    - Ensure you have SSH access to your cloud server.
    - Install Python and Node.js on the server.

2. **Clone the Repository:**

    ```sh
    git clone https://github.com/yourusername/autodocumentation-to-markdown.git
    cd autodocumentation-to-markdown
    ```

3. **Set Up the Environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

4. **Run the Script:**

    ```sh
    python main.py /path/to/repo
    ```

### Docker Containerization

1. **Create a Dockerfile:**

    ```Dockerfile
    FROM python:3.9-slim

    WORKDIR /app

    COPY requirements.txt .
    RUN pip install -r requirements.txt

    COPY . .

    CMD ["python", "main.py", "/path/to/repo"]
    ```

2. **Build the Docker Image:**

    ```sh
    docker build -t autodocumentation-to-markdown .
    ```

3. **Run the Docker Container:**

    ```sh
    docker run -v /path/to/repo:/path/to/repo autodocumentation-to-markdown
    ```

### CI/CD Integration

#### GitHub Actions Example

Create a `.github/workflows/docs.yml` file:

```yaml
name: Generate Documentation

on:
  push:
    branches:
      - main

jobs:
  generate-docs:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt

    - name: Generate documentation
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        source venv/bin/activate
        python main.py /path/to/repo

    - name: Commit documentation
      run: |
        git config --global user.name 'github-actions[bot]'
        git config --global user.email 'github-actions[bot]@users.noreply.github.com'
        git add output.md
        git commit -m "Update documentation"
        git push
```

## Troubleshooting

### Common Issues

- **Missing Output Files:** Ensure the output path is correct and the script has write permissions.
- **Permission Errors:** Run the script with appropriate permissions or use `sudo` if necessary.
- **API Rate Limits:** Adjust the concurrency level or implement rate limiting in the script.
- **Encoding Problems:** Ensure all files are encoded in UTF-8.

### Solutions

- **Check Logs:** Review the `docs_generation.log` file for detailed error messages.
- **Adjust Concurrency:** Lower the concurrency level to avoid API rate limits.
- **Validate Configuration:** Ensure the `config.json` file is correctly formatted and paths are accurate.

## Best Practices

- **Secure API Keys:** Store API keys securely in environment variables or a `.env` file.
- **Optimize Performance:** Adjust concurrency levels based on the API rate limits and system resources.
- **Regular Updates:** Keep the script and dependencies up to date to benefit from the latest features and security patches.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions, improvements, or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For more information, please refer to the [documentation](docs/README.md).