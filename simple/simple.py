import os
import sys
import shutil
import subprocess
import argparse
import fnmatch
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def clone_repo(repo_url, clone_dir):
    subprocess.run(['git', 'clone', repo_url, clone_dir], check=True)

def get_all_files(directory, ignore_patterns):
    """
    Gets a list of all files within a directory and its subdirectories, 
    excluding files and directories specified in the ignore_patterns.

    Args:
        directory (str): The path to the directory to search.
        ignore_patterns (list): A list of patterns to exclude files and directories.

    Returns:
        list: A list of absolute paths to the found files.
    """
    files_list = []

    for root, dirs, files in os.walk(directory, topdown=True):
        # Remove ignored directories from 'dirs'
        dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d), ignore_patterns)]

        for file in files:
            full_path = os.path.join(root, file)
            if not is_ignored(full_path, ignore_patterns):
                files_list.append(full_path)

    return files_list

def is_ignored(path, ignore_patterns):
    """
    Checks if a file or directory should be ignored based on the ignore patterns.

    Args:
        path (str): The path to the file or directory.
        ignore_patterns (list): A list of ignore patterns.

    Returns:
        bool: True if the path should be ignored, False otherwise.
    """
    for pattern in ignore_patterns:
        # Handle both directory patterns and file patterns
        if pattern.endswith('/') or pattern.endswith('\\'):  # Directory pattern
            if pattern in path:
                logging.debug(f"Ignoring directory: {path} matches pattern: {pattern}")
                return True
        elif '*' in pattern:  # Pattern with wildcard, potentially matching directories
            base_pattern = pattern.split('/', 1)[0]  # Get the part before the first '/'
            if base_pattern in path:
                logging.debug(f"Ignoring path: {path} matches pattern: {pattern}")
                return True
        else:  # File pattern or simple directory name
            if fnmatch.fnmatch(os.path.basename(path), pattern):
                logging.debug(f"Ignoring file: {path} matches pattern: {pattern}")
                return True
    logging.debug(f"Not ignoring: {path}")
    return False

def write_markdown(files_list, output_file, repo_dir):
    with open(output_file, 'w', encoding='utf-8') as md_file:
        for filepath in files_list:
            relative_path = os.path.relpath(filepath, repo_dir)
            md_file.write(f'## {relative_path}\n\n')
            # Determine language for syntax highlighting
            file_extension = os.path.splitext(filepath)[1][1:]
            language = language_from_extension(file_extension)
            md_file.write(f'```{language}\n')
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except (UnicodeDecodeError, FileNotFoundError):
                # Skip binary files or files that can't be decoded
                content = '[Binary file content not displayed or file not found]'
            md_file.write(content)
            md_file.write('\n```\n\n')

def language_from_extension(extension):
    language_extensions = {
        'py': 'python',
        'js': 'javascript',
        'java': 'java',
        'c': 'c',
        'cpp': 'cpp',
        'cs': 'csharp',
        'rb': 'ruby',
        'php': 'php',
        'go': 'go',
        'rs': 'rust',
        'sh': 'bash',
        'html': 'html',
        'css': 'css',
        'md': 'markdown',
        'json': 'json',
        'xml': 'xml',
        'yml': 'yaml',
        'yaml': 'yaml',
        'ts': 'typescript',
        'kt': 'kotlin',
        'swift': 'swift',
        'pl': 'perl',
        'r': 'r',
        # Add other extensions and languages as needed
    }
    return language_extensions.get(extension.lower(), '')

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, 'config.json')

    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            try:
                config = json.load(f)
                ignore_patterns = config.get('ignore', [])
                logging.info(f"Loaded ignore patterns: {ignore_patterns}")
                return ignore_patterns
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing {config_file}: {e}")
                sys.exit(1)
    else:
        logging.warning(f"Configuration file {config_file} not found. Proceeding without ignore patterns.")
        return []

def main():
    parser = argparse.ArgumentParser(description='Export source code from a GitHub repository or local directory to a Markdown file.')
    parser.add_argument('input_path', help='GitHub Repository URL or Local Directory Path')
    args = parser.parse_args()

    input_path = args.input_path

    # Load config file from the script's directory
    ignore_patterns = load_config()

    cleanup_needed = False

    # Add more logic here if needed

if __name__ == '__main__':
    main()