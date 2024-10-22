import os
import sys
import shutil
import subprocess
import argparse
import fnmatch
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clone_repo(repo_url, clone_dir):
    subprocess.run(['git', 'clone', repo_url, clone_dir], check=True)

def get_all_files(directory, ignore_patterns):
    files_list = []
    
    for root, dirs, files in os.walk(directory, topdown=True):
        # Remove ignored directories from 'dirs'
        dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d), ignore_patterns, directory)]
        
        for file in files:
            full_path = os.path.join(root, file)
            if not is_ignored(full_path, ignore_patterns, directory):
                files_list.append(full_path)
                
    return files_list

def is_ignored(path, ignore_patterns, base_dir):
    # Get the relative path from the base directory
    relative_path = os.path.relpath(path, base_dir)
    # Normalize path separators
    relative_path = relative_path.replace(os.path.sep, '/')
    
    for pattern in ignore_patterns:
        # Normalize pattern separators
        pattern = pattern.replace(os.path.sep, '/')
        
        # If the pattern contains wildcards, use fnmatch
        if '*' in pattern or '?' in pattern or '[' in pattern:
            if fnmatch.fnmatch(relative_path, pattern):
                logging.debug(f"Ignoring {path} due to wildcard pattern {pattern}")
                return True
        else:
            # Check if relative_path equals the pattern or starts with pattern + '/'
            if relative_path == pattern or relative_path.startswith(pattern + '/'):
                logging.debug(f"Ignoring {path} due to pattern {pattern}")
                return True
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

    # Determine if input_path is a URL or a local path
    if input_path.startswith('http://') or input_path.startswith('https://'):
        repo_url = input_path
        repo_dir = 'cloned_repo'
        logging.info(f"Cloning repository from {repo_url} into {repo_dir}")
        clone_repo(repo_url, repo_dir)
        cleanup_needed = True
    else:
        repo_dir = input_path
        if not os.path.isdir(repo_dir):
            logging.error(f"The directory {repo_dir} does not exist.")
            sys.exit(1)
        logging.info(f"Using local directory {repo_dir}")

    # Get all files, respecting ignore patterns
    files_list = get_all_files(repo_dir, ignore_patterns)
    logging.info(f"Found {len(files_list)} files after applying ignore patterns.")

    # Write the content to a Markdown file
    output_file = 'exported_code.md'
    write_markdown(files_list, output_file, repo_dir)
    logging.info(f"Exported code to {output_file}")

    # Clean up the cloned repository if needed
    if cleanup_needed:
        shutil.rmtree(repo_dir)
        logging.info(f"Cleaned up cloned repository at {repo_dir}")

if __name__ == '__main__':
    main()
