import os
import sys
import shutil
import subprocess
import argparse
import logging
import pathspec

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def clone_repo(repo_url, clone_dir):
    logging.debug(f"Cloning repository {repo_url} into {clone_dir}")
    subprocess.run(['git', 'clone', repo_url, clone_dir], check=True)
    logging.info(f"Cloned repository from {repo_url} into {clone_dir}")

def load_gitignore_patterns(base_dir):
    gitignore_file = os.path.join(base_dir, '.gitignore')
    all_patterns = []
    
    if os.path.exists(gitignore_file):
        logging.debug(f"Found .gitignore file at {gitignore_file}")
        with open(gitignore_file, 'r', encoding='utf-8') as f:
            patterns = f.read().splitlines()
        all_patterns.extend(patterns)
        logging.debug(f"Loaded {len(all_patterns)} patterns from {gitignore_file}")
    else:
        logging.warning(f"No .gitignore file found at {gitignore_file}")

    # Create PathSpec object for .gitignore patterns
    spec = pathspec.PathSpec.from_lines('gitwildmatch', all_patterns)
    logging.info(f"Loaded {len(all_patterns)} ignore patterns from .gitignore.")
    return spec

def is_ignored(path, spec, base_dir):
    if spec is None:
        return False
    # Get the relative path from the base directory
    relative_path = os.path.relpath(path, base_dir)
    relative_path = relative_path.replace(os.path.sep, '/')
    
    # Debugging the ignore logic
    if spec.match_file(relative_path):
        logging.debug(f"Path ignored by .gitignore: {relative_path}")
        return True
    logging.debug(f"Path not ignored: {relative_path}")
    return False

def get_all_files(directory, spec=None, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []

    files_list = []
    logging.debug(f"Starting file walk in {directory} with exclusions: {exclude_dirs}")
    for root, dirs, files in os.walk(directory, topdown=True):
        # Exclude specified directories explicitly, such as .git and .github
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not is_ignored(os.path.join(root, d), spec, directory)]
        logging.debug(f"Directories after exclusion in {root}: {dirs}")

        for file in files:
            full_path = os.path.join(root, file)
            if not is_ignored(full_path, spec, directory):
                logging.debug(f"Adding file: {full_path}")
                files_list.append(full_path)
            else:
                logging.debug(f"Skipping ignored file: {full_path}")
    
    logging.info(f"Retrieved {len(files_list)} files using os.walk.")
    return files_list

def write_markdown(files_list, output_file, repo_dir):
    logging.debug(f"Writing {len(files_list)} files to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as md_file:
        for filepath in files_list:
            relative_path = os.path.relpath(filepath, repo_dir)
            logging.debug(f"Processing file: {relative_path}")
            md_file.write(f'## {relative_path}\n\n')
            
            # Determine language for syntax highlighting
            file_extension = os.path.splitext(filepath)[1][1:]
            language = language_from_extension(file_extension)
            logging.debug(f"Detected language for {relative_path}: {language}")
            md_file.write(f'```{language}\n')
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                md_file.write(content)
            except (UnicodeDecodeError, FileNotFoundError) as e:
                logging.error(f"Failed to read file {filepath}: {e}")
                content = '[Binary file content not displayed or file not found]'
                md_file.write(content)
            md_file.write('\n```\n\n')
    logging.info(f"Exported code to {output_file}")

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
    language = language_extensions.get(extension.lower(), '')
    logging.debug(f"Mapped extension .{extension} to language: {language}")
    return language

def main():
    parser = argparse.ArgumentParser(description='Export source code from a GitHub repository or local directory to a Markdown file.')
    parser.add_argument('input_path', help='GitHub Repository URL or Local Directory Path')
    args = parser.parse_args()

    input_path = args.input_path
    cleanup_needed = False

    # Directories to always exclude
    always_exclude_dirs = ['.git', '.github']

    # Determine if input_path is a URL or a local path
    if input_path.startswith('http://') or input_path.startswith('https://'):
        repo_url = input_path
        repo_dir = 'cloned_repo'
        logging.debug(f"Input is a GitHub repository URL: {repo_url}")
        clone_repo(repo_url, repo_dir)
        cleanup_needed = True
    else:
        repo_dir = input_path
        if not os.path.isdir(repo_dir):
            logging.error(f"The directory {repo_dir} does not exist.")
            sys.exit(1)
        logging.info(f"Using local directory {repo_dir}")

    # Load .gitignore patterns
    spec = load_gitignore_patterns(repo_dir)

    # Get all files, respecting ignore patterns and always excluding certain directories
    files_list = get_all_files(repo_dir, spec, exclude_dirs=always_exclude_dirs)
    logging.info(f"Found {len(files_list)} files after applying ignore patterns and exclusions.")

    # Write the content to a Markdown file
    output_file = 'exported_code.md'
    write_markdown(files_list, output_file, repo_dir)

    # Clean up the cloned repository if needed
    if cleanup_needed:
        shutil.rmtree(repo_dir)
        logging.info(f"Cleaned up cloned repository at {repo_dir}")

if __name__ == '__main__':
    main()
