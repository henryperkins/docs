import os
import sys
import shutil
import subprocess
import argparse
import logging
import pathspec

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clone_repo(repo_url, clone_dir):
    subprocess.run(['git', 'clone', repo_url, clone_dir], check=True)
    logging.info(f"Cloned repository from {repo_url} into {clone_dir}")

def load_gitignore_patterns(base_dir):
    gitignore_files = []
    for root, dirs, files in os.walk(base_dir):
        if '.gitignore' in files:
            gitignore_files.append(os.path.join(root, '.gitignore'))

    all_patterns = []
    for gitignore_file in gitignore_files:
        with open(gitignore_file, 'r', encoding='utf-8') as f:
            patterns = f.read().splitlines()
        # Get the relative path from base_dir
        gitignore_dir = os.path.relpath(os.path.dirname(gitignore_file), base_dir)
        if gitignore_dir == '.':
            gitignore_dir = ''
        for pattern in patterns:
            pattern = pattern.strip()
            if not pattern or pattern.startswith('#'):
                continue
            # Adjust pattern paths for non-root .gitignore files
            if gitignore_dir != '':
                if not pattern.startswith('/'):
                    pattern = os.path.join(gitignore_dir, pattern)
                else:
                    pattern = os.path.join(gitignore_dir, pattern.lstrip('/'))
            all_patterns.append(pattern)
    # Create PathSpec
    spec = pathspec.PathSpec.from_lines('gitwildmatch', all_patterns)
    logging.info(f"Loaded {len(all_patterns)} ignore patterns from .gitignore files.")
    return spec

def is_ignored(path, spec, base_dir):
    if spec is None:
        return False
    # Get the relative path from the base directory
    relative_path = os.path.relpath(path, base_dir)
    relative_path = relative_path.replace(os.path.sep, '/')
    return spec.match_file(relative_path)

def get_all_files(directory, spec=None, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []

    if os.path.isdir(os.path.join(directory, '.git')):
        # Use git commands to get the list of files
        result_tracked = subprocess.run(['git', '-C', directory, 'ls-files'], stdout=subprocess.PIPE, text=True)
        tracked_files = result_tracked.stdout.strip().split('\n')
        result_untracked = subprocess.run(['git', '-C', directory, 'ls-files', '--others', '--exclude-standard'], stdout=subprocess.PIPE, text=True)
        untracked_files = result_untracked.stdout.strip().split('\n')
        all_files = tracked_files + untracked_files
        files_list = [os.path.join(directory, f) for f in all_files if f]
        logging.info(f"Retrieved {len(files_list)} files using git commands.")
    else:
        # Use os.walk to get all files, respecting .gitignore patterns
        files_list = []
        for root, dirs, files in os.walk(directory, topdown=True):
            # Exclude specified directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not is_ignored(os.path.join(root, d), spec, directory)]
            for file in files:
                full_path = os.path.join(root, file)
                if not is_ignored(full_path, spec, directory):
                    files_list.append(full_path)
        logging.info(f"Retrieved {len(files_list)} files using os.walk.")
    return files_list

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
    return language_extensions.get(extension.lower(), '')

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
        clone_repo(repo_url, repo_dir)
        cleanup_needed = True
    else:
        repo_dir = input_path
        if not os.path.isdir(repo_dir):
            logging.error(f"The directory {repo_dir} does not exist.")
            sys.exit(1)
        logging.info(f"Using local directory {repo_dir}")

    # Check if the directory is a git repository
    if os.path.isdir(os.path.join(repo_dir, '.git')):
        spec = None
        logging.info("Detected a Git repository. Using git commands to retrieve files.")
    else:
        # Load .gitignore patterns
        spec = load_gitignore_patterns(repo_dir)
        logging.info("Loaded .gitignore patterns.")

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
