import os
import sys
import shutil
import subprocess
import argparse
import fnmatch
import json
import logging
import pathspec

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clone_repo(repo_url, clone_dir):
    subprocess.run(['git', 'clone', repo_url, clone_dir], check=True)

def get_all_files(directory, ignore_spec):
    """
    Gets all files within a directory, respecting .gitignore rules.

    Args:
        directory: The directory to traverse.
        ignore_spec:  A pathspec.PathSpec object representing the .gitignore rules.

    Returns:
        A list of filepaths.
    """
    files_list = []
    for root, dirs, files in os.walk(directory, topdown=True):
        # Modify dirs in-place to respect .gitignore for directories
        dirs[:] = [d for d in dirs if not (ignore_spec and ignore_spec.match_file(os.path.relpath(os.path.join(root, d), directory).replace(os.path.sep, '/')))]
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, directory).replace(os.path.sep, '/')
            if not (ignore_spec and ignore_spec.match_file(relative_path)):
                files_list.append(full_path)
    return files_list


def write_markdown(files_list, output_file, repo_dir):
    with open(output_file, 'w', encoding='utf-8') as md_file:
        for filepath in files_list:
            relative_path = os.path.relpath(filepath, repo_dir)
            md_file.write(f'## {relative_path}\n\n')
            file_extension = os.path.splitext(filepath)[1][1:]
            language = language_from_extension(file_extension)
            md_file.write(f'```{language}\n')
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except (UnicodeDecodeError, FileNotFoundError):
                content = '[Binary file content not displayed or file not found]'
            md_file.write(content)
            md_file.write('\n```\n\n')


def language_from_extension(extension):
    language_extensions = {
        'py': 'python', 'js': 'javascript', 'java': 'java', 'c': 'c', 'cpp': 'cpp',
        'cs': 'csharp', 'rb': 'ruby', 'php': 'php', 'go': 'go', 'rs': 'rust',
        'sh': 'bash', 'html': 'html', 'css': 'css', 'md': 'markdown',
        'json': 'json', 'xml': 'xml', 'yml': 'yaml', 'yaml': 'yaml',
        'ts': 'typescript', 'kt': 'kotlin', 'swift': 'swift', 'pl': 'perl', 'r': 'r',
        # ... add more as needed
    }
    return language_extensions.get(extension.lower(), '')


def load_config_and_gitignore(repo_dir):
    """Loads ignore patterns from config.json and .gitignore."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, 'config.json')
    gitignore_path = os.path.join(repo_dir, '.gitignore')

    spec = None
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as gitignore_file:
            spec = pathspec.PathSpec.from_lines('gitwildmatch', gitignore_file)

    return spec



def main():
    parser = argparse.ArgumentParser(description='Export source code to a Markdown file.')
    parser.add_argument('input_path', help='GitHub Repository URL or Local Directory Path')
    args = parser.parse_args()
    input_path = args.input_path

    cleanup_needed = False
    if os.path.exists(input_path):
        repo_dir = os.path.abspath(input_path)
        repo_name = os.path.basename(os.path.normpath(repo_dir))
    else:
        repo_url = input_path
        repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        clone_dir = os.path.join('.', repo_name)
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
        try:
            clone_repo(repo_url, clone_dir)
            repo_dir = os.path.abspath(clone_dir)
            cleanup_needed = True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error cloning repository: {e}")
            sys.exit(1)


    ignore_spec = load_config_and_gitignore(repo_dir)
    files_list = get_all_files(repo_dir, ignore_spec)

    output_file = f'{repo_name}_source_code.md'  # Output in the current directory
    write_markdown(files_list, output_file, repo_dir)

    if cleanup_needed:
        shutil.rmtree(repo_dir)

    logging.info(f"Done! Source code has been written to {output_file}")


if __name__ == '__main__':
    main()
