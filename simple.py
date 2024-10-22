import os
import sys
import shutil
import subprocess
import argparse
import fnmatch
import json

def load_config():
    config_file = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            config = json.load(file)
        return config.get('ignore', [])
    else:
        print(f"Configuration file {config_file} not found. Proceeding without ignore patterns.")
        return []

def is_ignored(path, ignore_patterns):
    for pattern in ignore_patterns:
        if pattern.endswith('/') or pattern.endswith('\\'):
            if fnmatch.fnmatch(path + '/', pattern):
                return True
        else:
            if fnmatch.fnmatch(path, pattern):
                return True
    return False

def get_all_files(repo_dir, ignore_patterns):
    files_list = []
    for root, dirs, files in os.walk(repo_dir):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d), ignore_patterns)]
        for file in files:
            file_path = os.path.join(root, file)
            if not is_ignored(file_path, ignore_patterns):
                files_list.append(file_path)
    return files_list

def write_markdown(files_list, output_file, repo_dir):
    with open(output_file, 'w') as md_file:
        for file in files_list:
            relative_path = os.path.relpath(file, repo_dir)
            md_file.write(f"### {relative_path}\n")
            with open(file, 'r', encoding='utf-8', errors='ignore') as code_file:
                content = code_file.read()
                md_file.write(f"```\n{content}\n```\n\n")

def clone_repo(repo_url, clone_dir):
    subprocess.run(['git', 'clone', repo_url, clone_dir], check=True)

def main():
    parser = argparse.ArgumentParser(description='Export source code from a GitHub repository or local directory to a Markdown file.')
    parser.add_argument('input_path', help='GitHub Repository URL or Local Directory Path')
    args = parser.parse_args()

    input_path = args.input_path

    # Load config file from the script's directory
    ignore_patterns = load_config()

    cleanup_needed = False

    if os.path.exists(input_path):
        # Input is a local directory
        repo_dir = os.path.abspath(input_path)
        repo_name = os.path.basename(os.path.normpath(repo_dir))
        print(f"Using local directory: {repo_dir}")
    else:
        # Assume input is a GitHub repository URL
        repo_url = input_path
        repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        clone_dir = os.path.join('.', repo_name)

        # Remove the directory if it already exists
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)

        try:
            print(f"Cloning repository {repo_url}...")
            clone_repo(repo_url, clone_dir)
            repo_dir = os.path.abspath(clone_dir)
            cleanup_needed = True  # Mark that we need to delete this directory later
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            sys.exit(1)

    print("Collecting files...")
    files_list = get_all_files(repo_dir, ignore_patterns)

    # Hardcode the output file path to /home/ubuntu/
    output_file = f'/home/ubuntu/{repo_name}_source_code.md'

    print(f"Writing source code to {output_file}...")
    write_markdown(files_list, output_file, repo_dir)

    # Display the file size of the output document
    try:
        file_size = os.path.getsize(output_file)
        print(f"Output file size: {file_size / 1024:.2f} KB")  # Display size in kilobytes
    except OSError as e:
        print(f"Error getting file size: {e}")

    # Clean up cloned repository if needed
    if cleanup_needed:
        shutil.rmtree(repo_dir)

    print(f"Done! Source code has been written to {output_file}")

if __name__ == '__main__':
    main()