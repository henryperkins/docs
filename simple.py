import os
import sys
import shutil
import subprocess
import argparse
import logging
import pathspec
import json
import asyncio
import aiohttp
from dotenv import load_dotenv
from radon.complexity import cc_visit
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenAI Configuration
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    logging.error("API key is not set. Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

ENDPOINT = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

def clone_repo(repo_url, clone_dir):
    try:
        if os.path.exists(clone_dir):
            logging.debug(f"Removing existing directory: {clone_dir}")
            shutil.rmtree(clone_dir)
        
        logging.debug(f"Cloning repository {repo_url} into {clone_dir}")
        subprocess.run(['git', 'clone', repo_url, clone_dir], check=True)
        logging.info(f"Cloned repository from {repo_url} into {clone_dir}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to clone repository: {e}")
        sys.exit(1)

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

    spec = pathspec.PathSpec.from_lines('gitwildmatch', all_patterns)
    logging.info(f"Loaded {len(all_patterns)} ignore patterns from .gitignore.")
    return spec

def is_ignored(path, spec, base_dir):
    if spec is None:
        return False
    relative_path = os.path.relpath(path, base_dir)
    relative_path = relative_path.replace(os.path.sep, '/')
    
    if spec.match_file(relative_path):
        logging.debug(f"Path ignored by .gitignore: {relative_path}")
        return True
    logging.debug(f"Path not ignored: {relative_path}")
    return False

def get_all_files(directory, spec=None, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []

    files_list = []  # Initialize files_list as an empty list
    logging.debug(f"Starting file walk in {directory} with exclusions: {exclude_dirs}")
    for root, dirs, files in os.walk(directory, topdown=True):
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
    return files_list  # Ensure files_list is returned

def calculate_complexity(file_content):
    try:
        blocks = cc_visit(file_content)
        complexity_score = sum(block.complexity for block in blocks) / len(blocks) if blocks else 0
        return complexity_score
    except Exception as e:
        logging.error(f"Failed to calculate complexity: {e}")
        return 0

async def analyze_code(session, file_content):
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Analyze the following code and provide the output in JSON format with the following structure: "
                    "{\"summary\": \"A brief summary of the code.\", "
                    "\"changelog\": \"List of changes or updates.\", "
                    "\"glossary\": \"Definitions of functions, classes, and methods.\", "
                    "\"complexity_score\": \"A score representing the complexity of the code.\"}"
                )
            },
            {
                "role": "user",
                "content": file_content
            }
        ],
        "model": "gpt-4o-2024-08-06",
        "temperature": 0.2,
        "max_tokens": 6000
    }

    retries = 5
    for attempt in range(retries):
        try:
            async with session.post(ENDPOINT, headers=headers, json=payload) as response:
                if response.status == 429:
                    logging.warning("Rate limit hit, retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                response.raise_for_status()
                response_text = await response.json()
                logging.debug(f"API Response: {response_text}")
                
                # Validate JSON
                try:
                    analysis_json = json.loads(response_text["choices"][0]["message"]["content"].strip())
                    return analysis_json
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error: {e}")
                    logging.debug(f"Raw response: {response_text}")
                    return {"error": "Invalid JSON format."}
        except aiohttp.ClientError as e:
            logging.error(f"Failed to analyze code: {e}")
            return {"error": "Analysis failed."}
    return {"error": "Max retries exceeded."}

async def process_files(files_list, repo_dir, concurrency_limit):
    results = {}
    semaphore = asyncio.Semaphore(concurrency_limit)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for filepath in files_list:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    complexity_score = calculate_complexity(content)
                    task = asyncio.ensure_future(analyze_with_semaphore(session, content, semaphore))
                    tasks.append((filepath, task, complexity_score))
            except Exception as e:
                logging.error(f"Failed to process file {filepath}: {e}")
                results[filepath] = {"error": "File processing failed."}

        for filepath, task, complexity_score in tqdm(tasks, desc="Processing files"):
            analysis = await task
            analysis["complexity_score"] = complexity_score
            results[filepath] = analysis

    return results

async def analyze_with_semaphore(session, content, semaphore):
    async with semaphore:
        return await analyze_code(session, content)

def write_analysis_to_markdown(results, output_file_path, repo_dir):
    with open(output_file_path, 'w', encoding='utf-8') as md_file:
        for filepath, analysis in results.items():
            relative_path = os.path.relpath(filepath, repo_dir)
            md_file.write(f"# Analysis for {relative_path}\n\n")
            md_file.write("## Summary\n")
            md_file.write(analysis.get("summary", "Not available."))
            md_file.write("\n\n## Changelog\n")
            md_file.write(analysis.get("changelog", "Not available."))
            md_file.write("\n\n## Glossary\n")
            md_file.write(analysis.get("glossary", "Not available."))
            md_file.write("\n\n## Complexity Score\n")
            md_file.write(str(analysis.get("complexity_score", "Not available.")))
            md_file.write("\n\n---\n\n")

async def main():
    parser = argparse.ArgumentParser(description='Export source code from a GitHub repository or local directory to a single Markdown file.')
    parser.add_argument('input_path', help='GitHub Repository URL or Local Directory Path')
    parser.add_argument('output_file', help='File where the Markdown content will be saved')
    parser.add_argument('--concurrency', type=int, default=5, help='Number of concurrent requests to OpenAI')
    args = parser.parse_args()

    input_path = args.input_path
    output_file = args.output_file
    concurrency_limit = args.concurrency
    cleanup_needed = False

    always_exclude_dirs = ['.git', '.github']

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

    spec = load_gitignore_patterns(repo_dir)

    files_list = get_all_files(repo_dir, spec, exclude_dirs=always_exclude_dirs)
    logging.info(f"Found {len(files_list)} files after applying ignore patterns and exclusions.")

    results = await process_files(files_list, repo_dir, concurrency_limit)
    write_analysis_to_markdown(results, output_file, repo_dir)

    if cleanup_needed:
        try:
            shutil.rmtree(repo_dir)
            logging.info(f"Cleaned up cloned repository at {repo_dir}")
        except Exception as e:
            logging.error(f"Failed to clean up cloned repository: {e}")

if __name__ == '__main__':
    asyncio.run(main())
