#!/usr/bin/env python3

import os
import sys
import argparse
import asyncio
import aiohttp
import aiofiles
import ast
import re
import json
import logging
from typing import List, Optional, Tuple
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import lxml
from multiprocessing import Pool, cpu_count
from glob import glob
from os.path import join, basename, splitext
import gc

# Load environment variables from .env file
load_dotenv()

# Configure Logging
logging.basicConfig(
    filename='docs_generation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenAI API Configuration
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
openai_api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment or .env file

# Global variables for concurrency control
SEMAPHORE = None

DEFAULT_EXCLUDED_DIRS = ['.git', '__pycache__', 'node_modules']
DEFAULT_EXCLUDED_FILES = ['.DS_Store']

# Declare output_directory at the module level for accessibility in worker functions
output_directory = None

def get_language(ext: str) -> str:
    """
    Determines the programming language based on the file extension.

    Args:
        ext (str): File extension.

    Returns:
        str: Programming language.
    """
    language_mapping = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.go': 'go',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.rs': 'rust',
        '.html': 'html',
        '.htm': 'html'
    }
    return language_mapping.get(ext.lower(), 'plaintext')

def is_binary(file_path: str) -> bool:
    """
    Checks if a file is binary.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file is binary, False otherwise.
    """
    try:
        with open(file_path, 'rb') as file:
            chunk = file.read(1024)
            return b'\0' in chunk
    except Exception as e:
        logger.error(f"Error checking if file is binary '{file_path}': {e}")
        return True  # Assume binary if there's an error

def load_config(config_path: str, excluded_dirs: set, excluded_files: set) -> None:
    """
    Loads additional exclusions from a configuration JSON file.

    Args:
        config_path (str): Path to the configuration file.
        excluded_dirs (set): Set of directories to exclude.
        excluded_files (set): Set of files to exclude.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
        excluded_dirs.update(config.get('excluded_dirs', []))
        excluded_files.update(config.get('excluded_files', []))
    except Exception as e:
        logger.error(f"Error loading configuration file '{config_path}': {e}")

def get_all_html_files(directory: str) -> List[str]:
    """
    Returns a list of all HTML file paths in the directory recursively.

    Args:
        directory (str): The directory to search.

    Returns:
        list: A list of file paths.
    """
    pattern = os.path.join(directory, '**', '*.html')
    return glob(pattern, recursive=True)

def extract_text_from_html_file(file_path: str) -> str:
    """
    Extracts text content from an HTML file using BeautifulSoup with the lxml parser.

    Args:
        file_path (str): The path to the HTML file.

    Returns:
        str: The extracted text, or an empty string if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'lxml')
            return soup.get_text()
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return ''

def extract_and_save_text(file_path: str) -> None:
    """
    Extracts text from an HTML file and saves it as a .txt file in the output directory.

    Args:
        file_path (str): The path to the HTML file.
    """
    text = None
    try:
        text = extract_text_from_html_file(file_path)
        if text:
            base_name = os.path.basename(file_path)
            name, _ = os.path.splitext(base_name)
            output_file_path = os.path.join(output_directory, name + '.txt')
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(text)
            logger.info(f"Processed {file_path}")
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
    finally:
        del text
        gc.collect()

async def fetch_openai(session: aiohttp.ClientSession, content: str, language: str, retry: int = 3) -> Optional[str]:
    """
    Asynchronously fetches the generated documentation from OpenAI API.

    Args:
        session (aiohttp.ClientSession): The aiohttp session for making requests.
        content (str): The code content to generate documentation for.
        language (str): The programming language of the code.
        retry (int): Number of retry attempts.

    Returns:
        Optional[str]: Generated documentation if successful, None otherwise.
    """
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    Generate detailed Google-style docstrings/comments for the following {language} code. Ensure that each docstring includes descriptions of all parameters and return types where applicable.

    ```{language}
    {content}
    ```

    Please output the docstrings/comments in Markdown format.
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert software developer specializing in writing detailed Google-style docstrings and comments."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.2
    }

    for attempt in range(1, retry + 1):
        try:
            async with SEMAPHORE:
                async with session.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 200:
                        data = await response.json()
                        documentation = data['choices'][0]['message']['content'].strip()
                        logger.info("Successfully generated documentation from OpenAI.")
                        return documentation
                    elif response.status in {429, 500, 502, 503, 504}:
                        logger.warning(f"API rate limit or server error (status {response.status}). Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds.")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        error_text = await response.text()
                        logger.error(f"API request failed with status {response.status}: {error_text}")
                        return None
        except asyncio.TimeoutError:
            logger.error(f"Request timed out. Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds.")
            await asyncio.sleep(2 ** attempt)
        except aiohttp.ClientError as e:
            logger.error(f"API connection error: {e}. Attempt {attempt}/{retry}. Retrying in {2 ** attempt} seconds.")
            await asyncio.sleep(2 ** attempt)

    logger.error("Failed to generate documentation after multiple attempts.")
    return None

async def process_file(file_path: str, skip_types: List[str], output_file: str) -> None:
    """
    Processes a single file by reading its content, generating enhanced documentation,
    and writing the output to the specified markdown file.

    Args:
        file_path (str): Path to the file to process.
        skip_types (List[str]): List of file extensions to skip.
        output_file (str): Path to the output markdown file.
    """
    _, ext = os.path.splitext(file_path)
    logger.debug(f"Processing file: {file_path} with extension: {ext}")

    # Skip file types that are not meant for commenting (json, css, md, etc.)
    if ext in skip_types:
        logger.info(f"Skipping file: {file_path} (type {ext})")
        return

    # Check if the file is binary
    if is_binary(file_path):
        logger.warning(f"Skipped binary file: {file_path}")
        return

    try:
        # Read the content of the file
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as content_file:
            content = await content_file.read()
        logger.debug(f"Successfully read file: {file_path}")

        # Generate enhanced documentation for the file
        language = get_language(ext)
        logger.info(f"Generating documentation for file: {file_path}")
        
        async with aiohttp.ClientSession() as session:
            documentation = await fetch_openai(session, content, language)
            if not documentation:
                logger.error(f"Failed to generate documentation for '{file_path}'.")
                return

        # Write the file content and enhanced documentation to the output markdown file
        async with aiofiles.open(output_file, 'a', encoding='utf-8') as md_file:
            await md_file.write(f"## {file_path}\n\n")
            await md_file.write(f"```{language}\n{content}\n```\n\n")
            await md_file.write(f"### Generated Documentation:\n{documentation}\n\n")
        logger.info(f"Documentation for '{file_path}' written to {output_file}")
    except Exception as e:
        logger.error(f"Error processing file '{file_path}': {e}")

async def process_all_files(file_paths: List[str], skip_types: List[str], output_file: str) -> None:
    """
    Processes all files asynchronously.

    Args:
        file_paths (List[str]): List of file paths to process.
        skip_types (List[str]): List of file extensions to skip.
        output_file (str): Path to the output markdown file.
    """
    tasks = [process_file(file_path, skip_types, output_file) for file_path in file_paths]
    await tqdm.gather(*tasks, desc="Processing Files")

def main() -> None:
    """
    Main entry point of the script.
    """
    global output_directory, SEMAPHORE

    parser = argparse.ArgumentParser(
        description="Automatically generate and insert Google-style comments/docstrings into source files using GPT-4."
    )
    parser.add_argument(
        "input_directory",
        help="Path to the input directory containing HTML files"
    )
    parser.add_argument(
        "output_directory",
        help="Path to the output directory for processed files"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.json",
        help="Path to configuration file for additional exclusions"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent API requests"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_directory):
        logger.error(f"The input path '{args.input_directory}' is not a valid directory.")
        sys.exit(1)

    output_directory = args.output_directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    excluded_files = set(DEFAULT_EXCLUDED_FILES)
    skip_types = ['.json', '.css', '.md']

    # Load exclusions from config file if provided
    if args.config:
        load_config(args.config, excluded_dirs, excluded_files)

    SEMAPHORE = asyncio.Semaphore(args.concurrency)

    # Process HTML files
    html_files = get_all_html_files(args.input_directory)
    logger.info(f"Found {len(html_files)} HTML files to process.")

    for file_path in html_files:
        extract_and_save_text(file_path)

    # Process other file types
    other_files = [f for f in glob(join(args.input_directory, '**', '*'), recursive=True)
                   if os.path.isfile(f) and not any(d in f for d in excluded_dirs) and basename(f) not in excluded_files]
    other_files = [f for f in other_files if not f.lower().endswith(('.html', '.htm'))]
    logger.info(f"Found {len(other_files)} non-HTML files to process.")

    # Run the asynchronous processing for non-HTML files
    try:
        asyncio.run(process_all_files(other_files, skip_types, os.path.join(output_directory, "documentation.md")))
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
        sys.exit(1)

    logger.info("Documentation generation completed.")

if __name__ == "__main__":
    main()
