# language_functions/go_handler.py

import subprocess
import json
import logging
import re
from typing import Dict, Any
from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class GoHandler(BaseHandler):
    """Handler for the Go programming language."""

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """Extracts structure from Go code using the 'go ast' command."""
        try:
            process = subprocess.run(
                ["go", "ast", "-json"],
                input=code,
                capture_output=True,
                text=True,
                check=True
            )
            ast_json = process.stdout
            ast_data = json.loads(ast_json)

            structure = {
                "summary": "",
                "changes_made": [],
                "functions": [],
                "interfaces": [],  # Go interfaces
                "structs": []       # Go structs
            }

            def traverse_ast(node):
                """Recursively traverses the Go AST."""
                if node["Kind"] == "FuncDecl":
                    func = {
                        "name": node["Name"],
                        "args": [param.get("Names", [{}])[0].get("Name", "") for param in node.get("Type", {}).get("Params", {}).get("List", [])],
                        "docstring": node.get("Doc", {}).get("Text", "").strip(),
                        "async": False  # Go doesn't have async/await keywords
                    }
                    structure["functions"].append(func)
                elif node["Kind"] == "InterfaceType":
                    interface = {
                        "name": node["Name"],
                        "docstring": "",
                        "methods": []
                    }
                    if "Doc" in node and "Text" in node["Doc"]:
                        interface["docstring"] = node["Doc"]["Text"].strip()
                    for method in node.get("Methods", {}).get("List", []):
                        method_name = method["Names"][0]["Name"]
                        method_type = get_type_string(method["Type"])
                        interface["methods"].append({"name": method_name, "type": method_type, "description": ""})
                    structure["interfaces"].append(interface)
                elif node["Kind"] == "StructType":
                    struct = {
                        "name": node["Name"],
                        "docstring": "",
                        "fields": []
                    }
                    if "Doc" in node and "Text" in node["Doc"]:
                        struct["docstring"] = node["Doc"]["Text"].strip()
                    for field in node.get("Fields", {}).get("List", []):
                        field_name = field["Names"][0]["Name"]
                        field_type = get_type_string(field["Type"])
                        struct["fields"].append({"name": field_name, "type": field_type, "description": ""})
                    structure["structs"].append(struct)
                for child in node.values():
                    if isinstance(child, dict):
                        traverse_ast(child)
                    elif isinstance(child, list):
                        for item in child:
                            if isinstance(item, dict):
                                traverse_ast(item)

            traverse_ast(ast_data)
            return structure

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running 'go ast': {e.stderr}")
            return {}
        except Exception as e:
            logger.error(f"Error extracting Go structure: {e}", exc_info=True)
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """Inserts docstrings into Go code."""
        try:
            modified_code = code
            for func in documentation.get("functions", []):
                docstring = func.get("docstring", "").strip()
                if docstring:
                    pattern = r"(?P<indent>\s*)func\s+(?P<name>" + re.escape(func["name"]) + r")\s*\((?P<params>[^)]*)\)\s*(?P<return_type>.*?)\s*\{?"
                    match = re.search(pattern, modified_code)
                    if match:
                        indent = match.group("indent")
                        docstring_lines = docstring.splitlines()
                        formatted_docstring = "\n".join([f"{indent}// {line}" for line in docstring_lines])
                        modified_code = re.sub(pattern, f"{indent}{formatted_docstring}\n{match.group(0)}", modified_code)
            return modified_code

        except Exception as e:
            logger.error(f"Error inserting Go docstrings: {e}", exc_info=True)
            return code

    def validate_code(self, code: str) -> bool:
        """Validates Go code using the 'go vet' command."""
        try:
            process = subprocess.run(["go", "vet"], input=code, capture_output=True, text=True, check=False)
            if process.returncode == 0:
                logger.debug("Go code validation successful.")
                return True
            else:
                logger.error(f"Go code validation failed:\n{process.stderr}")
                return False
        except FileNotFoundError:
            logger.error("Go compiler not found. Please install Go.")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Go code validation: {e}")
            return False

def get_type_string(type_node):
    """Extracts the type string from a Go AST type node."""
    if isinstance(type_node, str):
        return type_node
    elif isinstance(type_node, dict):
        return type_node.get("Name", "") or get_type_string(type_node.get("Type", ""))
    return ""
