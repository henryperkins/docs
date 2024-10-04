# language_functions/cpp_handler.py

import subprocess
import json
import logging
import re
from typing import Dict, Any
from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class CppHandler(BaseHandler):
    """Handler for the C++ programming language."""

    def extract_structure(self, code: str, file_path: str) -> Dict[str, Any]:
        """Extracts structure from C++ code using libclang."""
        try:
            # Run clang command to get JSON AST
            process = subprocess.run(
                ["clang", "-Xclang", "-ast-dump=json", "-fsyntax-only", file_path],
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
                "classes": []
            }

            def traverse_ast(node):
                """Recursively traverses the C++ AST."""
                kind = node.get("kind", "")
                if kind == "FunctionDecl":
                    func = {
                        "name": node["name"],
                        "args": [param["name"] for param in node.get("inner", []) if param.get("kind", "") == "ParmVarDecl"],
                        "docstring": extract_docstring_from_comment(node.get("inner", [])).strip(),
                        "async": False  # C++ doesn't have async/await keywords
                    }
                    structure["functions"].append(func)
                elif kind == "CXXRecordDecl" and node.get("tagUsed", "") == "class":
                    cls = {
                        "name": node["name"],
                        "docstring": extract_docstring_from_comment(node.get("inner", [])),
                        "methods": [],
                        "attributes": []
                    }
                    for member in node.get("inner", []):
                        if member.get("kind", "") == "CXXMethodDecl":
                            method = {
                                "name": member["name"],
                                "docstring": extract_docstring_from_comment(member.get("inner", [])),
                                "parameters": [],
                                "return_value": {"type": member.get("type", {}).get("qualType", ""), "description": ""},
                                "examples": [],
                                "error_handling": "",
                                "assumptions_preconditions": ""
                            }
                            for param in member.get("inner", []):
                                if param.get("kind", "") == "ParmVarDecl":
                                    param_type = param.get("type", {}).get("qualType", "")
                                    method["parameters"].append({"name": param["name"], "type": param_type, "description": ""})
                            cls["methods"].append(method)
                        elif member.get("kind", "") == "FieldDecl":
                            attr = {
                                "name": member["name"],
                                "type": member.get("type", {}).get("qualType", ""),
                                "description": extract_docstring_from_comment(member.get("inner", []))
                            }
                            cls["attributes"].append(attr)
                    structure["classes"].append(cls)
                for child in node.get("inner", []):
                    traverse_ast(child)

            traverse_ast(ast_data)
            return structure

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running clang: {e.stderr}")
            return {}
        except Exception as e:
            logger.error(f"Error extracting C++ structure: {e}", exc_info=True)
            return {}

    def insert_docstrings(self, original_code: str, documentation: Dict[str, Any]) -> str:
        """Inserts Doxygen-style docstrings into C++ code."""
        try:
            modified_code = original_code

            for func in documentation.get("functions", []):
                docstring = func.get("docstring", "").strip()
                if docstring:
                    # Regex to find function definition (handles different return types and parameters)
                    pattern = r"(?P<indent>\s*)(?P<return_type>[\w\s\*\&]+)\s+(?P<name>" + re.escape(func["name"]) + r")\s*\((?P<params>[^)]*)\)\s*\{?"
                    match = re.search(pattern, modified_code)
                    if match:
                        indent = match.group("indent")
                        docstring_lines = docstring.splitlines()
                        formatted_docstring = "\n".join([f"{indent}/// {line}" for line in docstring_lines])
                        modified_code = re.sub(pattern, f"{indent}/**\n{formatted_docstring}\n{indent}*/\n{match.group(0)}", modified_code)

            for cls in documentation.get("classes", []):
                docstring = cls.get("docstring", "").strip()
                if docstring:
                    # Regex to find class definition
                    pattern = r"(?P<indent>\s*)class\s+(?P<name>" + re.escape(cls["name"]) + r")\s*\{?"
                    match = re.search(pattern, modified_code)
                    if match:
                        indent = match.group("indent")
                        docstring_lines = docstring.splitlines()
                        formatted_docstring = "\n".join([f"{indent}/// {line}" for line in docstring_lines])
                        modified_code = re.sub(pattern, f"{indent}/**\n{formatted_docstring}\n{indent}*/\n{match.group(0)}", modified_code)

                for method in cls.get("methods", []):
                    docstring = method.get("docstring", "").strip()
                    if docstring:
                        # Regex to find method definition (within the class)
                        pattern = r"(?P<indent>\s*)(?P<return_type>[\w\s\*\&]+)\s+(?P<class_name>" + re.escape(cls["name"]) + r")::(?P<name>" + re.escape(method["name"]) + r")\s*\((?P<params>[^)]*)\)\s*\{?"
                        match = re.search(pattern, modified_code)
                        if match:
                            indent = match.group("indent")
                            docstring_lines = docstring.splitlines()
                            formatted_docstring = "\n".join([f"{indent}/// {line}" for line in docstring_lines])
                            modified_code = re.sub(pattern, f"{indent}/**\n{formatted_docstring}\n{indent}*/\n{match.group(0)}", modified_code)

            return modified_code

        except Exception as e:
            logger.error(f"Error inserting C++ docstrings: {e}", exc_info=True)
            return original_code

    def validate_code(self, code: str, file_path: str = None) -> bool:
        """Validates C++ code using clang."""
        try:
            process = subprocess.run(
                ["clang++", "-fsyntax-only", "-std=c++17", file_path],  # Adjust the C++ standard if needed
                input=code,
                capture_output=True,
                text=True,
                check=False
            )
            if process.returncode == 0:
                logger.debug("C++ code validation successful.")
                return True
            else:
                logger.error(f"C++ code validation failed:\n{process.stderr}")
                return False
        except FileNotFoundError:
            logger.error("Clang compiler not found. Please install Clang.")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during C++ code validation: {e}")
            return False

def extract_docstring_from_comment(nodes):
    """Extracts docstring from comment nodes in the AST."""
    for node in nodes:
        if node.get("kind", "") == "FullComment":
            comment = node.get("text", "").strip()
            if comment.startswith("/**") and comment.endswith("*/"):
                return comment[3:-2].strip()
    return ""
