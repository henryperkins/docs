import os
import javalang
import logging
import subprocess
from typing import Dict, Any, Optional
from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class JavaHandler(BaseHandler):
    def __init__(self, function_schema):
        self.function_schema = function_schema

    # Other methods...
    """Handler for Java language."""

    def extract_structure(self, code: str) -> Dict[str, Any]:
        """Parses Java code to extract classes and methods."""
        logger.debug("Extracting Java code structure.")
        
        structure = {
            "classes": [],
            "functions": [],
        }
        
        try:
            tokens = javalang.tokenizer.tokenize(code)
            parser = javalang.parser.Parser(tokens)
            tree = parser.parse()
            
            for path, node in tree.filter(javalang.tree.ClassDeclaration):
                cls = {
                    "name": node.name,
                    "methods": [],
                    "docstring": node.documentation or ""
                }
                
                for method in node.methods:
                    func = {
                        "name": method.name,
                        "args": [param.name for param in method.parameters],
                        "docstring": method.documentation or "",
                    }
                    cls["methods"].append(func)
                
                structure["classes"].append(cls)
            
            for path, node in tree.filter(javalang.tree.MethodDeclaration):
                func = {
                    "name": node.name,
                    "args": [param.name for param in node.parameters],
                    "docstring": node.documentation or "",
                }
                structure["functions"].append(func)
            
            return structure
        
        except javalang.parser.JavaSyntaxError as e:
            logger.error(f"Failed to parse Java code: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error during Java code parsing: {e}")
            return {}

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """Inserts Javadoc comments into Java code based on the provided documentation."""
        logger.debug("Inserting Javadoc docstrings into Java code.")
        
        for class_doc in documentation.get("classes", []):
            class_name = class_doc.get("name")
            class_docstring = class_doc.get("docstring", "")
            code = self.insert_comment(code, class_name, class_docstring)
            
            for method_doc in class_doc.get("methods", []):
                method_name = method_doc.get("name")
                method_docstring = method_doc.get("docstring", "")
                code = self.insert_comment(code, method_name, method_docstring)
        
        return code

    def insert_comment(self, code: str, name: str, comment: str) -> str:
        """Inserts comment above the specified class/method name in the code."""
        comment_block = f"/**\n * {comment}\n */\n"
        code = code.replace(f"class {name}", f"{comment_block}class {name}")
        code = code.replace(f"void {name}", f"{comment_block}void {name}")
        return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates Java code by attempting to compile it.

        Args:
            code (str): The Java code to validate.
            file_path (Optional[str]): The path to the Java file being validated.

        Returns:
            bool: True if the code compiles successfully, False otherwise.
        """
        logger.debug('Starting Java code validation.')
        if not file_path:
            logger.warning('File path not provided for Java validation. Skipping compilation.')
            return True  # Assuming no compilation without a file

        try:
            # Write code to the specified file path
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)

            # Attempt to compile the Java file
            process = subprocess.run(
                ['javac', file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if process.returncode != 0:
                logger.error(f'Java compilation failed for {file_path}:\n{process.stderr}')
                return False
            else:
                logger.debug('Java compilation successful.')
                # Optionally, remove the .class file after validation
                class_file = file_path.replace('.java', '.class')
                if os.path.exists(class_file):
                    os.remove(class_file)
                    logger.debug(f'Removed compiled class file {class_file}.')
            return True
        except FileNotFoundError:
            logger.error("Java compiler (javac) not found. Please ensure JDK is installed and javac is in the PATH.")
            return False
        except Exception as e:
            logger.error(f'Unexpected error during Java code validation: {e}')
            return False