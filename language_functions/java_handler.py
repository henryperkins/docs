import javalang
import logging
import shutil
import subprocess
from typing import Dict, Any
from language_functions.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class JavaHandler(BaseHandler):
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

    def validate_code(self, code: str) -> bool:
        """Validates the modified Java code for syntax correctness using javac."""
        javac_path = shutil.which("javac")
        if not javac_path:
            logger.error("Java compiler (javac) not found. Please ensure JDK is installed and javac is in the PATH.")
            return False

        try:
            with open("temp.java", "w") as f:
                f.write(code)

            result = subprocess.run(["javac", "temp.java"], capture_output=True, text=True)

            if result.returncode == 0:
                return True
            else:
                logger.error(f"Syntax error in Java code: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Subprocess error during Java code validation: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during Java code validation: {e}")
        finally:
            try:
                os.remove("temp.java")
            except OSError as e:
                logger.warning(f"Failed to remove temporary file: {e}")

        return False
