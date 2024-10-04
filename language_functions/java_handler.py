import logging
from typing import Optional, Dict, Any
from language_functions.base_handler import BaseHandler
from parser.java_parser import extract_structure, JavaParser

logger = logging.getLogger(__name__)

class JavaHandler(BaseHandler):
    """Handler for Java language."""

    def extract_structure(self, code: str) -> Dict[str, Any]:
        """Parses Java code to extract classes and methods."""
        logger.debug("Extracting Java code structure.")
        structure = extract_structure(code)
        if structure is None:
            logger.error("Failed to extract structure from Java code.")
            return {}
        return structure

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """Inserts Javadoc comments into Java code based on the provided documentation."""
        logger.debug("Inserting Javadoc docstrings into Java code.")
        parser = JavaParser(code)
        parsed_code = parser.parse_code()

        for class_name, class_info in documentation.items():
            if class_name in parsed_code['classes']:
                class_def = parsed_code['classes'][class_name]
                class_doc = self.generate_javadoc(class_info)
                code = self.insert_comment(code, class_def['start'], class_doc)
                
                for method_name, method_info in class_info.get('methods', {}).items():
                    if method_name in class_def['methods']:
                        method_def = class_def['methods'][method_name]
                        method_doc = self.generate_javadoc(method_info)
                        code = self.insert_comment(code, method_def['start'], method_doc)

        return code

    def generate_javadoc(self, doc_info: Dict[str, Any]) -> str:
        """Generates Javadoc comment string."""
        javadoc = "/**\n"
        javadoc += f" * {doc_info.get('docstring', '')}\n"
        for arg in doc_info.get('args', []):
            javadoc += f" * @param {arg} Description.\n"  # Replace 'Description' as needed
        if doc_info.get('returns'):
            javadoc += f" * @return {doc_info['returns']}\n"
        javadoc += " */"
        return javadoc

    def insert_comment(self, code: str, position: int, comment: str) -> str:
        """Inserts comment at the specified position in the code."""
        before = code[:position]
        after = code[position:]
        return before + comment + "\n" + after

    def validate_code(self, code: str) -> bool:
        """Validates the modified Java code for syntax correctness."""
        try:
            structure = extract_structure(code)
            if structure:
                logger.debug("Java code validation successful.")
                return True
            else:
                logger.error("Java code validation failed: Structure extraction returned empty.")
                return False
        except Exception as e:
            logger.error(f"Java code validation failed: {e}")
            return False
