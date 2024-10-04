# language_functions/java_handler.py

import logging
from typing import Optional, Dict, Any
from language_functions.base_handler import BaseHandler
from parser.java_parser import extract_structure, insert_docstrings

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
        from parser.java_parser import JavaParser

        parser = JavaParser(code)
        parsed_code = parser.parse_code()

        for class_name, class_info in documentation.items():
            if class_name in parsed_code['classes']:
                class_def = parsed_code['classes'][class_name]
                class_doc = self.generate_javadoc(class_info.get('description', ''))
                code = self.insert_comment(code, class_def['start'], class_doc)
                
                for method_name, method_info in class_info.get('methods', {}).items():
                    if method_name in class_def['methods']:
                        method_def = class_def['methods'][method_name]
                        method_doc = self.generate_javadoc(method_info.get('description', ''), method_info.get('params', {}), method_info.get('returns', ''))
                        code = self.insert_comment(code, method_def['start'], method_doc)

        return code

    def generate_javadoc(self, description: str, params: Optional[Dict[str, str]] = None, returns: Optional[str] = None) -> str:
        """Generates Javadoc comment string."""
        javadoc = "/**\n"
        javadoc += f" * {description}\n"
        if params:
            for param, desc in params.items():
                javadoc += f" * @param {param} {desc}\n"
        if returns:
            javadoc += f" * @return {returns}\n"
        javadoc += " */"
        return javadoc

    def insert_comment(self, code: str, position: int, comment: str) -> str:
        """Inserts comment at the specified position in the code."""
        before = code[:position]
        after = code[position:]
        return before + comment + "\n" + after

    def validate_code(self, code: str) -> bool:
        """Validates the modified Java code for syntax correctness."""
        from parser.java_parser import extract_structure
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
