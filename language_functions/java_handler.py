# parser/java_parser.py

import javalang
import logging

logger = logging.getLogger(__name__)

def extract_structure(code):
    """
    Parses Java code to extract classes, methods, and fields.

    Parameters:
        code (str): The Java code as a string.

    Returns:
        dict: Structure containing classes and their methods.
    """
    try:
        tree = javalang.parse.parse(code)
        structure = {'classes': [], 'functions': []}

        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            cls = {
                'name': node.name,
                'docstring': extract_docstring(node),
                'methods': []
            }
            for method in node.methods:
                method_info = {
                    'name': method.name,
                    'args': [param.name for param in method.parameters],
                    'return_type': method.return_type.name if method.return_type else 'void',
                    'docstring': extract_docstring(method),
                    'modifiers': list(method.modifiers)
                }
                cls['methods'].append(method_info)
            structure['classes'].append(cls)
        
        # Top-level functions (not common in Java, but in some cases like scripts)
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if not isinstance(path[-2], javalang.tree.ClassDeclaration):
                func = {
                    'name': node.name,
                    'args': [param.name for param in node.parameters],
                    'return_type': node.return_type.name if node.return_type else 'void',
                    'docstring': extract_docstring(node),
                    'modifiers': list(node.modifiers)
                }
                structure['functions'].append(func)

        return structure
    except Exception as e:
        logger.error(f"Error parsing Java code: {e}")
        return None

def extract_docstring(node):
    """
    Extracts the docstring (comments) associated with a node.

    Parameters:
        node: A javalang AST node.

    Returns:
        str: The extracted docstring or an empty string.
    """
    # javalang does not support direct comment extraction.
    # So, we'll return an empty string here.
    # Alternatively, you can enhance this function to extract comments if needed.
    return ''

def insert_docstrings(code, documentation):
    """
    Inserts docstrings into Java code by adding comments above classes and methods.

    Parameters:
        code (str): Original Java code.
        documentation (dict): Documentation to insert.

    Returns:
        str: Code with inserted docstrings.
    """
    try:
        # Since modifying Java code while preserving original formatting is complex,
        # we'll perform a simple insertion by splitting the code into lines and
        # adding comments where appropriate.

        lines = code.split('\n')
        code_with_docs = []
        index = 0

        while index < len(lines):
            line = lines[index]
            stripped_line = line.strip()
            inserted = False

            # Check for class definitions
            if stripped_line.startswith('public class') or stripped_line.startswith('class'):
                class_name = stripped_line.split(' ')[-1].split('{')[0]
                class_doc = next((cls for cls in documentation.get('classes', []) if cls['name'] == class_name), None)
                if class_doc and class_doc.get('docstring'):
                    # Insert docstring above the class definition
                    docstring_lines = format_docstring_java(class_doc['docstring'])
                    code_with_docs.extend(docstring_lines)
                    inserted = True

            # Check for method definitions
            if ('(' in stripped_line and (stripped_line.endswith(') {') or stripped_line.endswith(');'))):
                method_signature = stripped_line
                method_name = extract_method_name(method_signature)
                parent_class = get_enclosing_class(lines, index)
                method_doc = None

                if parent_class:
                    class_doc = next((cls for cls in documentation.get('classes', []) if cls['name'] == parent_class), None)
                    if class_doc:
                        method_doc = next((m for m in class_doc.get('methods', []) if m['name'] == method_name), None)
                else:
                    method_doc = next((func for func in documentation.get('functions', []) if func['name'] == method_name), None)

                if method_doc and method_doc.get('docstring'):
                    # Insert docstring above the method definition
                    docstring_lines = format_docstring_java(method_doc['docstring'])
                    code_with_docs.extend(docstring_lines)
                    inserted = True

            code_with_docs.append(line)
            index += 1

        modified_code = '\n'.join(code_with_docs)
        return modified_code

    except Exception as e:
        logger.error(f"Error inserting Java docstrings: {e}")
        return code

def format_docstring_java(docstring):
    """
    Formats a docstring as a Java comment block.

    Parameters:
        docstring (str): The docstring to format.

    Returns:
        List[str]: Lines of the formatted docstring.
    """
    lines = docstring.strip().split('\n')
    formatted = ['/**']
    for line in lines:
        formatted.append(f' * {line}')
    formatted.append(' */')
    return formatted

def extract_method_name(signature):
    """
    Extracts the method name from a method signature.

    Parameters:
        signature (str): The method signature.

    Returns:
        str: The method name.
    """
    parts = signature.split('(')[0].split(' ')
    name = parts[-1]
    return name

def get_enclosing_class(lines, index):
    """
    Determines the enclosing class name for a given line index.

    Parameters:
        lines (List[str]): The lines of code.
        index (int): The current line index.

    Returns:
        str: The name of the enclosing class or None.
    """
    brace_count = 0
    while index >= 0:
        line = lines[index].strip()
        if line.endswith('}'):
            brace_count += line.count('}')
        if line.endswith('{'):
            brace_count -= line.count('{')
            if brace_count == -1:
                # Found the opening brace of the enclosing block
                if line.startswith('class '):
                    class_name = line.split(' ')[1].split('{')[0]
                    return class_name
        index -= 1
    return None
