import esprima

# A simple JavaScript function to parse
javascript_code = """
function add(a, b) {
  return a + b;
}
"""

# Parse the JavaScript code
parsed = esprima.parseScript(javascript_code, tolerant=True, comment=True, attachComment=True)

# Print the _fields attribute of each node in the AST
def print_fields(node):
    print(f"Node type: {node.type}, _fields: {node._fields}")
    for key, value in node.items():
        if isinstance(value, esprima.nodes.Node):
            print_fields(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, esprima.nodes.Node):
                    print_fields(item)

print_fields(parsed)
