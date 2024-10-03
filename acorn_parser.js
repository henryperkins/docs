const acorn = require('acorn');
const Ajv = require('ajv');
const { generate } = require('astring');

// Read data from stdin (code, documentation, and schema)
const inputData = JSON.parse(require('fs').readFileSync(0).toString());
const code = inputData.code;
const documentation = inputData.documentation;
const functionSchema = inputData.functionSchema;

// Create an Ajv validator instance
const ajv = new Ajv();

// Compile the schema
const validateFunction = ajv.compile(functionSchema);

// Parse the code using acorn
const ast = acorn.parse(code, { 
    ecmaVersion: 'latest', 
    sourceType: 'module',
    onComment: (block, text, start, end, loc) => {
        if (block && text.trim().startsWith('*')) {
            ast.comments.push({ type: 'Block', value: text, start, end, loc });
        }
    } 
});

// Extract functions and classes from the AST
const structure = {
  functions: [],
  classes: []
};

// Function to extract docstrings from leading comments
function extractDocstring(node) {
  let docstring = '';
  const leadingComments = node.leadingComments || [];
  for (const comment of leadingComments) {
    if (comment.type === 'Block' && comment.value.trim().startsWith('*')) {
      docstring += comment.value.replace(/^\s*\*\s?/gm, '').trim() + '\n'; // Remove leading "*" and whitespace
    }
  }
  return docstring.trim();
}

acorn.walk.simple(ast, {
  FunctionDeclaration(node) {
    structure.functions.push({
      name: node.id ? node.id.name : 'anonymous',
      args: node.params.map(param => param.name || 'param'),
      async: node.async,
      docstring: extractDocstring(node)
    });
  },
  VariableDeclaration(node) {
    node.declarations.forEach(declarator => {
      if (
        declarator.init &&
        (declarator.init.type === 'FunctionExpression' || declarator.init.type === 'ArrowFunctionExpression')
      ) {
        structure.functions.push({
          name: declarator.id.name,
          args: declarator.init.params.map(param => param.name || 'param'),
          async: declarator.init.async,
          docstring: extractDocstring(declarator)
        });
      }
    });
  },
  ClassDeclaration(node) {
    const methods = [];
    node.body.body.forEach(element => {
      if (element.type === 'MethodDefinition') {
        methods.push({
          name: element.key.name,
          args: element.value.params.map(param => param.name || 'param'),
          async: element.value.async,
          docstring: extractDocstring(element)
        });
      }
    });
    structure.classes.push({
      name: node.id.name,
      methods: methods,
      docstring: extractDocstring(node)
    });
  }
});

// Schema Validation:
structure.functions.forEach(func => {
  const valid = validateFunction(func);
  if (!valid) {
    console.error("Function data validation failed:", validateFunction.errors);
    // Handle the error appropriately (e.g., throw an exception)
    throw new Error("Function data validation failed");
  }
});

// Output the extracted structure as JSON to stdout
console.log(JSON.stringify(structure, null, 2));
