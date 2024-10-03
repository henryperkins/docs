const acorn = require('acorn');
const walk = require('acorn-walk');
const Ajv = require('ajv');
const fs = require('fs');

// Read data from stdin (code and schema)
const inputData = JSON.parse(fs.readFileSync(0).toString());
const code = inputData.code;

// Read schemas
const functionSchema = JSON.parse(fs.readFileSync('function_schema.json').toString());
const classSchema = JSON.parse(fs.readFileSync('class_schema.json').toString());
const methodSchema = JSON.parse(fs.readFileSync('method.schema.json').toString());

// Create an Ajv validator instance
const ajv = new Ajv({ allErrors: true });

// Add method schema to ajv
ajv.addSchema(methodSchema, 'https://example.com/method.schema.json');

// Compile the schemas
const validateFunction = ajv.compile(functionSchema);
const validateClass = ajv.compile(classSchema);

// Initialize an array to collect comments
const comments = [];

// Parse the code using acorn
const ast = acorn.parse(code, { 
    ecmaVersion: 'latest', 
    sourceType: 'module',
    locations: true,
    ranges: true,
    onComment: comments
});

// Function to associate comments with nodes
function attachComments(ast, comments) {
  let commentIndex = 0;
  let lastComment = null;

  walk.simple(ast, {
    Program(node) {
      node.body.forEach(child => {
        attachCommentToNode(child);
      });
    },
    FunctionDeclaration(node) {
      attachCommentToNode(node);
    },
    VariableDeclaration(node) {
      attachCommentToNode(node);
    },
    ClassDeclaration(node) {
      attachCommentToNode(node);
    }
  });

  function attachCommentToNode(node) {
    while (commentIndex < comments.length && comments[commentIndex].end <= node.start) {
      lastComment = comments[commentIndex];
      commentIndex++;
    }
    if (lastComment && lastComment.end <= node.start) {
      node.leadingComments = node.leadingComments || [];
      node.leadingComments.push(lastComment);
    }
  }
}

attachComments(ast, comments);

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
      docstring += comment.value.replace(/^\s*\*\s?/gm, '').trim() + '\n';
    }
  }
  return docstring.trim();
}

// Walk the AST to extract functions and classes
walk.simple(ast, {
  FunctionDeclaration(node) {
    const func = {
      name: node.id ? node.id.name : 'anonymous',
      args: node.params.map(param => param.name || 'param'),
      async: node.async || false,
      docstring: extractDocstring(node)
    };

    // Schema Validation
    const valid = validateFunction(func);
    if (!valid) {
      console.error("Function data validation failed:", validateFunction.errors);
      // Handle the error appropriately (e.g., throw an exception)
      throw new Error("Function data validation failed");
    }

    structure.functions.push(func);
  },
  VariableDeclaration(node) {
    node.declarations.forEach(declarator => {
      if (
        declarator.init &&
        (declarator.init.type === 'FunctionExpression' || declarator.init.type === 'ArrowFunctionExpression')
      ) {
        const func = {
          name: declarator.id.name,
          args: declarator.init.params.map(param => param.name || 'param'),
          async: declarator.init.async || false,
          docstring: extractDocstring(declarator)
        };

        // Schema Validation
        const valid = validateFunction(func);
        if (!valid) {
          console.error("Function data validation failed:", validateFunction.errors);
          throw new Error("Function data validation failed");
        }

        structure.functions.push(func);
      }
    });
  },
  ClassDeclaration(node) {
    const methods = [];
    node.body.body.forEach(element => {
      if (element.type === 'MethodDefinition') {
        const method = {
          name: element.key.name,
          args: element.value.params.map(param => param.name || 'param'),
          async: element.value.async || false,
          kind: element.kind,
          docstring: extractDocstring(element)
        };

        // Schema Validation for method
        const validMethod = ajv.validate('https://example.com/method.schema.json', method);
        if (!validMethod) {
          console.error("Method data validation failed:", ajv.errors);
          throw new Error("Method data validation failed");
        }

        methods.push(method);
      }
    });

    const cls = {
      name: node.id.name,
      methods: methods,
      docstring: extractDocstring(node)
    };

    // Schema Validation for class
    const validClass = validateClass(cls);
    if (!validClass) {
      console.error("Class data validation failed:", validateClass.errors);
      throw new Error("Class data validation failed");
    }

    structure.classes.push(cls);
  }
});

// Output the extracted structure as JSON to stdout
console.log(JSON.stringify(structure, null, 2));

