// acorn_parser.js

const fs = require('fs');
const { parse } = require('@typescript-eslint/typescript-estree');
const Ajv = require('ajv');

const ajv = new Ajv({ allErrors: true });

// Read data from stdin
let inputChunks = [];
process.stdin.on('data', chunk => {
  inputChunks.push(chunk);
});

process.stdin.on('end', () => {
  const inputData = JSON.parse(inputChunks.join(''));
  const { code, language, functionSchema } = inputData;

  // Parse the code
  let ast;
  try {
    ast = parse(code, {
      loc: true,
      range: true,
      comment: true,
      tokens: true,
      errorOnUnknownASTType: false,
      jsx: true,
    });
  } catch (e) {
    console.error(`Parsing error: ${e.message}`);
    process.exit(1);
  }

  // Build a map of comments
  const commentsMap = new Map();
  ast.comments.forEach(comment => {
    commentsMap.set(comment.range[1], comment);
  });

  // Function to extract docstrings from leading comments
  function extractDocstring(node) {
    if (node.leadingComments && node.leadingComments.length > 0) {
      const lastComment = node.leadingComments[node.leadingComments.length - 1];
      return lastComment.value.trim();
    }
    return null;
  }

  // Extract functions and classes from the AST
  const structure = {
    functions: [],
    classes: [],
  };

  function traverse(node) {
    switch (node.type) {
      case 'FunctionDeclaration':
        const func = {
          name: node.id ? node.id.name : 'anonymous',
          args: node.params.map(param => (param.name ? param.name : 'param')),
          async: node.async || false,
          docstring: extractDocstring(node),
        };
        structure.functions.push(func);
        break;
      case 'ClassDeclaration':
        const cls = {
          name: node.id ? node.id.name : 'anonymous',
          docstring: extractDocstring(node),
          methods: [],
        };
        if (node.body && node.body.body) {
          node.body.body.forEach(element => {
            if (element.type === 'MethodDefinition') {
              const method = {
                name: element.key.name,
                args: element.value.params.map(param =>
                  param.name ? param.name : 'param'
                ),
                async: element.value.async || false,
                kind: element.kind,
                docstring: extractDocstring(element),
              };
              cls.methods.push(method);
            }
          });
        }
        structure.classes.push(cls);
        break;
      default:
        break;
    }
    // Recurse on child nodes
    for (const key in node) {
      if (node.hasOwnProperty(key)) {
        const child = node[key];
        if (Array.isArray(child)) {
          child.forEach(c => {
            if (c && typeof c.type === 'string') {
              traverse(c);
            }
          });
        } else if (child && typeof child.type === 'string') {
          traverse(child);
        }
      }
    }
  }

  traverse(ast);

  // If functionSchema is provided, perform validation
  if (functionSchema) {
    try {
      const validate = ajv.compile(functionSchema);
      const valid = validate(structure);

      if (!valid) {
        console.error(
          'Validation errors:',
          JSON.stringify(validate.errors, null, 2)
        );
        process.exit(1);
      }
    } catch (e) {
      console.error(`Schema validation error: ${e.message}`);
      process.exit(1);
    }
  }

  // Output the extracted structure as JSON
  console.log(JSON.stringify(structure, null, 2));
});
