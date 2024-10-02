// scripts/extract_structure.js

const fs = require('fs');
const acorn = require('acorn');
const walk = require('acorn-walk');

const filePath = process.argv[2];

if (!filePath) {
  console.error('No file path provided.');
  process.exit(1);
}

fs.readFile(filePath, 'utf8', (err, code) => {
  if (err) {
    console.error('Error reading the file:', err);
    process.exit(1);
  }

  try {
    const parsed = acorn.parse(code, { ecmaVersion: 'latest', sourceType: 'module' });
    const structure = {
      functions: [],
      classes: []
    };

    walk.simple(parsed, {
      FunctionDeclaration(node) {
        structure.functions.push({
          name: node.id ? node.id.name : 'anonymous',
          args: node.params.map(param => param.name || 'param'),
          async: node.async,
          docstring: '' // Placeholder for docstring extraction if needed
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
              docstring: '' // Placeholder
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
              docstring: '' // Placeholder
            });
          }
        });
        structure.classes.push({
          name: node.id.name,
          methods: methods,
          docstring: '' // Placeholder
        });
      }
    });

    console.log(JSON.stringify(structure, null, 2));
  } catch (parseErr) {
    console.error('Error parsing JavaScript code:', parseErr);
    process.exit(1);
  }
});
