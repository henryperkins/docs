// acorn_inserter.js

const fs = require('fs');
const { parse } = require('@typescript-eslint/typescript-estree');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;

let inputChunks = [];
process.stdin.on('data', chunk => {
  inputChunks.push(chunk);
});

process.stdin.on('end', () => {
  const inputData = JSON.parse(inputChunks.join(''));
  const { code, documentation, language } = inputData;

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

  // Build a map of function and class names to their docstrings
  const docstringsMapping = {};

  // Process functions
  if (documentation.functions) {
    documentation.functions.forEach(funcDoc => {
      if (funcDoc.name && funcDoc.docstring) {
        docstringsMapping[funcDoc.name] = funcDoc.docstring;
      }
    });
  }

  // Process classes and their methods
  if (documentation.classes) {
    documentation.classes.forEach(classDoc => {
      if (classDoc.name && classDoc.docstring) {
        docstringsMapping[classDoc.name] = classDoc.docstring;
      }
      if (classDoc.methods) {
        classDoc.methods.forEach(methodDoc => {
          if (methodDoc.name && methodDoc.docstring) {
            const fullName = `${classDoc.name}.${methodDoc.name}`;
            docstringsMapping[fullName] = methodDoc.docstring;
          }
        });
      }
    });
  }

  // Function to insert docstrings as comments
  function insertDocstring(node, docstring) {
    const comment = {
      type: 'Block',
      value: `*\n * ${docstring.replace(/\n/g, '\n * ')}\n `,
    };
    if (!node.leadingComments) {
      node.leadingComments = [];
    }
    node.leadingComments.push(comment);
  }

  // Traverse the AST and insert docstrings
  try {
    traverse(ast, {
      enter(path) {
        const node = path.node;
        if (node.type === 'FunctionDeclaration' || node.type === 'FunctionExpression') {
          const name = node.id ? node.id.name : 'anonymous';
          if (docstringsMapping[name]) {
            insertDocstring(node, docstringsMapping[name]);
          }
        } else if (node.type === 'ClassDeclaration') {
          const className = node.id ? node.id.name : 'anonymous';
          if (docstringsMapping[className]) {
            insertDocstring(node, docstringsMapping[className]);
          }
          if (node.body && node.body.body) {
            node.body.body.forEach(element => {
              if (element.type === 'MethodDefinition') {
                const methodName = element.key.name;
                const fullName = `${className}.${methodName}`;
                if (docstringsMapping[fullName]) {
                  insertDocstring(element, docstringsMapping[fullName]);
                }
              }
            });
          }
        }
      },
    });
  } catch (e) {
    console.error(`Error traversing AST: ${e.message}`);
    process.exit(1);
  }

  // Generate code from modified AST, including comments
  let modifiedCode;
  try {
    const result = generate(ast, { comments: true });
    modifiedCode = result.code;
  } catch (e) {
    console.error(`Error generating code from AST: ${e.message}`);
    process.exit(1);
  }

  // Output the modified code
  console.log(modifiedCode);
});