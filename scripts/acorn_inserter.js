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
        // Generate JSDoc format
        const jsDoc = generateJSDoc(funcDoc);
        docstringsMapping[funcDoc.name] = jsDoc;
      }
    });
  }

  // Process classes and their methods
  if (documentation.classes) {
    documentation.classes.forEach(classDoc => {
      if (classDoc.name && classDoc.docstring) {
        const classJsDoc = generateJSDoc(classDoc, true); // true indicates class
        docstringsMapping[classDoc.name] = classJsDoc;
      }
      if (classDoc.methods) {
        classDoc.methods.forEach(methodDoc => {
          if (methodDoc.name && methodDoc.docstring) {
            const fullName = `${classDoc.name}.${methodDoc.name}`;
            const methodJsDoc = generateJSDoc(methodDoc);
            docstringsMapping[fullName] = methodJsDoc;
          }
        });
      }
    });
  }

  // Function to generate JSDoc formatted comments
  function generateJSDoc(doc, isClass = false) {
    let jsDoc = '/**\n';
    jsDoc += ` * ${doc.docstring}\n`;
    if (doc.args && doc.args.length > 0) {
      doc.args.forEach(arg => {
        jsDoc += ` * @param {type} ${arg} - Description.\n`; // Replace 'type' and 'Description' as needed
      });
    }
    if (doc.returns) {
      jsDoc += ` * @returns {type} Description.\n`; // Replace 'type' and 'Description' as needed
    }
    if (isClass) {
      jsDoc += ` * @class\n`;
    }
    jsDoc += ' */\n';
    return jsDoc;
  }

  // Function to insert JSDoc comments
  function insertJSDoc(node, jsDoc) {
    const comment = {
      type: 'CommentBlock',
      value: jsDoc.replace('/**', '').replace('*/', '').trim(),
    };
    if (!node.leadingComments) {
      node.leadingComments = [];
    }
    node.leadingComments.push(comment);
  }

  // Traverse the AST and insert JSDoc comments
  try {
    traverse(ast, {
      enter(path) {
        const node = path.node;
        if (node.type === 'FunctionDeclaration' || node.type === 'FunctionExpression') {
          const name = node.id ? node.id.name : 'anonymous';
          if (docstringsMapping[name]) {
            insertJSDoc(node, docstringsMapping[name]);
          }
        } else if (node.type === 'ClassDeclaration') {
          const className = node.id ? node.id.name : 'anonymous';
          if (docstringsMapping[className]) {
            insertJSDoc(node, docstringsMapping[className]);
          }
          if (node.body && node.body.body) {
            node.body.body.forEach(element => {
              if (element.type === 'MethodDefinition') {
                const methodName = element.key.name;
                const fullName = `${className}.${methodName}`;
                if (docstringsMapping[fullName]) {
                  insertJSDoc(element, docstringsMapping[fullName]);
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
