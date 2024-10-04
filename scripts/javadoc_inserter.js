// javadoc_inserter.js

const fs = require('fs');
const { parse } = require('@typescript-eslint/typescript-estree'); // Use appropriate parser for Java
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
        // Generate Javadoc format
        const javadoc = generateJavadoc(funcDoc);
        docstringsMapping[funcDoc.name] = javadoc;
      }
    });
  }

  // Process classes and their methods
  if (documentation.classes) {
    documentation.classes.forEach(classDoc => {
      if (classDoc.name && classDoc.docstring) {
        const classJavadoc = generateJavadoc(classDoc, true); // true indicates class
        docstringsMapping[classDoc.name] = classJavadoc;
      }
      if (classDoc.methods) {
        classDoc.methods.forEach(methodDoc => {
          if (methodDoc.name && methodDoc.docstring) {
            const fullName = `${classDoc.name}.${methodDoc.name}`;
            const methodJavadoc = generateJavadoc(methodDoc);
            docstringsMapping[fullName] = methodJavadoc;
          }
        });
      }
    });
  }

  // Function to generate Javadoc formatted comments
  function generateJavadoc(doc, isClass = false) {
    let javadoc = "/**\n";
    javadoc += ` * ${doc.docstring}\n`;
    if (doc.args && doc.args.length > 0) {
      doc.args.forEach(arg => {
        javadoc += ` * @param ${arg} Description.\n`; // Replace 'Description' as needed
      });
    }
    if (doc.returns) {
      javadoc += ` * @return Description.\n`; // Replace 'Description' as needed
    }
    if (isClass) {
      javadoc += ` * @class\n`;
    }
    javadoc += " */\n";
    return javadoc;
  }

  // Function to insert Javadoc comments
  function insertJavadoc(node, javadoc) {
    const comment = {
      type: 'CommentBlock',
      value: javadoc.replace('/**', '').replace('*/', '').trim(),
    };
    if (!node.leadingComments) {
      node.leadingComments = [];
    }
    node.leadingComments.push(comment);
  }

  // Traverse the AST and insert Javadoc comments
  try {
    traverse(ast, {
      enter(path) {
        const node = path.node;
        if (node.type === 'FunctionDeclaration' || node.type === 'FunctionExpression') {
          const name = node.id ? node.id.name : 'anonymous';
          if (docstringsMapping[name]) {
            insertJavadoc(node, docstringsMapping[name]);
          }
        } else if (node.type === 'ClassDeclaration') {
          const className = node.id ? node.id.name : 'anonymous';
          if (docstringsMapping[className]) {
            insertJavadoc(node, docstringsMapping[className]);
          }
          if (node.body && node.body.body) {
            node.body.body.forEach(element => {
              if (element.type === 'MethodDefinition') {
                const methodName = element.key.name;
                const fullName = `${className}.${methodName}`;
                if (docstringsMapping[fullName]) {
                  insertJavadoc(element, docstringsMapping[fullName]);
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
