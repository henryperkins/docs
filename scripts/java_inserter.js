// scripts/java_inserter.js

const fs = require('fs');
const javaParser = require('java-parser');
const path = require('path');

// Read input from stdin
let inputChunks = [];
process.stdin.on('data', (chunk) => {
  inputChunks.push(chunk);
});

process.stdin.on('end', () => {
  const inputData = inputChunks.join('');

  let parsedInput;
  try {
    parsedInput = JSON.parse(inputData);
  } catch (e) {
    console.error('Error parsing input JSON:', e.message);
    process.exit(1);
  }

  const { code, documentation, language } = parsedInput;

  if (language.toLowerCase() !== 'java') {
    console.error('Unsupported language:', language);
    process.exit(1);
  }

  let ast;
  try {
    ast = javaParser.parse(code);
  } catch (e) {
    console.error('Parsing error:', e.message);
    process.exit(1);
  }

  // Helper function to insert docstrings
  function insertDocstring(node, docstring) {
    if (node.documentation) {
      node.documentation = `/**\n * ${docstring}\n */`;
    } else {
      node.documentation = `/** ${docstring} */`;
    }
  }

  // Traverse documentation to insert into AST
  documentation.classes.forEach(clsDoc => {
    const cls = ast.children.find(child => child.node === 'ClassDeclaration' && child.name.identifier === clsDoc.name);
    if (cls) {
      insertDocstring(cls, clsDoc.docstring);
      cls.body.body.forEach(member => {
        if (member.node === 'MethodDeclaration') {
          const methodDoc = clsDoc.methods.find(m => m.name === member.name.identifier);
          if (methodDoc) {
            insertDocstring(member, methodDoc.docstring);
          }
        }
      });
    }
  });

  documentation.functions.forEach(funcDoc => {
    // Java functions are typically static methods; find and insert
    const cls = ast.children.find(child => child.node === 'ClassDeclaration');
    if (cls) {
      const method = cls.body.body.find(member => member.node === 'MethodDeclaration' && member.name.identifier === funcDoc.name);
      if (method) {
        insertDocstring(method, funcDoc.docstring);
      }
    }
  });

  // Note: java-parser does not support code generation. To output modified code,
  // consider using alternative libraries or integrating with Java tools that support AST modifications.

  // As a placeholder, output the original code
  // In a real-world scenario, you would need to use a Java code generation library
  // or implement a method to serialize the modified AST back to source code.
  console.log(code);
});
