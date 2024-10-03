// acorn_inserter.js

const acorn = require('acorn');
const walk = require('acorn-walk');
const astring = require('astring');
const fs = require('fs');

// Read data from stdin (code and documentation)
let inputChunks = [];
process.stdin.on('data', chunk => {
    inputChunks.push(chunk);
});

process.stdin.on('end', () => {
    const inputData = JSON.parse(inputChunks.join(''));
    const { code, documentation } = inputData;

    // Parse the code using acorn
    let ast;
    try {
        ast = acorn.parse(code, {
            ecmaVersion: 'latest',
            sourceType: 'module',
            locations: true,
            ranges: true,
            onComment: []
        });
    } catch (e) {
        console.error(`Acorn parsing error: ${e.message}`);
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
        const commentNode = {
            type: 'Block',
            value: `*\n * ${docstring.replace(/\n/g, '\n * ')}\n `,
            start: node.start,
            end: node.start,
            loc: node.loc,
            range: [node.start, node.start]
        };
        node.leadingComments = node.leadingComments || [];
        node.leadingComments.push(commentNode);
    }

    // Traverse the AST and insert docstrings
    walk.simple(ast, {
        FunctionDeclaration(node) {
            const name = node.id ? node.id.name : 'anonymous';
            if (docstringsMapping[name]) {
                insertDocstring(node, docstringsMapping[name]);
            }
        },
        VariableDeclaration(node) {
            node.declarations.forEach(declarator => {
                if (
                    declarator.init &&
                    (declarator.init.type === 'FunctionExpression' || declarator.init.type === 'ArrowFunctionExpression')
                ) {
                    const name = declarator.id.name;
                    if (docstringsMapping[name]) {
                        insertDocstring(declarator.init, docstringsMapping[name]);
                    }
                }
            });
        },
        ClassDeclaration(node) {
            const className = node.id.name;
            if (docstringsMapping[className]) {
                insertDocstring(node, docstringsMapping[className]);
            }
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
    });

    // Generate code from modified AST, including comments
    const modifiedCode = astring.generate(ast, {
        comments: true
    });

    // Output the modified code
    console.log(modifiedCode);
});
