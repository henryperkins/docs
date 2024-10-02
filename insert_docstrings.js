// insert_docstrings.js

const fs = require('fs');
const babelParser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generator = require('@babel/generator').default;
const t = require('@babel/types');

/**
 * Reads all data from stdin.
 * @returns {Promise<string>} The input code as a string.
 */
function readStdin() {
    return new Promise((resolve, reject) => {
        let data = '';
        process.stdin.setEncoding('utf-8');

        process.stdin.on('data', chunk => {
            data += chunk;
        });

        process.stdin.on('end', () => {
            resolve(data);
        });

        process.stdin.on('error', err => {
            reject(err);
        });
    });
}

/**
 * Inserts JSDoc comments into functions and classes that lack them.
 * @param {object} ast - The Abstract Syntax Tree of the code.
 */
function insertDocstrings(ast) {
    traverse(ast, {
        FunctionDeclaration(path) {
            if (!hasDocstring(path)) {
                const docComment = generateDocstringForFunction(path.node);
                path.addComment('leading', docComment);
                const funcName = path.node.id ? path.node.id.name : 'anonymous';
                const line = path.node.loc.start.line;
                console.log(`Inserted docstring in function: ${funcName} at line ${line}`);
            }
        },
        ClassDeclaration(path) {
            if (!hasDocstring(path)) {
                const docComment = generateDocstringForClass(path.node);
                path.addComment('leading', docComment);
                const className = path.node.id ? path.node.id.name : 'anonymous';
                const line = path.node.loc.start.line;
                console.log(`Inserted docstring in class: ${className} at line ${line}`);
            }
        },
        VariableDeclaration(path) {
            // Handle arrow functions and function expressions assigned to variables
            path.node.declarations.forEach(declaration => {
                if (
                    declaration.init &&
                    (declaration.init.type === 'ArrowFunctionExpression' ||
                        declaration.init.type === 'FunctionExpression')
                ) {
                    const funcPath = path.get('declarations').find(decl => decl.node === declaration).get('init');
                    if (!hasDocstring(funcPath)) {
                        const docComment = generateDocstringForFunction(declaration.init, declaration.id.name);
                        funcPath.addComment('leading', docComment);
                        const funcName = declaration.id.name;
                        const line = declaration.loc.start.line;
                        console.log(`Inserted docstring in function: ${funcName} at line ${line}`);
                    }
                }
            });
        },
    });
}

/**
 * Checks if a node already has a docstring.
 * @param {object} path - The Babel path object.
 * @returns {boolean} True if docstring exists, else False.
 */
function hasDocstring(path) {
    const leadingComments = path.node.leadingComments;
    if (leadingComments && leadingComments.length > 0) {
        // Check if the first leading comment is a JSDoc comment
        const firstComment = leadingComments[0];
        if (firstComment.type === 'CommentBlock' && firstComment.value.startsWith('*')) {
            return true;
        }
    }
    return false;
}

/**
 * Generates a JSDoc comment for a function.
 * @param {object} node - The function node.
 * @param {string} [funcName] - Optional function name for anonymous functions.
 * @returns {string} The JSDoc comment string.
 */
function generateDocstringForFunction(node, funcName) {
    const name = funcName || (node.id ? node.id.name : 'anonymous');
    const params = node.params.map(param => param.name || 'unknown');
    let doc = `*\n * ${name} - Description of the function.\n`;

    params.forEach(param => {
        doc += ` * @param {type} ${param} - Description of ${param}.\n`;
    });

    doc += ` * @returns {type} Description of return value.\n `;
    return doc;
}

/**
 * Generates a JSDoc comment for a class.
 * @param {object} node - The class node.
 * @returns {string} The JSDoc comment string.
 */
function generateDocstringForClass(node) {
    const name = node.id ? node.id.name : 'anonymous';
    let doc = `*\n * ${name} - Description of the class.\n`;

    // Optionally, add more details about class properties and methods
    doc += ` *\n * @class\n `;
    return doc;
}

/**
 * Main function to execute the script.
 */
async function main() {
    try {
        const inputCode = await readStdin();

        if (!inputCode.trim()) {
            console.error('No input provided.');
            process.exit(1);
        }

        const ast = babelParser.parse(inputCode, {
            sourceType: 'module',
            plugins: [
                'typescript',
                'jsx',
                'classProperties',
                'decorators-legacy',
                'dynamicImport',
                // Add other plugins as needed
            ],
        });

        insertDocstrings(ast);

        const output = generator(ast, {
            comments: true,
        }, inputCode).code;

        console.log(output);
    } catch (error) {
        console.error(`Error inserting docstrings into JS/TS file: ${error.message}`);
        process.exit(1);
    }
}

main();
