// extract_structure.js

const fs = require('fs');
const babelParser = require('@babel/parser');
const traverse = require('@babel/traverse').default;

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
 * Extracts the structure of the code, including functions and classes.
 * @param {object} ast - The Abstract Syntax Tree of the code.
 * @returns {object} The extracted structure.
 */
function extractStructure(ast) {
    const structure = {
        functions: [],
        classes: [],
    };

    traverse(ast, {
        FunctionDeclaration(path) {
            const func = {
                name: path.node.id.name,
                params: path.node.params.map(param => param.name || 'unknown'),
                line: path.node.loc.start.line,
            };
            structure.functions.push(func);
        },
        VariableDeclaration(path) {
            // Handle arrow functions assigned to variables
            path.node.declarations.forEach(declaration => {
                if (
                    declaration.init &&
                    (declaration.init.type === 'ArrowFunctionExpression' ||
                        declaration.init.type === 'FunctionExpression')
                ) {
                    const func = {
                        name: declaration.id.name,
                        params: declaration.init.params.map(param => param.name || 'unknown'),
                        line: declaration.loc.start.line,
                    };
                    structure.functions.push(func);
                }
            });
        },
        ClassDeclaration(path) {
            const cls = {
                name: path.node.id.name,
                methods: [],
                line: path.node.loc.start.line,
            };

            path.traverse({
                ClassMethod(classPath) {
                    const method = {
                        name: classPath.node.key.name,
                        params: classPath.node.params.map(param => param.name || 'unknown'),
                        kind: classPath.node.kind, // constructor, method, getter, setter
                        line: classPath.node.loc.start.line,
                    };
                    cls.methods.push(method);
                },
                ClassProperty(classPath) {
                    // Handle class properties if needed
                },
            });

            structure.classes.push(cls);
        },
    });

    return structure;
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

        const structure = extractStructure(ast);

        console.log(JSON.stringify(structure, null, 2));
    } catch (error) {
        console.error(`Error parsing JS/TS file: ${error.message}`);
        process.exit(1);
    }
}

main();
