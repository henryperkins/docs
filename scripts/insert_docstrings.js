// scripts/insert_docstrings.js

const fs = require('fs');
const path = require('path');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;
const t = require('@babel/types');
const prettier = require('prettier');

/**
 * Inserts docstrings into JS/TS code based on the provided documentation.
 * Usage: node insert_docstrings.js <code_file_path> <documentation_file_path>
 */
const [,, codeFilePath, docFilePath] = process.argv;

if (!codeFilePath || !docFilePath) {
    console.error('Usage: node insert_docstrings.js <code_file_path> <documentation_file_path>');
    process.exit(1);
}

try {
    const code = fs.readFileSync(codeFilePath, 'utf-8');
    const documentation = JSON.parse(fs.readFileSync(docFilePath, 'utf-8'));

    // Parse the code into an AST
    const ast = parser.parse(code, {
        sourceType: 'module',
        plugins: [
            'typescript',
            'classProperties',
            'decorators-legacy',
            'jsx',
            // Add other plugins as needed
        ],
    });

    // Traverse the AST and insert docstrings
    traverse(ast, {
        enter(path) {
            if (path.isFunctionDeclaration() || path.isClassDeclaration()) {
                const nodeName = path.node.id ? path.node.id.name : 'anonymous';
                let doc = 'No description provided.';

                if (path.isFunctionDeclaration()) {
                    const funcDoc = documentation.functions.find(f => f.name === nodeName);
                    if (funcDoc && funcDoc.docstring) {
                        doc = funcDoc.docstring;
                    }
                } else if (path.isClassDeclaration()) {
                    const classDoc = documentation.classes.find(c => c.name === nodeName);
                    if (classDoc && classDoc.docstring) {
                        doc = classDoc.docstring;
                    }
                }

                // Check if leading comments already exist
                if (!path.node.leadingComments || path.node.leadingComments.length === 0) {
                    const commentBlock = t.commentBlock(`* ${doc} `, true, true);
                    path.addComment('leading', commentBlock.value, true);
                    console.log(`Inserted docstring for ${path.node.type}: ${nodeName}`);
                }
            }

            // Handle methods within classes
            if (path.isClassMethod()) {
                const methodName = path.node.key.name;
                const classPath = path.findParent(p => p.isClassDeclaration());
                const className = classPath.node.id ? classPath.node.id.name : 'anonymous';
                let doc = 'No description provided.';

                const classDoc = documentation.classes.find(c => c.name === className);
                if (classDoc) {
                    const methodDoc = classDoc.methods.find(m => m.name === methodName);
                    if (methodDoc && methodDoc.docstring) {
                        doc = methodDoc.docstring;
                    }
                }

                // Check if leading comments already exist
                if (!path.node.leadingComments || path.node.leadingComments.length === 0) {
                    const commentBlock = t.commentBlock(`* ${doc} `, true, true);
                    path.addComment('leading', commentBlock.value, true);
                    console.log(`Inserted docstring for method: ${methodName} in class: ${className}`);
                }
            }
        }
    });

    // Generate the modified code from the AST
    let modifiedCode = generate(ast, { comments: true }, code).code;

    // Format the code using Prettier for consistency
    modifiedCode = prettier.format(modifiedCode, { parser: 'babel', singleQuote: true });

    // Output the modified code to stdout
    console.log(modifiedCode);

} catch (error) {
    console.error(`Error processing files: ${error.message}`);
    process.exit(1);
}
