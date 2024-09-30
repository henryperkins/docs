// insert_docstrings.js

const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const prettier = require('prettier');

/**
 * Inserts JSDoc comments into a JS/TS file based on the provided documentation.
 * @param {string} filePath - The path to the JS/TS file.
 * @param {string} docPath - The path to the JSON documentation file.
 */
function insertDocstrings(filePath, docPath) {
    const code = fs.readFileSync(filePath, 'utf8');
    const docContent = fs.readFileSync(docPath, 'utf8');
    const documentation = JSON.parse(docContent);
    const ext = path.extname(filePath).toLowerCase();
    let transformedCode = code;

    try {
        if (ext === '.ts' || ext === '.tsx' || ext === '.js' || ext === '.jsx') {
            const sourceFile = ts.createSourceFile(
                filePath,
                transformedCode,
                ts.ScriptTarget.Latest,
                true,
                ext.includes('ts') ? ts.ScriptKind.TS : ts.ScriptKind.JS
            );

            const printer = ts.createPrinter({ newLine: ts.NewLineKind.LineFeed });
            const transformer = (context) => (rootNode) => {
                function visit(node) {
                    // Insert JSDoc for functions
                    if (ts.isFunctionDeclaration(node) && node.name) {
                        const funcDoc = documentation.functions.find(f => f.name === node.name.text);
                        if (funcDoc && !hasJSDoc(node)) {
                            const jsDocComment = createJSDoc(funcDoc);
                            // Insert the JSDoc comment as a leading comment
                            const updatedNode = ts.addSyntheticLeadingComment(
                                node,
                                ts.SyntaxKind.MultiLineCommentTrivia,
                                `*\n * ${jsDocComment.replace(/\n/g, '\n * ')}\n `,
                                true
                            );
                            return updatedNode;
                        }
                    }

                    // Insert JSDoc for classes
                    if (ts.isClassDeclaration(node) && node.name) {
                        const classDoc = documentation.classes.find(c => c.name === node.name.text);
                        if (classDoc && !hasJSDoc(node)) {
                            const jsDocComment = createJSDoc(classDoc);
                            const updatedClass = ts.addSyntheticLeadingComment(
                                node,
                                ts.SyntaxKind.MultiLineCommentTrivia,
                                `*\n * ${jsDocComment.replace(/\n/g, '\n * ')}\n `,
                                true
                            );
                            // Insert JSDoc comments for class methods
                            const updatedMembers = node.members.map(member => {
                                if (ts.isMethodDeclaration(member) && member.name) {
                                    const methodDoc = classDoc.methods.find(m => m.name === member.name.text);
                                    if (methodDoc && !hasJSDoc(member)) {
                                        const methodJsDoc = createJSDoc(methodDoc);
                                        return ts.addSyntheticLeadingComment(
                                            member,
                                            ts.SyntaxKind.MultiLineCommentTrivia,
                                            `*\n * ${methodJsDoc.replace(/\n/g, '\n * ')}\n `,
                                            true
                                        );
                                    }
                                }
                                return member;
                            });

                            return ts.factory.updateClassDeclaration(
                                updatedClass,
                                updatedClass.modifiers,
                                updatedClass.name,
                                updatedClass.typeParameters,
                                updatedClass.heritageClauses,
                                updatedMembers
                            );
                        }
                    }

                    return ts.visitEachChild(node, visit, context);
                }
                return ts.visitNode(rootNode, visit);
            };

            const result = ts.transform(sourceFile, [transformer]);
            const transformedSourceFile = result.transformed[0];
            transformedCode = printer.printFile(transformedSourceFile);

            // Format the transformed code with Prettier
            transformedCode = prettier.format(transformedCode, {
                parser: ext.includes('ts') ? 'typescript' : 'babel',
                singleQuote: true,
                trailingComma: 'all',
            });
        }

        // Write the transformed code back to the file
        fs.writeFileSync(filePath, transformedCode, 'utf8');
        console.log(`Inserted docstrings into ${filePath}`);
    } catch (error) {
        console.error(`Error inserting docstrings into ${filePath}: ${error.message}`);
        process.exit(1);
    }
}

/**
 * Creates a JSDoc comment string based on documentation.
 * @param {object} doc - The documentation object.
 * @returns {string} - A formatted JSDoc comment string.
 */
function createJSDoc(doc) {
    let comment = `Summary: ${doc.description || doc.docstring}`;

    if (doc.parameters && doc.parameters.length > 0) {
        comment += `\n\n@parameters`;
        doc.parameters.forEach(param => {
            comment += `\n * @param {${param.type}} ${param.name} - ${param.description || ''}`;
        });
    }

    if (doc.returns) {
        comment += `\n\n@returns {${doc.returns.type}} - ${doc.returns.description || ''}`;
    }

    return comment;
}

/**
 * Checks if a node already has a JSDoc comment.
 * @param {ts.Node} node - The AST node.
 * @returns {boolean} - True if JSDoc exists, false otherwise.
 */
function hasJSDoc(node) {
    return !!(node.jsDoc && node.jsDoc.length > 0);
}

// Main Execution
if (require.main === module) {
    const args = process.argv.slice(2);
    if (args.length !== 2) {
        console.error('Usage: node insert_docstrings.js <path_to_js_or_ts_file> <path_to_doc_json>');
        process.exit(1);
    }

    const filePath = path.resolve(args[0]);
    const docPath = path.resolve(args[1]);

    if (!fs.existsSync(filePath)) {
        console.error(`JS/TS file not found: ${filePath}`);
        process.exit(1);
    }

    if (!fs.existsSync(docPath)) {
        console.error(`Documentation JSON file not found: ${docPath}`);
        process.exit(1);
    }

    insertDocstrings(filePath, docPath);
}