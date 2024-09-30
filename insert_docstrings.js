const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const prettier = require('prettier');

const function_schema = {
    // Your function schema details here
};

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

function hasJSDoc(node) {
    const comments = node.leadingComments;
    return comments && comments.some(comment => comment.type === 'CommentBlock' && comment.value.startsWith('*'));
}

async function insertDocstrings(filePath, documentation) {
    const code = fs.readFileSync(filePath, 'utf8');
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
                    if (ts.isFunctionDeclaration(node) && node.name) {
                        const funcDoc = documentation.functions.find(f => f.name === node.name.text);
                        if (funcDoc && !hasJSDoc(node)) {
                            const jsDocComment = createJSDoc(funcDoc);
                            const updatedNode = ts.addSyntheticLeadingComment(
                                node,
                                ts.SyntaxKind.MultiLineCommentTrivia,
                                `*\n * ${jsDocComment.replace(/\n/g, '\n * ')}\n `,
                                true
                            );
                            return updatedNode;
                        }
                    }

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

            transformedCode = prettier.format(transformedCode, {
                parser: ext.includes('ts') ? 'typescript' : 'babel',
                singleQuote: true,
                trailingComma: 'all',
            });
        }

        fs.writeFileSync(filePath, transformedCode, 'utf8');
        console.log(`Inserted docstrings into ${filePath}`);
    } catch (error) {
        console.error(`Error inserting docstrings into ${filePath}: ${error.message}`);
        process.exit(1);
    }
}

if (require.main === module) {
    const filePath = process.argv[2];
    const docPath = process.argv[3];

    if (!filePath || !docPath) {
        console.error('Usage: node insert_docstrings.js <path_to_js_or_ts_file> <path_to_doc_file>');
        process.exit(1);
    }

    const absoluteFilePath = path.resolve(filePath);
    const absoluteDocPath = path.resolve(docPath);

    if (!fs.existsSync(absoluteFilePath) || !fs.existsSync(absoluteDocPath)) {
        console.error(`File not found: ${absoluteFilePath} or ${absoluteDocPath}`);
        process.exit(1);
    }

    const docContent = fs.readFileSync(absoluteDocPath, 'utf8');
    const documentation = JSON.parse(docContent);

    insertDocstrings(absoluteFilePath, documentation);
}
