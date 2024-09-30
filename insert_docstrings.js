// insert_docstrings.js

const fs = require('fs');
const ts = require('typescript');

function insertDocstrings(filePath, docPath) {
    const code = fs.readFileSync(filePath, 'utf8');
    const documentation = JSON.parse(fs.readFileSync(docPath, 'utf8'));
    const sourceFile = ts.createSourceFile(filePath, code, ts.ScriptTarget.Latest, true);
    
    const printer = ts.createPrinter({ newLine: ts.NewLineKind.LineFeed });
    const transformer = (context) => (rootNode) => {
        function visit(node) {
            if (ts.isFunctionDeclaration(node) && node.name) {
                const doc = documentation.functions.find(f => f.name === node.name.text);
                if (doc && !ts.getJSDocTags(node).length) {
                    const jsDoc = ts.addSyntheticLeadingComment(
                        node,
                        ts.SyntaxKind.MultiLineCommentTrivia,
                        `* Summary: ${doc.docstring}\n * Changes: ${documentation.changes.join(', ')}`,
                        true
                    );
                    return jsDoc;
                }
            }
            if (ts.isClassDeclaration(node) && node.name) {
                const doc = documentation.classes.find(c => c.name === node.name.text);
                if (doc && !ts.getJSDocTags(node).length) {
                    const jsDoc = ts.addSyntheticLeadingComment(
                        node,
                        ts.SyntaxKind.MultiLineCommentTrivia,
                        `* Summary: ${documentation.summary}\n * Changes: ${documentation.changes.join(', ')}`,
                        true
                    );
                    return jsDoc;
                }
                const updatedMembers = node.members.map(member => {
                    if (ts.isMethodDeclaration(member) && member.name) {
                        const methodDoc = doc.methods.find(m => m.name === member.name.text);
                        if (methodDoc && !ts.getJSDocTags(member).length) {
                            const jsDoc = ts.addSyntheticLeadingComment(
                                member,
                                ts.SyntaxKind.MultiLineCommentTrivia,
                                `* Summary: ${methodDoc.docstring}\n * Changes: ${documentation.changes.join(', ')}`,
                                true
                            );
                            return jsDoc;
                        }
                    }
                    return member;
                });
                return ts.factory.updateClassDeclaration(
                    node,
                    node.modifiers,
                    node.name,
                    node.typeParameters,
                    node.heritageClauses,
                    updatedMembers
                );
            }
            return ts.visitEachChild(node, visit, context);
        }
        return ts.visitNode(rootNode, visit);
    };
    
    const result = ts.transform(sourceFile, [transformer]);
    const transformedSourceFile = result.transformed[0];
    const newCode = printer.printFile(transformedSourceFile);
    fs.writeFileSync(filePath, newCode, 'utf8');
}

const filePath = process.argv[2];
const docPath = process.argv[3];
insertDocstrings(filePath, docPath);
