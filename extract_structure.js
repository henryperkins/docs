// extract_structure.js

const fs = require('fs');
const ts = require('typescript');

function extractStructure(filePath) {
    const code = fs.readFileSync(filePath, 'utf8');
    const sourceFile = ts.createSourceFile(filePath, code, ts.ScriptTarget.Latest, true);
    
    const structure = {
        functions: [],
        classes: []
    };
    
    function visit(node) {
        if (ts.isFunctionDeclaration(node) && node.name) {
            const name = node.name.text;
            const params = node.parameters.map(param => param.name.getText());
            const docComment = ts.getJSDocTags(node).map(tag => tag.comment).join(' ');
            structure.functions.push({ name, params, docstring: docComment });
        }
        if (ts.isClassDeclaration(node) && node.name) {
            const name = node.name.text;
            const methods = [];
            node.members.forEach(member => {
                if (ts.isMethodDeclaration(member) && member.name) {
                    const methodName = member.name.getText();
                    const params = member.parameters.map(param => param.name.getText());
                    const docComment = ts.getJSDocTags(member).map(tag => tag.comment).join(' ');
                    methods.push({ name: methodName, params, docstring: docComment });
                }
            });
            structure.classes.push({ name, methods });
        }
        ts.forEachChild(node, visit);
    }
    
    visit(sourceFile);
    return structure;
}

const filePath = process.argv[2];
const structure = extractStructure(filePath);
console.log(JSON.stringify(structure, null, 2));
