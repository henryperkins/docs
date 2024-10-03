const acorn = require('acorn');
const { generate } = require('astring'); // Use astring for code generation

// Read data from stdin (code and documentation)
const inputData = JSON.parse(require('fs').readFileSync(0).toString());
const code = inputData.code;
const documentation = inputData.documentation;

// Parse the code using acorn
const ast = acorn.parse(code, { 
    ecmaVersion: 'latest', 
    sourceType: 'module', 
    onComment: (block, text, start, end, loc) => {
        // You might need to adjust this logic based on how you want to handle existing comments
        if (block && text.trim().startsWith('*')) {
            // Preserve existing docstring-style comments
            ast.comments.push({ type: 'Block', value: text, start, end, loc });
        }
    } 
});

// Function to insert docstrings into the AST
function insertDocstring(node, docstring) {
    if (!ast.comments) {
        ast.comments = [];
    }
    ast.comments.push({
        type: 'Block',
        value: `* ${docstring} `,
        start: node.start,
        end: node.start, // Place the comment right before the node
        loc: {
            start: { line: node.loc.start.line, column: node.loc.start.column },
            end: { line: node.loc.start.line, column: node.loc.start.column }
        }
    });
}

// Traverse the AST and insert docstrings
acorn.walk.simple(ast, {
    FunctionDeclaration(node) {
        const funcDoc = documentation.functions.find(f => f.name === node.id.name);
        if (funcDoc && funcDoc.docstring) {
            insertDocstring(node, funcDoc.docstring);
        }
    },
    ClassDeclaration(node) {
        const classDoc = documentation.classes.find(c => c.name === node.id.name);
        if (classDoc && classDoc.docstring) {
            insertDocstring(node, classDoc.docstring);
        }
        if (classDoc) {
            node.body.body.forEach(methodNode => {
                if (methodNode.type === 'MethodDefinition') {
                    const methodDoc = classDoc.methods.find(m => m.name === methodNode.key.name);
                    if (methodDoc && methodDoc.docstring) {
                        insertDocstring(methodNode, methodDoc.docstring);
                    }
                }
            });
        }
    }
});

// Generate the modified code from the AST
const modifiedCode = generate(ast, { comments: true });

// Output the modified code to stdout
console.log(modifiedCode);
