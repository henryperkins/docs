// scripts/insert_docstrings.js

const fs = require('fs');
const recast = require('recast');

const codeFilePath = process.argv[2];
const docFilePath = process.argv[3];

if (!codeFilePath || !docFilePath) {
  console.error('Usage: node insert_docstrings.js <code_file> <doc_file>');
  process.exit(1);
}

const code = fs.readFileSync(codeFilePath, 'utf8');
const documentation = JSON.parse(fs.readFileSync(docFilePath, 'utf8'));
const ast = recast.parse(code);

recast.types.visit(ast, {
  visitFunctionDeclaration(path) {
    const node = path.node;
    const doc = documentation.functions.find(f => f.name === node.id.name);
    if (doc) {
      const comment = {
        type: 'CommentBlock',
        value: `*\n * ${doc.docstring}\n `,
        leading: true,
      };
      node.comments = node.comments || [];
      node.comments.push(comment);
    }
    this.traverse(path);
  },
  visitClassDeclaration(path) {
    const node = path.node;
    const doc = documentation.classes.find(c => c.name === node.id.name);
    if (doc) {
      const comment = {
        type: 'CommentBlock',
        value: `*\n * ${doc.docstring}\n `,
        leading: true,
      };
      node.comments = node.comments || [];
      node.comments.push(comment);
    }
    this.traverse(path);
  },
});

const output = recast.print(ast).code;
console.log(output);
