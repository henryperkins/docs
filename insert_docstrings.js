// insert_docstrings.js
const ts = require('typescript');

// Read documentation data from stdin
let input = '';
process.stdin.on('data', function(chunk) {
  input += chunk;
});

process.stdin.on('end', function() {
  try {
    const documentation = JSON.parse(input);
    const sourceCode = documentation.source_code;

    const sourceFile = ts.createSourceFile(
      'file.ts',
      sourceCode,
      ts.ScriptTarget.Latest,
      /* setParentNodes */ true,
      ts.ScriptKind.TS
    );

    const inserts = [];

    // Create maps for quick access to documentation by function and class names
    const functionDocs = new Map();
    if (documentation.functions) {
      documentation.functions.forEach(func => {
        functionDocs.set(func.name + func.start, func);
      });
    }

    const classDocs = new Map();
    if (documentation.classes) {
      documentation.classes.forEach(cls => {
        const methodsMap = new Map();
        if (cls.methods) {
          cls.methods.forEach(method => {
            methodsMap.set(method.name + method.start, method);
          });
        }
        classDocs.set(cls.name + cls.start, {
          info: cls,
          methods: methodsMap
        });
      });
    }

    function visit(node) {
      if (ts.isFunctionDeclaration(node) || ts.isFunctionExpression(node) || ts.isArrowFunction(node)) {
        const name = node.name ? node.name.getText() : 'anonymous';
        const key = name + node.getStart();
        const docInfo = functionDocs.get(key);
        if (docInfo) {
          const jsDocComment = formatJSDoc(docInfo);
          inserts.push({
            position: node.getFullStart(),
            docstring: jsDocComment
          });
        }
      } else if (ts.isMethodDeclaration(node) && node.parent && ts.isClassDeclaration(node.parent)) {
        const className = node.parent.name ? node.parent.name.getText() : 'anonymous';
        const classKey = className + node.parent.getStart();
        const methodName = node.name.getText();
        const methodKey = methodName + node.getStart();
        const classDoc = classDocs.get(classKey);
        if (classDoc) {
          const methodDoc = classDoc.methods.get(methodKey);
          if (methodDoc) {
            const jsDocComment = formatJSDoc(methodDoc);
            inserts.push({
              position: node.getFullStart(),
              docstring: jsDocComment
            });
          }
        }
      } else if (ts.isClassDeclaration(node)) {
        const className = node.name ? node.name.getText() : 'anonymous';
        const classKey = className + node.getStart();
        const classDoc = classDocs.get(classKey);
        if (classDoc) {
          const jsDocComment = formatJSDoc(classDoc.info);
          inserts.push({
            position: node.getFullStart(),
            docstring: jsDocComment
          });
        }
      }
      ts.forEachChild(node, visit);
    }

    visit(sourceFile);

    // Sort inserts in reverse order to avoid shifting positions
    inserts.sort((a, b) => b.position - a.position);

    let updatedCode = sourceCode;
    inserts.forEach(insert => {
      updatedCode = insertAt(updatedCode, insert.position, insert.docstring);
    });

    console.log(updatedCode);

  } catch (error) {
    console.error(`Error processing docstrings: ${error.message}`);
    process.exit(1);
  }
});

function formatJSDoc(docInfo) {
  const lines = [];

  lines.push('/**');

  // Description
  if (docInfo.description) {
    docInfo.description.split('\n').forEach(line => {
      lines.push(` * ${line}`);
    });
  }

  // Parameters
  if (docInfo.parameters && docInfo.parameters.length > 0) {
    docInfo.parameters.forEach(param => {
      lines.push(` * @param {${param.type}} ${param.name} ${param.description}`);
    });
  }

  // Returns
  if (docInfo.returns && docInfo.returns.type) {
    lines.push(` * @returns {${docInfo.returns.type}} ${docInfo.returns.description}`);
  }

  // Decorators (if needed, can be included in the docstring)

  lines.push(' */\n');

  return lines.join('\n');
}

function insertAt(content, index, text) {
  return content.slice(0, index) + text + content.slice(index);
}
