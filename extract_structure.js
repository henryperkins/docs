// extract_structure.js
const fs = require('fs');
const ts = require('typescript');

const filePath = process.argv[2];
const code = fs.readFileSync(filePath, 'utf8');

// Determine the script kind based on the file extension
let scriptKind = ts.ScriptKind.TS;
if (filePath.endsWith(".tsx")) {
  scriptKind = ts.ScriptKind.TSX;
} else if (filePath.endsWith(".jsx")) {
  scriptKind = ts.ScriptKind.JSX;
} else if (filePath.endsWith(".js")) {
  scriptKind = ts.ScriptKind.JS;
}

function extractStructure(code, fileName) {
  try {
    const sourceFile = ts.createSourceFile(
      fileName,
      code,
      ts.ScriptTarget.Latest,
      /* setParentNodes */ true,
      scriptKind
    );

    const functions = [];
    const classes = [];

    function visit(node, parentClass = null) {
      switch (node.kind) {
        case ts.SyntaxKind.FunctionDeclaration:
          functions.push(extractFunctionInfo(node));
          break;
        case ts.SyntaxKind.VariableStatement:
          node.declarationList.declarations.forEach((declaration) => {
            if (
              declaration.initializer &&
              (ts.isArrowFunction(declaration.initializer) ||
                ts.isFunctionExpression(declaration.initializer))
            ) {
              functions.push(extractFunctionInfo(declaration.initializer, declaration.name.getText()));
            }
          });
          break;
        case ts.SyntaxKind.ClassDeclaration:
          const classInfo = extractClassInfo(node);
          classes.push(classInfo);
          node.members.forEach((member) => visit(member, classInfo));
          break;
        case ts.SyntaxKind.MethodDeclaration:
        case ts.SyntaxKind.MethodSignature:
        case ts.SyntaxKind.Constructor:
          if (parentClass) {
            parentClass.methods.push(extractFunctionInfo(node));
          }
          break;
        default:
          break;
      }
      ts.forEachChild(node, (child) => visit(child, parentClass));
    }

    function extractFunctionInfo(node, functionName = null) {
      const name = functionName || node.name?.getText() || "anonymous";
      const parameters = node.parameters.map((param) => {
        return {
          name: param.name.getText(),
          type: param.type?.getText() || "any",
          description: ""
        };
      });

      const decorators = [];
      if (node.modifiers) {
        node.modifiers.forEach((modifier) => {
          if (modifier.kind === ts.SyntaxKind.Decorator) {
            decorators.push(modifier.expression.getText());
          }
        });
      }

      const returnType = {
        type: node.type?.getText() || "any",
        description: ""
      };

      return {
        name: name,
        description: "", // Placeholder for description
        parameters: parameters,
        returns: returnType,
        decorators: decorators,
        examples: [],
        start: node.getStart(),
        end: node.getEnd(),
      };
    }

    function extractClassInfo(node) {
      const name = node.name?.getText() || "anonymous";
      const bases = node.heritageClauses?.map((clause) =>
        clause.types.map((type) => type.expression.getText())
      ).flat() || [];

      const decorators = [];
      if (node.modifiers) {
        node.modifiers.forEach((modifier) => {
          if (modifier.kind === ts.SyntaxKind.Decorator) {
            decorators.push(modifier.expression.getText());
          }
        });
      }

      return {
        name: name,
        description: "", // Placeholder for description
        bases: bases,
        decorators: decorators,
        methods: [],
        examples: [],
        start: node.getStart(),
        end: node.getEnd(),
      };
    }

    visit(sourceFile);

    return {
      language: "javascript", // or "typescript" based on your context
      functions: functions,
      classes: classes
    };
  } catch (error) {
    console.error(`Error parsing file '${fileName}': ${error.message}`);
    process.exit(1);
  }
}

const structure = extractStructure(code, filePath);
console.log(JSON.stringify(structure, null, 2));
