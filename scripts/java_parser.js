// scripts/java_parser.js

const fs = require('fs');
const javaParser = require('java-parser');
const Ajv = require('ajv');
const path = require('path');

// Initialize AJV for JSON schema validation
const ajv = new Ajv({ allErrors: true, strict: false });

// Load the unified function_schema.json
const schemaPath = path.join(__dirname, '../schemas/function_schema.json');
const functionSchema = JSON.parse(fs.readFileSync(schemaPath, 'utf-8'));
const validate = ajv.compile(functionSchema);

// Read input from stdin
let inputChunks = [];
process.stdin.on('data', (chunk) => {
  inputChunks.push(chunk);
});

process.stdin.on('end', () => {
  const inputData = inputChunks.join('');

  let parsedInput;
  try {
    parsedInput = JSON.parse(inputData);
  } catch (e) {
    console.error('Error parsing input JSON:', e.message);
    process.exit(1);
  }

  const { code, language } = parsedInput;

  if (language.toLowerCase() !== 'java') {
    console.error('Unsupported language:', language);
    process.exit(1);
  }

  let ast;
  try {
    ast = javaParser.parse(code);
  } catch (e) {
    console.error('Parsing error:', e.message);
    process.exit(1);
  }

  // Initialize the structure object
  const structure = {
    summary: "", // To be filled externally or manually
    changes_made: [], // To be filled externally or manually
    functions: [],
    classes: [],
    halstead: {
      volume: 0,
      difficulty: 0,
      effort: 0
    },
    maintainability_index: 0,
    variables: [],
    constants: []
  };

  // Helper function to extract docstrings (comments)
  function getDocstring(node) {
    if (node.documentation) {
      return node.documentation.trim();
    }
    return "";
  }

  // Traverse the AST to extract classes and functions
  const classes = ast.children.filter(child => child.node === 'ClassDeclaration');

  classes.forEach(cls => {
    const classInfo = {
      name: cls.name.identifier,
      docstring: getDocstring(cls),
      methods: []
    };

    cls.body.body.forEach(member => {
      if (member.node === 'MethodDeclaration') {
        const methodInfo = {
          name: member.name.identifier,
          docstring: getDocstring(member),
          args: member.parameters.map(param => param.name.identifier),
          async: false, // Java does not have async methods; can be extended if using CompletableFuture or similar
          type: 'instance', // Default to instance method; can be extended based on modifiers
          complexity: 1 // Placeholder: Cyclomatic complexity calculation requires further implementation
        };

        // Determine if the method is static
        if (member.modifiers && member.modifiers.includes('static')) {
          methodInfo.type = 'static';
        }

        classInfo.methods.push(methodInfo);
      } else if (member.node === 'FieldDeclaration') {
        member.declarators.forEach(decl => {
          const variableInfo = {
            name: decl.id.identifier,
            type: member.type.name.identifier,
            description: getDocstring(member),
            file: "Unknown", // File information can be added if available
            line: decl.position.start.line,
            link: "Unknown", // Link can be constructed based on repository URL
            example: "No example provided.",
            references: "No references."
          };

          // Determine if the field is a constant (e.g., final)
          if (member.modifiers && member.modifiers.includes('final')) {
            structure.constants.push(variableInfo);
          } else {
            structure.variables.push(variableInfo);
          }
        });
      }
    });

    structure.classes.push(classInfo);
  });

  // Traverse the AST to extract standalone functions (if any)
  // Note: Java primarily uses classes, but static methods can be considered standalone
  classes.forEach(cls => {
    cls.body.body.forEach(member => {
      if (member.node === 'MethodDeclaration' && member.modifiers && member.modifiers.includes('static')) {
        const functionInfo = {
          name: member.name.identifier,
          docstring: getDocstring(member),
          args: member.parameters.map(param => param.name.identifier),
          async: false, // Java does not support async directly
          complexity: 1 // Placeholder for cyclomatic complexity
        };
        structure.functions.push(functionInfo);
      }
    });
  });

  // Placeholder for Halstead metrics and Maintainability Index
  // These require detailed analysis and are not implemented here
  // They can be integrated using additional tools or libraries

  // Validate the structure against the schema
  const valid = validate(structure);
  if (!valid) {
    console.error('Validation errors:', validate.errors);
    process.exit(1);
  }

  // Output the structure as JSON
  console.log(JSON.stringify(structure, null, 2));
});
