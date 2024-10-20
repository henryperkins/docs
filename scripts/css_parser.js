// scripts/css_parser.js

const fs = require('fs');
const css = require('css');
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

  const { code, documentation, language } = parsedInput;

  if (language.toLowerCase() !== 'css') {
    console.error('Unsupported language:', language);
    process.exit(1);
  }

  let ast;
  try {
    ast = css.parse(code, { source: 'input.css' });
  } catch (e) {
    console.error('Parsing error:', e.message);
    process.exit(1);
  }

  // Initialize the structure object
  const structure = {
    summary: "", // To be filled externally or manually
    changes_made: [], // To be filled externally or manually
    functions: [], // Not typically applicable for CSS
    classes: [], // Not typically applicable for CSS
    halstead: {
      volume: 0,
      difficulty: 0,
      effort: 0
    },
    maintainability_index: 0,
    variables: [], // CSS Variables (Custom Properties) can be mapped here
    constants: []  // Not typically applicable for CSS
  };

  ast.stylesheet.rules.forEach(rule => {
    if (rule.type === 'rule') {
      const selectors = rule.selectors;
      const declarations = rule.declarations.map(decl => ({
        property: decl.property,
        value: decl.value
      }));

      // Find documentation for this rule
      const doc = documentation.rules.find(r => {
        // Simple matching; can be enhanced
        return r.selectors.some(sel => selectors.includes(sel));
      });

      if (doc && doc.docstring) {
        // Insert comment before the rule
        rule.comments = [`${doc.docstring}`];
      }

      // Variables (Custom Properties) handling
      selectors.forEach(sel => {
        if (sel.startsWith('--')) { // CSS Variable
          const varName = sel;
          rule.declarations.forEach(decl => {
            if (decl.property === varName) {
              const variableInfo = {
                name: varName,
                type: "CSS Variable",
                description: "No description provided.",
                file: "Unknown", // Can be set if file info is available
                line: decl.position ? decl.position.start.line : 0,
                link: "Unknown", // Can be constructed based on repository
                example: decl.value,
                references: "No references."
              };
              structure.variables.push(variableInfo);
            }
          });
        }
      });
    }
  });

  // Validate the structure against the schema
  const valid = validate(structure);
  if (!valid) {
    console.error('Validation errors:', validate.errors);
    process.exit(1);
  }

  // Output the structure as JSON
  console.log(JSON.stringify(structure, null, 2));
});
