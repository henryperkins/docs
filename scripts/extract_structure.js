// extract_structure.js

const fs = require('fs/promises');
const path = require('path');
const babelParser = require('@babel/parser');
const traverse = require('@babel/traverse').default;

/**
 * Extracts the structure of a JavaScript/TypeScript file, including functions and classes.
 * @param {string} filePath - Path to the JavaScript/TypeScript file.
 * @returns {Promise<Object>} - An object representing the code structure.
 */
async function extractStructure(filePath) {
  try {
    const code = await fs.readFile(filePath, 'utf-8');

    // Determine parser plugins based on file extension
    const ext = path.extname(filePath).toLowerCase();
    const isTypeScript = ext === '.ts' || ext === '.tsx';
    const isJSX = ext === '.jsx' || ext === '.tsx';

    const plugins = [
      'jsx',
      'classProperties',
      'dynamicImport',
      'objectRestSpread',
      'optionalChaining',
      'nullishCoalescingOperator',
      'decorators-legacy',
      'exportDefaultFrom',
      'exportNamespaceFrom',
      'topLevelAwait',
    ];

    if (isTypeScript) {
      plugins.push('typescript');
    }

    // Parse the code into an AST
    const ast = babelParser.parse(code, {
      sourceType: 'module',
      plugins: plugins,
    });

    const structure = {
      functions: [],
      classes: [],
    };

    traverse(ast, {
      // Extract standalone functions
      FunctionDeclaration(path) {
        const node = path.node;
        structure.functions.push({
          name: node.id ? node.id.name : 'anonymous',
          params: node.params.map((param) => getParamName(param)),
          location: node.loc,
        });
      },
      // Extract arrow functions assigned to variables
      VariableDeclaration(path) {
        const declarations = path.node.declarations;
        declarations.forEach((declarator) => {
          if (
            declarator.init &&
            (declarator.init.type === 'ArrowFunctionExpression' ||
              declarator.init.type === 'FunctionExpression')
          ) {
            structure.functions.push({
              name: declarator.id.name,
              params: declarator.init.params.map((param) => getParamName(param)),
              location: declarator.loc,
            });
          }
        });
      },
      // Extract class declarations
      ClassDeclaration(path) {
        const node = path.node;
        const classInfo = {
          name: node.id ? node.id.name : 'anonymous',
          methods: [],
          location: node.loc,
        };

        path.traverse({
          ClassMethod(methodPath) {
            const methodNode = methodPath.node;
            classInfo.methods.push({
              name: methodNode.key.name,
              params: methodNode.params.map((param) => getParamName(param)),
              location: methodNode.loc,
            });
          },
        });

        structure.classes.push(classInfo);
      },
      // Extract class expressions assigned to variables
      VariableDeclaration(path) {
        const declarations = path.node.declarations;
        declarations.forEach((declarator) => {
          if (declarator.init && declarator.init.type === 'ClassExpression') {
            const classNode = declarator.init;
            const classInfo = {
              name: declarator.id.name,
              methods: [],
              location: classNode.loc,
            };

            path.traverse({
              ClassMethod(methodPath) {
                const methodNode = methodPath.node;
                classInfo.methods.push({
                  name: methodNode.key.name,
                  params: methodNode.params.map((param) => getParamName(param)),
                  location: methodNode.loc,
                });
              },
            });

            structure.classes.push(classInfo);
          }
        });
      },
    });

    return structure;
  } catch (error) {
    console.error(`Error extracting structure from '${filePath}':`, error);
    process.exit(1);
  }
}

/**
 * Retrieves the name of a parameter, handling different parameter types.
 * @param {Object} param - The AST node representing the parameter.
 * @returns {string} - The name of the parameter.
 */
function getParamName(param) {
  switch (param.type) {
    case 'Identifier':
      return param.name;
    case 'AssignmentPattern':
      return param.left.name;
    case 'RestElement':
      return '...' + param.argument.name;
    case 'ObjectPattern':
      return '{' + param.properties.map((prop) => prop.key.name).join(', ') + '}';
    case 'ArrayPattern':
      return '[' + param.elements.map((el) => (el ? el.name : '')).join(', ') + ']';
    default:
      return 'unknown';
  }
}

/**
 * Writes the structure object to a JSON file.
 * @param {string} filePath - Path to the original JavaScript/TypeScript file.
 * @param {Object} structure - The extracted structure object.
 */
async function writeStructureToFile(filePath, structure) {
  try {
    const jsonContent = JSON.stringify(structure, null, 2);
    const outputPath = `${filePath}.structure.json`;
    await fs.writeFile(outputPath, jsonContent, 'utf-8');
    console.log(`Structure extracted and saved to '${outputPath}'`);
  } catch (error) {
    console.error(`Error writing structure to file for '${filePath}':`, error);
    process.exit(1);
  }
}

/**
 * Main function to handle command-line arguments and execute extraction.
 */
async function main() {
  const args = process.argv.slice(2);
  if (args.length !== 1) {
    console.error('Usage: node extract_structure.js /path/to/file.js');
    process.exit(1);
  }

  const filePath = args[0];

  // Validate file existence
  try {
    await fs.access(filePath);
  } catch (error) {
    console.error(`File '${filePath}' does not exist or is not accessible.`);
    process.exit(1);
  }

  // Extract structure
  const structure = await extractStructure(filePath);

  // Write structure to JSON file
  await writeStructureToFile(filePath, structure);
}

// Execute the main function
main();
