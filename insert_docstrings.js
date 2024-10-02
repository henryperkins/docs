// insert_docstrings.js

const fs = require('fs/promises');
const path = require('path');
const babelParser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;
const prettier = require('prettier');

/**
 * Inserts documentation into a JavaScript/TypeScript file based on the provided documentation JSON.
 * @param {string} filePath - Path to the JavaScript/TypeScript file.
 * @param {string} documentationPath - Path to the documentation JSON file.
 */
async function insertDocstrings(filePath, documentationPath) {
  try {
    const code = await fs.readFile(filePath, 'utf-8');
    const documentationContent = await fs.readFile(documentationPath, 'utf-8');
    const documentation = JSON.parse(documentationContent);

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

    traverse(ast, {
      // Insert docstrings for standalone functions
      FunctionDeclaration(path) {
        const node = path.node;
        const funcName = node.id ? node.id.name : 'anonymous';

        if (documentation.functions && documentation.functions[funcName]) {
          const doc = documentation.functions[funcName].docstring;

          if (doc) {
            const commentBlock = `*\n * ${doc}\n `;
            node.leadingComments = node.leadingComments || [];
            node.leadingComments.push({
              type: 'CommentBlock',
              value: commentBlock,
            });
          }
        }
      },
      // Insert comments for arrow functions assigned to variables
      VariableDeclaration(path) {
        const declarations = path.node.declarations;
        declarations.forEach((declarator) => {
          if (
            declarator.init &&
            (declarator.init.type === 'ArrowFunctionExpression' ||
              declarator.init.type === 'FunctionExpression')
          ) {
            const funcName = declarator.id.name;

            if (documentation.functions && documentation.functions[funcName]) {
              const doc = documentation.functions[funcName].docstring;

              if (doc) {
                const commentBlock = `*\n * ${doc}\n `;
                path.node.leadingComments = path.node.leadingComments || [];
                path.node.leadingComments.push({
                  type: 'CommentBlock',
                  value: commentBlock,
                });
              }
            }
          }
        });
      },
      // Insert docstrings for class declarations
      ClassDeclaration(path) {
        const node = path.node;
        const className = node.id ? node.id.name : 'anonymous';

        if (documentation.classes && documentation.classes[className]) {
          const doc = documentation.classes[className].docstring;

          if (doc) {
            const commentBlock = `*\n * ${doc}\n `;
            node.leadingComments = node.leadingComments || [];
            node.leadingComments.push({
              type: 'CommentBlock',
              value: commentBlock,
            });
          }
        }

        // Insert docstrings for class methods
        path.traverse({
          ClassMethod(methodPath) {
            const methodNode = methodPath.node;
            const methodName = methodNode.key.name;

            if (
              documentation.classes &&
              documentation.classes[className] &&
              documentation.classes[className].methods &&
              documentation.classes[className].methods[methodName]
            ) {
              const doc = documentation.classes[className].methods[methodName].docstring;

              if (doc) {
                const commentBlock = `*\n * ${doc}\n `;
                methodNode.leadingComments = methodNode.leadingComments || [];
                methodNode.leadingComments.push({
                  type: 'CommentBlock',
                  value: commentBlock,
                });
              }
            }
          },
        });
      },
    });

    // Generate the modified code from the AST
    const output = generate(ast, { comments: true }, code).code;

    // Format the code using Prettier
    const prettierConfig = await prettier.resolveConfig(filePath);
    const formattedCode = prettier.format(output, { ...prettierConfig, filepath: filePath });

    // Write the formatted code back to the file
    await fs.writeFile(filePath, formattedCode, 'utf-8');

    console.log(`Docstrings inserted successfully into '${filePath}'`);
  } catch (error) {
    console.error(`Error inserting docstrings into '${filePath}':`, error);
    process.exit(1);
  }
}

/**
 * Main function to handle command-line arguments and execute docstring insertion.
 */
async function main() {
  const args = process.argv.slice(2);
  if (args.length !== 2) {
    console.error('Usage: node insert_docstrings.js /path/to/file.js /path/to/file.js.documentation.json');
    process.exit(1);
  }

  const filePath = args[0];
  const documentationPath = args[1];

  // Validate file existence
  try {
    await fs.access(filePath);
  } catch (error) {
    console.error(`File '${filePath}' does not exist or is not accessible.`);
    process.exit(1);
  }

  // Validate documentation file existence
  try {
    await fs.access(documentationPath);
  } catch (error) {
    console.error(`Documentation file '${documentationPath}' does not exist or is not accessible.`);
    process.exit(1);
  }

  // Insert docstrings
  await insertDocstrings(filePath, documentationPath);
}

// Execute the main function
main();
