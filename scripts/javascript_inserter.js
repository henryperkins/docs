// src/inserters/javascript_inserter.js

const { BaseInserter } = require('../common/BaseInserter');
const { parse } = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;
const t = require('@babel/types');

class JavaScriptInserter extends BaseInserter {
    constructor() {
        super('javascript');
        this.babelOptions = {
            sourceType: 'module',
            plugins: [
                'jsx',
                'typescript',
                'decorators-legacy',
                'classProperties',
                'classPrivateProperties',
                'classPrivateMethods',
                'exportDefaultFrom',
                'exportNamespaceFrom',
                'dynamicImport',
                'nullishCoalescingOperator',
                'optionalChaining',
            ]
        };
    }

    generateJSDoc(element) {
        const lines = ['/**'];
        
        // Add description
        if (element.description) {
            lines.push(` * ${element.description}`);
        }

        // Add parameters
        if (element.params && element.params.length > 0) {
            lines.push(` *`);
            element.params.forEach(param => {
                lines.push(` * @param {${param.type || '*'}} ${param.name} ${param.description || ''}`);
            });
        }

        // Add return value
        if (element.returns) {
            lines.push(` * @returns {${element.returns.type || '*'}} ${element.returns.description || ''}`);
        }

        // Add examples
        if (element.examples && element.examples.length > 0) {
            lines.push(` * @example`);
            element.examples.forEach(example => {
                const exampleLines = example.split('\n');
                exampleLines.forEach(line => lines.push(` * ${line}`));
            });
        }

        lines.push(` */`);
        return lines.join('\n');
    }

    processDocumentation(documentation) {
        const docMap = new Map();

        // Process functions
        if (documentation.functions) {
            documentation.functions.forEach(func => {
                docMap.set(func.name, {
                    type: 'function',
                    doc: this.generateJSDoc(func)
                });
            });
        }

        // Process classes
        if (documentation.classes) {
            documentation.classes.forEach(cls => {
                docMap.set(cls.name, {
                    type: 'class',
                    doc: this.generateJSDoc(cls)
                });

                // Process class methods
                if (cls.methods) {
                    cls.methods.forEach(method => {
                        docMap.set(`${cls.name}.${method.name}`, {
                            type: 'method',
                            doc: this.generateJSDoc(method)
                        });
                    });
                }
            });
        }

        return docMap;
    }

    insertDocumentation(code, documentation) {
        try {
            // Parse the code into an AST
            const ast = parse(code, this.babelOptions);
            const docMap = this.processDocumentation(documentation);

            // Traverse and modify the AST
            traverse(ast, {
                FunctionDeclaration: (path) => {
                    this.insertFunctionDoc(path, docMap);
                },
                ClassDeclaration: (path) => {
                    this.insertClassDoc(path, docMap);
                },
                ClassMethod: (path) => {
                    this.insertMethodDoc(path, docMap);
                },
                ArrowFunctionExpression: (path) => {
                    this.insertArrowFunctionDoc(path, docMap);
                },
                VariableDeclaration: (path) => {
                    this.insertVariableDoc(path, docMap);
                }
            });

            // Generate the modified code
            const output = generate(ast, {
                retainLines: true,
                comments: true
            });

            return output.code;
        } catch (error) {
            throw new Error(`Failed to insert documentation: ${error.message}`);
        }
    }

    insertFunctionDoc(path, docMap) {
        if (path.node.id && path.node.id.name) {
            const funcName = path.node.id.name;
            const docInfo = docMap.get(funcName);
            
            if (docInfo) {
                this.addCommentToNode(path.node, docInfo.doc);
            }
        }
    }

    insertClassDoc(path, docMap) {
        if (path.node.id && path.node.id.name) {
            const className = path.node.id.name;
            const docInfo = docMap.get(className);
            
            if (docInfo) {
                this.addCommentToNode(path.node, docInfo.doc);
            }
        }
    }

    insertMethodDoc(path, docMap) {
        const classPath = path.findParent((path) => path.isClassDeclaration());
        if (classPath && classPath.node.id && path.node.key) {
            const className = classPath.node.id.name;
            const methodName = path.node.key.name;
            const fullName = `${className}.${methodName}`;
            const docInfo = docMap.get(fullName);
            
            if (docInfo) {
                this.addCommentToNode(path.node, docInfo.doc);
            }
        }
    }

    insertArrowFunctionDoc(path, docMap) {
        const parentPath = path.parentPath;
        if (parentPath.isVariableDeclarator() && parentPath.node.id) {
            const funcName = parentPath.node.id.name;
            const docInfo = docMap.get(funcName);
            
            if (docInfo) {
                this.addCommentToNode(parentPath.parentPath.node, docInfo.doc);
            }
        }
    }

    insertVariableDoc(path, docMap) {
        if (path.node.declarations && path.node.declarations.length === 1) {
            const declaration = path.node.declarations[0];
            if (declaration.id && declaration.id.name) {
                const varName = declaration.id.name;
                const docInfo = docMap.get(varName);
                
                if (docInfo) {
                    this.addCommentToNode(path.node, docInfo.doc);
                }
            }
        }
    }

    addCommentToNode(node, docString) {
        if (!node.leadingComments) {
            node.leadingComments = [];
        }
        
        // Remove existing JSDoc comments
        node.leadingComments = node.leadingComments.filter(
            comment => !comment.value.startsWith('*')
        );

        // Add new JSDoc comment
        node.leadingComments.push({
            type: 'CommentBlock',
            value: docString.slice(3, -2).trim() // Remove /** and */ and trim
        });
    }

    async process() {
        try {
            const input = await this.readStdin();
            const { code, documentation } = this.parseInput(input);
            return this.insertDocumentation(code, documentation);
        } catch (error) {
            console.error(`Error processing JavaScript file: ${error.message}`);
            process.exit(1);
        }
    }
}

// Create and run the inserter
if (require.main === module) {
    const inserter = new JavaScriptInserter();
    inserter.process()
        .then(result => {
            console.log(result);
        })
        .catch(error => {
            console.error(error);
            process.exit(1);
        });
}

module.exports = JavaScriptInserter;
