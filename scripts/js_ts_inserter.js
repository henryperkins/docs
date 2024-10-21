// js_ts_inserter.js

const fs = require('fs');
const babelParser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;
const t = require('@babel/types');

function generateJSDoc(description, params = [], returns = '', examples = []) {
    const lines = ['/**', ` * ${description}`];

    params.forEach(param => {
        lines.push(` * @param {${param.type || 'any'}} ${param.name} - ${param.description || ''}`);
    });

    if (returns) {
        lines.push(` * @returns {${returns.type || 'any'}} - ${returns.description || ''}`);
    }

    examples.forEach(example => {
        lines.push(' * @example');
        lines.push(` * ${example}`);
    });

    lines.push(' */');
    return lines.join('\n');
}

function insertJSDoc(code, documentation, language) {
    const isTypeScript = language === 'typescript';

    // Parse the code into an AST
    const ast = babelParser.parse(code, {
        sourceType: 'module',
        plugins: [
            'jsx',
            isTypeScript ? 'typescript' : null,
            'classProperties',
            'decorators-legacy',
        ].filter(Boolean),
    });

    const docsMap = new Map();

    // Map documentation to code elements
    if (documentation.functions) {
        documentation.functions.forEach(func => {
            docsMap.set(func.name, func);
        });
    }
    if (documentation.classes) {
        documentation.classes.forEach(cls => {
            docsMap.set(cls.name, cls);
            if (cls.methods) {
                cls.methods.forEach(method => {
                    docsMap.set(`${cls.name}.${method.name}`, method);
                });
            }
        });
    }

    // Traverse the AST to insert comments
    traverse(ast, {
        enter(path) {
            const node = path.node;
            if (node.type === 'FunctionDeclaration' && node.id) {
                const doc = docsMap.get(node.id.name);
                if (doc) {
                    const jsDocComment = generateJSDoc(doc.description, doc.params, doc.returns, doc.examples);
                    node.leadingComments = node.leadingComments || [];
                    node.leadingComments.push({
                        type: 'CommentBlock',
                        value: jsDocComment.replace(/^\/\*\*|\*\/$/g, '').trim(),
                    });
                }
            } else if (node.type === 'ClassDeclaration' && node.id) {
                const doc = docsMap.get(node.id.name);
                if (doc) {
                    const jsDocComment = generateJSDoc(doc.description, [], null, doc.examples);
                    node.leadingComments = node.leadingComments || [];
                    node.leadingComments.push({
                        type: 'CommentBlock',
                        value: jsDocComment.replace(/^\/\*\*|\*\/$/g, '').trim(),
                    });
                }
                // Handle class methods
                node.body.body.forEach(element => {
                    if (
                        (element.type === 'ClassMethod' || element.type === 'ClassPrivateMethod') &&
                        element.key.type === 'Identifier'
                    ) {
                        const methodName = `${node.id.name}.${element.key.name}`;
                        const doc = docsMap.get(methodName);
                        if (doc) {
                            const jsDocComment = generateJSDoc(doc.description, doc.params, doc.returns, doc.examples);
                            element.leadingComments = element.leadingComments || [];
                            element.leadingComments.push({
                                type: 'CommentBlock',
                                value: jsDocComment.replace(/^\/\*\*|\*\/$/g, '').trim(),
                            });
                        }
                    }
                });
            }
        },
    });

    // Generate the modified code
    const output = generate(ast, { comments: true }, code);
    return output.code;
}

function main() {
    const input = fs.readFileSync(0, 'utf-8');
    const data = JSON.parse(input);
    const code = data.code;
    const documentation = data.documentation;
    const language = data.language || 'javascript';
    const modifiedCode = insertJSDoc(code, documentation, language);
    console.log(modifiedCode);
}

main();