// js_ts_parser.js

const fs = require('fs');
const babelParser = require('@babel/parser');
const traverse = require('@babel/traverse').default;

function parseCode(code, language) {
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

    const structure = {
        classes: [],
        functions: [],
        variables: [],
        comments: [],
    };

    // Traverse the AST
    traverse(ast, {
        enter(path) {
            const node = path.node;
            if (node.leadingComments) {
                node.leadingComments.forEach(comment => {
                    structure.comments.push(comment.value.trim());
                });
            }

            if (node.type === 'ClassDeclaration' && node.id) {
                const classInfo = {
                    name: node.id.name,
                    methods: [],
                };
                node.body.body.forEach(element => {
                    if (
                        (element.type === 'ClassMethod' || element.type === 'ClassPrivateMethod') &&
                        element.key.type === 'Identifier'
                    ) {
                        const methodInfo = {
                            name: element.key.name,
                            params: element.params.map(param => {
                                if (param.type === 'Identifier') {
                                    return param.name;
                                } else if (param.type === 'AssignmentPattern' && param.left.type === 'Identifier') {
                                    return param.left.name;
                                }
                                return 'unknown';
                            }),
                        };
                        classInfo.methods.push(methodInfo);
                    }
                });
                structure.classes.push(classInfo);
            } else if (node.type === 'FunctionDeclaration' && node.id) {
                const functionInfo = {
                    name: node.id.name,
                    params: node.params.map(param => {
                        if (param.type === 'Identifier') {
                            return param.name;
                        } else if (param.type === 'AssignmentPattern' && param.left.type === 'Identifier') {
                            return param.left.name;
                        }
                        return 'unknown';
                    }),
                };
                structure.functions.push(functionInfo);
            } else if (node.type === 'VariableDeclaration') {
                node.declarations.forEach(declaration => {
                    if (declaration.id.type === 'Identifier') {
                        structure.variables.push(declaration.id.name);
                    } else if (declaration.id.type === 'ObjectPattern') {
                        declaration.id.properties.forEach(prop => {
                            if (prop.key && prop.key.type === 'Identifier') {
                                structure.variables.push(prop.key.name);
                            }
                        });
                    }
                });
            }
        },
    });

    return structure;
}

function main() {
    const input = fs.readFileSync(0, 'utf-8');
    const data = JSON.parse(input);
    const code = data.code;
    const language = data.language || 'javascript';
    const structure = parseCode(code, language);
    console.log(JSON.stringify(structure));
}

main();