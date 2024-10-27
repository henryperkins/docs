// js_ts_parser.js

const fs = require('fs');
const babelParser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;
const t = require('@babel/types');

function parseCode(code, language, options) {
    const isTypeScript = language === 'typescript';

    const ast = babelParser.parse(code, {
        sourceType: 'module',
        plugins: [
            'jsx',
            isTypeScript ? 'typescript' : null,
            'classProperties',
            'classPrivateProperties',
            'classPrivateMethods',
            'exportDefaultFrom',
            'exportNamespaceFrom',
            'dynamicImport',
            'nullishCoalescingOperator',
            'optionalChaining',
        ].filter(Boolean),
    });

    const codeStructure = {
        docstring_format: isTypeScript ? "TSDoc" : "JSDoc",
        summary: "",
        changes_made: [],
        functions: [],
        classes: [],
        variables: [],
        constants: [],
        imports: [],
        exports: [],
        metrics: {
            complexity: 0,
            halstead: {
                volume: 0,
                difficulty: 0,
                effort: 0
            },
            maintainability_index: 0,
            function_metrics: {}
        }
    };

    const CodeVisitor = {
        enter(path) {
            const node = path.node;
            if (node.type === 'ImportDeclaration') {
                node.specifiers.forEach(specifier => {
                    const importedName = specifier.imported.name;
                    const fullImportPath = specifier.imported.name;
                    codeStructure.imports.push(fullImportPath);
                });
            } else if (node.type === 'FunctionDeclaration' || node.type === 'ArrowFunctionExpression') {
                const functionInfo = {
                    name: node.id ? node.id.name : 'anonymous',
                    docstring: node.leadingComments ? node.leadingComments[0].value : '',
                    args: node.params.map(param => ({
                        name: param.name,
                        type: param.typeAnnotation ? param.typeAnnotation.typeAnnotation.type : 'any'
                    })),
                    async: node.async,
                    returns: node.returnType ? node.returnType.typeAnnotation.type : 'any',
                    complexity: 0,
                    halstead: {
                        volume: 0,
                        difficulty: 0,
                        effort: 0
                    }
                };
                codeStructure.functions.push(functionInfo);
            } else if (node.type === 'ClassDeclaration') {
                const classInfo = {
                    name: node.id.name,
                    docstring: node.leadingComments ? node.leadingComments[0].value : '',
                    methods: [],
                    complexity: 0,
                    halstead: {
                        volume: 0,
                        difficulty: 0,
                        effort: 0
                    }
                };
                codeStructure.classes.push(classInfo);
            } else if (node.type === 'VariableDeclaration') {
                node.declarations.forEach(declaration => {
                    const variableInfo = {
                        name: declaration.id.name,
                        type: declaration.id.typeAnnotation ? declaration.id.typeAnnotation.typeAnnotation.type : 'any',
                        value: declaration.init ? declaration.init.value : null,
                        docstring: declaration.leadingComments ? declaration.leadingComments[0].value : '',
                        complexity: 0,
                        halstead: {
                            volume: 0,
                            difficulty: 0,
                            effort: 0
                        }
                    };
                    codeStructure.variables.push(variableInfo);
                });
            } else if (node.type === 'ExportNamedDeclaration') {
                node.specifiers.forEach(specifier => {
                    const exportedName = specifier.exported.name;
                    codeStructure.exports.push(exportedName);
                });
            }
        }
    };

    traverse(ast, CodeVisitor);

    return codeStructure;
}

function main() {
    const input = fs.readFileSync(0, 'utf-8');
    const data = JSON.parse(input);
    const code = data.code;
    const language = data.language || 'javascript';
    const options = data.options || {};
    const parsedData = parseCode(code, language, options);
    console.log(JSON.stringify(parsedData, null, 2));
}

main();