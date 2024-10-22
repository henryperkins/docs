// js_ts_parser.js

const fs = require('fs');
const babelParser = require('@babel/parser');
const traverse = require('@babel/traverse').default;

function parseCode(code, language) {
    // Skip if file is empty or not text
    if (!code || typeof code !== 'string') {
        return {
            classes: [],
            functions: [],
            variables: [],
            constants: [],
            summary: "Empty or invalid file",
            changes_made: [],
            halstead: { volume: 0, difficulty: 0, effort: 0 },
            maintainability_index: 0
        };
    }

    // Configure parser options
    const parserOptions = {
        sourceType: 'module',
        errorRecovery: true,  // Continue parsing even if there are errors
        plugins: [
            'jsx',
            language === 'typescript' ? 'typescript' : null,
            'decorators-legacy',
            ['decorators', { decoratorsBeforeExport: true }],
            'classProperties',
            'classPrivateProperties',
            'classPrivateMethods',
            'exportDefaultFrom',
            'exportNamespaceFrom',
            'dynamicImport'
        ].filter(Boolean),
        tokens: true
    };

    // Parse the code
    let ast;
    try {
        ast = babelParser.parse(code, parserOptions);
    } catch (error) {
        console.error(`Parse error: ${error.message}`);
        return {
            classes: [],
            functions: [],
            variables: [],
            constants: [],
            summary: `Parse error: ${error.message}`,
            changes_made: [],
            halstead: { volume: 0, difficulty: 0, effort: 0 },
            maintainability_index: 0
        };
    }

    const structure = {
        classes: [],
        functions: [],
        variables: [],
        constants: [],
        summary: "",
        changes_made: [],
        halstead: {
            volume: 0,
            difficulty: 0,
            effort: 0
        },
        maintainability_index: 0
    };

    try {
        // Traverse the AST
        traverse(ast, {
            enter(path) {
                const node = path.node;

                // Extract comments
                if (node.leadingComments) {
                    structure.summary += node.leadingComments
                        .map(comment => comment.value.trim())
                        .join('\n');
                }

                // Handle classes
                if (node.type === 'ClassDeclaration' && node.id) {
                    const classInfo = {
                        name: node.id.name,
                        docstring: getDocstring(node),
                        methods: [],
                        complexity: 1 // Base complexity
                    };

                    // Process class methods
                    node.body.body.forEach(element => {
                        if ((element.type === 'ClassMethod' || element.type === 'ClassPrivateMethod') && 
                            element.key.type === 'Identifier') {
                            const methodInfo = {
                                name: element.key.name,
                                docstring: getDocstring(element),
                                args: getParams(element.params),
                                async: element.async,
                                type: getMethodType(element),
                                complexity: calculateComplexity(element)
                            };
                            classInfo.methods.push(methodInfo);
                            classInfo.complexity += methodInfo.complexity;
                        }
                    });

                    structure.classes.push(classInfo);
                }

                // Handle functions
                else if (node.type === 'FunctionDeclaration' && node.id) {
                    const functionInfo = {
                        name: node.id.name,
                        docstring: getDocstring(node),
                        args: getParams(node.params),
                        async: node.async,
                        complexity: calculateComplexity(node)
                    };
                    structure.functions.push(functionInfo);
                }

                // Handle variables and constants
                else if (node.type === 'VariableDeclaration') {
                    node.declarations.forEach(declaration => {
                        const varInfo = {
                            name: getVariableName(declaration.id),
                            type: declaration.init ? declaration.init.type : 'undefined',
                            description: getDocstring(node),
                            file: 'current file',
                            line: node.loc ? node.loc.start.line : 0,
                            link: 'N/A',
                            example: getExample(declaration),
                            references: []
                        };

                        if (node.kind === 'const') {
                            structure.constants.push(varInfo);
                        } else {
                            structure.variables.push(varInfo);
                        }
                    });
                }
            }
        });

        // Calculate basic halstead metrics
        const metrics = calculateHalsteadMetrics(ast);
        structure.halstead = metrics.halstead;
        structure.maintainability_index = metrics.maintainability;

        return structure;

    } catch (error) {
        console.error(`Error during traversal: ${error.message}`);
        return {
            classes: [],
            functions: [],
            variables: [],
            constants: [],
            summary: `Error during traversal: ${error.message}`,
            changes_made: [],
            halstead: { volume: 0, difficulty: 0, effort: 0 },
            maintainability_index: 0
        };
    }
}

// Helper functions
function getDocstring(node) {
    if (node.leadingComments && node.leadingComments.length > 0) {
        return node.leadingComments[node.leadingComments.length - 1].value.trim();
    }
    return '';
}

function getParams(params) {
    return params.map(param => {
        if (param.type === 'Identifier') {
            return param.name;
        } else if (param.type === 'AssignmentPattern' && param.left.type === 'Identifier') {
            return param.left.name;
        }
        return 'unknown';
    });
}

function getMethodType(node) {
    if (node.static) return 'static';
    if (node.kind === 'get') return 'getter';
    if (node.kind === 'set') return 'setter';
    return 'instance';
}

function getVariableName(id) {
    if (id.type === 'Identifier') {
        return id.name;
    } else if (id.type === 'ObjectPattern') {
        return id.properties.map(prop => 
            prop.key && prop.key.type === 'Identifier' ? prop.key.name : 'unknown'
        ).join(', ');
    }
    return 'unknown';
}

function getExample(declaration) {
    if (declaration.init) {
        return declaration.init.type;
    }
    return 'No example available';
}

function calculateComplexity(node) {
    let complexity = 1;
    traverse(node, {
        enter(path) {
            if (path.isIfStatement() || 
                path.isWhileStatement() || 
                path.isForStatement() || 
                path.isForInStatement() || 
                path.isForOfStatement() || 
                path.isSwitchCase() || 
                path.isConditionalExpression()) {
                complexity++;
            }
        }
    });
    return complexity;
}

function calculateHalsteadMetrics(ast) {
    // Simple implementation - could be enhanced
    return {
        halstead: {
            volume: ast.program.body.length * 10,
            difficulty: ast.program.body.length * 2,
            effort: ast.program.body.length * 20
        },
        maintainability: Math.max(0, 100 - ast.program.body.length)
    };
}

// Main execution
try {
    const input = fs.readFileSync(0, 'utf-8');
    const data = JSON.parse(input);
    const structure = parseCode(data.code, data.language || 'javascript');
    console.log(JSON.stringify(structure, null, 2));
} catch (error) {
    console.error(`Fatal error: ${error.message}`);
    process.exit(1);
}