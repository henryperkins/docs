// acorn_parser.js

const acorn = require('acorn');
const walk = require('acorn-walk');
const fs = require('fs');
const Ajv = require('ajv');

// Set up Ajv validator
const ajv = new Ajv({ allErrors: true });

// Read data from stdin (code, language, functionSchema)
let inputChunks = [];
process.stdin.on('data', chunk => {
    inputChunks.push(chunk);
});

process.stdin.on('end', () => {
    const inputData = JSON.parse(inputChunks.join(''));
    const { code, language, functionSchema } = inputData;

    // Initialize comments array
    let comments = [];

    // Parse the code using acorn
    let ast;
    try {
        ast = acorn.parse(code, {
            ecmaVersion: 'latest',
            sourceType: 'module',
            locations: true,
            ranges: true,
            onComment: comments
        });
    } catch (e) {
        console.error(`Acorn parsing error: ${e.message}`);
        process.exit(1);
    }

    // Attach comments to nodes
    acorn.addComments(ast, comments);

    // Function to extract docstrings from leading comments
    function extractDocstring(node) {
        if (node.leadingComments && node.leadingComments.length > 0) {
            const lastComment = node.leadingComments[node.leadingComments.length - 1];
            return lastComment.value.trim();
        }
        return null;
    }

    // Extract functions and classes from the AST
    const structure = {
        functions: [],
        classes: []
    };

    walk.simple(ast, {
        FunctionDeclaration(node) {
            const func = {
                name: node.id ? node.id.name : 'anonymous',
                args: node.params.map(param => param.name || 'param'),
                async: node.async || false,
                docstring: extractDocstring(node)
            };
            structure.functions.push(func);
        },
        VariableDeclaration(node) {
            node.declarations.forEach(declarator => {
                if (
                    declarator.init &&
                    (declarator.init.type === 'FunctionExpression' || declarator.init.type === 'ArrowFunctionExpression')
                ) {
                    const func = {
                        name: declarator.id.name,
                        args: declarator.init.params.map(param => param.name || 'param'),
                        async: declarator.init.async || false,
                        docstring: extractDocstring(declarator)
                    };
                    structure.functions.push(func);
                }
            });
        },
        ClassDeclaration(node) {
            const cls = {
                name: node.id.name,
                docstring: extractDocstring(node),
                methods: []
            };

            node.body.body.forEach(element => {
                if (element.type === 'MethodDefinition') {
                    const method = {
                        name: element.key.name,
                        args: element.value.params.map(param => param.name || 'param'),
                        async: element.value.async || false,
                        kind: element.kind,
                        docstring: extractDocstring(element)
                    };
                    cls.methods.push(method);
                }
            });
            structure.classes.push(cls);
        }
    });

    // If functionSchema is provided, perform validation
    if (functionSchema) {
        try {
            // Compile the schema
            const validate = ajv.compile(functionSchema);

            // Validate the structure
            const valid = validate(structure);

            if (!valid) {
                console.error('Validation errors:', JSON.stringify(validate.errors, null, 2));
                process.exit(1);
            }
        } catch (e) {
            console.error(`Schema validation error: ${e.message}`);
            process.exit(1);
        }
    }

    // Output the extracted structure as JSON
    console.log(JSON.stringify(structure, null, 2));
});
