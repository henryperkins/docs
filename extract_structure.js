// extract_structure.js

const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const babelParser = require('@babel/parser');

/**
 * Parses a TypeScript or JavaScript file and extracts its structure.
 * @param {string} filePath - The path to the JS/TS file.
 * @returns {object} - The extracted structure of the code.
 */
function extractStructure(filePath) {
    const code = fs.readFileSync(filePath, 'utf8');
    const ext = path.extname(filePath).toLowerCase();
    let structure = {
        functions: [],
        classes: []
    };

    try {
        if (ext === '.ts' || ext === '.tsx') {
            // Use TypeScript Compiler API
            const sourceFile = ts.createSourceFile(filePath, code, ts.ScriptTarget.Latest, true);
            ts.forEachChild(sourceFile, visitTS);
        } else {
            // Use Babel Parser for JavaScript
            const ast = babelParser.parse(code, {
                sourceType: 'module',
                plugins: [
                    'jsx',
                    'typescript',
                    'classProperties',
                    'decorators-legacy',
                    'objectRestSpread',
                    'dynamicImport',
                    'optionalChaining',
                    'nullishCoalescingOperator'
                ]
            });
            traverseJS(ast, structure);
        }
    } catch (error) {
        console.error(`Error parsing file ${filePath}: ${error.message}`);
        process.exit(1);
    }

    return structure;
}

/**
 * Visits nodes in TypeScript AST to extract functions and classes.
 * @param {ts.Node} node - The current AST node.
 */
function visitTS(node) {
    if (ts.isFunctionDeclaration(node) && node.name) {
        const funcName = node.name.text;
        const params = node.parameters.map(param => param.name.getText());
        const jsDoc = getJSDoc(node);
        structure.functions.push({
            name: funcName,
            params: params,
            docstring: jsDoc
        });
    } else if (ts.isClassDeclaration(node) && node.name) {
        const className = node.name.text;
        const classDoc = getJSDoc(node);
        const methods = [];
        node.members.forEach(member => {
            if (ts.isMethodDeclaration(member) && member.name) {
                const methodName = member.name.getText();
                const methodParams = member.parameters.map(param => param.name.getText());
                const methodDoc = getJSDoc(member);
                methods.push({
                    name: methodName,
                    params: methodParams,
                    docstring: methodDoc
                });
            }
        });
        structure.classes.push({
            name: className,
            docstring: classDoc,
            methods: methods
        });
    }
    ts.forEachChild(node, visitTS);
}

/**
 * Retrieves JSDoc comments from a TypeScript node.
 * @param {ts.Node} node - The AST node.
 * @returns {string} - The JSDoc comment or an empty string.
 */
function getJSDoc(node) {
    const jsDocs = node.jsDoc;
    if (jsDocs && jsDocs.length > 0) {
        return jsDocs[0].comment || '';
    }
    return '';
}

/**
 * Traverses JavaScript AST to extract functions and classes.
 * @param {object} ast - The Babel AST object.
 * @param {object} structure - The structure object to populate.
 */
function traverseJS(ast, structure) {
    const traverse = require('@babel/traverse').default;

    traverse(ast, {
        FunctionDeclaration(path) {
            const funcName = path.node.id.name;
            const params = path.node.params.map(param => param.name);
            const jsDoc = getJSJSDoc(path);
            structure.functions.push({
                name: funcName,
                params: params,
                docstring: jsDoc
            });
        },
        ClassDeclaration(path) {
            const className = path.node.id.name;
            const classDoc = getJSJSDoc(path);
            const methods = [];
            path.traverse({
                ClassMethod(methodPath) {
                    const methodName = methodPath.node.key.name;
                    const methodParams = methodPath.node.params.map(param => param.name);
                    const methodDoc = getJSJSDoc(methodPath);
                    methods.push({
                        name: methodName,
                        params: methodParams,
                        docstring: methodDoc
                    });
                }
            });
            structure.classes.push({
                name: className,
                docstring: classDoc,
                methods: methods
            });
        }
    });
}

/**
 * Retrieves JSDoc comments from a JavaScript path.
 * @param {object} path - The Babel path object.
 * @returns {string} - The JSDoc comment or an empty string.
 */
function getJSJSDoc(path) {
    const comments = path.node.leadingComments;
    if (comments && comments.length > 0) {
        const jsDoc = comments.find(comment => comment.type === 'CommentBlock' && comment.value.startsWith('*'));
        if (jsDoc) {
            // Clean up the JSDoc comment
            return jsDoc.value.replace(/^\*/, '').trim();
        }
    }
    return '';
}

// Main Execution
if (require.main === module) {
    const filePath = process.argv[2];
    if (!filePath) {
        console.error('Usage: node extract_structure.js <path_to_js_or_ts_file>');
        process.exit(1);
    }

    const absolutePath = path.resolve(filePath);
    if (!fs.existsSync(absolutePath)) {
        console.error(`File not found: ${absolutePath}`);
        process.exit(1);
    }

    const structure = extractStructure(absolutePath);
    console.log(JSON.stringify(structure, null, 2));
}