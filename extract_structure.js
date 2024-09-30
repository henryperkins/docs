const babelParser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const fs = require('fs');
const path = require('path');
const ts = require('typescript');

const function_schema = {
    // Your function schema details here
};

async function extractStructure(filePath) {
    const code = fs.readFileSync(filePath, 'utf8');
    let structure = { functions: [], classes: [] };

    try {
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
    } catch (error) {
        console.error(`Error parsing file ${filePath}: ${error.message}`);
        process.exit(1);
    }

    return structure;
}

function traverseJS(ast, structure) {
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

function getJSJSDoc(path) {
    const comments = path.node.leadingComments;
    if (comments && comments.length > 0) {
        const jsDoc = comments.find(comment => comment.type === 'CommentBlock' && comment.value.startsWith('*'));
        if (jsDoc) {
            return jsDoc.value.replace(/^\*/, '').trim();
        }
    }
    return '';
}

module.exports = { extractStructure };

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

    extractStructure(absolutePath).then(structure => {
        console.log(JSON.stringify(structure, null, 2));
    });
}
