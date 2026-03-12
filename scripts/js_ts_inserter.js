#!/usr/bin/env node

/**
 * js_ts_inserter.js
 *
 * Inserts JSDoc/TSDoc comments into JavaScript/TypeScript source code.
 *
 * Reads JSON from stdin:
 *   { code, documentation, language, options }
 *
 * documentation.functions[]: { name, docstring, args, async, ... }
 * documentation.classes[]:   { name, docstring, methods[] }
 *
 * Outputs modified code to stdout.
 */

const { parse } = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;

// --- Babel parser plugins (broad compatibility) ---
const BABEL_PLUGINS = [
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
    'topLevelAwait',
    'importAssertions',
];

/**
 * Normalise a docstring from the LLM into the inner content of a block comment.
 * The LLM may return:
 *   - A full JSDoc block:  "/** ... * /"
 *   - Just the description text: "Does something useful."
 *
 * We always want the inner content (without the leading / ** and trailing * /)
 * because @babel/types stores it that way in `CommentBlock.value`.
 */
function normaliseDocstring(raw) {
    if (!raw || typeof raw !== 'string') return null;

    let s = raw.trim();

    // Strip outer /** ... */
    if (s.startsWith('/**')) s = s.slice(3);
    if (s.startsWith('/*'))  s = s.slice(2);
    if (s.endsWith('*/'))    s = s.slice(0, -2);

    s = s.trim();
    if (!s) return null;

    // Ensure it reads like a JSDoc inner block:  *\n * line\n
    // If it already has leading asterisks on each line, keep them.
    // Otherwise, format it.
    const lines = s.split('\n');
    const alreadyFormatted = lines.every(l => l.trimStart().startsWith('*') || l.trim() === '');

    if (alreadyFormatted) {
        return '*\n' + lines.map(l => ' ' + l.trimEnd()).join('\n') + '\n ';
    }

    // Plain text — wrap it
    const wrapped = lines.map(l => ` * ${l}`).join('\n');
    return `*\n${wrapped}\n `;
}

/**
 * Build a Map<name, commentValue> from the LLM documentation structure.
 * Keys: function name, class name, "ClassName.methodName"
 */
function buildDocMap(documentation) {
    const map = new Map();

    // Functions
    if (Array.isArray(documentation.functions)) {
        for (const func of documentation.functions) {
            if (!func.name) continue;
            const val = normaliseDocstring(func.docstring);
            if (val) map.set(func.name, val);
        }
    }

    // Classes + methods
    if (Array.isArray(documentation.classes)) {
        for (const cls of documentation.classes) {
            if (!cls.name) continue;
            const val = normaliseDocstring(cls.docstring);
            if (val) map.set(cls.name, val);

            if (Array.isArray(cls.methods)) {
                for (const method of cls.methods) {
                    if (!method.name) continue;
                    const mval = normaliseDocstring(method.docstring);
                    if (mval) map.set(`${cls.name}.${method.name}`, mval);
                }
            }
        }
    }

    return map;
}

/**
 * Returns true if the node already has a JSDoc leading comment.
 */
function hasJSDoc(node) {
    return (node.leadingComments || []).some(
        c => c.type === 'CommentBlock' && c.value.trimStart().startsWith('*')
    );
}

/**
 * Attach a JSDoc block comment as a leading comment on the node.
 * If preserveExisting is false, remove any existing JSDoc first.
 */
function attachComment(node, commentValue, preserveExisting) {
    if (preserveExisting && hasJSDoc(node)) return;

    if (!node.leadingComments) node.leadingComments = [];

    // Remove existing JSDoc comments
    node.leadingComments = node.leadingComments.filter(
        c => !(c.type === 'CommentBlock' && c.value.trimStart().startsWith('*'))
    );

    node.leadingComments.push({
        type: 'CommentBlock',
        value: commentValue,
    });
}

/**
 * Main: parse code, traverse AST, attach docstrings, generate output.
 */
function insertDocstrings(code, documentation, options = {}) {
    const preserveExisting = options.preserveExisting !== false;

    const ast = parse(code, {
        sourceType: 'module',
        plugins: BABEL_PLUGINS,
        errorRecovery: true,       // don't crash on minor issues
        allowReturnOutsideFunction: true,
    });

    const docMap = buildDocMap(documentation);
    if (docMap.size === 0) return code; // nothing to insert

    traverse(ast, {
        // --- Named function declarations ---
        FunctionDeclaration(path) {
            const name = path.node.id && path.node.id.name;
            if (name && docMap.has(name)) {
                attachComment(path.node, docMap.get(name), preserveExisting);
            }
        },

        // --- Export default function ---
        ExportDefaultDeclaration(path) {
            const decl = path.node.declaration;
            if (decl && decl.type === 'FunctionDeclaration' && decl.id) {
                const name = decl.id.name;
                if (docMap.has(name)) {
                    attachComment(path.node, docMap.get(name), preserveExisting);
                }
            }
        },

        // --- Arrow functions assigned to variables (const foo = () => {}) ---
        VariableDeclaration(path) {
            if (path.node.declarations.length !== 1) return;
            const decl = path.node.declarations[0];
            if (!decl.id || !decl.id.name) return;
            const init = decl.init;
            if (!init) return;

            const isFunc =
                init.type === 'ArrowFunctionExpression' ||
                init.type === 'FunctionExpression';

            if (isFunc && docMap.has(decl.id.name)) {
                attachComment(path.node, docMap.get(decl.id.name), preserveExisting);
            }
        },

        // --- Class declarations ---
        ClassDeclaration(path) {
            const name = path.node.id && path.node.id.name;
            if (name && docMap.has(name)) {
                attachComment(path.node, docMap.get(name), preserveExisting);
            }
        },

        // --- Class methods ---
        ClassMethod(path) {
            const classPath = path.findParent(p => p.isClassDeclaration());
            if (!classPath || !classPath.node.id) return;
            const className = classPath.node.id.name;
            const methodName = path.node.key && path.node.key.name;
            if (!methodName) return;

            const fullName = `${className}.${methodName}`;
            if (docMap.has(fullName)) {
                attachComment(path.node, docMap.get(fullName), preserveExisting);
            } else if (docMap.has(methodName)) {
                // Fallback: match by method name alone
                attachComment(path.node, docMap.get(methodName), preserveExisting);
            }
        },
    });

    const output = generate(ast, {
        retainLines: false,
        comments: true,
        jsescOption: { minimal: true },
    });

    return output.code;
}

// --- stdin/stdout interface ---
function readStdin() {
    return new Promise((resolve, reject) => {
        let data = '';
        process.stdin.setEncoding('utf-8');
        process.stdin.on('data', chunk => { data += chunk; });
        process.stdin.on('end', () => resolve(data));
        process.stdin.on('error', reject);
    });
}

async function main() {
    try {
        const raw = await readStdin();
        const input = JSON.parse(raw);
        const { code, documentation, options } = input;

        if (!code) {
            process.stderr.write('Error: no code provided\n');
            process.exit(1);
        }

        const result = insertDocstrings(code, documentation || {}, options || {});
        // Output as JSON with { code: "..." } so the Python caller can parse it
        process.stdout.write(JSON.stringify({ code: result }));
    } catch (err) {
        process.stderr.write(`Error: ${err.message}\n`);
        process.exit(1);
    }
}

main();
