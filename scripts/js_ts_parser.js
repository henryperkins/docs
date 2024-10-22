// Enhanced JavaScript/TypeScript parser with comprehensive analysis capabilities

const babelParser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const t = require('@babel/types');
const tsEstree = require('@typescript-eslint/typescript-estree');
const escomplex = require('typhonjs-escomplex');

class JSTSParser {
    constructor(options = {}) {
        this.options = {
            sourceType: 'module',
            errorRecovery: true,
            ...options
        };
    }

    parse(code, language = 'javascript') {
        try {
            const isTypeScript = language === 'typescript';
            const ast = this._parseCode(code, isTypeScript);
            const structure = this._initializeStructure();
            
            // Calculate metrics first
            const metrics = this._calculateMetrics(code, isTypeScript);
            Object.assign(structure, metrics);
            
            this._traverseAST(ast, structure, isTypeScript);
            
            return structure;
        } catch (error) {
            console.error(`Parse error: ${error.message}`);
            return this._getEmptyStructure(error.message);
        }
    }

    _parseCode(code, isTypeScript) {
        if (isTypeScript) {
            return tsEstree.parse(code, {
                jsx: true,
                tokens: true,
                loc: true,
                range: true,
                comment: true,
            });
        }

        return babelParser.parse(code, {
            sourceType: this.options.sourceType,
            plugins: this._getBabelPlugins(isTypeScript),
            errorRecovery: this.options.errorRecovery,
            tokens: true,
        });
    }

    _calculateMetrics(code, isTypeScript) {
        try {
            const analysis = escomplex.analyzeModule(code, {
                sourceType: 'module',
                useTypeScriptEstree: isTypeScript,
                loc: true,
                newmi: true,
                skipCalculation: false
            });

            return {
                complexity: analysis.aggregate.cyclomatic,
                maintainability_index: analysis.maintainability,
                halstead: {
                    volume: analysis.aggregate.halstead.volume,
                    difficulty: analysis.aggregate.halstead.difficulty,
                    effort: analysis.aggregate.halstead.effort
                },
                function_metrics: analysis.methods.reduce((acc, method) => {
                    acc[method.name] = {
                        complexity: method.cyclomatic,
                        sloc: method.sloc,
                        params: method.params
                    };
                    return acc;
                }, {})
            };
        } catch (error) {
            console.error(`Metrics calculation error: ${error.message}`);
            return {
                complexity: 0,
                maintainability_index: 0,
                halstead: { volume: 0, difficulty: 0, effort: 0 },
                function_metrics: {}
            };
        }
    }

    _traverseAST(ast, structure, isTypeScript) {
        const visitors = {
            // Class handling
            ClassDeclaration: (path) => {
                const classInfo = this._extractClassInfo(path, isTypeScript);
                structure.classes.push(classInfo);
            },

            // Function handling
            FunctionDeclaration: (path) => {
                const functionInfo = this._extractFunctionInfo(path, isTypeScript);
                structure.functions.push(functionInfo);
            },

            // Variable declarations
            VariableDeclaration: (path) => {
                const declarations = this._extractVariableInfo(path, isTypeScript);
                const collection = path.node.kind === 'const' ? 
                    structure.constants : structure.variables;
                collection.push(...declarations);
            },

            // Import/Export handling
            ImportDeclaration: (path) => {
                const importInfo = this._extractImportInfo(path);
                structure.imports.push(importInfo);
            },

            ExportDefaultDeclaration: (path) => {
                const exportInfo = this._extractExportInfo(path, true);
                structure.exports.push(exportInfo);
            },

            ExportNamedDeclaration: (path) => {
                const exportInfo = this._extractExportInfo(path, false);
                structure.exports.push(exportInfo);
            },

            // React component handling
            JSXElement: (path) => {
                if (this._isReactComponent(path)) {
                    const componentInfo = this._extractReactComponentInfo(path);
                    structure.react_components.push(componentInfo);
                }
            },

            // TypeScript specific handlers
            TSInterfaceDeclaration: (path) => {
                if (isTypeScript) {
                    const interfaceInfo = this._extractInterfaceInfo(path);
                    structure.interfaces.push(interfaceInfo);
                }
            },

            TSTypeAliasDeclaration: (path) => {
                if (isTypeScript) {
                    const typeInfo = this._extractTypeInfo(path);
                    structure.types.push(typeInfo);
                }
            }
        };

        traverse(ast, visitors);
    }

    _extractClassInfo(path, isTypeScript) {
        const node = path.node;
        const className = node.id?.name || 'Anonymous';
        const decorators = this._extractDecorators(node);
        const superClass = node.superClass?.name;
        const implementedInterfaces = isTypeScript ? 
            this._extractImplementedInterfaces(node) : [];

        const methods = node.body.body
            .filter(member => t.isClassMethod(member) || t.isClassPrivateMethod(member))
            .map(method => this._extractMethodInfo(method, isTypeScript));

        const properties = node.body.body
            .filter(member => t.isClassProperty(member) || t.isClassPrivateProperty(member))
            .map(prop => this._extractPropertyInfo(prop, isTypeScript));

        return {
            name: className,
            superClass,
            interfaces: implementedInterfaces,
            decorators,
            methods,
            properties,
            docstring: this._extractDocstring(node),
            isAbstract: node.abstract || false,
            isExported: this._isExported(path)
        };
    }

    _extractMethodInfo(node, isTypeScript) {
        const methodName = this._getMethodName(node);
        const params = this._extractParameters(node.params, isTypeScript);
        const returnType = isTypeScript ? this._extractReturnType(node) : null;
        const decorators = this._extractDecorators(node);
        const accessibility = this._getAccessibility(node);
        const isAsync = node.async || false;
        const isStatic = node.static || false;
        const isAbstract = node.abstract || false;

        return {
            name: methodName,
            params,
            returnType,
            decorators,
            accessibility,
            isAsync,
            isStatic,
            isAbstract,
            docstring: this._extractDocstring(node),
            complexity: this._calculateMethodComplexity(node)
        };
    }

    _extractReactComponentInfo(path) {
        const component = path.findParent(p => 
            t.isFunctionDeclaration(p) || 
            t.isArrowFunctionExpression(p) || 
            t.isClassDeclaration(p)
        );

        if (!component) return null;

        const props = this._extractReactProps(component);
        const hooks = this._extractReactHooks(component);
        const state = this._extractReactState(component);
        const effects = this._extractReactEffects(component);

        return {
            name: component.node.id?.name || 'Anonymous',
            type: t.isClassDeclaration(component) ? 'class' : 'function',
            props,
            hooks,
            state,
            effects,
            docstring: this._extractDocstring(component.node),
            isExported: this._isExported(component)
        };
    }

    _extractReactProps(component) {
        const props = [];

        if (t.isClassDeclaration(component)) {
            // Handle class component props
            const constructor = component.node.body.body
                .find(node => t.isClassMethod(node) && node.kind === 'constructor');
            
            if (constructor && constructor.params[0]) {
                const propsParam = constructor.params[0];
                props.push(...this._extractPropsFromTypeAnnotation(propsParam));
            }
        } else {
            // Handle functional component props
            const param = component.node.params[0];
            if (param) {
                props.push(...this._extractPropsFromTypeAnnotation(param));
            }
        }

        return props;
    }

    _extractPropsFromTypeAnnotation(param) {
        if (!param.typeAnnotation) return [];

        const propsType = param.typeAnnotation.typeAnnotation;
        if (!t.isTSTypeLiteral(propsType)) return [];

        return propsType.members.map(member => ({
            name: member.key.name,
            type: this._getTypeString(member.typeAnnotation.typeAnnotation),
            required: !member.optional,
            defaultValue: this._getDefaultValue(member)
        }));
    }

    _extractReactHooks(component) {
        const hooks = [];
        traverse(component.node, {
            CallExpression(path) {
                if (t.isIdentifier(path.node.callee) && 
                    path.node.callee.name.startsWith('use')) {
                    hooks.push({
                        name: path.node.callee.name,
                        dependencies: this._extractHookDependencies(path.node)
                    });
                }
            }
        });
        return hooks;
    }

    _extractReactEffects(component) {
        const effects = [];
        traverse(component.node, {
            CallExpression(path) {
                if (t.isIdentifier(path.node.callee) && 
                    path.node.callee.name === 'useEffect') {
                    effects.push({
                        dependencies: this._extractHookDependencies(path.node),
                        cleanup: this._hasEffectCleanup(path.node)
                    });
                }
            }
        });
        return effects;
    }

    _getBabelPlugins(isTypeScript) {
        const plugins = [
            'jsx',
            'decorators-legacy',
            ['decorators', { decoratorsBeforeExport: true }],
            'classProperties',
            'classPrivateProperties',
            'classPrivateMethods',
            'exportDefaultFrom',
            'exportNamespaceFrom',
            'dynamicImport',
            'nullishCoalescing',
            'optionalChaining',
        ];

        if (isTypeScript) {
            plugins.push('typescript');
        }

        return plugins;
    }

    _initializeStructure() {
        return {
            classes: [],
            functions: [],
            variables: [],
            constants: [],
            imports: [],
            exports: [],
            interfaces: [],
            types: [],
            react_components: [],
            complexity: 0,
            maintainability_index: 0,
            halstead: {
                volume: 0,
                difficulty: 0,
                effort: 0
            },
            summary: "",
            function_metrics: {}
        };
    }

    _getEmptyStructure(reason = '') {
        return {
            ...this._initializeStructure(),
            summary: `Empty structure: ${reason}`
        };
    }

    _extractDocstring(node) {
        const comments = node.leadingComments || [];
        for (const comment of comments) {
            if (comment.type === 'CommentBlock' && 
                (comment.value.startsWith('*') || comment.value.startsWith('/'))) {
                return comment.value.replace(/^\*+/, '').trim();
            }
        }
        return '';
    }
}

module.exports = JSTSParser;