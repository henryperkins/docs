// Enhanced JavaScript/TypeScript parser with comprehensive analysis capabilities

const babelParser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const t = require('@babel/types');
const generate = require('@babel/generator').default;
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

    parse(code, language = 'javascript', filePath = 'unknown') {
        try {
            const isTypeScript = language === 'typescript' || filePath.endsWith('.ts') || filePath.endsWith('.tsx');

            const ast = this._parseCode(code, isTypeScript);
            const structure = this._initializeStructure();

            const metrics = this._calculateMetrics(code, isTypeScript, filePath);
            Object.assign(structure, metrics);

            this._traverseAST(ast, structure, isTypeScript);
            return structure;

        } catch (error) {
            console.error(`Parse error in ${filePath}: ${error.message}`);
            return this._getEmptyStructure(error.message);
        }
    }

    _parseCode(code, isTypeScript) {
        const parserOptions = {
            sourceType: this.options.sourceType,
            plugins: this._getBabelPlugins(isTypeScript),
            errorRecovery: this.options.errorRecovery,
            tokens: true,
            ...this.options
        };

        try {
            if (isTypeScript) {
                return tsEstree.parse(code, { jsx: true, ...parserOptions });
            } else {
                return babelParser.parse(code, parserOptions);
            }
        } catch (error) {
            console.error("Parsing failed:", error);
            throw error;
        }
    }

    _calculateMetrics(code, isTypeScript, filePath) {
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
            console.error(`Metrics calculation error in ${filePath}: ${error.message}`);
            return {
                complexity: 0,
                maintainability_index: 0,
                halstead: { volume: 0, difficulty: 0, effort: 0 },
                function_metrics: {}
            };
        }
    }

    _traverseAST(ast, structure, isTypeScript) {
        traverse(ast, {
            ClassDeclaration: (path) => {
                structure.classes.push(this._extractClassInfo(path.node, path, isTypeScript));
            },
            FunctionDeclaration: (path) => {
                structure.functions.push(this._extractFunctionInfo(path.node, path, isTypeScript));
            },
            VariableDeclaration: (path) => {
                const declarations = this._extractVariableInfo(path.node, path, isTypeScript);
                const collection = path.node.kind === 'const' ? structure.constants : structure.variables;
                collection.push(...declarations);
            },
            ImportDeclaration: (path) => {
                structure.imports.push(this._extractImportInfo(path.node));
            },
            ExportDefaultDeclaration: (path) => {
                structure.exports.push(this._extractExportInfo(path.node, true));
            },
            ExportNamedDeclaration: (path) => {
                const exportInfo = this._extractExportInfo(path.node, false);
                if (Array.isArray(exportInfo)) {
                    structure.exports.push(...exportInfo);
                } else if (exportInfo) {
                    structure.exports.push(exportInfo);
                }
            },
            JSXElement: (path) => {
                if (this._isReactComponent(path)) {
                    structure.react_components.push(this._extractReactComponentInfo(path));
                }
            },
            TSInterfaceDeclaration: isTypeScript ? (path) => {
                structure.interfaces.push(this._extractInterfaceInfo(path.node));
            } : null,
            TSTypeAliasDeclaration: isTypeScript ? (path) => {
                structure.types.push(this._extractTypeAliasInfo(path.node));
            } : null,
            ArrowFunctionExpression: (path) => {
                structure.functions.push(this._extractFunctionInfo(path.node, path, isTypeScript));
            },
            ...this._getAdditionalVisitors(isTypeScript)
        });
    }

    _extractClassInfo(node, path, isTypeScript) {
        return {
            name: node.id.name,
            methods: node.body.body
                .filter(member => t.isClassMethod(member) || t.isClassPrivateMethod(member))
                .map(method => this._extractMethodInfo(method, isTypeScript)),
            properties: node.body.body
                .filter(member => t.isClassProperty(member) || t.isClassPrivateProperty(member))
                .map(prop => this._extractPropertyInfo(prop, isTypeScript)),

            superClass: node.superClass?.name,
            decorators: this._extractDecorators(node),
            docstring: this._extractDocstring(node),
            isAbstract: node.abstract || false,
            isExported: this._isExported(path),
            implements: isTypeScript ? this._extractImplementedInterfaces(node) : []
        };
    }

    _extractFunctionInfo(node, path, isTypeScript) {
        const functionName = node.id ? node.id.name : (node.key && node.key.name) || 'anonymous';
        const params = this._extractParameters(node.params, isTypeScript);
        const returnType = isTypeScript ? this._getTypeString(node.returnType) : null;
        const async = node.async || false;
        const generator = node.generator || false;

        return {
            name: functionName,
            params,
            returnType,
            docstring: this._extractDocstring(node),
            isExported: this._isExported(path),
            async: async,
            generator: generator,
            complexity: this.options.function_metrics && this.options.function_metrics[functionName] ? this.options.function_metrics[functionName].complexity : null
        };
    }

    _extractVariableInfo(node, path, isTypeScript) {
        return node.declarations.map(declarator => {
            const varName = declarator.id.name;
            const varType = isTypeScript ? this._getTypeString(declarator.id.typeAnnotation) : null;
            const defaultValue = this._getDefaultValue(declarator.init);

            return {
                name: varName,
                type: varType,
                defaultValue: defaultValue,
                docstring: this._extractDocstring(declarator),
                isExported: this._isExported(path)
            };
        });
    }

    _extractImportInfo(node) {
        const source = node.source.value;
        const specifiers = node.specifiers.map(specifier => {
            if (t.isImportSpecifier(specifier)) {
                return {
                    type: 'named',
                    imported: specifier.imported.name,
                    local: specifier.local.name,
                };
            } else if (t.isImportDefaultSpecifier(specifier)) {
                return {
                    type: 'default',
                    local: specifier.local.name
                };
            } else if (t.isImportNamespaceSpecifier(specifier)) {
                return {
                    type: 'namespace',
                    local: specifier.local.name
                };
            }
        });
        return { source, specifiers };
    }

    _extractExportInfo(node, isDefault) {
        if (isDefault) {
            const declaration = node.declaration;
            return {
                type: 'default',
                name: this._getDeclarationName(declaration),
                declaration: generate(declaration).code
            };
        } else if (node.declaration) {
            const declaration = node.declaration;
            const declarations = t.isVariableDeclaration(declaration) ? declaration.declarations : [declaration];
            return declarations.map(decl => ({
                type: 'named',
                name: this._getDeclarationName(decl),
                declaration: generate(decl).code
            }));
        } else if (node.specifiers && node.specifiers.length > 0) {
            return node.specifiers.map(specifier => ({
                type: 'named',
                exported: specifier.exported.name,
                local: specifier.local.name
            }));
        }
        return null;
    }

    _getDeclarationName(declaration) {
        if (t.isIdentifier(declaration)) {
            return declaration.name;
        } else if (t.isFunctionDeclaration(declaration) || t.isClassDeclaration(declaration)) {
            return declaration.id?.name || null;
        } else if (t.isVariableDeclarator(declaration)) {
            return declaration.id.name;
        }
        return null;
    }

    _extractInterfaceInfo(node) {
        const interfaceName = node.id.name;
        const properties = node.body.body.map(property => {
            return {
                name: property.key.name,
                type: this._getTypeString(property.typeAnnotation),
                docstring: this._extractDocstring(property)
            };
        });
        return { name: interfaceName, properties };
    }

    _extractTypeAliasInfo(node) {
        return {
            name: node.id.name,
            type: this._getTypeString(node.typeAnnotation)
        };
    }

    _extractReactComponentInfo(path) {
        const component = path.findParent(p =>
            t.isFunctionDeclaration(p) ||
            t.isArrowFunctionExpression(p) ||
            t.isClassDeclaration(p) ||
            t.isVariableDeclarator(p)
        );

        if (!component) return null;

        const componentName = this._getComponentName(component.node);
        const props = this._extractReactProps(component);
        const hooks = this._extractReactHooks(component);
        const state = this._extractReactState(component);
        const effects = this._extractReactEffects(component);
        const isExportedComponent = this._isExported(component);

        return {
            name: componentName,
            props,
            hooks,
            state,
            effects,
            docstring: this._extractDocstring(component.node),
            isExported: isExportedComponent,
            type: this._getReactComponentType(component.node)
        };
    }

    _getComponentName(node) {
        if (t.isVariableDeclarator(node)) {
            return node.id.name;
        } else if (t.isFunctionDeclaration(node) || t.isClassDeclaration(node)) {
            return node.id?.name || null;
        }
        return 'anonymous';
    }

    _getReactComponentType(node) {
        if (t.isClassDeclaration(node)) {
            return 'class';
        } else if (t.isFunctionDeclaration(node) || t.isArrowFunctionExpression(node)) {
            return 'function';
        } else if (t.isVariableDeclarator(node)) {
            return 'variable';
        }
        return null;
    }

    _extractReactProps(componentPath) {
        const component = componentPath.node;
        let props = [];

        if (t.isClassDeclaration(component)) {
            const constructor = component.body.body.find(member => t.isClassMethod(member) && member.kind === 'constructor');
            if (constructor && constructor.params.length > 0) {
                props = this._extractPropsFromParam(constructor.params[0]);
            }
        } else if (t.isFunctionDeclaration(component) || t.isArrowFunctionExpression(component)) {
            if (component.params.length > 0) {
                props = this._extractPropsFromParam(component.params[0]);
            }
        } else if (t.isVariableDeclarator(component)) {
            if (component.init && (t.isArrowFunctionExpression(component.init) || t.isFunctionExpression(component.init))) {
                if (component.init.params.length > 0) {
                    props = this._extractPropsFromParam(component.init.params[0]);
                }
            }
        }

        return props;
    }

    _extractPropsFromParam(param) {
        if (param.typeAnnotation) {
            const typeAnnotation = param.typeAnnotation.typeAnnotation;
            if (t.isTSTypeLiteral(typeAnnotation)) {
                return typeAnnotation.members.map(member => ({
                    name: member.key.name,
                    type: this._getTypeString(member.typeAnnotation),
                    required: !member.optional,
                    defaultValue: this._getDefaultValue(member)
                }));
            } else if (t.isTSTypeReference(typeAnnotation) && t.isIdentifier(typeAnnotation.typeName)) {
                return [{ name: param.name, type: typeAnnotation.typeName.name, required: !param.optional }];
            }
        } else if (t.isObjectPattern(param)) {
            return param.properties.map(prop => ({
                name: prop.key.name,
                type: this._getTypeString(prop.value?.typeAnnotation),
                required: true
            }));
        }
        return [];
    }

    _extractReactHooks(componentPath) {
        const hooks = [];
        componentPath.traverse({
            CallExpression(path) {
                if (t.isIdentifier(path.node.callee) && path.node.callee.name.startsWith('use')) {
                    const hookName = path.node.callee.name;
                    const dependencies = this._extractHookDependencies(path.node);
                    hooks.push({ name: hookName, dependencies });
                }
            }
        });
        return hooks;
    }

    _extractHookDependencies(node) {
        if (node.arguments && node.arguments.length > 1 && t.isArrayExpression(node.arguments[1])) {
            return node.arguments[1].elements.map(element => generate(element).code);
        }
        return [];
    }

    _extractReactEffects(componentPath) {
        const effects = [];
        componentPath.traverse({
            CallExpression(path) {
                if (t.isIdentifier(path.node.callee) && path.node.callee.name === 'useEffect') {
                    const dependencies = this._extractHookDependencies(path.node);
                    const cleanup = this._hasEffectCleanup(path.node);
                    effects.push({ dependencies, cleanup });
                }
            }
        });
        return effects;
    }

    _hasEffectCleanup(node) {
        if (node.arguments && node.arguments.length > 0 && t.isArrowFunctionExpression(node.arguments[0]) && node.arguments[0].body) {
            const body = node.arguments[0].body;
            return t.isBlockStatement(body) && body.body.some(statement => t.isReturnStatement(statement) && statement.argument !== null);
        }
        return false;
    }

    _extractReactState(componentPath) {
        const state = [];
        componentPath.traverse({
            CallExpression(path) {
                if (t.isMemberExpression(path.node.callee) &&
                    t.isIdentifier(path.node.callee.object, { name: 'React' }) &&
                    t.isIdentifier(path.node.callee.property, { name: 'useState' })) {

                    const initialValue = path.node.arguments[0];
                    state.push({
                        initialValue: generate(initialValue).code
                    });
                } else if (t.isIdentifier(path.node.callee, { name: 'useState' })) {
                    const initialValue = path.node.arguments[0];
                    state.push({
                        initialValue: generate(initialValue).code
                    });
                }
            }
        });
        return state;
    }

    _getDefaultValue(node) {
        if (!node) return null;
        return generate(node).code;
    }

    _getTypeString(typeAnnotation) {
        if (!typeAnnotation) return null;
        if (t.isTSTypeReference(typeAnnotation)) {
            return generate(typeAnnotation.typeName).code;
        } else if (t.isTSLiteralType(typeAnnotation)) {
            return generate(typeAnnotation.literal).code;
        } else if (t.isTSTypeAnnotation(typeAnnotation)) {
            return this._getTypeString(typeAnnotation.typeAnnotation);
        }
        return null;
    }

    _extractParameters(params, isTypeScript) {
        return params.map(param => {
            return {
                name: param.name,
                type: isTypeScript ? this._getTypeString(param.typeAnnotation) : null,
                defaultValue: this._getDefaultValue(param.defaultValue)
            };
        });
    }

    _extractReturnType(node) {
        return this._getTypeString(node.returnType);
    }

    _extractDecorators(node) {
        return (node.decorators || []).map(decorator => generate(decorator.expression).code);
    }

    _getAccessibility(node) {
        return node.accessibility || 'public';
    }

    _getMethodName(node) {
        if (node.key && t.isIdentifier(node.key)) {
            return node.key.name;
        } else if (node.key && t.isPrivateName(node.key)) {
            return `#${node.key.id.name}`;
        }
        return null;
    }

    _calculateMethodComplexity(node) {
        return null;
    }

    _isExported(path) {
        let parent = path.parentPath;
        while (parent) {
            if (parent.isExportNamedDeclaration() || parent.isExportDefaultDeclaration()) {
                return true;
            }
            parent = parent.parentPath;
        }
        return false;
    }

    _isReactComponent(path) {
        return t.isJSXIdentifier(path.node.openingElement.name);
    }

    _getBabelPlugins(isTypeScript) {
        const plugins = [
            'jsx',
            'decorators-legacy',
            ['decorators', { decoratorsBeforeExport: true }],
            'classProperties', 'classPrivateProperties', 'classPrivateMethods',
            'exportDefaultFrom', 'exportNamespaceFrom', 'dynamicImport',
            'nullishCoalescing', 'optionalChaining', 'asyncGenerators', 'bigInt',
            'classProperties', 'doExpressions', 'dynamicImport', 'exportDefaultFrom',
            'exportNamespaceFrom', 'functionBind', 'functionSent', 'importMeta',
            'logicalAssignment', 'numericSeparator', 'nullishCoalescingOperator',
            'optionalCatchBinding', 'optionalChaining', 'partialApplication',
            'throwExpressions', "pipelineOperator", "recordAndTuple"
        ];

        if (isTypeScript) {
            plugins.push('typescript');
        }
        return plugins;
    }

    _getAdditionalVisitors(isTypeScript) {
        if (isTypeScript) {
            return {
                TSEnumDeclaration(path) {
                    this.node.enums.push({
                        name: path.node.id.name,
                        members: path.node.members.map(member => ({
                            name: member.id.name,
                            initializer: member.initializer ? generate(member.initializer).code : null
                        }))
                    });
                },
                TSTypeAliasDeclaration(path) {
                    this.node.types.push({
                        name: path.node.id.name,
                        type: generate(path.node.typeAnnotation).code
                    });
                },
                TSInterfaceDeclaration(path) {
                    this.node.interfaces.push({
                        name: path.node.id.name,
                    });
                },
            };
        }
        return {};
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
            enums: [],
            react_components: [],
            complexity: 0,
            maintainability_index: 0,
            halstead: { volume: 0, difficulty: 0, effort: 0 },
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
        const leadingComments = node.leadingComments || [];
        const docstringComment = leadingComments.find(comment => comment.type === 'CommentBlock' && comment.value.trim().startsWith('*'));
        return docstringComment ? docstringComment.value.replace(/^\*\s?/gm, '').trim() : '';
    }

    _extractPropertyInfo(node, isTypeScript) {
        const propertyName = node.key.name;
        const propertyType = isTypeScript ? this._getTypeString(node.typeAnnotation) : null;
        const defaultValue = this._getDefaultValue(node.value);
        const accessibility = this._getAccessibility(node);
        const isStatic = node.static || false;
        const decorators = this._extractDecorators(node);
        const docstring = this._extractDocstring(node);

        return {
            name: propertyName,
            type: propertyType,
            defaultValue: defaultValue,
            accessibility,
            isStatic,
            decorators,
            docstring
        };
    }

    _extractImplementedInterfaces(node) {
        return (node.implements || []).map(i => i.id.name);
    }
}

module.exports = JSTSParser;