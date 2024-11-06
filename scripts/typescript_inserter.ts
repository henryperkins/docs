// src/inserters/typescript_inserter.ts

import { BaseInserter } from '../common/BaseInserter';
import * as ts from 'typescript';
import { parse } from '@babel/parser';
import traverse from '@babel/traverse';
import generate from '@babel/generator';
import * as t from '@babel/types';

interface TypeScriptDocumentation {
    functions: Array<{
        name: string;
        description: string;
        params: Array<{
            name: string;
            type: string;
            description: string;
        }>;
        returns?: {
            type: string;
            description: string;
        };
        typeParameters?: Array<{
            name: string;
            constraint?: string;
            default?: string;
        }>;
    }>;
    classes: Array<{
        name: string;
        description: string;
        typeParameters?: Array<{
            name: string;
            constraint?: string;
            default?: string;
        }>;
        methods: Array<{
            name: string;
            description: string;
            params: Array<{
                name: string;
                type: string;
                description: string;
            }>;
            returns?: {
                type: string;
                description: string;
            };
            typeParameters?: Array<{
                name: string;
                constraint?: string;
                default?: string;
            }>;
        }>;
        properties: Array<{
            name: string;
            type: string;
            description: string;
            visibility?: 'public' | 'private' | 'protected';
        }>;
    }>;
    interfaces: Array<{
        name: string;
        description: string;
        typeParameters?: Array<{
            name: string;
            constraint?: string;
            default?: string;
        }>;
        properties: Array<{
            name: string;
            type: string;
            description: string;
            optional?: boolean;
        }>;
        extends?: string[];
    }>;
    types: Array<{
        name: string;
        description: string;
        typeParameters?: Array<{
            name: string;
            constraint?: string;
            default?: string;
        }>;
        type: string;
    }>;
}

export class TypeScriptInserter extends BaseInserter {
    private babelOptions = {
        sourceType: 'module',
        plugins: [
            'typescript',
            'decorators-legacy',
            ['decorators', { decoratorsBeforeExport: true }],
            'classProperties',
            'classPrivateProperties',
            'classPrivateMethods',
            'exportDefaultFrom',
            'exportNamespaceFrom',
            'dynamicImport',
            'nullishCoalescingOperator',
            'optionalChaining',
        ]
    };

    constructor() {
        super('typescript');
    }

    private generateTSDoc(element: any): string {
        const lines: string[] = ['/**'];
        
        // Add description
        if (element.description) {
            lines.push(` * ${element.description}`);
            lines.push(' *');
        }

        // Add type parameters
        if (element.typeParameters?.length > 0) {
            element.typeParameters.forEach((tp: any) => {
                let typeParamDoc = ` * @template ${tp.name}`;
                if (tp.constraint) {
                    typeParamDoc += ` - extends ${tp.constraint}`;
                }
                if (tp.default) {
                    typeParamDoc += ` - defaults to ${tp.default}`;
                }
                lines.push(typeParamDoc);
            });
            lines.push(' *');
        }

        // Add parameters
        if (element.params?.length > 0) {
            element.params.forEach((param: any) => {
                lines.push(` * @param ${param.name} - ${param.description}`);
            });
            lines.push(' *');
        }

        // Add return type
        if (element.returns) {
            lines.push(` * @returns ${element.returns.description}`);
        }

        // Add examples if present
        if (element.examples?.length > 0) {
            lines.push(' * @example');
            element.examples.forEach((example: string) => {
                example.split('\n').forEach(line => {
                    lines.push(` * ${line}`);
                });
            });
        }

        lines.push(' */');
        return lines.join('\n');
    }

    private processDocumentation(documentation: TypeScriptDocumentation): Map<string, any> {
        const docMap = new Map<string, any>();

        // Process interfaces
        documentation.interfaces?.forEach(iface => {
            docMap.set(iface.name, {
                type: 'interface',
                doc: this.generateTSDoc(iface)
            });
        });

        // Process types
        documentation.types?.forEach(type => {
            docMap.set(type.name, {
                type: 'type',
                doc: this.generateTSDoc(type)
            });
        });

        // Process functions
        documentation.functions?.forEach(func => {
            docMap.set(func.name, {
                type: 'function',
                doc: this.generateTSDoc(func)
            });
        });

        // Process classes
        documentation.classes?.forEach(cls => {
            docMap.set(cls.name, {
                type: 'class',
                doc: this.generateTSDoc(cls)
            });

            // Process class methods
            cls.methods?.forEach(method => {
                docMap.set(`${cls.name}.${method.name}`, {
                    type: 'method',
                    doc: this.generateTSDoc(method)
                });
            });

            // Process class properties
            cls.properties?.forEach(prop => {
                docMap.set(`${cls.name}.${prop.name}`, {
                    type: 'property',
                    doc: this.generateTSDoc(prop)
                });
            });
        });

        return docMap;
    }

    protected insertDocumentation(code: string, documentation: TypeScriptDocumentation): string {
        try {
            const ast = parse(code, this.babelOptions);
            const docMap = this.processDocumentation(documentation);

            traverse(ast, {
                InterfaceDeclaration: (path) => {
                    this.insertInterfaceDoc(path, docMap);
                },
                TypeAlias: (path) => {
                    this.insertTypeDoc(path, docMap);
                },
                FunctionDeclaration: (path) => {
                    this.insertFunctionDoc(path, docMap);
                },
                ClassDeclaration: (path) => {
                    this.insertClassDoc(path, docMap);
                },
                ClassProperty: (path) => {
                    this.insertPropertyDoc(path, docMap);
                },
                ClassMethod: (path) => {
                    this.insertMethodDoc(path, docMap);
                },
                ArrowFunctionExpression: (path) => {
                    this.insertArrowFunctionDoc(path, docMap);
                }
            });

            const output = generate(ast, {
                retainLines: true,
                comments: true
            });

            return output.code;
        } catch (error) {
            throw new Error(`Failed to insert documentation: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    private insertInterfaceDoc(path: any, docMap: Map<string, any>): void {
        if (path.node.id?.name) {
            const docInfo = docMap.get(path.node.id.name);
            if (docInfo) {
                this.addCommentToNode(path.node, docInfo.doc);
            }
        }
    }

    private insertTypeDoc(path: any, docMap: Map<string, any>): void {
        if (path.node.id?.name) {
            const docInfo = docMap.get(path.node.id.name);
            if (docInfo) {
                this.addCommentToNode(path.node, docInfo.doc);
            }
        }
    }

    private insertFunctionDoc(path: any, docMap: Map<string, any>): void {
        if (path.node.id?.name) {
            const docInfo = docMap.get(path.node.id.name);
            if (docInfo) {
                this.addCommentToNode(path.node, docInfo.doc);
            }
        }
    }

    private insertClassDoc(path: any, docMap: Map<string, any>): void {
        if (path.node.id?.name) {
            const docInfo = docMap.get(path.node.id.name);
            if (docInfo) {
                this.addCommentToNode(path.node, docInfo.doc);
            }
        }
    }

    private insertPropertyDoc(path: any, docMap: Map<string, any>): void {
        const classPath = path.findParent((p: any) => p.isClassDeclaration());
        if (classPath?.node.id?.name && path.node.key?.name) {
            const fullName = `${classPath.node.id.name}.${path.node.key.name}`;
            const docInfo = docMap.get(fullName);
            if (docInfo) {
                this.addCommentToNode(path.node, docInfo.doc);
            }
        }
    }

    private insertMethodDoc(path: any, docMap: Map<string, any>): void {
        const classPath = path.findParent((p: any) => p.isClassDeclaration());
        if (classPath?.node.id?.name && path.node.key?.name) {
            const fullName = `${classPath.node.id.name}.${path.node.key.name}`;
            const docInfo = docMap.get(fullName);
            if (docInfo) {
                this.addCommentToNode(path.node, docInfo.doc);
            }
        }
    }

    private insertArrowFunctionDoc(path: any, docMap: Map<string, any>): void {
        const parentPath = path.parentPath;
        if (parentPath.isVariableDeclarator() && parentPath.node.id?.name) {
            const docInfo = docMap.get(parentPath.node.id.name);
            if (docInfo) {
                this.addCommentToNode(parentPath.parentPath.node, docInfo.doc);
            }
        }
    }

    private addCommentToNode(node: any, docString: string): void {
        if (!node.leadingComments) {
            node.leadingComments = [];
        }
        
        // Remove existing TSDoc comments
        node.leadingComments = node.leadingComments.filter(
            (comment: any) => !comment.value.startsWith('*')
        );

        // Add new TSDoc comment
        node.leadingComments.push({
            type: 'CommentBlock',
            value: docString.slice(3, -2).trim() // Remove /** and */ and trim
        });
    }

    public async process(): Promise<string> {
        try {
            const input = await this.readStdin();
            const { code, documentation } = this.parseInput(input);
            return this.insertDocumentation(code, documentation);
        } catch (error) {
            console.error(`Error processing TypeScript file: ${error instanceof Error ? error.message : String(error)}`);
            process.exit(1);
        }
    }
}

// Run the inserter if called directly
if (require.main === module) {
    const inserter = new TypeScriptInserter();
    inserter.process()
        .then(result => {
            console.log(result);
        })
        .catch(error => {
            console.error(error);
            process.exit(1);
        });
}
