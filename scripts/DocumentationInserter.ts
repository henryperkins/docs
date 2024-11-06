// src/common/DocumentationInserter.ts

export interface CodeElement {
    name: string;
    type: 'function' | 'class' | 'method' | 'variable';
    description?: string;
    params?: Array<{
        name: string;
        type: string;
        description: string;
    }>;
    returns?: {
        type: string;
        description: string;
    };
    examples?: string[];
}

export interface Documentation {
    elements: CodeElement[];
    metadata?: Record<string, any>;
}

export interface DocumentationInserter {
    insertDocumentation(code: string, documentation: Documentation): string;
    generateDocstring(element: CodeElement): string;
}

export abstract class BaseDocumentationInserter implements DocumentationInserter {
    protected abstract formatDocstring(content: string): string;
    
    public generateDocstring(element: CodeElement): string {
        const lines: string[] = [];
        
        // Add description
        if (element.description) {
            lines.push(element.description);
        }
        
        // Add parameters
        if (element.params && element.params.length > 0) {
            lines.push('');
            lines.push('Parameters:');
            element.params.forEach(param => {
                lines.push(`  ${param.name} (${param.type}): ${param.description}`);
            });
        }
        
        // Add return value
        if (element.returns) {
            lines.push('');
            lines.push('Returns:');
            lines.push(`  ${element.returns.type}: ${element.returns.description}`);
        }
        
        // Add examples
        if (element.examples && element.examples.length > 0) {
            lines.push('');
            lines.push('Examples:');
            element.examples.forEach(example => {
                lines.push(`  ${example}`);
            });
        }
        
        return this.formatDocstring(lines.join('\n'));
    }
    
    abstract insertDocumentation(code: string, documentation: Documentation): string;
}
