// src/common/BaseInserter.ts

interface DocumentationInput {
    code: string;
    documentation: any;
    language: string;
}

export class BaseInserter {
    protected supportedLanguage: string;

    constructor(language: string) {
        this.supportedLanguage = language.toLowerCase();
    }

    protected async readStdin(): Promise<string> {
        const chunks: Buffer[] = [];
        
        return new Promise((resolve, reject) => {
            process.stdin
                .on('data', (chunk: Buffer) => chunks.push(chunk))
                .on('end', () => resolve(Buffer.concat(chunks).toString()))
                .on('error', reject);
        });
    }

    protected parseInput(input: string): DocumentationInput {
        try {
            const parsedInput = JSON.parse(input);
            this.validateInput(parsedInput);
            return parsedInput;
        } catch (error) {
            throw new Error(`Failed to parse input: ${error.message}`);
        }
    }

    protected validateInput(input: DocumentationInput): void {
        if (!input.code || typeof input.code !== 'string') {
            throw new Error('Invalid or missing code in input');
        }
        if (!input.documentation) {
            throw new Error('Missing documentation in input');
        }
        if (!input.language || input.language.toLowerCase() !== this.supportedLanguage) {
            throw new Error(`Unsupported language: ${input.language}`);
        }
    }

    public async process(): Promise<string> {
        const input = await this.readStdin();
        const { code, documentation, language } = this.parseInput(input);
        return this.insertDocumentation(code, documentation);
    }

    protected insertDocumentation(code: string, documentation: any): string {
        // To be implemented by language-specific inserters
        throw new Error('insertDocumentation must be implemented by subclass');
    }
}
