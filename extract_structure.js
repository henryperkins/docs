// extract_structure.js
const ts = require("typescript");
const fs = require("fs");

const filePath = process.argv[2];
const fileContent = fs.readFileSync(filePath, "utf8");

function extractStructure(code, fileName) {
    const sourceFile = ts.createSourceFile(
        fileName,
        code,
        ts.ScriptTarget.Latest,
        true
    );

    const functions = [];
    const classes = [];

    function visit(node, parentClass = null) {
        switch (node.kind) {
            case ts.SyntaxKind.FunctionDeclaration:
                functions.push({
                    name: node.name?.text || "anonymous",
                    args: node.parameters.map((param) => param?.name?.text || ""), // Handle missing param names
                    returnType: node.type?.getText() || "any",
                    docstring: node.jsDoc?.[0]?.comment || "",
                    start: node.getStart(),
                    end: node.getEnd(),
                });
                break;
            case ts.SyntaxKind.ClassDeclaration:
                const classInfo = {
                    name: node.name?.text || "", // Handle missing class names
                    bases: node.heritageClauses?.map((clause) =>
                        clause.types.map((type) => type.expression.getText())
                    ).flat() || [],  // Simplified base class extraction
                    methods: [],
                    docstring: node.jsDoc?.[0]?.comment || "",
                    start: node.getStart(),
                    end: node.getEnd(),
                };
                classes.push(classInfo);
                node.members.forEach((member) => visit(member, classInfo));
                break;
            case ts.SyntaxKind.MethodDeclaration:
            case ts.SyntaxKind.MethodSignature:
                parentClass?.methods.push({ // Use optional chaining for parentClass
                    name: node.name?.text || "",  // Handle missing method names
                    args: node.parameters.map((param) => param?.name?.text || ""), // Handle missing param names
                    returnType: node.type?.getText() || "any",
                    docstring: node.jsDoc?.[0]?.comment || "", //  Extract docstrings for methods!
                    start: node.getStart(),
                    end: node.getEnd(),
                });
                break;

        }
        ts.forEachChild(node, (child) => visit(child, parentClass)); // Recursive traversal
    }

    visit(sourceFile);

    return { functions, classes };
}

const structure = extractStructure(fileContent, filePath);
console.log(JSON.stringify(structure, null, 2));