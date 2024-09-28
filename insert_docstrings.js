// insert_docstrings.js
const ts = require("typescript");

// Read docstrings from stdin
const docstrings = JSON.parse(process.stdin.read());
const sourceCode = docstrings.source_code; // Get original source code

// Function to insert docstrings into the source code
function insertDocstrings(code, docstrings) {
    let newCode = code;
    const inserts = [];

    // Process functions and classes
    for (const itemType of ["functions", "classes"]) {
        for (const item of docstrings[itemType] || []) {
            if (item.docstring) {
                const insertPosition = item.start;
                const formattedDocstring = formatJSDoc(item.docstring);
                inserts.push({ position: insertPosition, content: formattedDocstring });

                // Process methods for classes
                if (itemType === "classes") {
                    for (const method of item.methods || []) {
                        if (method.docstring) {
                            const methodInsertPosition = method.start;
                            const formattedMethodDocstring = formatJSDoc(method.docstring);
                            inserts.push({ position: methodInsertPosition, content: formattedMethodDocstring });
                        }
                    }
                }
            }
        }
    }

    // Sort inserts in reverse order to avoid position conflicts
    inserts.sort((a, b) => b.position - a.position);

    // Insert docstrings into the code
    for (const insert of inserts) {
        newCode = newCode.slice(0, insert.position) + insert.content + "\n" + newCode.slice(insert.position);
    }

    return newCode;
}


function formatJSDoc(docstring) {
    const lines = docstring.split("\n");
    const formattedLines = lines.map((line) => ` * ${line}`);
    return `/**\n${formattedLines.join("\n")}\n */`;
}


const updatedCode = insertDocstrings(sourceCode, docstrings);
console.log(updatedCode); // Print the updated code to stdout