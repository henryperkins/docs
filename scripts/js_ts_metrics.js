// js_ts_metrics.js

const fs = require('fs');
const escomplex = require('typhonjs-escomplex');

function calculateMetrics(code, language) {
    const isTypeScript = language === 'typescript';

    // Analyze the code
    const analysis = escomplex.analyzeModule(code, {
        esmImportExport: true,
        jsx: true,
        loc: true,
        newmi: true,
        skipCalculation: false,
        ignoreErrors: false,
        useTypeScriptEstree: isTypeScript,
    });

    // Extract relevant metrics
    const metrics = {
        aggregate: analysis.aggregate,
        functions: analysis.functions.map(func => ({
            name: func.name,
            cyclomatic: func.cyclomatic,
            halstead: func.halstead,
            paramCount: func.params,
            lineStart: func.lineStart,
            lineEnd: func.lineEnd,
        })),
        maintainability: analysis.maintainability,
    };

    return metrics;
}

function main() {
    const input = fs.readFileSync(0, 'utf-8');
    const data = JSON.parse(input);
    const code = data.code;
    const language = data.language || 'javascript';
    const metrics = calculateMetrics(code, language);
    console.log(JSON.stringify(metrics));
}

main();