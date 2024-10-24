const escomplex = require('typhonjs-escomplex');

let inputData = '';

process.stdin.on('data', (chunk) => {
    inputData += chunk;
});

process.stdin.on('end', () => {
    try {
        const input = JSON.parse(inputData);
        const code = input.code;
        const options = input.options;

        const analysis = escomplex.analyzeModule(code, options);

        const halstead = analysis.aggregate.halstead;
        const functionsMetrics = analysis.functions.reduce((acc, method) => {
            acc[method.name] = {
                complexity: method.cyclomatic,
                sloc: method.sloc,
                params: method.params,
                halstead: method.halstead
            };
            return acc;
        }, {});

        const result = {
            complexity: analysis.aggregate.cyclomatic,
            maintainability: analysis.maintainability,
            halstead: {
                volume: halstead.volume,
                difficulty: halstead.difficulty,
                effort: halstead.effort
            },
            functions: functionsMetrics
        };

        console.log(JSON.stringify(result));

    } catch (error) {
        console.error(`Metrics calculation error: ${error.message}`);
        const defaultMetrics = {
            complexity: 0,
            maintainability: 0,
            halstead: { volume: 0, difficulty: 0, effort: 0 },
            functions: {}
        };
        console.log(JSON.stringify(defaultMetrics));
    }
});
