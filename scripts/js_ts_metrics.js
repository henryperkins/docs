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
            // Decode the Halstead properties
            const decodedHalstead = {
                operators: method.halstead.operators.map(op => op.toString()),
                operands: method.halstead.operands.map(op => op.toString()),
                length: method.halstead.length,
                vocabulary: method.halstead.vocabulary,
                difficulty: method.halstead.difficulty,
                volume: method.halstead.volume,
                effort: method.halstead.effort,
                bugs: method.halstead.bugs,
                time: method.halstead.time
            };

            acc[method.name] = {
                complexity: method.cyclomatic,
                sloc: method.sloc,
                params: method.params,
                halstead: decodedHalstead // Assign the decoded object
            };
            return acc;
        }, {}); // Initialize accumulator as an empty object

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
            functions: {} // Initialize functions as an empty object
        };
        console.log(JSON.stringify(defaultMetrics));
    }
});