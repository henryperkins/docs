// Sample JavaScript code adhering to the schema

// Function example
async function sampleFunction(arg1, arg2) {
    /**
     * This is a sample function.
     * @param {string} arg1 - The first argument.
     * @param {string} arg2 - The second argument.
     */
    console.log(arg1, arg2);
}

// Class example
class SampleClass {
    /**
     * This is a sample class.
     */
    constructor() {
        this.name = 'Sample';
    }

    // Method example
    async sampleMethod(arg1) {
        /**
         * This is a sample method.
         * @param {string} arg1 - The argument.
         */
        console.log(arg1);
    }
}

// Exporting for testing purposes
module.exports = { sampleFunction, SampleClass };