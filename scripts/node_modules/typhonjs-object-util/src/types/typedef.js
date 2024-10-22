/**
 * @typedef {object} ValidationEntry - Provides data for a validation check.
 *
 * @property {string}               [type] - Optionally tests with a typeof check.
 *
 * @property {Array<*>|function|Set<*>}  [expected] - Optional array, function, or set of expected values to test against.
 *
 * @property {string}               [message] - Optional message to include.
 *
 * @property {boolean}              [required=true] - When false if the accessor is missing validation is skipped.
 *
 * @property {boolean}              [error=true] - When true and error is thrown otherwise a boolean is returned.
 */
