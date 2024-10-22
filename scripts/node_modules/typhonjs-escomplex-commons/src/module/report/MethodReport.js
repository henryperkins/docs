import AggregateReport        from './AggregateReport';
import AnalyzeError           from '../../analyze/AnalyzeError';

// import NestedMethodReport     from './NestedMethodReport';

import ObjectUtil             from '../../utils/ObjectUtil';

/**
 * Provides the method report object which stores data pertaining to a single method / function.
 */
export default class MethodReport extends AggregateReport
{
   /**
    * Initializes method report.
    *
    * @param {string}   name - Name of the method.
    * @param {number}   paramNames - Array of any associated parameter names.
    * @param {number}   lineStart - Start line of method.
    * @param {number}   lineEnd - End line of method.
    */
   constructor(name = void 0, paramNames = [], lineStart = 0, lineEnd = 0)
   {
      super(lineStart, lineEnd);

      /**
       * Stores any analysis errors.
       * @type {Array}
       */
      this.errors = [];

      /**
       * Stores the end line for the method.
       * @type {number}
       */
      this.lineEnd = lineEnd;

      /**
       * Stores the start line for the method.
       * @type {number}
       */
      this.lineStart = lineStart;

      /**
       * The name of the method.
       * @type {string}
       */
      this.name = name;

      /**
       * Stores any parameter names.
       * @type {Array<string>}
       */
      this.paramNames = paramNames;

      /**
       * The number of parameters for the method.
       * @type {number}
       */
      this.paramCount = paramNames.length;
   }

   /**
    * Clears all errors stored in the method report.
    *
    * @param {boolean}  clearChildren - (Optional) If false then nested method errors are not cleared; default (true).
    */
   clearErrors(clearChildren = true)
   {
      this.errors = [];

      if (clearChildren && Array.isArray(this.nestedMethods))
      {
         this.nestedMethods.forEach((report) => { report.clearErrors(); });
      }
   }

   // TODO: Remove? Old implementation
   // /**
   //  * Gets all errors stored in the method report.
   //  *
   //  * @param {object}   options - Optional parameters.
   //  * @property {boolean}  includeChildren - If false then module errors are not included; default (true).
   //  * @property {boolean}  includeReports - If true then results will be an array of object hashes containing `source`
   //  *                                      (the source report object of the error) and `error`
   //  *                                      (an AnalyzeError instance) keys; default (false).
   //  *
   //  * @returns {Array<AnalyzeError|{error: AnalyzeError, source: *}>}
   //  */
   // getErrors(options = { includeChildren: true, includeReports: false })
   // {
   //   /* istanbul ignore if */
   //   if (typeof options !== 'object') { throw new TypeError(`getErrors error: 'options' is not an 'object'.`); }
   //
   //   // By default set includeChildren to true.
   //   /* istanbul ignore if */
   //   if (typeof options.includeChildren !== 'boolean') { options.includeChildren = true; }
   //
   //   // If `includeReports` is true then return an object hash with the source and error otherwise return the error.
   //   return options.includeReports ? this.errors.map((entry) => { return { error: entry, source: this }; }) :
   //    [].concat(...this.errors);
   // }

   /**
    * Gets all errors stored in the method report and by default any nested methods.
    *
    * @param {object}   options - Optional parameters.
    * @property {boolean}  includeChildren - If false then module errors are not included; default (true).
    * @property {boolean}  includeReports - If true then results will be an array of object hashes containing `source`
    *                                      (the source report object of the error) and `error`
    *                                      (an AnalyzeError instance) keys; default (false).
    *
    * @returns {Array<AnalyzeError|{error: AnalyzeError, source: *}>}
    */
   getErrors(options = { includeChildren: true, includeReports: false })
   {
      /* istanbul ignore if */
      if (typeof options !== 'object') { throw new TypeError(`getErrors error: 'options' is not an 'object'.`); }

      // By default set includeChildren to true.
      /* istanbul ignore if */
      if (typeof options.includeChildren !== 'boolean') { options.includeChildren = true; }

      // If `includeReports` is true then return an object hash with the source and error otherwise return the error.
      let errors = options.includeReports ? this.errors.map((entry) => { return { error: entry, source: this }; }) :
       [].concat(...this.errors);

      // If `includeChildren` is true then traverse all children reports for errors.
      if (options.includeChildren && Array.isArray(this.nestedMethods))
      {
         // Add class to all children errors.
         if (options.includeReports)
         {
            const childErrors = [];

            this.nestedMethods.forEach((report) => { childErrors.push(...report.getErrors(options)); });

            // Add module to object hash.
            childErrors.forEach((error) => { error.method = this; });

            // Push to all module errors.
            errors.push(...childErrors);
         }
         else
         {
            this.nestedMethods.forEach((report) => { errors.push(...report.getErrors(options)); });
         }
      }

      // If `options.query` is defined then filter errors against the query object.
      if (typeof options.query === 'object')
      {
         errors = errors.filter((error) => ObjectUtil.safeEqual(options.query, error));
      }

      return errors;
   }

   /**
    * Returns the name / id associated with this report.
    * @returns {string}
    */
   getName()
   {
      return this.name;
   }

   /**
    * Deserializes a JSON object representing a ClassMethodReport.
    *
    * @param {ClassMethodReport|ModuleMethodReport}   targetObject - A target object to hydrate.
    *
    * @param {object}   jsonObject - A JSON object of a class or module method report that was previously serialized.
    *
    * @returns {ClassMethodReport|ModuleMethodReport}
    * @protected
    */
   static _parse(targetObject, jsonObject)
   {
      /* istanbul ignore if */
      if (typeof jsonObject !== 'object') { throw new TypeError(`parse error: 'jsonObject' is not an 'object'.`); }

      const methodReport = Object.assign(targetObject, jsonObject);

      if (methodReport.errors.length > 0)
      {
         methodReport.errors = methodReport.errors.map((error) => AnalyzeError.parse(error));
      }

      // TODO unimplemented yet!
      // if (methodReport.nestedMethods.length > 0)
      // {
      //    methodReport.nestedMethods = methodReport.nestedMethods.map((method) => NestedMethodReport.parse(method));
      // }

      return methodReport;
   }
}
