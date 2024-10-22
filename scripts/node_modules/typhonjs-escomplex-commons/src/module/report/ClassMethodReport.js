import MethodReport     from './MethodReport';

import ReportType       from '../../types/ReportType';
import TransformFormat  from '../../transform/TransformFormat';

/**
 * Provides the class method report object which stores data pertaining to a single method / function.
 */
export default class ClassMethodReport extends MethodReport
{
   /**
    * Initializes class module method report.
    *
    * @param {string}   name - Name of the method.
    * @param {number}   paramNames - Array of any associated parameter names.
    * @param {number}   lineStart - Start line of method.
    * @param {number}   lineEnd - End line of method.
    */
   constructor(name, paramNames, lineStart, lineEnd)
   {
      super(name, paramNames, lineStart, lineEnd);

      /**
       * Stores the max nested method depth.
       * @type {number}
       */
      this.maxNestedMethodDepth = 0;

      /**
       * Stores all nested method data.
       * @type {Array<NestedMethodReport>}
       */
      this.nestedMethods = [];
   }

   /**
    * Returns the enum for the report type.
    * @returns {CLASS_METHOD}
    */
   get type() { return ReportType.CLASS_METHOD; }

   /**
    * Returns the supported transform formats.
    *
    * @returns {Object[]}
    */
   static getFormats()
   {
      return TransformFormat.getFormats(ReportType.CLASS_METHOD);
   }

   /**
    * Deserializes a JSON object representing a ClassMethodReport.
    *
    * @param {object}   object - A JSON object of a ClassMethodReport that was previously serialized.
    *
    * @returns {ClassMethodReport}
    */
   static parse(object) { return this._parse(new ClassMethodReport(), object); }
}
