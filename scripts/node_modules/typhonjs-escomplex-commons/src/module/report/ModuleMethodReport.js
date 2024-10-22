import MethodReport     from './MethodReport';

import ReportType       from '../../types/ReportType';
import TransformFormat  from '../../transform/TransformFormat';

/**
 * Provides the module method report object which stores data pertaining to a single method / function.
 */
export default class ModuleMethodReport extends MethodReport
{
   /**
    * Initializes module method report.
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
    * @returns {ReportType}
    */
   get type() { return ReportType.MODULE_METHOD; }

   /**
    * Returns the supported transform formats.
    *
    * @returns {Object[]}
    */
   static getFormats()
   {
      return TransformFormat.getFormats(ReportType.MODULE_METHOD);
   }

   /**
    * Deserializes a JSON object representing a ModuleMethodReport.
    *
    * @param {object}   object - A JSON object of a ModuleMethodReport that was previously serialized.
    *
    * @returns {ModuleMethodReport}
    */
   static parse(object) { return this._parse(new ModuleMethodReport(), object); }
}
