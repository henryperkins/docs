import TransformFormat  from '../../transform/TransformFormat';

/**
 * Provides several helper methods to work with method oriented data stored as `this.aggregate` in `ClassReport` /
 * `ModuleReport` and directly in `ClassMethodReport` / `ModuleMethodReport`.
 */
export default class AbstractReport
{
   /**
    * If given assigns the method report to an internal variable. This is used by `ClassReport` and `ModuleReport`
    * which stores a `AggregateReport` respectively in `this.aggregate`.
    *
    * @param {AggregateReport}   aggregateReport - An AggregateReport to associate with this report.
    */
   constructor(aggregateReport = void 0)
   {
      /**
       * Stores any associated `AggregateReport`.
       * @type {AggregateReport}
       */
      this.aggregate = aggregateReport;
   }

   /**
    * Returns the associated `AggregateReport` or `this`. Both ClassReport and ModuleReport have an
    * `aggregate` AggregateReport.
    *
    * @returns {AggregateReport}
    */
   get aggregateReport() { return typeof this.aggregate !== 'undefined' ? this.aggregate : this; }

   /**
    * Formats this report given the type.
    *
    * @param {string}   name - The name of formatter to use.
    *
    * @param {object}   options - (Optional) One or more optional parameters to pass to the formatter.
    *
    * @returns {string}
    */
   toFormat(name, options = void 0)
   {
      return TransformFormat.format(this, name, options);
   }
}
