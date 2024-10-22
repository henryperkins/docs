import AbstractReport   from './AbstractReport';
import HalsteadData     from './HalsteadData';

/**
 * Provides the aggregate report object which stores base data pertaining to a single method / function or cumulative
 * aggregate data for a ModuleReport / ClassReport.
 */
export default class AggregateReport extends AbstractReport
{
   /**
    * Initializes aggregate report.
    *
    * @param {number}   lineStart - Start line of method.
    * @param {number}   lineEnd - End line of method.
    * @param {number}   baseCyclomatic - The initial base cyclomatic complexity of the report. Module and class reports
    *                                    start at 0.
    */
   constructor(lineStart = 0, lineEnd = 0, baseCyclomatic = 1)
   {
      super();

      /**
       * The cyclomatic complexity of the method / report.
       * @type {number}
       */
      this.cyclomatic = baseCyclomatic;

      /**
       * The cyclomatic density of the method.
       * @type {number}
       */
      this.cyclomaticDensity = 0;

      /**
       * Stores the Halstead data instance.
       * @type {HalsteadData}
       */
      this.halstead = new HalsteadData();

      /**
       * The number of parameters for the method or aggregate report.
       * @type {number}
       */
      this.paramCount = 0;

      /**
       * The source lines of code for the method.
       * @type {{logical: number, physical: number}}
       */
      this.sloc = { logical: 0, physical: lineEnd - lineStart + 1 };
   }
}
