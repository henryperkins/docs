/**
 * Provides a typhonjs-escomplex-module / ESComplexModule plugin which gathers and calculates all default metrics.
 *
 * @see https://www.npmjs.com/package/typhonjs-escomplex-commons
 * @see https://www.npmjs.com/package/typhonjs-escomplex-module
 */
export default class ModuleMetricPostAverage
{
   /**
    * Coordinates calculating all metrics. All module and class methods are traversed. If there are no module or class
    * methods respectively the aggregate MethodReport is used for calculations.
    *
    * @param {ModuleReport}   moduleReport - The ModuleReport being processed.
    * @param {object}         settings - Settings for module processing.
    *
    * @private
    */
   static calculate(moduleReport, settings)
   {
      // Handle module report.
      const moduleAggregateAverage = moduleReport.aggregateAverage;

      ModuleMetricPostAverage.calculateMaintainabilityIndex(moduleReport, settings, moduleAggregateAverage.cyclomatic,
       moduleAggregateAverage.halstead.effort, moduleAggregateAverage.sloc.logical);

      // Handle module class reports.
      moduleReport.classes.forEach((classReport) =>
      {
         // const classAggregateAverage = classReport.aggregateAverage;
         //
         // ModuleMetricPostAverage.calculateMaintainabilityIndex(classReport, settings, classAggregateAverage.cyclomatic,
         // classAggregateAverage.halstead.effort, classAggregateAverage.sloc.logical);

         const classMethodAverages = classReport.methodAverage;

         ModuleMetricPostAverage.calculateMaintainabilityIndex(classReport, settings, classMethodAverages.cyclomatic,
          classMethodAverages.halstead.effort, classMethodAverages.sloc.logical);
      });
   }

   /**
    * Designed in 1991 by Paul Oman and Jack Hagemeister at the University of Idaho, this metric is calculated at the
    * whole program or module level from averages of the other 3 metrics, using the following formula:
    * ```
    * 171 -
    * (3.42 * ln(mean effort)) -
    * (0.23 * ln(mean cyclomatic complexity)) -
    * (16.2 * ln(mean logical LOC))
    * ```
    * Values are on a logarithmic scale ranging from negative infinity up to 171, with greater numbers indicating a
    * higher level of maintainability. In their original paper, Oman and Hagemeister identified 65 as the threshold
    * value below which a program should be considered difficult to maintain.
    *
    * Note: Bare module reports with no branching control flow or module methods and class reports with no class
    * methods start with a base cyclomatic complexity of 0. It is necessary to handle this case as
    * Math.log(0) === -Infinity.
    *
    * @param {ClassReport|ModuleReport}   report - A ClassReport or ModuleReport to perform calculations on.
    * @param {object}               settings - Settings for module processing.
    * @param {number}               averageCyclomatic - Average cyclomatic metric across a ClassReport / ModuleReport.
    * @param {number}               averageEffort - Average Halstead effort across a ClassReport / ModuleReport.
    * @param {number}               averageLoc - Average logical SLOC across a ClassReport / ModuleReport.
    *
    * @private
    */
   static calculateMaintainabilityIndex(report, settings, averageCyclomatic, averageEffort, averageLoc)
   {
      report.maintainability =
       171
       - (3.42 * Math.log(averageEffort))
       - (0.23 * averageCyclomatic === 0 ? 0 : Math.log(averageCyclomatic))
       - (16.2 * Math.log(averageLoc));

      /* istanbul ignore if */
      if (report.maintainability > 171) { report.maintainability = 171; }

      /* istanbul ignore if */
      if (settings.newmi) { report.maintainability = Math.max(0, (report.maintainability * 100) / 171); }
   }
}
