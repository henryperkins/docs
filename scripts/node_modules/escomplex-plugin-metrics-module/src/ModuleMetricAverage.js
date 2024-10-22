import ObjectUtil    from 'typhonjs-escomplex-commons/src/utils/ObjectUtil';

/**
 * Provides a typhonjs-escomplex-module / ESComplexModule plugin which gathers and calculates all default metrics.
 *
 * @see https://www.npmjs.com/package/typhonjs-escomplex-commons
 * @see https://www.npmjs.com/package/typhonjs-escomplex-module
 */
export default class ModuleMetricAverage
{
   /**
    * Coordinates calculating all metrics. All module and class methods are traversed. If there are no module or class
    * methods respectively the aggregate MethodReport is used for calculations.
    *
    * @param {ModuleReport}   moduleReport - The ModuleReport being processed.
    * @param {object}         settings - Settings for module processing.
    */
   static calculate(moduleReport)
   {
      let moduleMethodCount = moduleReport.methods.length;
      let moduleAggregateDivisor = moduleMethodCount + 1; // Include the module as a potential control path.

      // Handle module methods.
      moduleReport.methods.forEach((methodReport) =>
      {
         ObjectUtil.safeBatchSet(moduleReport.methodAverage, moduleReport.methodAverage.keys, methodReport, 'add');
      });

      // Handle module class reports.
      moduleReport.classes.forEach((classReport) =>
      {
         const classMethodCount = classReport.methods.length;
         const classAggregateDivisor = classMethodCount + 1;

         moduleMethodCount += classMethodCount;
         moduleAggregateDivisor += classMethodCount;  // Soon to add + 1 for the Class aggregate

         // Process all class methods.
         classReport.methods.forEach((methodReport) =>
         {
            ObjectUtil.safeBatchSet(moduleReport.methodAverage, classReport.methodAverage.keys, methodReport, 'add');
            ObjectUtil.safeBatchSet(classReport.methodAverage, classReport.methodAverage.keys, methodReport, 'add');
         });

         // Calculate the pure average method data only if there are class methods.
         if (classMethodCount !== 0)
         {
            ObjectUtil.safeBatchSet(classReport.methodAverage, classReport.methodAverage.keys,
             classMethodCount, 'div');
         }

         // Calculate average class aggregate method data by adding the aggregate & dividing by the class divisor.
         ObjectUtil.safeBatchSet(classReport.aggregateAverage, classReport.aggregateAverage.keys,
          classReport.aggregate, 'add');

         ObjectUtil.safeBatchSet(classReport.aggregateAverage, classReport.aggregateAverage.keys,
          classAggregateDivisor, 'div');
      });

      // Calculate the pure average method data only if there are module methods.
      if (moduleMethodCount !== 0)
      {
         ObjectUtil.safeBatchSet(moduleReport.methodAverage, moduleReport.methodAverage.keys, moduleMethodCount, 'div');
      }

      // Calculate average module aggregate method data by adding the aggregate & dividing by the module divisor.
      ObjectUtil.safeBatchSet(moduleReport.aggregateAverage, moduleReport.aggregateAverage.keys,
       moduleReport.aggregate, 'add');

      ObjectUtil.safeBatchSet(moduleReport.aggregateAverage, moduleReport.aggregateAverage.keys,
       moduleAggregateDivisor, 'div');
   }
}
