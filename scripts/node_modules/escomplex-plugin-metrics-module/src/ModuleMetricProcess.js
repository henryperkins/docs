import HalsteadArray from 'typhonjs-escomplex-commons/src/module/traits/HalsteadArray';

/**
 * Provides the main processing of syntax data for all default metrics gathered.
 */
export default class ModuleMetricProcess
{
   /**
    * Potentially adds given dependencies for tracking.
    *
    * @param {ModuleReport}         moduleReport - The ModuleReport being processed.
    * @param {object|Array<object>} dependencies - Dependencies to add.
    */
   static addDependencies(moduleReport, dependencies)
   {
      if (typeof dependencies === 'object' || Array.isArray(dependencies))
      {
         moduleReport.dependencies = moduleReport.dependencies.concat(dependencies);
      }
   }

   /**
    * Creates a moduleReport scope when a class or method is entered.
    *
    * @param {ModuleReport}         moduleReport - The ModuleReport being processed.
    * @param {ModuleScopeControl}   scopeControl - The associated module report scope control.
    * @param {object}               newScope - An object hash defining the new scope including:
    * @param {object}               node - Current AST node.
    * @param {object}               parent - Parent AST node.
    * ```
    * (string) type - Type of report scope being created.
    * ```
    */
   static preScopeCreated(moduleReport, scopeControl, newScope = {}, node, parent)
   {
      if (typeof newScope !== 'object')
      {
         throw new TypeError(`preScopeCreated error: 'newScope' is not an 'object'.`);
      }

      if (typeof newScope.type !== 'string')
      {
         throw new TypeError(`preScopeCreated error: 'newScope.type' is not a 'string'.`);
      }

      switch (newScope.type)
      {
         case 'class':
         case 'method':

            // Increments logical SLOC for previous report scopes.
            if (Number.isInteger(newScope.lloc))
            {
               // Increments current module report associated aggregate report parameter count.
               moduleReport.aggregateReport.sloc.logical += newScope.lloc;

               const classReport = scopeControl.getCurrentClassReport();
               const methodReport = scopeControl.getCurrentMethodReport();

               if (classReport) { classReport.aggregateReport.sloc.logical += newScope.lloc; }
               if (methodReport) { methodReport.sloc.logical += newScope.lloc; }
            }

            if (newScope.operands instanceof HalsteadArray)
            {
               const identifiers = newScope.operands.valueOf(node, parent);

               identifiers.forEach((identifier) =>
               {
                  ModuleMetricProcess.halsteadItemEncountered(moduleReport, scopeControl, 'operands', identifier);
               });
            }

            if (newScope.operators instanceof HalsteadArray)
            {
               const identifiers = newScope.operators.valueOf(node, parent);

               identifiers.forEach((identifier) =>
               {
                  ModuleMetricProcess.halsteadItemEncountered(moduleReport, scopeControl, 'operators', identifier);
               });
            }
            break;
      }
   }

   /**
    * Creates a moduleReport scope when a class or method is entered.
    *
    * @param {ModuleReport}         moduleReport - The ModuleReport being processed.
    * @param {ModuleScopeControl}   scopeControl - The associated module report scope control.
    * @param {object}               newScope - An object hash defining the new scope including:
    * ```
    * (string) type - Type of report scope being created.
    * (string) name - Name of the class or method.
    * (number) lineStart - Start line of method.
    * (number) lineEnd - End line of method.
    * (Array<string>) paramNames - (For method scopes) An array of parameters names for method.
    * ```
    */
   static postScopeCreated(moduleReport, scopeControl, newScope = {})
   {
      if (typeof newScope !== 'object')
      {
         throw new TypeError(`postScopeCreated error: 'newScope' is not an 'object'.`);
      }

      if (typeof newScope.type !== 'string')
      {
         throw new TypeError(`postScopeCreated error: 'newScope.type' is not a 'string'.`);
      }

      switch (newScope.type)
      {
         case 'class':
            break;

         case 'method':
         {
            if (!Number.isInteger(newScope.cyclomatic))
            {
               throw new TypeError(`postScopeCreated error: 'newScope.cyclomatic' is not an 'integer'.`);
            }

            if (!Array.isArray(newScope.paramNames))
            {
               throw new TypeError(`postScopeCreated error: 'newScope.paramNames' is not an 'array'.`);
            }

            const classReport = scopeControl.getCurrentClassReport();
            const methodReport = scopeControl.getCurrentMethodReport();

            // Increments current module report associated aggregate report cyclomatic count.
            moduleReport.aggregateReport.cyclomatic += newScope.cyclomatic;

            // Increments current module report associated aggregate report parameter count.
            moduleReport.aggregateReport.paramCount += newScope.paramNames.length;

            if (classReport)
            {
               // Increments current class report associated aggregate report cyclomatic count.
               classReport.aggregateReport.cyclomatic += newScope.cyclomatic;

               // Increments current class report associated aggregate report parameter count.
               classReport.aggregateReport.paramCount += newScope.paramNames.length;
            }

            if (Number.isInteger(newScope.postLloc))
            {
               moduleReport.aggregateReport.sloc.logical += newScope.postLloc;
               if (classReport) { classReport.aggregateReport.sloc.logical += newScope.postLloc; }
               if (methodReport) { methodReport.sloc.logical += newScope.postLloc; }
            }
            break;
         }
      }
   }

   /**
    * Increments the Halstead `metric` for the given `identifier` for the ModuleReport and any current class or method
    * report being tracked.
    *
    * @param {ModuleReport}         moduleReport - The ModuleReport being processed.
    * @param {ModuleScopeControl}   scopeControl - The associated module report scope control.
    * @param {string}               metric - A Halstead metric name.
    * @param {string}               identifier - A Halstead identifier name.
    */
   static halsteadItemEncountered(moduleReport, scopeControl, metric, identifier)
   {
      const currentClassReport = scopeControl.getCurrentClassReport();
      const currentMethodReport = scopeControl.getCurrentMethodReport();

      ModuleMetricProcess.incrementHalsteadItems(moduleReport, metric, identifier);

      if (currentClassReport) { ModuleMetricProcess.incrementHalsteadItems(currentClassReport, metric, identifier); }

      if (currentMethodReport) { ModuleMetricProcess.incrementHalsteadItems(currentMethodReport, metric, identifier); }
   }

   /**
    * Increments the cyclomatic metric for the ModuleReport and any current class or method report being tracked.
    *
    * @param {ModuleReport}         moduleReport - The ModuleReport being processed.
    * @param {ModuleScopeControl}   scopeControl - The associated module report scope control.
    * @param {number}   amount - Amount to increment.
    */
   static incrementCyclomatic(moduleReport, scopeControl, amount)
   {
      const currentClassReport = scopeControl.getCurrentClassReport();
      const currentMethodReport = scopeControl.getCurrentMethodReport();

      moduleReport.aggregate.cyclomatic += amount;

      if (currentClassReport) { currentClassReport.aggregate.cyclomatic += amount; }
      if (currentMethodReport) { currentMethodReport.cyclomatic += amount; }
   }

   /**
    * Increments the logical SLOC (source lines of code) metric for the ModuleReport and any current class or method
    * report being tracked.
    *
    * @param {ModuleReport}         moduleReport - The ModuleReport being processed.
    * @param {ModuleScopeControl}   scopeControl - The associated module report scope control.
    * @param {number}   amount - Amount to increment.
    */
   static incrementLogicalSloc(moduleReport, scopeControl, amount)
   {
      const currentClassReport = scopeControl.getCurrentClassReport();
      const currentMethodReport = scopeControl.getCurrentMethodReport();

      moduleReport.aggregate.sloc.logical += amount;

      if (currentClassReport) { currentClassReport.aggregate.sloc.logical += amount; }
      if (currentMethodReport)
      {
// if (amount > 0)
// {
//   console.log('!! MMP - incrementLogicalSloc (method) - node type: ' + nodeType + '; amount: ' + amount);
// }

         currentMethodReport.sloc.logical += amount;
      }
   }

   /**
    * Increments the associated aggregate report Halstead items including distinct and total counts.
    *
    * @param {ModuleReport|ClassReport|MethodReport}  report - The report being processed.
    * @param {string}                                 metric - A Halstead metric name.
    * @param {string}                                 identifier - A Halstead identifier name.
    */
   static incrementHalsteadItems(report, metric, identifier)
   {
      // Increments the associated aggregate report HalsteadData for distinct identifiers.
      if (report.aggregateReport.halstead[metric].identifiers.indexOf(identifier) === -1)
      {
         report.aggregateReport.halstead[metric].identifiers.push(identifier);
         report.aggregateReport.halstead[metric]['distinct'] += 1;
      }

      // Increment total halstead items
      report.aggregateReport.halstead[metric]['total'] += 1;
   }

   /**
    * Processes all TraitHalstead identifier data.
    *
    * @param {ModuleReport}         moduleReport - The ModuleReport being processed.
    * @param {ModuleScopeControl}   scopeControl - The associated module report scope control.
    * @param {object}               syntax - The associated syntax being processed for current node.
    * @param {object}               node - The node being entered.
    * @param {object}               parent - The parent node of the node being entered.
    */
   static processSyntax(moduleReport, scopeControl, syntax, node, parent)
   {
      for (const key in syntax)
      {
         const trait = syntax[key];

         switch (trait.metric)
         {
            case 'cyclomatic':
               ModuleMetricProcess.incrementCyclomatic(moduleReport, scopeControl, trait.valueOf(node, parent));
               break;

            case 'dependencies':
               ModuleMetricProcess.addDependencies(moduleReport, trait.valueOf(node, parent));
               break;

            case 'lloc':
// if (trait.valueOf(node, parent) > 0)
// {
//   console.log('!! MMP - increment sloc - node type: ' + node.type + '; value: ' + trait.valueOf(node, parent));
// }
               ModuleMetricProcess.incrementLogicalSloc(moduleReport, scopeControl, trait.valueOf(node, parent));
               break;
         }

         // Process operands / operators HalsteadArray entries.
         if (trait instanceof HalsteadArray)
         {
            const identifiers = trait.valueOf(node, parent);

            identifiers.forEach((identifier) =>
            {
               ModuleMetricProcess.halsteadItemEncountered(moduleReport, scopeControl, trait.metric, identifier);
            });
         }
      }
   }
}
