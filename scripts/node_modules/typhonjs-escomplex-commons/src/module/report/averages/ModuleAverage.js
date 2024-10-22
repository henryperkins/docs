import MethodAverage from './MethodAverage';

import ObjectUtil    from '../../../utils/ObjectUtil';

/**
 * Provides all the averaged module metric data.
 */
export default class ModuleAverage
{
   /**
    * Initializes the default averaged module data.
    */
   constructor()
   {
      this.methodAverage = new MethodAverage();

      /**
       * Measures the average method maintainability index for the module / file.
       * @type {number}
       */
      this.maintainability = 0;
   }

   /**
    * Returns the object accessor list / keys for ModuleAverage.
    *
    * @returns {Array<string>}
    */
   get keys()
   {
      return s_AVERAGE_KEYS;
   }
}

/**
 * Defines the default module average accessor list / keys.
 * @type {Array<string>}
 * @ignore
 */
const s_AVERAGE_KEYS = ObjectUtil.getAccessorList(new ModuleAverage());
