import Enum from '../utils/Enum';

/**
 * Defines ReportType enum.
 */
export default class ReportType extends Enum {}

ReportType.initEnum(
{
   CLASS: { description: 'Class' },
   CLASS_METHOD: { description: 'Class Method' },
   MODULE_METHOD: { description: 'Module Method' },
   MODULE: { description: 'Module' },
   NESTED_METHOD: { description: 'Nested Method' },
   PROJECT: { description: 'Project' }
});
