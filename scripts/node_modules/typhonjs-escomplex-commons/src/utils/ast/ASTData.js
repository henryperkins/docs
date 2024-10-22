/**
 * Defines the output data from parsing an AST tree.
 */
export default class ASTData
{
   /**
    * Initializes ASTData
    */
   constructor()
   {
      this.source = '';
      this.operands = [];
      this.operators = [];
   }

   /**
    * Appends a string.
    *
    * @param {string} string - A string to append.
    */
   write(string)
   {
      this.source += string;
   }

   /**
    * Convert to string
    *
    * @returns {string|*|string}
    */
   toString()
   {
      return this.source;
   }
}