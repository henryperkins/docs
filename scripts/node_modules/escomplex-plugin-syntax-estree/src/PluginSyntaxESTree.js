import AbstractSyntaxLoader   from 'typhonjs-escomplex-commons/src/module/plugin/syntax/AbstractSyntaxLoader';

import ASTGenerator           from 'typhonjs-escomplex-commons/src/utils/ast/ASTGenerator';

import HalsteadArray          from 'typhonjs-escomplex-commons/src/module/traits/HalsteadArray';
import TraitUtil              from 'typhonjs-escomplex-commons/src/module/traits/TraitUtil';

import actualize              from 'typhonjs-escomplex-commons/src/module/traits/actualize';

/**
 * Provides an typhonjs-escomplex-module / ESComplexModule plugin which loads syntax definitions for trait resolution
 * for all ESTree AST nodes up to and including ES6.
 *
 * @see https://www.npmjs.com/package/typhonjs-escomplex-module
 */
export default class PluginSyntaxESTree extends AbstractSyntaxLoader
{
   // ESComplexModule plugin callbacks ------------------------------------------------------------------------------

   /**
    * Loads any default settings that are not already provided by any user options.
    *
    * @param {object}   ev - escomplex plugin event data.
    *
    * The following options are:
    * ```
    * (boolean)   commonjs - Boolean indicating that source code being processed is CommonJS; defaults to false.
    *
    * (function)  dependencyResolver - Provides a function to resolve dynamic dependencies; defaults to undefined.
    *
    * (boolean|object)  esmImportExport - As a boolean indicating whether ES Modules import / export statements
    *                                     contribute to modules logical SLOC & Halstead operators / operands; defaults
    *                                     to false. As an object separate boolean entries to set logical SLOC (lloc)
    *                                     and operators / operands (halstead) are available to set these parameters
    *                                     separately.
    *
    * (boolean)   forin - Boolean indicating whether for...in / for...of loops should be considered a source of
    *                     cyclomatic complexity; defaults to false.
    *
    * (boolean)   logicalor - Boolean indicating whether operator || should be considered a source of cyclomatic
    *                         complexity; defaults to true.
    *
    * (boolean)   switchcase - Boolean indicating whether switch statements should be considered a source of cyclomatic
    *                          complexity; defaults to true.
    *
    * (boolean|object)  templateExpressions - As a boolean indicating whether template literal expressions
    *                                     contribute to logical SLOC & Halstead operators / operands; defaults
    *                                     to `true`. As an object separate boolean entries to set logical SLOC (lloc)
    *                                     and operators / operands (halstead) are available to set these parameters
    *                                     separately. When `lloc` is true tagged template expressions
    *                                     increment lloc by 1. If `halstead` is true then operators are added for '``'
    *                                     and each instance of a `${...}` expression.

    * (boolean)   trycatch - Boolean indicating whether catch clauses should be considered a source of cyclomatic
    *                        complexity; defaults to false.
    * ```
    */
   onConfigure(ev)
   {
      ev.data.settings.commonjs = typeof ev.data.options.commonjs === 'boolean' ? ev.data.options.commonjs : false;

      ev.data.settings.dependencyResolver = typeof ev.data.options.dependencyResolver === 'function' ?
       ev.data.options.dependencyResolver : void 0;

      if (typeof ev.data.options.esmImportExport === 'object')
      {
         ev.data.settings.esmImportExport =
         {
            halstead: typeof ev.data.options.esmImportExport.halstead === 'boolean' ?
             ev.data.options.esmImportExport.halstead : false,

            lloc: typeof ev.data.options.esmImportExport.lloc === 'boolean' ?
               ev.data.options.esmImportExport.lloc : false,
         };
      }
      else
      {
         // Catch all boolean to set `esmImportExport.halstead` & `esmImportExport.lloc` if a boolean is supplied.
         const value = typeof ev.data.options.esmImportExport === 'boolean' ? ev.data.options.esmImportExport : false;
         ev.data.settings.esmImportExport = { halstead: value, lloc: value };
      }

      ev.data.settings.forin = typeof ev.data.options.forin === 'boolean' ? ev.data.options.forin : false;

      ev.data.settings.logicalor = typeof ev.data.options.logicalor === 'boolean' ? ev.data.options.logicalor : true;

      ev.data.settings.switchcase = typeof ev.data.options.switchcase === 'boolean' ? ev.data.options.switchcase : true;

      if (typeof ev.data.options.templateExpression === 'object')
      {
         ev.data.settings.templateExpression =
         {
            halstead: typeof ev.data.options.templateExpression.halstead === 'boolean' ?
             ev.data.options.templateExpression.halstead : true,

            lloc: typeof ev.data.options.templateExpression.lloc === 'boolean' ?
             ev.data.options.templateExpression.lloc : true,
         };
      }
      else
      {
         // Catch all boolean to set `templateExpression.halstead` & `templateExpression.lloc` if a boolean is supplied.
         const value = typeof ev.data.options.templateExpression === 'boolean' ? ev.data.options.templateExpression :
          true;

         ev.data.settings.templateExpression = { halstead: value, lloc: value };
      }

      ev.data.settings.trycatch = typeof ev.data.options.trycatch === 'boolean' ? ev.data.options.trycatch : false;
   }

   // Core / ES5 ESTree AST nodes -----------------------------------------------------------------------------------

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#arrayexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ArrayExpression()
   {
      return actualize(0, 0, (node) =>
      {
         const operators = ['[]'];
         if (Array.isArray(node.elements) && node.elements.length > 0)  // Add length - 1 commas
         {
            operators.push(...(new Array(node.elements.length - 1).fill(',')));
         }
         return operators;
      });
   }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#assignmentexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   AssignmentExpression()
   {
      return actualize((node, parent) =>
      {
         // Handle multiple assignment such as `a = b = c = 1` which resolves logical SLOC of 3.
         // The top most `ExpressionStatement` already provides 1 lloc and `AssignmentExpression` for 'c' & '1' are
         // additionally counted as additional lloc.
         return parent && parent.type !== 'ExpressionStatement' ? 1 : 0;
      },
      0,
      (node) => { return node.operator; });
   }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#blockstatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   BlockStatement() { return actualize(0, 0); }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#binaryexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   BinaryExpression() { return actualize(0, 0, (node) => { return node.operator; }); }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#breakstatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   BreakStatement() { return actualize(1, 0, 'break'); }

   /**
    * ES5 Node
    *
    * Processes CommonJS dependencies if settings.commonjs is set to true. An optional function
    * settings.dependencyResolver may be used to resolve dynamic dependencies.

    * @param {object}   settings - escomplex settings
    *
    * @see https://github.com/estree/estree/blob/master/es5.md#callexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   CallExpression(settings)
   {
      return actualize((node, parent) =>                                           // lloc
      {
         // Handles nested CallExpression nodes. Since ExpressionStatement provides 1 lloc already the first
         // CallExpression ignores posting an additional lloc; subsequent CallExpression nodes do contribute 1 lloc.
         // Likewise the first CallExpression after a YieldExpression doesn't contribute 1 lloc.
         return parent && parent.type !== 'ExpressionStatement' && parent.type !== 'YieldExpression' ? 1 : 0;
      },
      0,                                                                           // cyclomatic
      '()',                                                                        // operators
      void 0,                                                                      // operands
      void 0,                                                                      // ignoreKeys
      void 0,                                                                      // newScope
      (node) =>
      {
         // Only process CJS dependencies if settings.commonjs is true.
         if (settings.commonjs && node.callee.type === 'Identifier' && node.callee.name === 'require' &&
          node.arguments.length === 1)
         {
            const dependency = node.arguments[0];

            let dependencyPath = '* dynamic dependency *';

            if (dependency.type === 'Literal' || dependency.type === 'StringLiteral')
            {
               dependencyPath = typeof settings.dependencyResolver === 'function' ?
                settings.dependencyResolver(dependency.value) : dependency.value;
            }

            return { line: node.loc.start.line, path: dependencyPath, type: 'cjs' };
         }
      });
   }

   /**
    * ES5 Node
    * @param {object}   settings - escomplex settings
    * @see https://github.com/estree/estree/blob/master/es5.md#catchclause
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   CatchClause(settings) { return actualize(1, () => { return settings.trycatch ? 1 : 0; }, 'catch'); }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#conditionalexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ConditionalExpression() { return actualize(0, 1, ':?'); }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#continuestatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ContinueStatement() { return actualize(1, 0, 'continue'); }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#dowhilestatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   DoWhileStatement() { return actualize(2, (node) => { return node.test ? 1 : 0; }, 'dowhile'); }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#emptystatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   EmptyStatement() { return actualize(0, 0); }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#expressionstatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ExpressionStatement()
   {
      return actualize((node) =>
      {
         // Ignore adding 1 lloc when the child expression is an ArrowFunctionExpression which is invalid / no-op.
         return typeof node.expression === 'object' && node.expression.type !== 'ArrowFunctionExpression' ? 1 : 0;
      },
      0);
   }

   /**
    * ES5 Node
    * @param {object}   settings - escomplex settings
    * @see https://github.com/estree/estree/blob/master/es5.md#forinstatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ForInStatement(settings)
   {
      // return actualize(1, () => { return settings.forin ? 1 : 0; }, 'forin');

      return actualize(1, () => { return settings.forin ? 1 : 0; }, (node) =>
      {
         const operators = ['forin'];
         const childNodes = [];

         if (node.left) { childNodes.push(node.left); }
         if (node.right) { childNodes.push(node.right); }

         if (childNodes.length > 0) { operators.push(...ASTGenerator.parseNodes(childNodes).operators); }

         return operators;
      },
      (node) =>
      {
         const operands = [];
         const childNodes = [];

         if (node.left) { childNodes.push(node.left); }
         if (node.right) { childNodes.push(node.right); }

         if (childNodes.length > 0) { operands.push(...ASTGenerator.parseNodes(childNodes).operands); }

         return operands;
      },
      ['left', 'right']);
   }

   /**
    * ES5 Node
    *
    * Note: ASTGenerator is necessary to pick up operators / operands of `init, test, update` child nodes while
    * excluding traversal / LLOC counts for these nodes.
    *
    * @see https://github.com/estree/estree/blob/master/es5.md#forstatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ForStatement()
   {
      return actualize(1, (node) => { return node.test ? 1 : 0; }, (node) =>
      {
         const operators = ['for'];
         const childNodes = [];

         if (node.init) { childNodes.push(node.init); }
         if (node.test) { childNodes.push(node.test); }
         if (node.update) { childNodes.push(node.update); }

         if (childNodes.length > 0) { operators.push(...ASTGenerator.parseNodes(childNodes).operators); }

         return operators;
      },
      (node) =>
      {
         const operands = [];
         const childNodes = [];

         if (node.init) { childNodes.push(node.init); }
         if (node.test) { childNodes.push(node.test); }
         if (node.update) { childNodes.push(node.update); }

         if (childNodes.length > 0) { operands.push(...ASTGenerator.parseNodes(childNodes).operands); }

         return operands;
      },
      ['init', 'test', 'update']);
   }

   /**
    * ES5 Node
    *
    * Note: The function name (node.id) is returned as an operand and excluded from traversal as to not be included in
    * the function operand calculations.
    *
    * @see https://github.com/estree/estree/blob/master/es5.md#functiondeclaration
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   FunctionDeclaration()
   {
      return actualize(0, 0, void 0, void 0, ['id', 'params', 'defaults'], (node, parent) =>
      {
         const name = s_SAFE_COMPUTED_NAME(node, parent);
         const operands = s_SAFE_COMPUTED_OPERANDS(node, parent);
         const operators = [];

         if (parent && parent.type === 'MethodDefinition')
         {
            if (typeof node.generator === 'boolean' && node.generator) { operators.push('function*'); }

            if (typeof parent.computed === 'boolean' && parent.computed)
            {
               operators.push(...ASTGenerator.parse(parent.key).operators);
            }
         }
         else
         {
            operators.push(typeof node.generator === 'boolean' && node.generator ? 'function*' : 'function');
         }

         const paramNames = [];

         // Parse params; must use ASTGenerator
         if (Array.isArray(node.params))
         {
            node.params.forEach((param) =>
            {
               const parsedParams = ASTGenerator.parse(param);
               operands.push(...parsedParams.operands);
               operators.push(...parsedParams.operators);

               // For param names only the left hand node of AssignmentPattern must be considered.
               if (param.type === 'AssignmentPattern')
               {
                  paramNames.push(...ASTGenerator.parse(param.left).operands);
               }
               else
               {
                  paramNames.push(...parsedParams.operands);
               }
            });
         }

         // Esprima default values; param default values are stored in a non-standard `defaults` node.
         if (Array.isArray(node.defaults))
         {
            node.defaults.forEach((value) =>
            {
               if (value !== null && typeof value === 'object')
               {
                  const parsedParams = ASTGenerator.parse(value);
                  operands.push(...parsedParams.operands);
                  operators.push(...parsedParams.operators);
                  operators.push('='); // Must also manually push '='
               }
            });
         }

         return {
            type: 'method',
            name,
            cyclomatic: 1,
            lineStart: node.loc.start.line,
            lineEnd: node.loc.end.line,
            lloc: 1,
            operands: new HalsteadArray('operands', operands),
            operators: new HalsteadArray('operators', operators),
            paramNames
         };
      });
   }

   // FunctionDeclaration()
   // {
   //   return actualize(1, 0, (node, parent) =>
   //   {
   //      const operators = parent && parent.type === 'MethodDefinition' && typeof parent.computed === 'boolean' &&
   //       parent.computed ? [...ASTGenerator.parse(parent.key).operators] : [];
   //
   //       if (parent && parent.type === 'MethodDefinition')
   //       {
   //          if (typeof node.generator === 'boolean' && node.generator)
   //          {
   //             operators.push('function*');
   //          }
   //       }
   //       else
   //       {
   //          operators.push(typeof node.generator === 'boolean' && node.generator ? 'function*' : 'function');
   //       }
   //
   //       return operators;
   //   },
   //   (node, parent) => { return s_SAFE_COMPUTED_OPERANDS(node, parent); },
   //   'id',
   //   (node, parent) =>
   //   {
   //      return {
   //         type: 'method',
   //         name: s_SAFE_COMPUTED_NAME(node, parent),
   //         lineStart: node.loc.start.line,
   //         lineEnd: node.loc.end.line,
   //         paramCount: node.params.length
   //      };
   //   });
   // }

   /**
    * ES5 Node
    *
    * Note: The function name (node.id) is returned as an operand and excluded from traversal as to not be included in
    * the function operand calculations.
    *
    * @see https://github.com/estree/estree/blob/master/es5.md#functionexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   FunctionExpression()
   {
      return actualize(0, 0, void 0, void 0, ['id', 'params', 'defaults'], (node, parent) =>
      {
         const name = s_SAFE_COMPUTED_NAME(node, parent);
         const operands = s_SAFE_COMPUTED_OPERANDS(node, parent);
         const operators = [];

         if (parent && parent.type === 'MethodDefinition')
         {
            if (typeof node.generator === 'boolean' && node.generator) { operators.push('function*'); }

            if (typeof parent.computed === 'boolean' && parent.computed)
            {
               operators.push(...ASTGenerator.parse(parent.key).operators);
            }
         }
         else
         {
            operators.push(typeof node.generator === 'boolean' && node.generator ? 'function*' : 'function');
         }

         const paramNames = [];

         // Parse params; must use ASTGenerator
         if (Array.isArray(node.params))
         {
            node.params.forEach((param) =>
            {
               const parsedParams = ASTGenerator.parse(param);
               operands.push(...parsedParams.operands);
               operators.push(...parsedParams.operators);

               // For param names only the left hand node of AssignmentPattern must be considered.
               if (param.type === 'AssignmentPattern')
               {
                  paramNames.push(...ASTGenerator.parse(param.left).operands);
               }
               else
               {
                  paramNames.push(...parsedParams.operands);
               }
            });
         }

         // Esprima default values; param default values are stored in a non-standard `defaults` node.
         if (Array.isArray(node.defaults))
         {
            node.defaults.forEach((value) =>
            {
               if (value !== null && typeof value === 'object')
               {
                  const parsedParams = ASTGenerator.parse(value);
                  operands.push(...parsedParams.operands);
                  operators.push(...parsedParams.operators);
                  operators.push('='); // Must also manually push '='
               }
            });
         }

         return {
            type: 'method',
            name,
            cyclomatic: 1,
            lineStart: node.loc.start.line,
            lineEnd: node.loc.end.line,
            lloc: 1,
            operands: new HalsteadArray('operands', operands),
            operators: new HalsteadArray('operators', operators),
            paramNames
         };
      });
   }
   // FunctionExpression()
   // {
   //   return actualize(1, 0, (node, parent) =>
   //   {
   //      const operators = parent && parent.type === 'MethodDefinition' && typeof parent.computed === 'boolean' &&
   //       parent.computed ? [...ASTGenerator.parse(parent.key).operators] : [];
   //
   //      if (parent && parent.type === 'MethodDefinition')
   //      {
   //         if (typeof node.generator === 'boolean' && node.generator)
   //         {
   //            operators.push('function*');
   //         }
   //      }
   //      else
   //      {
   //         operators.push(typeof node.generator === 'boolean' && node.generator ? 'function*' : 'function');
   //      }
   //
   //      return operators;
   //   },
   //   (node, parent) =>
   //   {
   //      return s_SAFE_COMPUTED_OPERANDS(node, parent);
   //   },
   //   'id',
   //   (node, parent) =>
   //   {
   //      return {
   //         type: 'method',
   //         name: s_SAFE_COMPUTED_NAME(node, parent),
   //         lineStart: node.loc.start.line,
   //         lineEnd: node.loc.end.line,
   //         paramCount: node.params.length
   //      };
   //   });
   // }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#identifier
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   Identifier() { return actualize(0, 0, void 0, (node) => { return node.name; }); }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#ifstatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   IfStatement()
   {
      return actualize((node) => { return node.alternate ? 2 : 1; }, 1,
       ['if', { identifier: 'else', filter: (node) => { return !!node.alternate; } }]);
   }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#labeledstatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   LabeledStatement() { return actualize(0, 0); }

   /**
    * ES5 Node
    *
    * Avoid conflicts between string literals and identifiers.
    *
    * @see https://github.com/estree/estree/blob/master/es5.md#literal
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   Literal()
   {
      return actualize(0, 0, void 0, (node) =>
      {
         const operands = [];

         if (typeof node.regex === 'object' && typeof node.raw !== 'undefined')
         {
            operands.push(node.raw);
         }
         else
         {
            operands.push(typeof node.value === 'string' ? `"${node.value}"` : node.value);
         }

         return operands;
      });
   }

   /**
    * ES5 Node
    * @param {object}   settings - escomplex settings
    * @see https://github.com/estree/estree/blob/master/es5.md#logicalexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   LogicalExpression(settings)
   {
      return actualize(0, (node) =>
      {
         const isAnd = node.operator === '&&';
         const isOr = node.operator === '||';
         return (isAnd || (settings.logicalor && isOr)) ? 1 : 0;
      },
      (node) => { return node.operator; });
   }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#memberexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   MemberExpression()
   {
      return actualize(0, 0, (node) => { return typeof node.computed === 'boolean' && node.computed ? '[]' : '.'; });
   }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#newexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   NewExpression()
   {
      return actualize((node) => { return node.callee.type === 'FunctionExpression' ? 1 : 0; }, 0, 'new');
   }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#objectexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ObjectExpression() { return actualize(0, 0, '{}'); }

   /**
    * ES5 Node
    *
    * Note that w/ ES6+ `:` may be omitted and the Property node defines `shorthand` to indicate this case.
    *
    * @see https://github.com/estree/estree/blob/master/es5.md#property
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   Property()
   {
      return actualize(1, 0, (node) =>
      {
         return typeof node.shorthand === 'undefined' ? ':' :
          typeof node.shorthand === 'boolean' && !node.shorthand ? ':' : void 0;
      });
   }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#returnstatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ReturnStatement() { return actualize(1, 0, 'return'); }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#sequenceexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   SequenceExpression() { return actualize(0, 0); }

   /**
    * ES5 Node
    * @param {object}   settings - escomplex settings
    * @see https://github.com/estree/estree/blob/master/es5.md#switchcase
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   SwitchCase(settings)
   {
      return actualize(1, (node) => { return settings.switchcase && node.test ? 1 : 0; },
       (node) => { return node.test ? 'case' : 'default'; });
   }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#switchstatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   SwitchStatement() { return actualize(1, 0, 'switch'); }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#thisexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ThisExpression() { return actualize(0, 0, 'this'); }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#throwstatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ThrowStatement() { return actualize(1, 0, 'throw'); }

   /**
    * ES5 Node
    *
    * Note: esprima has duplicate nodes the catch block; `handler` is the actual ESTree spec.
    *
    * @see https://github.com/estree/estree/blob/master/es5.md#trystatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   TryStatement()
   {
      return actualize(1, 0, (node) => { return node.finalizer ? ['try', 'finally'] : ['try']; }, void 0,
       ['guardedHandlers', 'handlers']);
   }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#unaryexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   UnaryExpression()
   {
      return actualize(0, 0, (node) => { return `${node.operator} (${node.prefix ? 'pre' : 'post'}fix)`; });
   }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#updateexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   UpdateExpression()
   {
      return actualize(0, 0, (node) => { return `${node.operator} (${node.prefix ? 'pre' : 'post'}fix)`; });
   }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#variabledeclaration
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   VariableDeclaration() { return actualize(0, 0, (node) => { return node.kind; }); }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#variabledeclarator
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   VariableDeclarator()
   {
      return actualize(1, 0, { identifier: '=', filter: (node) => { return !!node.init; } });
   }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#whilestatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   WhileStatement() { return actualize(1, (node) => { return node.test ? 1 : 0; }, 'while'); }

   /**
    * ES5 Node
    * @see https://github.com/estree/estree/blob/master/es5.md#withstatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   WithStatement() { return actualize(1, 0, 'with'); }

   // ES6 ESTree AST nodes ------------------------------------------------------------------------------------------

   /**
    * ES6 Node
    * @see https://github.com/estree/estree/blob/master/es2015.md#assignmentpattern
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   AssignmentPattern() { return actualize(1, 0, (node) => { return node.operator; }); }

   /**
    * ES6 Node
    * @see https://github.com/estree/estree/blob/master/es2015.md#arraypattern
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ArrayPattern()
   {
      return actualize(0, 0, (node) =>
      {
         const operators = ['[]'];
         if (Array.isArray(node.elements) && node.elements.length > 0)  // Add length - 1 commas
         {
            operators.push(...(new Array(node.elements.length - 1).fill(',')));
         }
         return operators;
      });
   }

   /**
    * ES6 Node
    *
    * Note: If the parent node is an ExpressionStatement the arrow function is essentially a no-op / invalid. In this
    * case no metrics are gathered, but a method is still registered with no data. Further analysis looking for methods
    * with no logical SLOC will reveal this case.
    *
    * @see https://github.com/estree/estree/blob/master/es2015.md#arrowfunctionexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ArrowFunctionExpression()
   {
      return actualize(0, 0, void 0, void 0, (node, parent) =>
      {
         // If not valid do not traverse any children.
         return parent && parent.type !== 'ExpressionStatement' ? ['id', 'params', 'defaults'] : null;
      },
      (node, parent) =>
      {
         const isValid = parent && parent.type !== 'ExpressionStatement';

         const name = TraitUtil.safeName(node.id);
         const operands = [];
         const operators = isValid ? ['function=>'] : [];

         const paramNames = [];

         // Parse params; must use ASTGenerator
         if (isValid && Array.isArray(node.params))
         {
            node.params.forEach((param) =>
            {
               const parsedParams = ASTGenerator.parse(param);
               operands.push(...parsedParams.operands);
               operators.push(...parsedParams.operators);

               // For param names only the left hand node of AssignmentPattern must be considered.
               if (param.type === 'AssignmentPattern')
               {
                  paramNames.push(...ASTGenerator.parse(param.left).operands);
               }
               else
               {
                  paramNames.push(...parsedParams.operands);
               }
            });
         }

         // Esprima default values; param default values are stored in a non-standard `defaults` node.
         if (isValid && Array.isArray(node.defaults))
         {
            node.defaults.forEach((value) =>
            {
               if (value !== null && typeof value === 'object')
               {
                  const parsedParams = ASTGenerator.parse(value);
                  operands.push(...parsedParams.operands);
                  operators.push(...parsedParams.operators);
                  operators.push('='); // Must also manually push '='
               }
            });
         }

         return {
            type: 'method',
            name,
            cyclomatic: isValid ? 1 : 0,
            lineStart: node.loc.start.line,
            lineEnd: node.loc.end.line,
            lloc: isValid ? 1 : 0,
            operands: isValid ? new HalsteadArray('operands', operands) : void 0,
            operators: isValid ? new HalsteadArray('operators', operators) : void 0,
            paramNames,

            // ArrowFunctionExpressions without brackets have an implicit `return` expression so post increment lloc.
            postLloc: isValid && typeof node.body === 'object' && node.body.type !== 'BlockStatement' ? 1 : 0

            // NOTE (altered 12/8/18): ESTree has expression entry which was removed in Babel Parser / 7.0 so
            // node.body.type must be checked to determine if an expression or block / return statement.
            // Reference: https://github.com/babel/babel/issues/6773
            // postLloc: isValid && typeof node.expression === 'boolean' && node.expression ? 1 : 0
         };
      });
   }

   /**
    * ES6 Node
    * @see https://github.com/estree/estree/blob/master/es2015.md#classbody
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ClassBody() { return actualize(0, 0); }

   // /**
   // * ES6 Node
   // * @see https://github.com/estree/estree/blob/master/es2015.md#classdeclaration
   // * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
   // */
   // ClassDeclaration()
   // {
   //   return actualize(1, 0, 'class', void 0, void 0, (node) =>
   //   {
   //      return {
   //         type: 'class',
   //         name: TraitUtil.safeName(node.id),
   //         lineStart: node.loc.start.line,
   //         lineEnd: node.loc.end.line
   //      };
   //   });
   // }
   //
   // /**
   // * ES6 Node
   // * @see https://github.com/estree/estree/blob/master/es2015.md#classexpression
   // * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
   // */
   // ClassExpression()
   // {
   //   return actualize(1, 0, 'class', void 0, void 0, (node) =>
   //   {
   //      return {
   //         type: 'class',
   //         name: TraitUtil.safeName(node.id),
   //         lineStart: node.loc.start.line,
   //         lineEnd: node.loc.end.line
   //      };
   //   });
   // }

   /**
    * ES6 Node
    * @see https://github.com/estree/estree/blob/master/es2015.md#classdeclaration
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ClassDeclaration()
   {
      return actualize(0, 0, void 0, void 0, ['id', 'superClass'], (node) =>
      {
         const name = TraitUtil.safeName(node.id);
         const operands = name !== '<anonymous>' ? [name] : [];
         const operators = ['class'];
         let superClassName;

         if (typeof node.superClass !== 'undefined' && node.superClass !== null)
         {
            const parsed = ASTGenerator.parse(node.superClass);

            operators.push('extends');

            operands.push(...parsed.operands);
            operators.push(...parsed.operators);

            // The following will pick up a Identifier or a computed value (string); computed expressions return
            // `<computed~expression>`.
            superClassName = node.superClass.type === 'Identifier' ? TraitUtil.safeName(node.superClass) :
             `<computed~${parsed.source}>`;
         }

         return {
            type: 'class',
            name,
            superClassName,
            lineStart: node.loc.start.line,
            lineEnd: node.loc.end.line,
            lloc: 1,
            operands: new HalsteadArray('operands', operands),
            operators: new HalsteadArray('operators', operators)
         };
      });
   }

   /**
    * ES6 Node
    * @see https://github.com/estree/estree/blob/master/es2015.md#classexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ClassExpression()
   {
      return actualize(0, 0, void 0, void 0, ['id', 'superClass'], (node) =>
      {
         const name = TraitUtil.safeName(node.id);
         const operands = name !== '<anonymous>' ? [name] : [];
         const operators = ['class'];
         let superClassName;

         if (typeof node.superClass !== 'undefined' && node.superClass !== null)
         {
            const parsed = ASTGenerator.parse(node.superClass);

            operators.push('extends');

            operands.push(...parsed.operands);
            operators.push(...parsed.operators);

            // The following will pick up a Identifier or a computed value (string); computed expressions return
            // `<computed~expression>`.
            superClassName = node.superClass.type === 'Identifier' ? TraitUtil.safeName(node.superClass) :
             `<computed~${parsed.source}>`;
         }

         return {
            type: 'class',
            name: TraitUtil.safeName(node.id),
            superClassName,
            lineStart: node.loc.start.line,
            lineEnd: node.loc.end.line,
            lloc: 1,
            operands: new HalsteadArray('operands', operands),
            operators: new HalsteadArray('operators', operators)
         };
      });
   }

   /**
    * ES6 Node
    * @param {object} settings - Runtime settings of escomplex.
    * @see https://github.com/estree/estree/blob/master/es2015.md#exportalldeclaration
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ExportAllDeclaration(settings)
   {
      return actualize(settings.esmImportExport.lloc ? 1 : 0, 0, settings.esmImportExport.halstead ? ['export', '*'] :
       void 0, void 0, settings.esmImportExport.halstead ? void 0 : null);
   }

   /**
    * ES6 Node
    * @param {object} settings - Runtime settings of escomplex.
    * @see https://github.com/estree/estree/blob/master/es2015.md#exportdefaultdeclaration
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ExportDefaultDeclaration(settings)
   {
      return actualize(settings.esmImportExport.lloc ? 1 : 0, 0, settings.esmImportExport.halstead ?
       ['export', 'default'] : void 0);
   }

   /**
    * ES6 Node
    * @param {object} settings - Runtime settings of escomplex.
    * @see https://github.com/estree/estree/blob/master/es2015.md#exportnameddeclaration
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ExportNamedDeclaration(settings)
   {
      return actualize(settings.esmImportExport.lloc ? 1 : 0, 0, settings.esmImportExport.halstead ? ['export', '{}'] :
       void 0, void 0, settings.esmImportExport.halstead ? void 0 : ['source', 'specifiers']);
   }

   /**
    * ES6 Node
    * @param {object} settings - Runtime settings of escomplex.
    * @see https://github.com/estree/estree/blob/master/es2015.md#exportspecifier
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ExportSpecifier(settings)
   {
      return actualize(0, 0, !settings.esmImportExport.halstead ? void 0 : (node) =>
      {
         return node.exported.name !== node.local.name ? 'as' : void 0;
      },
      void 0,
      !settings.esmImportExport.halstead ? null : (node) =>
      {
         return node.exported.name === node.local.name ? ['local'] : void 0;
      });
   }

   /**
    * ES6 Node
    * @see https://github.com/estree/estree/blob/master/es2015.md#forofstatement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ForOfStatement()
   {
      return actualize(1, (node) => { return node.test ? 1 : 0; }, (node) =>
      {
         const operators = ['forof'];
         const childNodes = [];

         if (node.left) { childNodes.push(node.left); }
         if (node.right) { childNodes.push(node.right); }

         if (childNodes.length > 0) { operators.push(...ASTGenerator.parseNodes(childNodes).operators); }

         return operators;
      },
      (node) =>
      {
         const operands = [];
         const childNodes = [];

         if (node.left) { childNodes.push(node.left); }
         if (node.right) { childNodes.push(node.right); }

         if (childNodes.length > 0) { operands.push(...ASTGenerator.parseNodes(childNodes).operands); }

         return operands;
      },
      ['left', 'right']);
   }

   /**
    * ES6 Node
    * @param {object}   settings - escomplex settings
    * @see https://github.com/estree/estree/blob/master/es2015.md#importdeclaration
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ImportDeclaration(settings)
   {
      return actualize(settings.esmImportExport.lloc ? 1 : 0, 0,
       settings.esmImportExport.halstead ? ['import', 'from'] : void 0, void 0, settings.esmImportExport.halstead ?
        void 0 : null, void 0, (node) =>
      {
         const dependencyPath = typeof settings.dependencyResolver === 'function' ?
          settings.dependencyResolver(node.source.value) : node.source.value;

         return { line: node.source.loc.start.line, path: dependencyPath, type: 'esm' };
      });
   }

   /**
    * ES6 Node
    * @see https://github.com/estree/estree/blob/master/es2015.md#importdefaultspecifier
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ImportDefaultSpecifier() { return actualize(0, 0); }

   /**
    * ES6 Node
    * @param {object} settings - Runtime settings of escomplex.
    * @see https://github.com/estree/estree/blob/master/es2015.md#importnamespacespecifier
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ImportNamespaceSpecifier(settings)
   {
      return actualize(0, 0, settings.esmImportExport.halstead ? ['import', '*', 'as'] : void 0, void 0,
       settings.esmImportExport.halstead ? void 0 : null);
   }

   /**
    * ES6 Node
    * @param {object} settings - Runtime settings of escomplex.
    * @see https://github.com/estree/estree/blob/master/es2015.md#importspecifier
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ImportSpecifier(settings)
   {
      return actualize(0, 0, !settings.esmImportExport.halstead ? void 0 : (node) =>
      {
         return node.imported.name === node.local.name ? '{}' : ['{}', 'as'];
      },
      void 0,
      settings.esmImportExport.halstead ? void 0 : null);
   }

   /**
    * ES6 Node
    *
    * Note: esprima doesn't follow the ESTree spec and `meta` & `property` are strings instead of Identifier nodes.
    *
    * @see https://github.com/estree/estree/blob/master/es2015.md#metaproperty
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   MetaProperty()
   {
      return actualize(0, 0, '.', (node) =>
      {
         return typeof node.meta === 'string' && typeof node.property === 'string' ? [node.meta, node.property] :
          void 0;
      });
   }

   /**
    * ES6 Node
    *
    * Note: must skip as the following FunctionExpression assigns the name.
    *
    * @see https://github.com/estree/estree/blob/master/es2015.md#methoddefinition
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   MethodDefinition()
   {
      return actualize(0, 0, (node) =>
      {
         const operators = [];
         if (node.kind && (node.kind === 'get' || node.kind === 'set')) { operators.push(node.kind); }
         if (typeof node.static === 'boolean' && node.static) { operators.push('static'); }
         return operators;
      },
      void 0,
      'key');
   }

   /**
    * ES6 Node
    * @see https://github.com/estree/estree/blob/master/es2015.md#objectpattern
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   ObjectPattern() { return actualize(0, 0, '{}'); }

   /**
    * ES6 Node
    * @see https://github.com/estree/estree/blob/master/es2015.md#restelement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   RestElement() { return actualize(0, 0, '... (rest)'); }

   /**
    * ES6 Node
    * @see https://github.com/estree/estree/blob/master/es2015.md#spreadelement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   SpreadElement() { return actualize(0, 0, '... (spread)'); }

   /**
    * ES6 Node
    * @see https://github.com/estree/estree/blob/master/es2015.md#super
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   Super() { return actualize(0, 0, 'super'); }

   /**
    * ES6 Node
    * @param {object} settings - Runtime settings of escomplex.
    * @see https://github.com/estree/estree/blob/master/es2015.md#taggedtemplateexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   TaggedTemplateExpression(settings) { return actualize(settings.templateExpression.lloc ? 1 : 0, 0); }

   /**
    * ES6 Node
    * @see https://github.com/estree/estree/blob/master/es2015.md#templateelement
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   TemplateElement()
   {
      return actualize(0, 0, void 0, (node) => { return node.value.cooked !== '' ? node.value.cooked : void 0; });
   }

   /**
    * ES6 Node
    * @param {object} settings - Runtime settings of escomplex.
    * @see https://github.com/estree/estree/blob/master/es2015.md#templateliteral
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   TemplateLiteral(settings)
   {
      return actualize(0, 0,
      !settings.templateExpression.halstead ? [] : (node) =>
      {
         const operators = ['``'];

         if (Array.isArray(node.expressions) && node.expressions.length > 0)  // Add length '${}' elements.
         {
            operators.push(...(new Array(node.expressions.length).fill('${}')));
         }

         return operators;
      });
   }

   /**
    * ES6 Node
    * @see https://github.com/estree/estree/blob/master/es2015.md#yieldexpression
    * @returns {{lloc: *, cyclomatic: *, operators: *, operands: *, ignoreKeys: *, newScope: *, dependencies: *}}
    */
   YieldExpression()
   {
      return actualize((node, parent) =>
      {
         // The top most `ExpressionStatement` already provides 1 lloc.
         return parent && parent.type !== 'ExpressionStatement' ? 1 : 0;
      },
      0,
      (node) =>
      {
         return typeof node.delegate === 'boolean' && node.delegate ? 'yield*' : 'yield';
      });
   }
}

/**
 * Provides a utility method that determines the name of a method for ESTree / Babylon AST nodes. For ESTree the
 * parent node must be accessed for class methods. If the name is a computed value and not a string literal then
 * `astParse` is invoked to determine the computed name and is output as `<computed~${computed expression}>`.
 *
 * Note; ESTree has a parent node which defines the method name with a child FunctionExpression /
 * FunctionDeclaration. Babylon AST only has ClassMethod with a child `key` providing the method name.
 *
 * @param {object}   node - The current AST node.
 * @param {object}   parent - The parent AST node.
 *
 * @returns {string}
 */
function s_SAFE_COMPUTED_NAME(node, parent)
{
   let name;

   // Handle ESTree case.
   if (parent && parent.type === 'MethodDefinition')
   {
      if (typeof parent.computed === 'boolean' && parent.computed)
      {
         // The following will pick up a single literal computed value (string); expressions return
         // `<computed~expression>`.
         name = parent.key.type === 'Literal' ? TraitUtil.safeValue(parent.key) :
          `<computed~${ASTGenerator.parse(parent.key).source}>`;
      }
      else // Parent is not computed and `parent.key` is an `Identifier` node.
      {
         name = TraitUtil.safeName(parent.key);
      }
   }

   // Last chance assignment handles other node types such as FunctionExpression / ArrowFunctionExpression.
   if (typeof name !== 'string')
   {
      name = TraitUtil.safeName(node.id || node.key);
   }

   return name;
}

/**
 * Provides a utility method that determines the operands of a method for ESTree AST nodes. For ESTree the
 * parent node must be accessed for class methods. If the name is a computed value and not a string literal then
 * `astParse` is invoked to determine the computed operands.
 *
 * Note; ESTree has a parent node which defines the method name with a child FunctionExpression /
 * FunctionDeclaration. Babylon AST only has ClassMethod with a child `key` providing the method name.
 *
 * @param {object}   node - The current AST node.
 * @param {object}   parent - The parent AST node.
 *
 * @returns {Array<*>}
 */
function s_SAFE_COMPUTED_OPERANDS(node, parent)
{
   const operands = [];

   if (parent && parent.type === 'MethodDefinition')
   {
      if (typeof parent.computed === 'boolean' && parent.computed)
      {
         // The following will pick up a single literal computed value (string).
         if (parent.key.type === 'Literal')
         {
            operands.push(TraitUtil.safeValue(parent.key));
         }
         else // Fully evaluate AST node and children for computed operands.
         {
            operands.push(...ASTGenerator.parse(parent.key).operands);
         }
      }
      else // Parent is not computed and `parent.key` is an `Identifier` node.
      {
         operands.push(TraitUtil.safeName(parent.key));
      }
   }

   // Only add further operands if the parent is not MethodDefinition and `node.id` or `node.key` is defined.
   if (operands.length === 0 &&
    ((typeof node.id !== 'undefined' && node.id !== null) || (typeof node.key !== 'undefined' && node.key !== null)))
   {
      operands.push(TraitUtil.safeName(node.id || node.key));
   }

   return operands;
}
