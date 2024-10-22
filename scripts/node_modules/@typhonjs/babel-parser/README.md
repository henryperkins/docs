![@typhonjs/babel-parser](https://i.imgur.com/hN409kK.png)

[![NPM](https://img.shields.io/npm/v/@typhonjs/babel-parser.svg?label=npm)](https://www.npmjs.com/package/@typhonjs/babel-parser)
[![Code Style](https://img.shields.io/badge/code%20style-allman-yellowgreen.svg?style=flat)](https://en.wikipedia.org/wiki/Indent_style#Allman_style)
[![License](https://img.shields.io/badge/license-MPLv2-yellowgreen.svg?style=flat)](https://github.com/typhonjs-node-ast/babel-parser/blob/master/LICENSE)
[![Gitter](https://img.shields.io/gitter/room/typhonjs/TyphonJS.svg)](https://gitter.im/typhonjs/TyphonJS)

[![Build Status](https://travis-ci.org/typhonjs-node-ast/babel-parser.svg?branch=master)](https://travis-ci.org/typhonjs-node-ast/babel-parser)
[![Coverage](https://img.shields.io/codecov/c/github/typhonjs-node-ast/babel-parser.svg)](https://codecov.io/github/typhonjs-node-ast/babel-parser)
[![Dependency Status](https://david-dm.org/typhonjs-node-ast/babel-parser/status.svg)](https://david-dm.org/typhonjs-node-ast/babel-parser)

Provides a front end for Javascript / Typescript AST generation by Babel parser with TyphonJS plugin support. By default all Babel parser plugins are enabled except for `flow` and there is a handy override mechanism to change relevant defaults without having to provide a full set of options to the parser. 

This NPM module can be installed as a dependency in `package.json` as follows:
```js
"dependencies": {
  "@typhonjs/babel-parser": "^0.2.0"
}
```

Please see [Babel Parser Docs](https://babeljs.io/docs/en/babel-parser) for specific plugin information. By default `babel-parser` enables all plugins except for `flow` as it is incompatible with the `typescript` plugin. Also by default the `decorators` plugins is enabled and not compatible with `decorators-legacy`. 

The default Babel parser options is as follows:
```
const s_DEFAULT_BABELPARSER_OPTIONS =
{
   plugins: ['asyncGenerators', 'bigInt', 'classProperties', 'classPrivateProperties', 'classPrivateMethods',
    ['decorators', { decoratorsBeforeExport: false }], 'doExpressions', 'dynamicImport',
     'exportDefaultFrom', 'exportNamespaceFrom',  'functionBind', 'functionSent', 'importMeta',
      'jsx', 'logicalAssignment', 'nullishCoalescingOperator', 'numericSeparator', 'objectRestSpread',
       'optionalCatchBinding', 'optionalChaining', ['pipelineOperator', { proposal: 'minimal' }], 'throwExpressions',
        'typescript']
};
```

There is a way to provide additional override directives to `babel-parser` which modifies the default babel parser options above. This only activates when no manual parser options are provided. A third parameter can be passed into `parse` which will modify the default parameters above. For example Flow is supported by passing in `{ flow: true }` as the override object. This allows flow to be enabled and typescript to be disabled without providing the full babel parser options manually. A few other optional overrides are available:

```
{
   decoratorsBeforeExport: <boolean>,   // Sets the associated configuration value
   decoratorsLegacy: true,              // Removes the proposal decorators plugin for `decorators-legacy`
   flow: true,                          // Enables flow / disables typescript plugins
   pipelineOperatorProposal: <string>   // Sets the proposal field of pipelineOperator plugin
}
```


An ES6 example follows:
```js
import BabelParser from '@typhonjs/babel-parser';

// Basic usage to parse text / source code with default options.
const ast = BabelParser.parse(`<some JS or Typescript source code>`);

// Providing custom options
const parserOptions = { plugins: [<any parser options desired to be enabled>] };
const ast = BabelParser.parse(`<some JS or Typescript source code>`, parserOptions);

// Basic usage with default options, but with an override to disable Typescript and enable Flow plugins
const ast = BabelParser.parse(`<some JS w/ Flow typing>`, void 0, { flow: true });
```

`@typhonjs/babel-parser` may be loaded as a TyphonJS plugin with [typhonjs-plugin-manager](https://www.npmjs.com/package/typhonjs-plugin-manager) and if an eventbus is associated the following event categories are registered:

`typhonjs:babel:parser:parse` - invokes `BabelParser.parse`
