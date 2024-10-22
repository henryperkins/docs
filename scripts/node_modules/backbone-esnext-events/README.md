![backbone-esnext-events](http://i.imgur.com/gfeYvBh.png)

[![NPM](https://img.shields.io/npm/v/backbone-esnext-events.svg?label=npm)](https://www.npmjs.com/package/backbone-esnext-events)
[![Documentation](http://docs.typhonjs.io/typhonjs-backbone-esnext/backbone-esnext-events/badge.svg)](http://docs.typhonjs.io/typhonjs-backbone-esnext/backbone-esnext-events/)
[![Code Style](https://img.shields.io/badge/code%20style-allman-yellowgreen.svg?style=flat)](https://en.wikipedia.org/wiki/Indent_style#Allman_style)
[![License](https://img.shields.io/badge/license-MPLv2-yellowgreen.svg?style=flat)](https://github.com/typhonjs-backbone/typhonjs-core-backbone-events/blob/master/LICENSE)
[![Gitter](https://img.shields.io/gitter/room/typhonjs/TyphonJS.svg)](https://gitter.im/typhonjs/TyphonJS)

[![Build Status](https://travis-ci.org/typhonjs-backbone-esnext/backbone-esnext-events.svg)](https://travis-ci.org/typhonjs-backbone-esnext/backbone-esnext-events)
[![Coverage](https://img.shields.io/codecov/c/github/typhonjs-backbone-esnext/backbone-esnext-events.svg)](https://codecov.io/github/typhonjs-backbone-esnext/backbone-esnext-events)
[![Dependency Status](https://david-dm.org/typhonjs-backbone-esnext/backbone-esnext-events.svg)](https://david-dm.org/typhonjs-backbone-esnext/backbone-esnext-events)

Separates 'Events' support from [backbone-esnext](https://github.com/typhonjs-backbone-esnext) in addition to adding TyphonJS extensions found in [TyphonEvents](https://github.com/typhonjs-backbone-esnext/backbone-esnext-events/blob/master/src/TyphonEvents.js). The events dispatch functionality is useful well outside the context of Backbone and is utilized across several TyphonJS repos. It should be noted that there are no dependencies with backbone-esnext-events and it can be used independently in any project without pulling in Underscore like Backbone does. 

The default trigger mechanism work justs as it does with Backbone:
- `trigger` - Invokes all targets matched with a one way message. 

TyphonEvents adds new functionality for triggering events. The following are new trigger mechanisms:

- `triggerDefer` - Defers invoking `trigger` to the next clock tick.
- `triggerSync` - Synchronously invokes all targets matched and passes back a single value or an array of results to the callee.
- `triggerAsync` - Asynchronously invokes all targets matched and passes back a promise resolved with a single value or an array of results through `Promise.all` which returns a single promise to the callee.

To import TyphonEvents and create a new instance: 
```
import Events from 'backbone-esnext-events';

const eventbus = new Events();

// or extend a class to add event functionality
 
export default class MyThing extends Events {}
```

Please see [backbone-esnext-eventbus](https://www.npmjs.com/package/backbone-esnext-eventbus) for a module which provides a default main eventbus instance for ease of use across modules.