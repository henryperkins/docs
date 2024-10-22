![typhonjs-plugin-manager](https://i.imgur.com/rCbwc2o.png)

[![NPM](https://img.shields.io/npm/v/typhonjs-plugin-manager.svg?label=npm)](https://www.npmjs.com/package/typhonjs-plugin-manager)
[![Documentation](http://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/badge.svg)](http://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/)
[![Code Style](https://img.shields.io/badge/code%20style-allman-yellowgreen.svg?style=flat)](https://en.wikipedia.org/wiki/Indent_style#Allman_style)
[![License](https://img.shields.io/badge/license-MPLv2-yellowgreen.svg?style=flat)](https://github.com/typhonjs-node-plugin/typhonjs-plugin-manager/blob/master/LICENSE)
[![Gitter](https://img.shields.io/gitter/room/typhonjs/TyphonJS.svg)](https://gitter.im/typhonjs/TyphonJS)

[![Build Status](https://travis-ci.org/typhonjs-node-plugin/typhonjs-plugin-manager.svg?branch=master)](https://travis-ci.org/typhonjs-node-plugin/typhonjs-plugin-manager)
[![Coverage](https://img.shields.io/codecov/c/github/typhonjs-node-plugin/typhonjs-plugin-manager.svg)](https://codecov.io/github/typhonjs-node-plugin/typhonjs-plugin-manager)
[![Dependency Status](https://www.versioneye.com/user/projects/575e79a77757a00041b3ba3f/badge.svg?style=flat)](https://www.versioneye.com/user/projects/575e79a77757a00041b3ba3f)

Provides a lightweight plugin manager for Node / NPM with optional `backbone-esnext-events`
integration for plugins in a safe and protected manner across NPM modules, local files, and preloaded object
instances. This pattern facilitates message passing between modules versus direct dependencies / method invocation.

It isn't necessary to use an eventbus associated with the plugin manager though invocation then relies on invoking
methods directly with the plugin manager instance.

When passing in an eventbus from `backbone-esnext-events` the plugin manager will register by default under these
event categories:

`plugins:add` - invokes [PluginManager#add](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-add)

`plugins:add:all` - invokes [PluginManager#addAll](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-addAll)

`plugins:async:add` - invokes [PluginManager#addAsync](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-addAsync)

`plugins:async:add:all` - invokes [PluginManager#addAllAsync](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-addAllAsync)

`plugins:async:destroy:manager` - invokes [PluginManager#destroyAsync](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-destroyAsync)

`plugins:async:invoke` - invokes [PluginManager#invokeAsync](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-invokeAsync)

`plugins:async:invoke:event` - invokes [PluginManager#invokeAsyncEvent](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-invokeAsyncEvent)

`plugins:async:remove` - invokes [PluginManager#removeAsync](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-removeAsync)

`plugins:async:remove:all` - invokes [PluginManager#removeAllAsync](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-removeAllAsync)

`plugins:create:event:proxy` - invokes [PluginManager#createEventProxy](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-createEventProxy)

`plugins:destroy:manager` - invokes [PluginManager#destroy](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-destroy)

`plugins:get:all:plugin:data` - invokes [PluginManager#getAllPluginData](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-getAllPluginData)

`plugins:get:extra:event:data` - invokes [PluginManager#getExtraEventData](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-getExtraEventData)

`plugins:get:method:names` - invokes [PluginManager#getMethodNames](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-getMethodNames)

`plugins:get:options` - invokes [PluginManager#getOptions](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-getOptions)

`plugins:get:plugin:data` - invokes [PluginManager#getPluginData](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-getPluginData)

`plugins:get:plugin:enabled` - invokes [PluginManager#getPluginEnabled](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-getPluginEnabled)

`plugins:get:plugin:method:names` - invokes [PluginManager#getPluginMethodNames](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-getPluginMethodNames)

`plugins:get:plugin:names` - invokes [PluginManager#getPluginNames](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-getPluginNames)

`plugins:get:plugin:options` - invokes [PluginManager#getPluginOptions](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-getPluginOptions)

`plugins:get:plugins:enabled` - invokes [PluginManager#getPluginsEnabled](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-getPluginsEnabled)

`plugins:has:method` - invokes [PluginManager#hasMethod](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-hasMethod)

`plugins:has:plugin` - invokes [PluginManager#hasPlugin](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-hasPlugin)

`plugins:has:plugin:method` - invokes [PluginManager#hasPluginMethod](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-hasPluginMethod)

`plugins:invoke` - invokes [PluginManager#invoke](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-invoke)

`plugins:is:valid:config` - invokes [PluginManager#isValidConfig](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-isValidConfig)

`plugins:remove` - invokes [PluginManager#remove](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-remove)

`plugins:remove:all` - invokes [PluginManager#removeAll](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-removeAll)

`plugins:set:extra:event:data` - invokes [PluginManager#setExtraEventData](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-setExtraEventData)

`plugins:set:plugin:enabled` - invokes [PluginManager#setPluginEnabled](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-setPluginEnabled)

`plugins:set:plugins:enabled` - invokes [PluginManager#setPluginsEnabled](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-setPluginsEnabled)

`plugins:sync:invoke` - invokes [PluginManager#invokeSync](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-invokeSync)

`plugins:sync:invoke:event` - invokes [PluginManager#invokeSyncEvent](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-invokeSyncEvent)

Automatically when a plugin is loaded and unloaded respective callbacks `onPluginLoad` and `onPluginUnload` will
be attempted to be invoked on the plugin. This is an opportunity for the plugin to receive any associated eventbus
and wire itself into it. It should be noted that a protected proxy around the eventbus is passed to the plugins
such that when the plugin is removed automatically all events registered on the eventbus are cleaned up without
a plugin author needing to do this manually in the `onPluginUnload` callback. This solves any dangling event binding
issues.

The plugin manager also supports asynchronous operation with the methods ending in `Async` along with event bindings
that include `async`. For asynchronous variations of `add`, `destroy`, and `remove` the lifecycle methods
`onPluginLoad` and `onPluginUnload` will be awaited on such that if a plugin returns a Promise or is an async method
then it must complete before execution continues. One can use Promises to interact with the plugin manager
asynchronously, but usage via async / await is recommended.

If eventbus functionality is enabled it is important especially if using a process / global level eventbus such as
`backbone-esnext-eventbus` to call [PluginManager#destroy](https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginManager.js~PluginManager.html#instance-method-destroy) to clean up all plugin eventbus resources and
the plugin manager event bindings.

Please see the following NPM modules for eventbus info:
- [backbone-esnext-events](https://www.npmjs.com/package/backbone-esnext-events)
- [backbone-esnext-eventbus](https://www.npmjs.com/package/backbone-esnext-eventbus)

Examples follow:
```
import Events        from 'backbone-esnext-events';   // Imports the TyphonEvents class for local usage.
::or alternatively::
import eventbus      from 'backbone-esnext-eventbus'; // Imports a global / process level eventbus.

import PluginManager from 'typhonjs-plugin-manager';

const pluginManager = new PluginManager({ eventbus });

pluginManager.add({ name: 'an-npm-plugin-enabled-module' });
pluginManager.add({ name: 'my-local-module', target: './myModule.js' });

// Let's say an-npm-plugin-enabled-module responds to 'cool:event' which returns 'true'.
// Let's say my-local-module responds to 'hot:event' which returns 'false'.
// Both of the plugin / modules will have 'onPluginLoaded' invoked with a proxy to the eventbus and any plugin
// options defined.

// One can then use the eventbus functionality to invoke associated module / plugin methods even retrieving results.
assert(eventbus.triggerSync('cool:event') === true);
assert(eventbus.triggerSync('hot:event') === false);

// One can also indirectly invoke any method of the plugin via:
eventbus.triggerSync('plugins:invoke:sync:event', 'aCoolMethod'); // Any plugin with a method named `aCoolMethod` is invoked.
eventbus.triggerSync('plugins:invoke:sync:event', 'aCoolMethod', {}, {}, 'an-npm-plugin-enabled-module'); // specific invocation.

// The 3rd parameter defines a pass through object hash and the 4th will make a copy of the hash sending a single
// event / object hash to the invoked method.

// -----------------------

// Given that `backbone-esnext-eventbus` defines a global / process level eventbus you can import it in an entirely
// different file or even NPM module and invoke methods of loaded plugins like this:

import eventbus from 'backbone-esnext-eventbus';

eventbus.triggerSync('plugins:invoke', 'aCoolMethod'); // Any plugin with a method named `aCoolMethod` is invoked.

assert(eventbus.triggerSync('cool:event') === true);

eventbus.trigger('plugins:remove', 'an-npm-plugin-enabled-module'); // Removes the plugin and unregisters events.

assert(eventbus.triggerSync('cool:event') === true); // Will now fail!

// In this case though when using the global eventbus be mindful to always call `pluginManager.destroy()` in the main
// thread of execution scope to remove all plugins and the plugin manager event bindings!
```
