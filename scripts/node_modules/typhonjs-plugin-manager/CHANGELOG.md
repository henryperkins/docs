## 0.2.0 (2017-08-25)
- Fine tuned internal plugin entry data.
- Added optional module data to be associated with plugin data when added.
- Added deep freeze of plugin data / typhonjs-object-util.
- Breaking change (minor impact): Updated data passed to global event bindings when plugins are added, removed, enabled.

## 0.1.11 (2017-07-14)
- Breaking change (minor impact): updated event bindings for async / sync methods

## 0.1.10 (2017-07-12)
- added support for asynchronous usage: please see `addAsync`, `addAllAsync`, `destroyAsync`, 
`setEventbusAsync`, `removeAsync`, `removeAllAsync` and related event bindings. These methods 
that invoke plugin lifecycle methods `onPluginLoad` / `onPluginUnload` will asynchronously invoke
them such that if a plugin returns a Promise or is `async` then execution awaits until completed.

- It is recommended that you use async / await to integrate asynchronous usage.

## 0.1.9 (2017-05-30)
- added basic plugin invoke method and event binding that does not return results.
- reorganized method signature for invokeAsync / invokeSync to take a method name and then optional 
  arguments / plugin names  

## 0.1.7 (2017-03-10)
- updated event bindings.

## 0.1.6 (2017-03-09)
- added getPluginEventNames / `plugins:get:plugin:event:names` event.
- added getPluginsByEventName / `plugins:get:plugins:by:event:name` event.
- added getPluginsEventNames / `plugins:get:plugins:event:names` event.

## 0.1.5 (2017-03-07)
added `plugins:destroy:manager` event.

## 0.1.2 (2017-02-20)
Added `createEventProxy` returning an EventProxy instance of any assigned eventbus to PluginManager.

## 0.1.0 (2017-02-11)
Initial feature complete implementation; pending full testsuite for coverage & more docs.
