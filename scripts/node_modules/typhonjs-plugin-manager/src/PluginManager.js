import path          from 'path';
import ObjectUtil    from 'typhonjs-object-util';

import EventProxy    from 'backbone-esnext-events/src/EventProxy';

import PluginEntry   from './PluginEntry.js';
import PluginEvent   from './PluginEvent.js';

/**
 * Provides a lightweight plugin manager for Node / NPM with optional `backbone-esnext-events`
 * integration for plugins in a safe and protected manner across NPM modules, local files, and preloaded object
 * instances. This pattern facilitates message passing between modules versus direct dependencies / method invocation.
 *
 * It isn't necessary to use an eventbus associated with the plugin manager though invocation then relies on invoking
 * methods directly with the plugin manager instance.
 *
 * When passing in an eventbus from `backbone-esnext-events` the plugin manager will register by default under these
 * event categories:
 *
 * `plugins:add` - {@link PluginManager#add}
 *
 * `plugins:add:all` - {@link PluginManager#addAll}
 *
 * `plugins:async:add` - {@link PluginManager#addAsync}
 *
 * `plugins:async:add:all` - {@link PluginManager#addAllAsync}
 *
 * `plugins:async:destroy:manager` - {@link PluginManager#destroyAsync}
 *
 * `plugins:async:invoke` - {@link PluginManager#invokeAsync}
 *
 * `plugins:async:invoke:event` - {@link PluginManager#invokeAsyncEvent}
 *
 * `plugins:async:remove` - {@link PluginManager#removeAsync}
 *
 * `plugins:async:remove:all` - {@link PluginManager#removeAllAsync}
 *
 * `plugins:create:event:proxy` - {@link PluginManager#createEventProxy}
 *
 * `plugins:destroy:manager` - {@link PluginManager#destroy}
 *
 * `plugins:get:all:plugin:data` - {@link PluginManager#getAllPluginData}
 *
 * `plugins:get:extra:event:data` - {@link PluginManager#getExtraEventData}
 *
 * `plugins:get:method:names` - {@link PluginManager#getMethodNames}
 *
 * `plugins:get:options` - {@link PluginManager#getOptions}
 *
 * `plugins:get:plugin:data` - {@link PluginManager#getPluginData}
 *
 * `plugins:get:plugin:enabled` - {@link PluginManager#getPluginEnabled}
 *
 * `plugins:get:plugin:event:names` - {@link PluginManager#getPluginEventNames}
 *
 * `plugins:get:plugin:method:names` - {@link PluginManager#getPluginMethodNames}
 *
 * `plugins:get:plugin:names` - {@link PluginManager#getPluginNames}
 *
 * `plugins:get:plugin:options` - {@link PluginManager#getPluginOptions}
 *
 * `plugins:get:plugins:enabled` - {@link PluginManager#getPluginsEnabled}
 *
 * `plugins:get:plugins:by:event:name` - {@link PluginManager#getPluginsByEventName}
 *
 * `plugins:get:plugins:event:names` - {@link PluginManager#getPluginsEventNames}
 *
 * `plugins:has:method` - {@link PluginManager#hasMethod}
 *
 * `plugins:has:plugin` - {@link PluginManager#hasPlugin}
 *
 * `plugins:has:plugin:method` - {@link PluginManager#hasPluginMethod}
 *
 * `plugins:invoke` - {@link PluginManager#invoke}
 *
 * `plugins:is:valid:config` - {@link PluginManager#isValidConfig}
 *
 * `plugins:remove` - {@link PluginManager#remove}
 *
 * `plugins:remove:all` - {@link PluginManager#removeAll}
 *
 * `plugins:set:extra:event:data` - {@link PluginManager#setExtraEventData}
 *
 * `plugins:set:options` - {@link PluginManager#setOptions}
 *
 * `plugins:set:plugin:enabled` - {@link PluginManager#setPluginEnabled}
 *
 * `plugins:set:plugins:enabled` - {@link PluginManager#setPluginsEnabled}
 *
 * `plugins:sync:invoke` - {@link PluginManager#invokeSync}
 *
 * `plugins:sync:invoke:event` - {@link PluginManager#invokeSyncEvent}
 *
 * Automatically when a plugin is loaded and unloaded respective callbacks `onPluginLoad` and `onPluginUnload` will
 * be attempted to be invoked on the plugin. This is an opportunity for the plugin to receive any associated eventbus
 * and wire itself into it. It should be noted that a protected proxy around the eventbus is passed to the plugins
 * such that when the plugin is removed automatically all events registered on the eventbus are cleaned up without
 * a plugin author needing to do this manually in the `onPluginUnload` callback. This solves any dangling event binding
 * issues.
 *
 * The plugin manager also supports asynchronous operation with the methods ending in `Async` along with event bindings
 * that include `async`. For asynchronous variations of `add`, `destroy`, and `remove` the lifecycle methods
 * `onPluginLoad` and `onPluginUnload` will be awaited on such that if a plugin returns a Promise or is an async method
 * then it must complete before execution continues. One can use Promises to interact with the plugin manager
 * asynchronously, but usage via async / await is recommended.
 *
 * If eventbus functionality is enabled it is important especially if using a process / global level eventbus such as
 * `backbone-esnext-eventbus` to call {@link PluginManager#destroy} to clean up all plugin eventbus resources and
 * the plugin manager event bindings.
 *
 * @see https://www.npmjs.com/package/backbone-esnext-events
 * @see https://www.npmjs.com/package/backbone-esnext-eventbus
 *
 * @example
 * import Events        from 'backbone-esnext-events';   // Imports the TyphonEvents class for local usage.
 * ::or alternatively::
 * import eventbus      from 'backbone-esnext-eventbus'; // Imports a global / process level eventbus.
 *
 * import PluginManager from 'typhonjs-plugin-manager';
 *
 * const pluginManager = new PluginManager({ eventbus });
 *
 * pluginManager.add({ name: 'an-npm-plugin-enabled-module' });
 * pluginManager.add({ name: 'my-local-module', target: './myModule.js' });
 *
 * // Let's say an-npm-plugin-enabled-module responds to 'cool:event' which returns 'true'.
 * // Let's say my-local-module responds to 'hot:event' which returns 'false'.
 * // Both of the plugin / modules will have 'onPluginLoaded' invoked with a proxy to the eventbus and any plugin
 * // options defined.
 *
 * // One can then use the eventbus functionality to invoke associated module / plugin methods even retrieving results.
 * assert(eventbus.triggerSync('cool:event') === true);
 * assert(eventbus.triggerSync('hot:event') === false);
 *
 * // One can also indirectly invoke any method of the plugin via:
 * eventbus.triggerSync('plugins:invoke:sync:event', 'aCoolMethod'); // Any plugin with a method named `aCoolMethod` is invoked.
 * eventbus.triggerSync('plugins:invoke:sync:event', 'aCoolMethod', {}, {}, 'an-npm-plugin-enabled-module'); // specific invocation.
 *
 * // The 3rd parameter will make a copy of the hash and the 4th defines a pass through object hash sending a single
 * // event / object hash to the invoked method.
 *
 * // -----------------------
 *
 * // Given that `backbone-esnext-eventbus` defines a global / process level eventbus you can import it in an entirely
 * // different file or even NPM module and invoke methods of loaded plugins like this:
 *
 * import eventbus from 'backbone-esnext-eventbus';
 *
 * eventbus.triggerSync('plugins:invoke', 'aCoolMethod'); // Any plugin with a method named `aCoolMethod` is invoked.
 *
 * assert(eventbus.triggerSync('cool:event') === true);
 *
 * eventbus.trigger('plugins:remove', 'an-npm-plugin-enabled-module'); // Removes the plugin and unregisters events.
 *
 * assert(eventbus.triggerSync('cool:event') === true); // Will now fail!
 *
 * // In this case though when using the global eventbus be mindful to always call `pluginManager.destroy()` in the main
 * // thread of execution scope to remove all plugins and the plugin manager event bindings!
 */
export default class PluginManager
{
   /**
    * Instantiates PluginManager
    *
    * @param {object}   [options] - Provides various configuration options:
    *
    * @param {TyphonEvents}   [options.eventbus] - An instance of 'backbone-esnext-events' used as the plugin eventbus.
    *
    * @param {string}   [options.eventPrepend='plugin'] - A customized name to prepend PluginManager events on the
    *                                                     eventbus.
    *
    * @param {boolean}  [options.throwNoMethod=false] - If true then when a method fails to be invoked by any plugin
    *                                                   an exception will be thrown.
    *
    * @param {boolean}  [options.throwNoPlugin=false] - If true then when no plugin is matched to be invoked an
    *                                                   exception will be thrown.
    *
    *
    * @param {object}   [extraEventData] - Provides additional optional data to attach to PluginEvent callbacks.
    */
   constructor(options = {}, extraEventData = void 0)
   {
      if (typeof options !== 'object') { throw new TypeError(`'options' is not an object.`); }

      /**
       * Stores the plugins by name with an associated PluginEntry.
       * @type {Map<string, PluginEntry>}
       * @private
       */
      this._pluginMap = new Map();

      /**
       * Stores any associated eventbus.
       * @type {TyphonEvents}
       * @private
       */
      this._eventbus = null;

      /**
       * Stores any extra options / data to add to PluginEvent callbacks.
       * @type {Object}
       * @private
       */
      this._extraEventData = extraEventData;

      /**
       * Defines options for throwing exceptions. Turned off by default.
       * @type {PluginManagerOptions}
       * @private
       */
      this._options =
      {
         pluginsEnabled: true,
         noEventAdd: false,
         noEventDestroy: false,
         noEventOptions: true,
         noEventRemoval: false,
         throwNoMethod: false,
         throwNoPlugin: false
      };

      if (typeof options.eventbus === 'object') { this.setEventbus(options.eventbus, options.eventPrepend); }

      this.setOptions(options);
   }

   /**
    * Adds a plugin by the given configuration parameters. A plugin `name` is always required. If no other options
    * are provided then the `name` doubles as the NPM module / local file to load. The loading first checks for an
    * existing `instance` to use as the plugin. Then the `target` is chosen as the NPM module / local file to load.
    * By passing in `options` this will be stored and accessible to the plugin during all callbacks.
    *
    * @param {PluginConfig}   pluginConfig - Defines the plugin to load.
    *
    * @param {object}         [moduleData] - Optional object hash to associate with plugin.
    *
    * @returns {PluginData|undefined}
    */
   add(pluginConfig, moduleData)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof pluginConfig !== 'object') { throw new TypeError(`'pluginConfig' is not an 'object'.`); }

      if (typeof pluginConfig.name !== 'string')
      {
         throw new TypeError(`'pluginConfig.name' is not a 'string' for entry: ${JSON.stringify(pluginConfig)}.`);
      }

      if (typeof pluginConfig.target !== 'undefined' && typeof pluginConfig.target !== 'string')
      {
         throw new TypeError(`'pluginConfig.target' is not a string for entry: ${JSON.stringify(pluginConfig)}.`);
      }

      if (typeof pluginConfig.options !== 'undefined' && typeof pluginConfig.options !== 'object')
      {
         throw new TypeError(`'pluginConfig.options' is not an 'object' for entry: ${JSON.stringify(pluginConfig)}.`);
      }

      if (typeof moduleData !== 'undefined' && typeof moduleData !== 'object')
      {
         throw new TypeError(`'moduleData' is not an 'object' for entry: ${JSON.stringify(pluginConfig)}.`);
      }

      // If a plugin with the same name already exists post a warning and exit early.
      if (this._pluginMap.has(pluginConfig.name))
      {
         // Please note that a plugin or other logger must be setup on the associated eventbus.
         if (this._eventbus !== null && typeof this._eventbus !== 'undefined')
         {
            this._eventbus.trigger('log:warn', `A plugin already exists with name: ${pluginConfig.name}.`);
         }

         return void 0;
      }

      let instance, target, type;

      // Use an existing instance of a plugin; a static class is assumed when instance is a function.
      if (typeof pluginConfig.instance === 'object' || typeof pluginConfig.instance === 'function')
      {
         instance = pluginConfig.instance;

         target = pluginConfig.name;

         type = 'instance';
      }
      else
      {
         // If a target is defined use it instead of the name.
         target = pluginConfig.target || pluginConfig.name;

         if (target.match(/^[.\/\\]/))
         {
            instance = require(path.resolve(target)); // eslint-disable global-require

            type = 'require-path';
         }
         else
         {
            instance = require(target);               // eslint-disable global-require

            type = 'require-module';
         }
      }

      // Create an object hash with data describing the plugin, manager, and any extra module data.
      const pluginData = JSON.parse(JSON.stringify(
      {
         manager:
         {
            eventPrepend: this._eventPrepend
         },

         module: moduleData || {},

         plugin:
         {
            name: pluginConfig.name,
            scopedName: `${this._eventPrepend}:${pluginConfig.name}`,
            target,
            targetEscaped: PluginEntry.escape(target),
            type,
            options: pluginConfig.options || {}
         }
      }));

      ObjectUtil.deepFreeze(pluginData, ['eventPrepend', 'scopedName']);

      const eventProxy = this._eventbus !== null && typeof this._eventbus !== 'undefined' ?
       new EventProxy(this._eventbus) : void 0;

      const entry = new PluginEntry(pluginConfig.name, pluginData, instance, eventProxy);

      this._pluginMap.set(pluginConfig.name, entry);

      // Invoke private module method which allows skipping optional error checking.
      s_INVOKE_SYNC_EVENTS('onPluginLoad', {}, {}, this._extraEventData, pluginConfig.name, this._pluginMap,
       this._options, false);

      // Invoke `typhonjs:plugin:manager:plugin:added` allowing external code to react to plugin addition.
      if (this._eventbus)
      {
         this._eventbus.trigger(`typhonjs:plugin:manager:plugin:added`, pluginData);
      }

      return pluginData;
   }

   /**
    * Adds a plugin by the given configuration parameters. A plugin `name` is always required. If no other options
    * are provided then the `name` doubles as the NPM module / local file to load. The loading first checks for an
    * existing `instance` to use as the plugin. Then the `target` is chosen as the NPM module / local file to load.
    * By passing in `options` this will be stored and accessible to the plugin during all callbacks.
    *
    * @param {PluginConfig}   pluginConfig - Defines the plugin to load.
    *
    * @param {object}         [moduleData] - Optional object hash to associate with plugin.
    *
    * @returns {Promise<PluginData|undefined>}
    */
   async addAsync(pluginConfig, moduleData)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof pluginConfig !== 'object') { throw new TypeError(`'pluginConfig' is not an 'object'.`); }

      if (typeof pluginConfig.name !== 'string')
      {
         throw new TypeError(`'pluginConfig.name' is not a 'string' for entry: ${JSON.stringify(pluginConfig)}.`);
      }

      if (typeof pluginConfig.target !== 'undefined' && typeof pluginConfig.target !== 'string')
      {
         throw new TypeError(`'pluginConfig.target' is not a string for entry: ${JSON.stringify(pluginConfig)}.`);
      }

      if (typeof pluginConfig.options !== 'undefined' && typeof pluginConfig.options !== 'object')
      {
         throw new TypeError(`'pluginConfig.options' is not an 'object' for entry: ${JSON.stringify(pluginConfig)}.`);
      }

      if (typeof moduleData !== 'undefined' && typeof moduleData !== 'object')
      {
         throw new TypeError(`'moduleData' is not an 'object' for entry: ${JSON.stringify(pluginConfig)}.`);
      }

      // If a plugin with the same name already exists post a warning and exit early.
      if (this._pluginMap.has(pluginConfig.name))
      {
         // Please note that a plugin or other logger must be setup on the associated eventbus.
         if (this._eventbus !== null && typeof this._eventbus !== 'undefined')
         {
            this._eventbus.trigger('log:warn', `A plugin already exists with name: ${pluginConfig.name}.`);
         }

         return void 0;
      }

      let instance, target, type;

      // Use an existing instance of a plugin; a static class is assumed when instance is a function.
      if (typeof pluginConfig.instance === 'object' || typeof pluginConfig.instance === 'function')
      {
         instance = pluginConfig.instance;

         target = pluginConfig.name;

         type = 'instance';
      }
      else
      {
         // If a target is defined use it instead of the name.
         target = pluginConfig.target || pluginConfig.name;

         if (target.match(/^[.\/\\]/))
         {
            instance = require(path.resolve(target)); // eslint-disable global-require

            type = 'require-path';
         }
         else
         {
            instance = require(target);               // eslint-disable global-require

            type = 'require-module';
         }
      }

      // Create an object hash with data describing the plugin, manager, and any extra module data.
      const pluginData = JSON.parse(JSON.stringify(
      {
         manager:
         {
            eventPrepend: this._eventPrepend
         },

         module: moduleData || {},

         plugin:
         {
            name: pluginConfig.name,
            scopedName: `${this._eventPrepend}:${pluginConfig.name}`,
            target,
            targetEscaped: PluginEntry.escape(target),
            type,
            options: pluginConfig.options || {}
         }
      }));

      ObjectUtil.deepFreeze(pluginData, ['eventPrepend', 'scopedName']);

      const eventProxy = this._eventbus !== null && typeof this._eventbus !== 'undefined' ?
       new EventProxy(this._eventbus) : void 0;

      const entry = new PluginEntry(pluginConfig.name, pluginData, instance, eventProxy);

      this._pluginMap.set(pluginConfig.name, entry);

      // Invoke private module method which allows skipping optional error checking.
      await s_INVOKE_ASYNC_EVENTS('onPluginLoad', {}, {}, this._extraEventData, pluginConfig.name, this._pluginMap,
       this._options, false);

      // Invoke `typhonjs:plugin:manager:plugin:added` allowing external code to react to plugin addition.
      if (this._eventbus)
      {
         await this._eventbus.triggerAsync(`typhonjs:plugin:manager:plugin:added`, pluginData);
      }

      return pluginData;
   }

   /**
    * Initializes multiple plugins in a single call.
    *
    * @param {Array<PluginConfig>}  pluginConfigs - An array of plugin config object hash entries.
    *
    * @param {object}               [moduleData] - Optional object hash to associate with all plugins.
    *
    * @returns {Array<PluginData>}
    */
   addAll(pluginConfigs = [], moduleData)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (!Array.isArray(pluginConfigs)) { throw new TypeError(`'plugins' is not an array.`); }

      const pluginsData = [];

      for (const pluginConfig of pluginConfigs)
      {
         const result = this.add(pluginConfig, moduleData);

         if (result) { pluginsData.push(result); }
      }

      return pluginsData;
   }

   /**
    * Initializes multiple plugins in a single call.
    *
    * @param {Array<PluginConfig>}  pluginConfigs - An array of plugin config object hash entries.
    *
    * @param {object}               [moduleData] - Optional object hash to associate with all plugins.
    *
    * @returns {Promise<Array<PluginData>>}
    */
   addAllAsync(pluginConfigs = [], moduleData)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (!Array.isArray(pluginConfigs)) { throw new TypeError(`'plugins' is not an array.`); }

      const pluginsData = [];

      for (const pluginConfig of pluginConfigs)
      {
         const result = this.addAsync(pluginConfig, moduleData);

         if (result) { pluginsData.push(result); }
      }

      return Promise.all(pluginsData);
   }

   /**
    * Provides the eventbus callback which may prevent addition if optional `noEventAdd` is enabled. This disables
    * the ability for plugins to be added via events preventing any external code adding plugins in this manner.
    *
    * @param {PluginConfig}   pluginConfig - Defines the plugin to load.
    *
    * @param {object}         [moduleData] - Optional object hash to associate with all plugins.
    *
    * @returns {PluginData|undefined} - Operation success.
    * @private
    */
   _addEventbus(pluginConfig, moduleData)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      return !this._options.noEventAdd ? this.add(pluginConfig, moduleData) : void 0;
   }

   /**
    * Provides the eventbus callback which may prevent addition if optional `noEventAdd` is enabled. This disables
    * the ability for plugins to be added via events preventing any external code adding plugins in this manner.
    *
    * @param {PluginConfig}   pluginConfig - Defines the plugin to load.
    *
    * @param {object}         [moduleData] - Optional object hash to associate with all plugins.
    *
    * @returns {Promise<PluginData|undefined>} - Operation success.
    * @private
    */
   _addEventbusAsync(pluginConfig, moduleData)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      return !this._options.noEventAdd ? this.addAsync(pluginConfig, moduleData) : Promise.resolve(void 0);
   }

   /**
    * Provides the eventbus callback which may prevent addition if optional `noEventAdd` is enabled. This disables
    * the ability for plugins to be added via events preventing any external code adding plugins in this manner.
    *
    * @param {Array<PluginConfig>}  pluginConfigs - An array of plugin config object hash entries.
    *
    * @param {object}               [moduleData] - Optional object hash to associate with all plugins.
    *
    * @returns {Array<PluginData>}
    * @private
    */
   _addAllEventbus(pluginConfigs, moduleData)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (!this._options.noEventAdd) { return this.addAll(pluginConfigs, moduleData); }
   }

   /**
    * Provides the eventbus callback which may prevent addition if optional `noEventAdd` is enabled. This disables
    * the ability for plugins to be added via events preventing any external code adding plugins in this manner.
    *
    * @param {Array<PluginConfig>}  pluginConfigs - An array of plugin config object hash entries.
    *
    * @param {object}               [moduleData] - Optional object hash to associate with all plugins.
    *
    * @returns {Promise<Array<PluginData>>}
    * @private
    */
   _addAllEventbusAsync(pluginConfigs, moduleData)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (!this._options.noEventAdd) { return this.addAllAsync(pluginConfigs, moduleData); }
   }

   /**
    * If an eventbus is assigned to this plugin manager then a new EventProxy wrapping this eventbus is returned.
     *
    * @returns {EventProxy}
    */
   createEventProxy()
   {
      return this._eventbus !== null ? new EventProxy(this._eventbus) : void 0;
   }

   /**
    * Destroys all managed plugins after unloading them.
    */
   destroy()
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      this.removeAll();

      if (this._eventbus !== null && typeof this._eventbus !== 'undefined')
      {
         this._eventbus.off(`${this._eventPrepend}:add`, this._addEventbus, this);
         this._eventbus.off(`${this._eventPrepend}:add:all`, this._addAllEventbus, this);
         this._eventbus.off(`${this._eventPrepend}:async:add`, this._addEventbusAsync, this);
         this._eventbus.off(`${this._eventPrepend}:async:add:all`, this._addAllEventbusAsync, this);
         this._eventbus.off(`${this._eventPrepend}:async:destroy:manager`, this._destroyEventbusAsync, this);
         this._eventbus.off(`${this._eventPrepend}:async:invoke`, this.invokeAsync, this);
         this._eventbus.off(`${this._eventPrepend}:async:invoke:event`, this.invokeAsyncEvent, this);
         this._eventbus.off(`${this._eventPrepend}:async:remove`, this._removeEventbusAsync, this);
         this._eventbus.off(`${this._eventPrepend}:async:remove:all`, this._removeAllEventbusAsync, this);
         this._eventbus.off(`${this._eventPrepend}:create:event:proxy`, this.createEventProxy, this);
         this._eventbus.off(`${this._eventPrepend}:destroy:manager`, this._destroyEventbus, this);
         this._eventbus.off(`${this._eventPrepend}:get:all:plugin:data`, this.getAllPluginData, this);
         this._eventbus.off(`${this._eventPrepend}:get:extra:event:data`, this.getExtraEventData, this);
         this._eventbus.off(`${this._eventPrepend}:get:method:names`, this.getMethodNames, this);
         this._eventbus.off(`${this._eventPrepend}:get:options`, this.getOptions, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugin:enabled`, this.getPluginEnabled, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugin:data`, this.getPluginData, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugin:event:names`, this.getPluginEventNames, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugin:method:names`, this.getPluginMethodNames, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugin:names`, this.getPluginNames, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugin:options`, this.getPluginOptions, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugins:enabled`, this.getPluginsEnabled, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugins:by:event:name`, this.getPluginsByEventName, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugins:event:names`, this.getPluginsEventNames, this);
         this._eventbus.off(`${this._eventPrepend}:has:method`, this.hasMethod, this);
         this._eventbus.off(`${this._eventPrepend}:has:plugin`, this.hasPlugin, this);
         this._eventbus.off(`${this._eventPrepend}:has:plugin:method`, this.hasPluginMethod, this);
         this._eventbus.off(`${this._eventPrepend}:invoke`, this.invoke, this);
         this._eventbus.off(`${this._eventPrepend}:is:valid:config`, this.isValidConfig, this);
         this._eventbus.off(`${this._eventPrepend}:remove`, this._removeEventbus, this);
         this._eventbus.off(`${this._eventPrepend}:remove:all`, this._removeAllEventbus, this);
         this._eventbus.off(`${this._eventPrepend}:set:extra:event:data`, this.setExtraEventData, this);
         this._eventbus.off(`${this._eventPrepend}:set:options`, this._setOptionsEventbus, this);
         this._eventbus.off(`${this._eventPrepend}:set:plugin:enabled`, this.setPluginEnabled, this);
         this._eventbus.off(`${this._eventPrepend}:set:plugins:enabled`, this.setPluginsEnabled, this);
         this._eventbus.off(`${this._eventPrepend}:sync:invoke`, this.invokeSync, this);
         this._eventbus.off(`${this._eventPrepend}:sync:invoke:event`, this.invokeSyncEvent, this);
      }

      this._pluginMap = null;
      this._eventbus = null;
   }

   /**
    * Destroys all managed plugins after unloading them.
    */
   async destroyAsync()
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      await this.removeAll();

      if (this._eventbus !== null && typeof this._eventbus !== 'undefined')
      {
         this._eventbus.off(`${this._eventPrepend}:add`, this._addEventbus, this);
         this._eventbus.off(`${this._eventPrepend}:add:all`, this._addAllEventbus, this);
         this._eventbus.off(`${this._eventPrepend}:async:add`, this._addEventbusAsync, this);
         this._eventbus.off(`${this._eventPrepend}:async:add:all`, this._addAllEventbusAsync, this);
         this._eventbus.off(`${this._eventPrepend}:async:destroy:manager`, this._destroyEventbusAsync, this);
         this._eventbus.off(`${this._eventPrepend}:async:invoke`, this.invokeAsync, this);
         this._eventbus.off(`${this._eventPrepend}:async:invoke:event`, this.invokeAsyncEvent, this);
         this._eventbus.off(`${this._eventPrepend}:async:remove`, this._removeEventbusAsync, this);
         this._eventbus.off(`${this._eventPrepend}:async:remove:all`, this._removeAllEventbusAsync, this);
         this._eventbus.off(`${this._eventPrepend}:create:event:proxy`, this.createEventProxy, this);
         this._eventbus.off(`${this._eventPrepend}:destroy:manager`, this._destroyEventbus, this);
         this._eventbus.off(`${this._eventPrepend}:get:all:plugin:data`, this.getAllPluginData, this);
         this._eventbus.off(`${this._eventPrepend}:get:extra:event:data`, this.getExtraEventData, this);
         this._eventbus.off(`${this._eventPrepend}:get:method:names`, this.getMethodNames, this);
         this._eventbus.off(`${this._eventPrepend}:get:options`, this.getOptions, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugin:enabled`, this.getPluginEnabled, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugin:data`, this.getPluginData, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugin:event:names`, this.getPluginEventNames, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugin:method:names`, this.getPluginMethodNames, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugin:names`, this.getPluginNames, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugin:options`, this.getPluginOptions, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugins:enabled`, this.getPluginsEnabled, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugins:by:event:name`, this.getPluginsByEventName, this);
         this._eventbus.off(`${this._eventPrepend}:get:plugins:event:names`, this.getPluginsEventNames, this);
         this._eventbus.off(`${this._eventPrepend}:has:method`, this.hasMethod, this);
         this._eventbus.off(`${this._eventPrepend}:has:plugin`, this.hasPlugin, this);
         this._eventbus.off(`${this._eventPrepend}:has:plugin:method`, this.hasPluginMethod, this);
         this._eventbus.off(`${this._eventPrepend}:invoke`, this.invoke, this);
         this._eventbus.off(`${this._eventPrepend}:is:valid:config`, this.isValidConfig, this);
         this._eventbus.off(`${this._eventPrepend}:remove`, this._removeEventbus, this);
         this._eventbus.off(`${this._eventPrepend}:remove:all`, this._removeAllEventbus, this);
         this._eventbus.off(`${this._eventPrepend}:set:extra:event:data`, this.setExtraEventData, this);
         this._eventbus.off(`${this._eventPrepend}:set:options`, this._setOptionsEventbus, this);
         this._eventbus.off(`${this._eventPrepend}:set:plugin:enabled`, this.setPluginEnabled, this);
         this._eventbus.off(`${this._eventPrepend}:set:plugins:enabled`, this.setPluginsEnabled, this);
         this._eventbus.off(`${this._eventPrepend}:sync:invoke`, this.invokeSync, this);
         this._eventbus.off(`${this._eventPrepend}:sync:invoke:event`, this.invokeSyncEvent, this);
      }

      this._pluginMap = null;
      this._eventbus = null;
   }

   /**
    * Provides the eventbus callback which may prevent plugin manager destruction if optional `noEventDestroy` is
    * enabled. This disables the ability for the plugin manager to be destroyed via events preventing any external
    * code removing plugins in this manner.
    *
    * @private
    */
   _destroyEventbus()
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (!this._options.noEventDestroy) { this.destroy(); }
   }

   /**
    * Provides the eventbus callback which may prevent plugin manager destruction if optional `noEventDestroy` is
    * enabled. This disables the ability for the plugin manager to be destroyed via events preventing any external
    * code removing plugins in this manner.
    *
    * @private
    */
   async _destroyEventbusAsync()
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (!this._options.noEventDestroy) { await this.destroyAsync(); }
   }

   /**
    * Returns the enabled state of a plugin.
    *
    * @param {string}   pluginName - Plugin name to set state.
    *
    * @returns {boolean} - Operation success.
    */
   getPluginEnabled(pluginName)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof pluginName !== 'string') { throw new TypeError(`'pluginName' is not a string.`); }

      const entry = this._pluginMap.get(pluginName);

      return entry instanceof PluginEntry && entry.enabled;
   }

   /**
    * Returns the event binding names registered on any associated plugin EventProxy.
    *
    * @param {string}   pluginName - Plugin name to set state.
    *
    * @returns {string[]} - Event binding names registered from the plugin.
    */
   getPluginEventNames(pluginName)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof pluginName !== 'string') { throw new TypeError(`'pluginName' is not a string.`); }

      const entry = this._pluginMap.get(pluginName);

      return entry instanceof PluginEntry && entry._eventProxy ? entry._eventProxy.getEventNames() : [];
   }

   /**
    * Returns the enabled state of a list of plugins.
    *
    * @param {Array<string>}  pluginNames - An array / iterable of plugin names.
    *
    * @returns {Array<{pluginName: string, enabled: boolean}>} A list of objects with plugin name and enabled state.
    */
   getPluginsEnabled(pluginNames)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      const results = [];

      for (const pluginName of pluginNames)
      {
         results.push({ pluginName, enabled: this.getPluginEnabled(pluginName) });
      }

      return results;
   }

   /**
    * Returns the event binding names registered from each plugin.
    *
    * @param {string|string[]} [nameOrList] - An array / iterable of plugin names.
    *
    * @returns {Array<{pluginName: string, events: string[]}>} A list of objects with plugin name and event binding
    *                                                          names registered from the plugin.
    */
   getPluginsEventNames(nameOrList)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof nameOrList === 'undefined') { nameOrList = this._pluginMap.keys(); }
      if (typeof nameOrList === 'string') { nameOrList = [nameOrList]; }

      const results = [];

      for (const pluginName of nameOrList)
      {
         results.push({ pluginName, events: this.getPluginEventNames(pluginName) });
      }

      return results;
   }

   /**
    * Returns the plugin names that registered the given event binding name.
    *
    * @param {string} eventName - An event name that plugins may have registered.
    *
    * @returns {Array<string[]>} A list of plugin names that has registered the given event name.
    */
   getPluginsByEventName(eventName)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof eventName !== 'string') { throw new TypeError(`'eventName' is not a 'string'.`); }

      const results = [];

      const pluginEventNames = this.getPluginsEventNames();

      for (const entry of pluginEventNames)
      {
         if (entry.events.indexOf(eventName) >= 0) { results.push(entry.pluginName); }
      }

      return results;
   }

   /**
    * Returns all plugin data or if a boolean is passed in will return plugin data by current enabled state.
    *
    * @param {boolean|undefined} enabled - If enabled is a boolean it will return plugins given their enabled state.
    *
    * @returns {Array<PluginData>}
    */
   getAllPluginData(enabled = void 0)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof enabled !== 'boolean' && typeof enabled !== 'undefined')
      {
         throw new TypeError(`'enabled' is not a 'boolean' or 'undefined'.`);
      }

      const results = [];

      // Return all plugin data if enabled is not defined.
      const allPlugins = typeof enabled === 'undefined';

      for (const entry of this._pluginMap.values())
      {
         if (allPlugins || entry.enabled === enabled)
         {
            results.push(this.getPluginData(entry.name));
         }
      }

      return results;
   }

   /**
    * Returns any associated eventbus.
    *
    * @returns {TyphonEvents|null}
    */
   getEventbus()
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      return this._eventbus;
   }

   /**
    * Returns any extra event data associated with PluginEvents.
    *
    * @returns {*}
    */
   getExtraEventData()
   {
      return this._extraEventData;
   }

   /**
    * Returns all method names or if a boolean is passed in will return method names for plugins by current enabled
    * state.
    *
    * @param {boolean|undefined} enabled - If enabled is a boolean it will return plugin methods names given their
    *                                      enabled state.
    *
    * @param {string|undefined}  pluginName - If a string then just this plugins methods names are returned.
    *
    * @returns {Array<string>}
    */
   getMethodNames(enabled = void 0, pluginName = void 0)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof enabled !== 'boolean' && typeof enabled !== 'undefined')
      {
         throw new TypeError(`'enabled' is not a 'boolean' or 'undefined'.`);
      }

      const results = {};
      const allEnabled = typeof enabled === 'undefined';
      const allNames = typeof pluginName === 'undefined';

      for (const entry of this._pluginMap.values())
      {
         if (entry.instance && (allEnabled || entry.enabled === enabled) && (allNames || entry.name === pluginName))
         {
            for (const name of s_GET_ALL_PROPERTY_NAMES(entry.instance))
            {
               // Skip any names that are not a function or are the constructor.
               if (entry.instance[name] instanceof Function && name !== 'constructor') { results[name] = true; }
            }
         }
      }

      return Object.keys(results);
   }

   /**
    * Returns a copy of the plugin manager options.
    *
    * @returns {PluginManagerOptions}
    */
   getOptions()
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      return JSON.parse(JSON.stringify(this._options));
   }

   /**
    * Gets the plugin data for a plugin by name.
    *
    * @param {string}   pluginName - A plugin name.
    *
    * @returns {PluginData|undefined}
    */
   getPluginData(pluginName)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof pluginName !== 'string') { throw new TypeError(`'pluginName' is not a string.`); }

      const entry = this._pluginMap.get(pluginName);

      if (entry instanceof PluginEntry)
      {
         return JSON.parse(JSON.stringify(entry.data));
      }

      return void 0;
   }

   /**
    * Returns all plugin names or if a boolean is passed in will return plugin names by current enabled state.
    *
    * @param {boolean|undefined} enabled - If enabled is a boolean it will return plugins given their enabled state.
    *
    * @returns {Array<{plugin: string, method: string}>}
    */
   getPluginMethodNames(enabled = void 0)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof enabled !== 'boolean' && typeof enabled !== 'undefined')
      {
         throw new TypeError(`'enabled' is not a 'boolean' or 'undefined'.`);
      }

      const results = [];
      const allPlugins = typeof enabled === 'undefined';

      for (const entry of this._pluginMap.values())
      {
         if (entry.instance && (allPlugins || entry.enabled === enabled))
         {
            for (const name of s_GET_ALL_PROPERTY_NAMES(entry.instance))
            {
               // Skip any names that are not a function or are the constructor.
               if (entry.instance[name] instanceof Function && name !== 'constructor')
               {
                  results.push({ plugin: entry.name, method: name });
               }
            }
         }
      }

      return results;
   }

   /**
    * Returns all plugin names or if a boolean is passed in will return plugin names by current enabled state.
    *
    * @param {boolean|undefined} enabled - If enabled is a boolean it will return plugins given their enabled state.
    *
    * @returns {Array<string>}
    */
   getPluginNames(enabled = void 0)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof enabled !== 'boolean' && typeof enabled !== 'undefined')
      {
         throw new TypeError(`'enabled' is not a 'boolean' or 'undefined'.`);
      }

      // Return all plugin names if enabled is not defined.
      if (enabled === void 0) { return Array.from(this._pluginMap.keys()); }

      const results = [];

      for (const entry of this._pluginMap.values())
      {
         if (entry.enabled === enabled) { results.push(entry.name); }
      }

      return results;
   }

   /**
    * Returns a copy of the given plugin options.
    *
    * @param {string}   pluginName - Plugin name to retrieve.
    *
    * @returns {*}
    */
   getPluginOptions(pluginName)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof pluginName !== 'string') { throw new TypeError(`'pluginName' is not a string.`); }

      let result;

      const entry = this._pluginMap.get(pluginName);

      if (entry instanceof PluginEntry) { result = JSON.parse(JSON.stringify(entry.data.plugin.options)); }

      return result;
   }

   /**
    * Returns true if there is at least one plugin loaded with the given method name.
    *
    * @param {string}   methodName - Method name to test.
    *
    * @returns {boolean} - True method is found.
    */
   hasMethod(methodName)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof methodName !== 'string') { throw new TypeError(`'methodName' is not a string.`); }

      for (const plugin of this._pluginMap.values())
      {
         if (typeof plugin.instance[methodName] === 'function') { return true; }
      }

      return false;
   }

   /**
    * Returns true if there is a plugin loaded with the given plugin name.
    *
    * @param {string}   pluginName - Plugin name to test.
    *
    * @returns {boolean} - True if a plugin exists.
    */
   hasPlugin(pluginName)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof pluginName !== 'string') { throw new TypeError(`'pluginName' is not a string.`); }

      return this._pluginMap.has(pluginName);
   }

   /**
    * Returns true if there is a plugin loaded with the given plugin name that also has a method with the given
    * method name.
    *
    * @param {string}   pluginName - Plugin name to test.
    * @param {string}   methodName - Method name to test.
    *
    * @returns {boolean} - True if a plugin and method exists.
    */
   hasPluginMethod(pluginName, methodName)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof pluginName !== 'string') { throw new TypeError(`'pluginName' is not a string.`); }
      if (typeof methodName !== 'string') { throw new TypeError(`'methodName' is not a string.`); }

      const plugin = this._pluginMap.get(pluginName);

      return plugin instanceof PluginEntry && typeof plugin[methodName] === 'function';
   }

   /**
    * This dispatch method simply invokes any plugin targets for the given methodName..
    *
    * @param {string}               methodName - Method name to invoke.
    *
    * @param {*|Array<*>}           [args] - Optional arguments. An array will be spread as multiple arguments.
    *
    * @param {string|Array<string>} [nameOrList] - An optional plugin name or array / iterable of plugin names to
    *                                              invoke.
    */
   invoke(methodName, args = void 0, nameOrList = void 0)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof methodName !== 'string') { throw new TypeError(`'methodName' is not a string.`); }

      if (typeof nameOrList === 'undefined') { nameOrList = this._pluginMap.keys(); }

      if (typeof nameOrList !== 'string' && !Array.isArray(nameOrList) &&
       typeof nameOrList[Symbol.iterator] !== 'function')
      {
         throw new TypeError(`'nameOrList' is not a string, array, or iterator.`);
      }

      // Track if a plugin method is invoked.
      let hasMethod = false;
      let hasPlugin = false;

      // Early out if plugins are not enabled.
      if (!this._options.pluginsEnabled) { return; }

      if (typeof nameOrList === 'string')
      {
         const plugin = this._pluginMap.get(nameOrList);

         if (plugin instanceof PluginEntry && plugin.enabled && plugin.instance)
         {
            hasPlugin = true;

            if (typeof plugin.instance[methodName] === 'function')
            {
               Array.isArray(args) ? plugin.instance[methodName](...args) : plugin.instance[methodName](args);

               hasMethod = true;
            }
         }
      }
      else
      {
         for (const name of nameOrList)
         {
            const plugin = this._pluginMap.get(name);

            if (plugin instanceof PluginEntry && plugin.enabled && plugin.instance)
            {
               hasPlugin = true;

               if (typeof plugin.instance[methodName] === 'function')
               {
                  Array.isArray(args) ? plugin.instance[methodName](...args) : plugin.instance[methodName](args);

                  hasMethod = true;
               }
            }
         }
      }

      if (this._options.throwNoPlugin && !hasPlugin)
      {
         throw new Error(`PluginManager failed to find any target plugins.`);
      }

      if (this._options.throwNoMethod && !hasMethod)
      {
         throw new Error(`PluginManager failed to invoke '${methodName}'.`);
      }
   }

   /**
    * This dispatch method uses ES6 Promises and adds any returned results to an array which is added to a Promise.all
    * construction which passes back a Promise which waits until all Promises complete. Any target invoked may return a
    * Promise or any result. This is very useful to use for any asynchronous operations.
    *
    * @param {string}               methodName - Method name to invoke.
    *
    * @param {*|Array<*>}           [args] - Optional arguments. An array will be spread as multiple arguments.
    *
    * @param {string|Array<string>} [nameOrList] - An optional plugin name or array / iterable of plugin names to
    *                                              invoke.
    *
    * @returns {Promise<*|Array<*>>}
    */
   invokeAsync(methodName, args = void 0, nameOrList = void 0)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof methodName !== 'string') { throw new TypeError(`'methodName' is not a string.`); }

      if (typeof nameOrList === 'undefined') { nameOrList = this._pluginMap.keys(); }

      if (typeof nameOrList !== 'string' && !Array.isArray(nameOrList) &&
       typeof nameOrList[Symbol.iterator] !== 'function')
      {
         throw new TypeError(`'nameOrList' is not a string, array, or iterator.`);
      }

      // Track if a plugin method is invoked.
      let hasMethod = false;
      let hasPlugin = false;

      // Capture results.
      let result = void 0;
      const results = [];

      // Early out if plugins are not enabled.
      if (!this._options.pluginsEnabled) { return result; }

      try
      {
         if (typeof nameOrList === 'string')
         {
            const plugin = this._pluginMap.get(nameOrList);

            if (plugin instanceof PluginEntry && plugin.enabled && plugin.instance)
            {
               hasPlugin = true;

               if (typeof plugin.instance[methodName] === 'function')
               {
                  result = Array.isArray(args) ? plugin.instance[methodName](...args) :
                   plugin.instance[methodName](args);

                  // If we received a valid result return immediately.
                  if (result !== null || typeof result !== 'undefined') { results.push(result); }

                  hasMethod = true;
               }
            }
         }
         else
         {
            for (const name of nameOrList)
            {
               const plugin = this._pluginMap.get(name);

               if (plugin instanceof PluginEntry && plugin.enabled && plugin.instance)
               {
                  hasPlugin = true;

                  if (typeof plugin.instance[methodName] === 'function')
                  {
                     result = Array.isArray(args) ? plugin.instance[methodName](...args) :
                      plugin.instance[methodName](args);

                     // If we received a valid result return immediately.
                     if (result !== null || typeof result !== 'undefined') { results.push(result); }

                     hasMethod = true;
                  }
               }
            }
         }

         if (this._options.throwNoPlugin && !hasPlugin)
         {
            return Promise.reject(new Error(`PluginManager failed to find any target plugins.`));
         }

         if (this._options.throwNoMethod && !hasMethod)
         {
            return Promise.reject(new Error(`PluginManager failed to invoke '${methodName}'.`));
         }
      }
      catch (error)
      {
         return Promise.reject(error);
      }

      // If there are multiple results then use Promise.all otherwise Promise.resolve.
      return results.length > 1 ? Promise.all(results) : Promise.resolve(result);
   }

   /**
    * This dispatch method synchronously passes to and returns from any invoked targets a PluginEvent.
    *
    * @param {string}               methodName - Method name to invoke.
    *
    * @param {object}               [copyProps={}] - plugin event object.
    *
    * @param {object}               [passthruProps={}] - if true, event has plugin option.
    *
    * @param {string|Array<string>} [nameOrList] - An optional plugin name or array / iterable of plugin names to
    *                                              invoke.
    *
    * @returns {Promise<PluginEvent>}
    */
   invokeAsyncEvent(methodName, copyProps = {}, passthruProps = {}, nameOrList = void 0)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof nameOrList === 'undefined') { nameOrList = this._pluginMap.keys(); }

      // Early out if plugins are not enabled.
      if (!this._options.pluginsEnabled) { return Promise.resolve(); }

      // Invokes the private internal async events method with optional error checking enabled.
      return s_INVOKE_ASYNC_EVENTS(methodName, copyProps, passthruProps, this._extraEventData, nameOrList,
       this._pluginMap, this._options);
   }

   /**
    * This dispatch method synchronously passes back a single value or an array with all results returned by any
    * invoked targets.
    *
    * @param {string}               methodName - Method name to invoke.
    *
    * @param {*|Array<*>}           [args] - Optional arguments. An array will be spread as multiple arguments.
    *
    * @param {string|Array<string>} [nameOrList] - An optional plugin name or array / iterable of plugin names to
    *                                              invoke.
    *
    * @returns {*|Array<*>}
    */
   invokeSync(methodName, args = void 0, nameOrList = void 0)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof methodName !== 'string') { throw new TypeError(`'methodName' is not a string.`); }

      if (typeof nameOrList === 'undefined') { nameOrList = this._pluginMap.keys(); }

      if (typeof nameOrList !== 'string' && !Array.isArray(nameOrList) &&
       typeof nameOrList[Symbol.iterator] !== 'function')
      {
         throw new TypeError(`'nameOrList' is not a string, array, or iterator.`);
      }

      // Track if a plugin method is invoked.
      let hasMethod = false;
      let hasPlugin = false;

      // Capture results.
      let result = void 0;
      const results = [];

      // Early out if plugins are not enabled.
      if (!this._options.pluginsEnabled) { return result; }

      if (typeof nameOrList === 'string')
      {
         const plugin = this._pluginMap.get(nameOrList);

         if (plugin instanceof PluginEntry && plugin.enabled && plugin.instance)
         {
            hasPlugin = true;

            if (typeof plugin.instance[methodName] === 'function')
            {
               result = Array.isArray(args) ? plugin.instance[methodName](...args) : plugin.instance[methodName](args);

               // If we received a valid result return immediately.
               if (result !== null || typeof result !== 'undefined') { results.push(result); }

               hasMethod = true;
            }
         }
      }
      else
      {
         for (const name of nameOrList)
         {
            const plugin = this._pluginMap.get(name);

            if (plugin instanceof PluginEntry && plugin.enabled && plugin.instance)
            {
               hasPlugin = true;

               if (typeof plugin.instance[methodName] === 'function')
               {
                  result = Array.isArray(args) ? plugin.instance[methodName](...args) :
                   plugin.instance[methodName](args);

                  // If we received a valid result return immediately.
                  if (result !== null || typeof result !== 'undefined') { results.push(result); }

                  hasMethod = true;
               }
            }
         }
      }

      if (this._options.throwNoPlugin && !hasPlugin)
      {
         throw new Error(`PluginManager failed to find any target plugins.`);
      }

      if (this._options.throwNoMethod && !hasMethod)
      {
         throw new Error(`PluginManager failed to invoke '${methodName}'.`);
      }

      // Return the results array if there are more than one or just a single result.
      return results.length > 1 ? results : result;
   }

   /**
    * This dispatch method synchronously passes to and returns from any invoked targets a PluginEvent.
    *
    * @param {string}               methodName - Method name to invoke.
    *
    * @param {object}               [copyProps={}] - plugin event object.
    *
    * @param {object}               [passthruProps={}] - if true, event has plugin option.
    *
    * @param {string|Array<string>} [nameOrList] - An optional plugin name or array / iterable of plugin names to
    *                                              invoke.
    *
    * @returns {PluginEvent|undefined}
    */
   invokeSyncEvent(methodName, copyProps = {}, passthruProps = {}, nameOrList = void 0)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof nameOrList === 'undefined') { nameOrList = this._pluginMap.keys(); }

      // Early out if plugins are not enabled.
      if (!this._options.pluginsEnabled) { return void 0; }

      // Invokes the private internal sync events method with optional error checking enabled.
      return s_INVOKE_SYNC_EVENTS(methodName, copyProps, passthruProps, this._extraEventData, nameOrList,
       this._pluginMap, this._options);
   }

   /**
    * Performs validation of a PluginConfig.
    *
    * @param {PluginConfig}   pluginConfig - A PluginConfig to validate.
    *
    * @returns {boolean} True if the given PluginConfig is valid.
    */
   isValidConfig(pluginConfig)
   {
      if (typeof pluginConfig !== 'object') { return false; }

      if (typeof pluginConfig.name !== 'string') { return false; }

      if (typeof pluginConfig.target !== 'undefined' && typeof pluginConfig.target !== 'string') { return false; }

      if (typeof pluginConfig.options !== 'undefined' && typeof pluginConfig.options !== 'object') { return false; }

      return true;
   }

   /**
    * Sets the eventbus associated with this plugin manager. If any previous eventbus was associated all plugin manager
    * events will be removed then added to the new eventbus. If there are any existing plugins being managed their
    * events will be removed from the old eventbus and then `onPluginLoad` will be called with the new eventbus.
    *
    * @param {TyphonEvents}   targetEventbus - The target eventbus to associate.
    *
    * @param {string}         [eventPrepend='plugins'] - An optional string to prepend to all of the event binding
    *                                                    targets.
    *
    * @returns {PluginManager}
    */
   setEventbus(targetEventbus, eventPrepend = 'plugins')
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof targetEventbus !== 'object') { throw new TypeError(`'targetEventbus' is not an 'object'.`); }
      if (typeof eventPrepend !== 'string') { throw new TypeError(`'eventPrepend' is not a 'string'.`); }

      // Early escape if the targetEventbus is the same as the current eventbus.
      if (targetEventbus === this._eventbus) { return this; }

      const oldPrepend = this._eventPrepend;

      /**
       * Stores the prepend string for eventbus registration.
       * @type {string}
       * @private
       */
      this._eventPrepend = eventPrepend;

      // Unload and reload any existing plugins from the old eventbus to the target eventbus.
      if (this._pluginMap.size > 0)
      {
         // Invoke private module method which allows skipping optional error checking.
         s_INVOKE_SYNC_EVENTS('onPluginUnload', {}, {}, this._extraEventData, this._pluginMap.keys(),
          this._pluginMap, this._options, false);

         for (const entry of this._pluginMap.values())
         {
            // Automatically remove any potential reference to a stored event proxy instance.
            try
            {
               entry.instance._eventbus = void 0;
            }
            catch (err) { /* nop */ }

            entry.data.manager.eventPrepend = eventPrepend;
            entry.data.plugin.scopedName = `${eventPrepend}:${entry.name}`;

            if (entry.eventProxy instanceof EventProxy) { entry.eventProxy.destroy(); }

            entry.eventProxy = new EventProxy(targetEventbus);
         }

         // Invoke private module method which allows skipping optional error checking.
         s_INVOKE_SYNC_EVENTS('onPluginLoad', {}, {}, this._extraEventData, this._pluginMap.keys(),
          this._pluginMap, this._options, false);

         for (const entry of this._pluginMap.values())
         {
            // Invoke `typhonjs:plugin:manager:eventbus:changed` allowing external code to react to plugin
            // changing eventbus.
            if (this._eventbus)
            {
               this._eventbus.trigger(`typhonjs:plugin:manager:eventbus:changed`, Object.assign({
                  oldEventbus: this._eventbus,
                  oldManagerEventPrepend: oldPrepend,
                  oldScopedName: `${oldPrepend}:${entry.name}`,
                  newEventbus: targetEventbus,
                  newManagerEventPrepend: eventPrepend,
                  newScopedName: `${eventPrepend}:${entry.name}`
               }, JSON.parse(JSON.stringify(entry.data))));
            }
         }
      }

      if (this._eventbus !== null)
      {
         this._eventbus.off(`${oldPrepend}:add`, this._addEventbus, this);
         this._eventbus.off(`${oldPrepend}:add:all`, this._addAllEventbus, this);
         this._eventbus.off(`${oldPrepend}:async:add`, this._addEventbusAsync, this);
         this._eventbus.off(`${oldPrepend}:async:add:all`, this._addAllEventbusAsync, this);
         this._eventbus.off(`${oldPrepend}:async:destroy:manager`, this._destroyEventbusAsync, this);
         this._eventbus.off(`${oldPrepend}:async:invoke`, this.invokeAsync, this);
         this._eventbus.off(`${oldPrepend}:async:invoke:event`, this.invokeAsyncEvent, this);
         this._eventbus.off(`${oldPrepend}:async:remove`, this._removeEventbusAsync, this);
         this._eventbus.off(`${oldPrepend}:async:remove:all`, this._removeAllEventbusAsync, this);
         this._eventbus.off(`${oldPrepend}:create:event:proxy`, this.createEventProxy, this);
         this._eventbus.off(`${oldPrepend}:destroy:manager`, this._destroyEventbus, this);
         this._eventbus.off(`${oldPrepend}:get:all:plugin:data`, this.getAllPluginData, this);
         this._eventbus.off(`${oldPrepend}:get:extra:event:data`, this.getExtraEventData, this);
         this._eventbus.off(`${oldPrepend}:get:method:names`, this.getMethodNames, this);
         this._eventbus.off(`${oldPrepend}:get:options`, this.getOptions, this);
         this._eventbus.off(`${oldPrepend}:get:plugin:enabled`, this.getPluginEnabled, this);
         this._eventbus.off(`${oldPrepend}:get:plugin:data`, this.getPluginData, this);
         this._eventbus.off(`${oldPrepend}:get:plugin:event:names`, this.getPluginEventNames, this);
         this._eventbus.off(`${oldPrepend}:get:plugin:method:names`, this.getPluginMethodNames, this);
         this._eventbus.off(`${oldPrepend}:get:plugin:names`, this.getPluginNames, this);
         this._eventbus.off(`${oldPrepend}:get:plugin:options`, this.getPluginOptions, this);
         this._eventbus.off(`${oldPrepend}:get:plugins:enabled`, this.getPluginsEnabled, this);
         this._eventbus.off(`${oldPrepend}:get:plugins:by:event:name`, this.getPluginsByEventName, this);
         this._eventbus.off(`${oldPrepend}:get:plugins:event:names`, this.getPluginsEventNames, this);
         this._eventbus.off(`${oldPrepend}:has:method`, this.hasMethod, this);
         this._eventbus.off(`${oldPrepend}:has:plugin`, this.hasPlugin, this);
         this._eventbus.off(`${oldPrepend}:has:plugin:method`, this.hasPluginMethod, this);
         this._eventbus.off(`${oldPrepend}:invoke`, this.invoke, this);
         this._eventbus.off(`${oldPrepend}:is:valid:config`, this.isValidConfig, this);
         this._eventbus.off(`${oldPrepend}:remove`, this._removeEventbus, this);
         this._eventbus.off(`${oldPrepend}:remove:all`, this._removeAllEventbus, this);
         this._eventbus.off(`${oldPrepend}:set:extra:event:data`, this.setExtraEventData, this);
         this._eventbus.off(`${oldPrepend}:set:options`, this._setOptionsEventbus, this);
         this._eventbus.off(`${oldPrepend}:set:plugin:enabled`, this.setPluginEnabled, this);
         this._eventbus.off(`${oldPrepend}:set:plugins:enabled`, this.setPluginsEnabled, this);
         this._eventbus.off(`${oldPrepend}:sync:invoke`, this.invokeSync, this);
         this._eventbus.off(`${oldPrepend}:sync:invoke:event`, this.invokeSyncEvent, this);

         // Invoke `typhonjs:plugin:manager:eventbus:removed` allowing external code to react to eventbus removal.
         this._eventbus.trigger(`typhonjs:plugin:manager:eventbus:removed`,
         {
            oldEventbus: this._eventbus,
            oldEventPrepend: oldPrepend,
            newEventbus: targetEventbus,
            newEventPrepend: eventPrepend
         });
      }

      targetEventbus.on(`${eventPrepend}:add`, this._addEventbus, this);
      targetEventbus.on(`${eventPrepend}:add:all`, this._addAllEventbus, this);
      targetEventbus.on(`${eventPrepend}:async:add`, this._addEventbusAsync, this);
      targetEventbus.on(`${eventPrepend}:async:add:all`, this._addAllEventbusAsync, this);
      targetEventbus.on(`${eventPrepend}:async:destroy:manager`, this._destroyEventbusAsync, this);
      targetEventbus.on(`${eventPrepend}:async:invoke`, this.invokeAsync, this);
      targetEventbus.on(`${eventPrepend}:async:invoke:event`, this.invokeAsyncEvent, this);
      targetEventbus.on(`${eventPrepend}:async:remove`, this._removeEventbusAsync, this);
      targetEventbus.on(`${eventPrepend}:async:remove:all`, this._removeAllEventbusAsync, this);
      targetEventbus.on(`${eventPrepend}:create:event:proxy`, this.createEventProxy, this);
      targetEventbus.on(`${eventPrepend}:destroy:manager`, this._destroyEventbus, this);
      targetEventbus.on(`${eventPrepend}:get:all:plugin:data`, this.getAllPluginData, this);
      targetEventbus.on(`${eventPrepend}:get:extra:event:data`, this.getExtraEventData, this);
      targetEventbus.on(`${eventPrepend}:get:method:names`, this.getMethodNames, this);
      targetEventbus.on(`${eventPrepend}:get:options`, this.getOptions, this);
      targetEventbus.on(`${eventPrepend}:get:plugin:data`, this.getPluginData, this);
      targetEventbus.on(`${eventPrepend}:get:plugin:enabled`, this.getPluginEnabled, this);
      targetEventbus.on(`${eventPrepend}:get:plugin:event:names`, this.getPluginEventNames, this);
      targetEventbus.on(`${eventPrepend}:get:plugin:method:names`, this.getPluginMethodNames, this);
      targetEventbus.on(`${eventPrepend}:get:plugin:names`, this.getPluginNames, this);
      targetEventbus.on(`${eventPrepend}:get:plugin:options`, this.getPluginOptions, this);
      targetEventbus.on(`${eventPrepend}:get:plugins:enabled`, this.getPluginsEnabled, this);
      targetEventbus.on(`${eventPrepend}:get:plugins:by:event:name`, this.getPluginsByEventName, this);
      targetEventbus.on(`${eventPrepend}:get:plugins:event:names`, this.getPluginsEventNames, this);
      targetEventbus.on(`${eventPrepend}:has:method`, this.hasMethod, this);
      targetEventbus.on(`${eventPrepend}:has:plugin`, this.hasPlugin, this);
      targetEventbus.on(`${eventPrepend}:has:plugin:method`, this.hasPluginMethod, this);
      targetEventbus.on(`${eventPrepend}:invoke`, this.invoke, this);
      targetEventbus.on(`${eventPrepend}:is:valid:config`, this.isValidConfig, this);
      targetEventbus.on(`${eventPrepend}:remove`, this._removeEventbus, this);
      targetEventbus.on(`${eventPrepend}:remove:all`, this._removeAllEventbus, this);
      targetEventbus.on(`${eventPrepend}:set:extra:event:data`, this.setExtraEventData, this);
      targetEventbus.on(`${eventPrepend}:set:options`, this._setOptionsEventbus, this);
      targetEventbus.on(`${eventPrepend}:set:plugin:enabled`, this.setPluginEnabled, this);
      targetEventbus.on(`${eventPrepend}:set:plugins:enabled`, this.setPluginsEnabled, this);
      targetEventbus.on(`${eventPrepend}:sync:invoke`, this.invokeSync, this);
      targetEventbus.on(`${eventPrepend}:sync:invoke:event`, this.invokeSyncEvent, this);

      // Invoke `typhonjs:plugin:manager:eventbus:set` allowing external code to react to eventbus set.
      targetEventbus.trigger('typhonjs:plugin:manager:eventbus:set',
      {
         oldEventbus: this._eventbus,
         oldEventPrepend: oldPrepend,
         newEventbus: targetEventbus,
         newEventPrepend: eventPrepend
      });

      this._eventbus = targetEventbus;

      return this;
   }

   /**
    * Sets the eventbus associated with this plugin manager. If any previous eventbus was associated all plugin manager
    * events will be removed then added to the new eventbus. If there are any existing plugins being managed their
    * events will be removed from the old eventbus and then `onPluginLoad` will be called with the new eventbus.
    *
    * @param {TyphonEvents}   targetEventbus - The target eventbus to associate.
    *
    * @param {string}         [eventPrepend='plugins'] - An optional string to prepend to all of the event binding
    *                                                    targets.
    *
    * @returns {Promise<PluginManager>}
    */
   async setEventbusAsync(targetEventbus, eventPrepend = 'plugins')
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof targetEventbus !== 'object') { throw new TypeError(`'targetEventbus' is not an 'object'.`); }
      if (typeof eventPrepend !== 'string') { throw new TypeError(`'eventPrepend' is not a 'string'.`); }

      // Early escape if the targetEventbus is the same as the current eventbus.
      if (targetEventbus === this._eventbus) { return this; }

      const oldPrepend = this._eventPrepend;

      /**
       * Stores the prepend string for eventbus registration.
       * @type {string}
       * @private
       */
      this._eventPrepend = eventPrepend;

      // Unload and reload any existing plugins from the old eventbus to the target eventbus.
      if (this._pluginMap.size > 0)
      {
         // Invoke private module method which allows skipping optional error checking.
         await s_INVOKE_ASYNC_EVENTS('onPluginUnload', {}, {}, this._extraEventData, this._pluginMap.keys(),
          this._pluginMap, this._options, false);

         for (const entry of this._pluginMap.values())
         {
            // Automatically remove any potential reference to a stored event proxy instance.
            try
            {
               entry.instance._eventbus = void 0;
            }
            catch (err) { /* nop */ }

            entry.data.manager.eventPrepend = eventPrepend;
            entry.data.plugin.scopedName = `${eventPrepend}:${entry.name}`;

            if (entry.eventProxy instanceof EventProxy) { entry.eventProxy.destroy(); }

            entry.eventProxy = new EventProxy(targetEventbus);
         }

         // Invoke private module method which allows skipping optional error checking.
         await s_INVOKE_ASYNC_EVENTS('onPluginLoad', {}, {}, this._extraEventData, this._pluginMap.keys(),
          this._pluginMap, this._options, false);

         for (const entry of this._pluginMap.values())
         {
            // Invoke `typhonjs:plugin:manager:eventbus:changed` allowing external code to react to plugin
            // changing eventbus.
            if (this._eventbus)
            {
               this._eventbus.trigger(`typhonjs:plugin:manager:eventbus:changed`, Object.assign({
                  oldEventbus: this._eventbus,
                  oldManagerEventPrepend: oldPrepend,
                  oldScopedName: `${oldPrepend}:${entry.name}`,
                  newEventbus: targetEventbus,
                  newManagerEventPrepend: eventPrepend,
                  newScopedName: `${eventPrepend}:${entry.name}`
               }, JSON.parse(JSON.stringify(entry.data))));
            }
         }
      }

      if (this._eventbus !== null)
      {
         this._eventbus.off(`${oldPrepend}:add`, this._addEventbus, this);
         this._eventbus.off(`${oldPrepend}:add:all`, this._addAllEventbus, this);
         this._eventbus.off(`${oldPrepend}:async:add`, this._addEventbusAsync, this);
         this._eventbus.off(`${oldPrepend}:async:add:all`, this._addAllEventbusAsync, this);
         this._eventbus.off(`${oldPrepend}:async:destroy:manager`, this._destroyEventbusAsync, this);
         this._eventbus.off(`${oldPrepend}:async:invoke`, this.invokeAsync, this);
         this._eventbus.off(`${oldPrepend}:async:invoke:event`, this.invokeAsyncEvent, this);
         this._eventbus.off(`${oldPrepend}:async:remove`, this._removeEventbusAsync, this);
         this._eventbus.off(`${oldPrepend}:async:remove:all`, this._removeAllEventbusAsync, this);
         this._eventbus.off(`${oldPrepend}:create:event:proxy`, this.createEventProxy, this);
         this._eventbus.off(`${oldPrepend}:destroy:manager`, this._destroyEventbus, this);
         this._eventbus.off(`${oldPrepend}:get:all:plugin:data`, this.getAllPluginData, this);
         this._eventbus.off(`${oldPrepend}:get:extra:event:data`, this.getExtraEventData, this);
         this._eventbus.off(`${oldPrepend}:get:method:names`, this.getMethodNames, this);
         this._eventbus.off(`${oldPrepend}:get:options`, this.getOptions, this);
         this._eventbus.off(`${oldPrepend}:get:plugin:enabled`, this.getPluginEnabled, this);
         this._eventbus.off(`${oldPrepend}:get:plugin:data`, this.getPluginData, this);
         this._eventbus.off(`${oldPrepend}:get:plugin:event:names`, this.getPluginEventNames, this);
         this._eventbus.off(`${oldPrepend}:get:plugin:method:names`, this.getPluginMethodNames, this);
         this._eventbus.off(`${oldPrepend}:get:plugin:names`, this.getPluginNames, this);
         this._eventbus.off(`${oldPrepend}:get:plugin:options`, this.getPluginOptions, this);
         this._eventbus.off(`${oldPrepend}:get:plugins:enabled`, this.getPluginsEnabled, this);
         this._eventbus.off(`${oldPrepend}:get:plugins:by:event:name`, this.getPluginsByEventName, this);
         this._eventbus.off(`${oldPrepend}:get:plugins:event:names`, this.getPluginsEventNames, this);
         this._eventbus.off(`${oldPrepend}:has:method`, this.hasMethod, this);
         this._eventbus.off(`${oldPrepend}:has:plugin`, this.hasPlugin, this);
         this._eventbus.off(`${oldPrepend}:has:plugin:method`, this.hasPluginMethod, this);
         this._eventbus.off(`${oldPrepend}:invoke`, this.invoke, this);
         this._eventbus.off(`${oldPrepend}:is:valid:config`, this.isValidConfig, this);
         this._eventbus.off(`${oldPrepend}:remove`, this._removeEventbus, this);
         this._eventbus.off(`${oldPrepend}:remove:all`, this._removeAllEventbus, this);
         this._eventbus.off(`${oldPrepend}:set:extra:event:data`, this.setExtraEventData, this);
         this._eventbus.off(`${oldPrepend}:set:options`, this._setOptionsEventbus, this);
         this._eventbus.off(`${oldPrepend}:set:plugin:enabled`, this.setPluginEnabled, this);
         this._eventbus.off(`${oldPrepend}:set:plugins:enabled`, this.setPluginsEnabled, this);
         this._eventbus.off(`${oldPrepend}:sync:invoke`, this.invokeSync, this);
         this._eventbus.off(`${oldPrepend}:sync:invoke:event`, this.invokeSyncEvent, this);

         // Invoke `typhonjs:plugin:manager:eventbus:removed` allowing external code to react to eventbus removal.
         this._eventbus.trigger(`typhonjs:plugin:manager:eventbus:removed`,
         {
            oldEventbus: this._eventbus,
            oldEventPrepend: oldPrepend,
            newEventbus: targetEventbus,
            newEventPrepend: eventPrepend
         });
      }

      targetEventbus.on(`${eventPrepend}:add`, this._addEventbus, this);
      targetEventbus.on(`${eventPrepend}:add:all`, this._addAllEventbus, this);
      targetEventbus.on(`${eventPrepend}:async:add`, this._addEventbusAsync, this);
      targetEventbus.on(`${eventPrepend}:async:add:all`, this._addAllEventbusAsync, this);
      targetEventbus.on(`${eventPrepend}:async:destroy:manager`, this._destroyEventbusAsync, this);
      targetEventbus.on(`${eventPrepend}:async:invoke`, this.invokeAsync, this);
      targetEventbus.on(`${eventPrepend}:async:invoke:event`, this.invokeAsyncEvent, this);
      targetEventbus.on(`${eventPrepend}:async:remove`, this._removeEventbusAsync, this);
      targetEventbus.on(`${eventPrepend}:async:remove:all`, this._removeAllEventbusAsync, this);
      targetEventbus.on(`${eventPrepend}:create:event:proxy`, this.createEventProxy, this);
      targetEventbus.on(`${eventPrepend}:destroy:manager`, this._destroyEventbus, this);
      targetEventbus.on(`${eventPrepend}:get:all:plugin:data`, this.getAllPluginData, this);
      targetEventbus.on(`${eventPrepend}:get:extra:event:data`, this.getExtraEventData, this);
      targetEventbus.on(`${eventPrepend}:get:method:names`, this.getMethodNames, this);
      targetEventbus.on(`${eventPrepend}:get:options`, this.getOptions, this);
      targetEventbus.on(`${eventPrepend}:get:plugin:data`, this.getPluginData, this);
      targetEventbus.on(`${eventPrepend}:get:plugin:enabled`, this.getPluginEnabled, this);
      targetEventbus.on(`${eventPrepend}:get:plugin:event:names`, this.getPluginEventNames, this);
      targetEventbus.on(`${eventPrepend}:get:plugin:method:names`, this.getPluginMethodNames, this);
      targetEventbus.on(`${eventPrepend}:get:plugin:names`, this.getPluginNames, this);
      targetEventbus.on(`${eventPrepend}:get:plugin:options`, this.getPluginOptions, this);
      targetEventbus.on(`${eventPrepend}:get:plugins:enabled`, this.getPluginsEnabled, this);
      targetEventbus.on(`${eventPrepend}:get:plugins:by:event:name`, this.getPluginsByEventName, this);
      targetEventbus.on(`${eventPrepend}:get:plugins:event:names`, this.getPluginsEventNames, this);
      targetEventbus.on(`${eventPrepend}:has:method`, this.hasMethod, this);
      targetEventbus.on(`${eventPrepend}:has:plugin`, this.hasPlugin, this);
      targetEventbus.on(`${eventPrepend}:has:plugin:method`, this.hasPluginMethod, this);
      targetEventbus.on(`${eventPrepend}:invoke`, this.invoke, this);
      targetEventbus.on(`${eventPrepend}:is:valid:config`, this.isValidConfig, this);
      targetEventbus.on(`${eventPrepend}:remove`, this._removeEventbus, this);
      targetEventbus.on(`${eventPrepend}:remove:all`, this._removeAllEventbus, this);
      targetEventbus.on(`${eventPrepend}:set:extra:event:data`, this.setExtraEventData, this);
      targetEventbus.on(`${eventPrepend}:set:options`, this._setOptionsEventbus, this);
      targetEventbus.on(`${eventPrepend}:set:plugin:enabled`, this.setPluginEnabled, this);
      targetEventbus.on(`${eventPrepend}:set:plugins:enabled`, this.setPluginsEnabled, this);
      targetEventbus.on(`${eventPrepend}:sync:invoke`, this.invokeSync, this);
      targetEventbus.on(`${eventPrepend}:sync:invoke:event`, this.invokeSyncEvent, this);

      // Invoke `typhonjs:plugin:manager:eventbus:set` allowing external code to react to eventbus set.
      targetEventbus.trigger('typhonjs:plugin:manager:eventbus:set',
      {
         oldEventbus: this._eventbus,
         oldEventPrepend: oldPrepend,
         newEventbus: targetEventbus,
         newEventPrepend: eventPrepend
      });

      this._eventbus = targetEventbus;

      return this;
   }

   /**
    * Sets any extra event data attached to PluginEvent `extra` field.
    *
    * @param {*}  extraEventData - Adds extra data to PluginEvent `extra` field.
    */
   setExtraEventData(extraEventData = void 0)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      this._extraEventData = extraEventData;
   }

   /**
    * Set optional parameters. All parameters are off by default.
    *
    * @param {PluginManagerOptions} options - Defines optional parameters to set.
    */
   setOptions(options = {})
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof options !== 'object') { throw new TypeError(`'options' is not an object.`); }

      if (typeof options.pluginsEnabled === 'boolean') { this._options.pluginsEnabled = options.pluginsEnabled; }
      if (typeof options.noEventAdd === 'boolean') { this._options.noEventAdd = options.noEventAdd; }
      if (typeof options.noEventDestroy === 'boolean') { this._options.noEventDestroy = options.noEventDestroy; }
      if (typeof options.noEventOptions === 'boolean') { this._options.noEventOptions = options.noEventOptions; }
      if (typeof options.noEventRemoval === 'boolean') { this._options.noEventRemoval = options.noEventRemoval; }
      if (typeof options.throwNoMethod === 'boolean') { this._options.throwNoMethod = options.throwNoMethod; }
      if (typeof options.throwNoPlugin === 'boolean') { this._options.throwNoPlugin = options.throwNoPlugin; }
   }

   /**
    * Provides the eventbus callback which may prevent plugin manager options being set if optional `noEventOptions` is
    * enabled. This disables the ability for the plugin manager options to be set via events preventing any external
    * code modifying options.
    *
    * @param {PluginManagerOptions} options - Defines optional parameters to set.
    *
    * @private
    */
   _setOptionsEventbus(options = {})
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (!this._options.noEventOptions) { this.setOptions(options); }
   }

   /**
    * Enables or disables a single plugin.
    *
    * @param {string}   pluginName - Plugin name to set state.
    * @param {boolean}  enabled - The new enabled state.
    *
    * @returns {boolean} - Operation success.
    */
   setPluginEnabled(pluginName, enabled)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof pluginName !== 'string') { throw new TypeError(`'pluginName' is not a string.`); }
      if (typeof enabled !== 'boolean') { throw new TypeError(`'enabled' is not a boolean.`); }

      const entry = this._pluginMap.get(pluginName);

      if (entry instanceof PluginEntry)
      {
         entry.enabled = enabled;

         // Invoke `typhonjs:plugin:manager:plugin:enabled` allowing external code to react to plugin enabled state.
         if (this._eventbus)
         {
            this._eventbus.trigger(`typhonjs:plugin:manager:plugin:enabled`, Object.assign({
               enabled
            }, JSON.parse(JSON.stringify(entry.data))));
         }

         return true;
      }

      return false;
   }

   /**
    * Enables or disables a set of plugins given an array or iterabe of plugin names.
    *
    * @param {Array<string>}  pluginNames - An array / iterable of plugin names.
    * @param {boolean}        enabled - The new enabled state.
    *
    * @returns {boolean} - Operation success.
    */
   setPluginsEnabled(pluginNames, enabled)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (typeof enabled !== 'boolean') { throw new TypeError(`'enabled' is not a boolean.`); }

      let success = true;

      for (const pluginName of pluginNames)
      {
         if (!this.setPluginEnabled(pluginName, enabled)) { success = false; }
      }

      return success;
   }

   /**
    * Removes a plugin by name after unloading it and clearing any event bindings automatically.
    *
    * @param {string}   pluginName - The plugin name to remove.
    *
    * @returns {boolean} - Operation success.
    */
   remove(pluginName)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      const entry = this._pluginMap.get(pluginName);

      if (entry instanceof PluginEntry)
      {
         // Invoke private module method which allows skipping optional error checking.
         s_INVOKE_SYNC_EVENTS('onPluginUnload', {}, {}, this._extraEventData, pluginName, this._pluginMap,
          this._options, false);

         // Automatically remove any potential reference to a stored event proxy instance.
         try
         {
            entry.instance._eventbus = void 0;
         }
         catch (err) { /* nop */ }

         if (entry.eventProxy instanceof EventProxy) { entry.eventProxy.destroy(); }

         const pluginData = this.getPluginData(pluginName);

         this._pluginMap.delete(pluginName);

         // Invoke `typhonjs:plugin:manager:plugin:removed` allowing external code to react to plugin removed.
         if (this._eventbus)
         {
            this._eventbus.trigger(`typhonjs:plugin:manager:plugin:removed`, pluginData);
         }

         return true;
      }

      return false;
   }

   /**
    * Removes a plugin by name after unloading it and clearing any event bindings automatically.
    *
    * @param {string}   pluginName - The plugin name to remove.
    *
    * @returns {Promise<boolean>} - Operation success.
    */
   async removeAsync(pluginName)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      const entry = this._pluginMap.get(pluginName);

      if (entry instanceof PluginEntry)
      {
         // Invoke private module method which allows skipping optional error checking.
         await s_INVOKE_ASYNC_EVENTS('onPluginUnload', {}, {}, this._extraEventData, pluginName, this._pluginMap,
          this._options, false);

         // Automatically remove any potential reference to a stored event proxy instance.
         try
         {
            entry.instance._eventbus = void 0;
         }
         catch (err) { /* nop */ }

         if (entry.eventProxy instanceof EventProxy) { entry.eventProxy.destroy(); }

         const pluginData = this.getPluginData(pluginName);

         this._pluginMap.delete(pluginName);

         // Invoke `typhonjs:plugin:manager:plugin:removed` allowing external code to react to plugin removed.
         if (this._eventbus)
         {
            await this._eventbus.triggerAsync(`typhonjs:plugin:manager:plugin:removed`, pluginData);
         }

         return true;
      }

      return false;
   }

   /**
    * Removes all plugins after unloading them and clearing any event bindings automatically.
    */
   removeAll()
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      for (const pluginName of this._pluginMap.keys())
      {
         this.remove(pluginName);
      }

      this._pluginMap.clear();
   }

   /**
    * Removes all plugins after unloading them and clearing any event bindings automatically.
    *
    * @returns {Promise.<*>}
    */
   removeAllAsync()
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      const values = [];

      for (const pluginName of this._pluginMap.keys())
      {
         values.push(this.remove(pluginName));
      }

      this._pluginMap.clear();

      return Promise.all(values);
   }

   /**
    * Provides the eventbus callback which may prevent removal if optional `noEventRemoval` is enabled. This disables
    * the ability for plugins to be removed via events preventing any external code removing plugins in this manner.
    *
    * @param {string}   pluginName - The plugin name to remove.
    *
    * @returns {boolean} - Operation success.
    * @private
    */
   _removeEventbus(pluginName)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      return !this._options.noEventRemoval ? this.remove(pluginName) : false;
   }

   /**
    * Provides the eventbus callback which may prevent removal if optional `noEventRemoval` is enabled. This disables
    * the ability for plugins to be removed via events preventing any external code removing plugins in this manner.
    *
    * @param {string}   pluginName - The plugin name to remove.
    *
    * @returns {Promise<boolean>} - Operation success.
    * @private
    */
   async _removeEventbusAsync(pluginName)
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      return !this._options.noEventRemoval ? await this.removeAsync(pluginName) : Promise.resolve(false);
   }

   /**
    * Provides the eventbus callback which may prevent removal if optional `noEventRemoval` is enabled. This disables
    * the ability for plugins to be removed via events preventing any external code removing plugins in this manner.
    *
    * @private
    */
   _removeAllEventbus()
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (!this._options.noEventRemoval) { this.removeAll(); }
   }

   /**
    * Provides the eventbus callback which may prevent removal if optional `noEventRemoval` is enabled. This disables
    * the ability for plugins to be removed via events preventing any external code removing plugins in this manner.
    *
    * @private
    */
   async _removeAllEventbusAsync()
   {
      if (this._pluginMap === null) { throw new ReferenceError('This PluginManager instance has been destroyed.'); }

      if (!this._options.noEventRemoval) { await this.removeAll(); }
   }
}

/**
 * Private implementation to invoke asynchronous events. This allows internal calls in PluginManager for
 * `onPluginLoad` and `onPluginUnload` callbacks to bypass optional error checking.
 *
 * This dispatch method asynchronously passes to and returns from any invoked targets a PluginEvent. Any invoked plugin
 * may return a Promise which is awaited upon by `Promise.all` before returning the PluginEvent data via a Promise.
 *
 * @param {string}                     methodName - Method name to invoke.
 *
 * @param {object}                     copyProps - plugin event object.
 *
 * @param {object}                     passthruProps - if true, event has plugin option.
 *
 * @param {*}                          extraEventData - Optional extra data attached to all plugin events.
 *
 * @param {string|Array<string>}       nameOrList - An optional plugin name or array / iterable of plugin names to
 *                                                  invoke.
 *
 * @param {Map<string, PluginEvent>}   pluginMap - Stores the plugins by name with an associated PluginEntry.
 *
 * @param {object}                     options - Defines options for throwing exceptions. Turned off by default.
 *
 * @param {boolean}                    [performErrorCheck=true] - If false optional error checking is disabled.
 *
 * @returns {Promise<PluginEvent>}
 */
const s_INVOKE_ASYNC_EVENTS = async (methodName, copyProps = {}, passthruProps = {}, extraEventData, nameOrList,
 pluginMap, options, performErrorCheck = true) =>
{
   if (typeof methodName !== 'string') { throw new TypeError(`'methodName' is not a string.`); }
   if (typeof passthruProps !== 'object') { throw new TypeError(`'passthruProps' is not an object.`); }
   if (typeof copyProps !== 'object') { throw new TypeError(`'copyProps' is not an object.`); }

   if (typeof nameOrList !== 'string' && !Array.isArray(nameOrList) &&
    typeof nameOrList[Symbol.iterator] !== 'function')
   {
      throw new TypeError(`'nameOrList' is not a string, array, or iterator.`);
   }

   // Track how many plugins were invoked.
   let pluginInvokeCount = 0;
   const pluginInvokeNames = [];

   // Track if a plugin method is invoked
   let hasMethod = false;
   let hasPlugin = false;

   // Create plugin event.
   const ev = new PluginEvent(copyProps, passthruProps, extraEventData);

   const results = [];

   if (typeof nameOrList === 'string')
   {
      const entry = pluginMap.get(nameOrList);

      if (entry instanceof PluginEntry && entry.enabled && entry.instance)
      {
         hasPlugin = true;

         if (typeof entry.instance[methodName] === 'function')
         {
            ev.eventbus = entry.eventProxy;
            ev.pluginName = entry.name;
            ev.pluginOptions = entry.data.plugin.options;

            const result = entry.instance[methodName](ev);

            if (typeof result !== 'undefined' && result !== null) { results.push(result); }

            hasMethod = true;
            pluginInvokeCount++;
            pluginInvokeNames.push(entry.name);
         }
      }
   }
   else
   {
      for (const name of nameOrList)
      {
         const entry = pluginMap.get(name);

         if (entry instanceof PluginEntry && entry.enabled && entry.instance)
         {
            hasPlugin = true;

            if (typeof entry.instance[methodName] === 'function')
            {
               ev.eventbus = entry.eventProxy;
               ev.pluginName = entry.name;
               ev.pluginOptions = entry.data.plugin.options;

               const result = entry.instance[methodName](ev);

               if (typeof result !== 'undefined' && result !== null) { results.push(result); }

               hasMethod = true;
               pluginInvokeCount++;
               pluginInvokeNames.push(entry.name);
            }
         }
      }
   }

   if (performErrorCheck && options.throwNoPlugin && !hasPlugin)
   {
      throw new Error(`PluginManager failed to find any target plugins.`);
   }

   if (performErrorCheck && options.throwNoMethod && !hasMethod)
   {
      throw new Error(`PluginManager failed to invoke '${methodName}'.`);
   }

   // Add meta data for plugin invoke count.
   ev.data.$$plugin_invoke_count = pluginInvokeCount;
   ev.data.$$plugin_invoke_names = pluginInvokeNames;

   await Promise.all(results);

   return ev.data;
};

/**
 * Private implementation to invoke synchronous events. This allows internal calls in PluginManager for
 * `onPluginLoad` and `onPluginUnload` callbacks to bypass optional error checking.
 *
 * This dispatch method synchronously passes to and returns from any invoked targets a PluginEvent.
 *
 * @param {string}                     methodName - Method name to invoke.
 *
 * @param {object}                     copyProps - plugin event object.
 *
 * @param {object}                     passthruProps - if true, event has plugin option.
 *
 * @param {*}                          extraEventData - Optional extra data attached to all plugin events.
 *
 * @param {string|Array<string>}       nameOrList - An optional plugin name or array / iterable of plugin names to
 *                                                  invoke.
 *
 * @param {Map<string, PluginEvent>}   pluginMap - Stores the plugins by name with an associated PluginEntry.
 *
 * @param {object}                     options - Defines options for throwing exceptions. Turned off by default.
 *
 * @param {boolean}                    [performErrorCheck=true] - If false optional error checking is disabled.
 *
 * @returns {PluginEvent}
 */
const s_INVOKE_SYNC_EVENTS = (methodName, copyProps = {}, passthruProps = {}, extraEventData, nameOrList, pluginMap,
 options, performErrorCheck = true) =>
{
   if (typeof methodName !== 'string') { throw new TypeError(`'methodName' is not a string.`); }
   if (typeof passthruProps !== 'object') { throw new TypeError(`'passthruProps' is not an object.`); }
   if (typeof copyProps !== 'object') { throw new TypeError(`'copyProps' is not an object.`); }

   if (typeof nameOrList !== 'string' && !Array.isArray(nameOrList) &&
    typeof nameOrList[Symbol.iterator] !== 'function')
   {
      throw new TypeError(`'nameOrList' is not a string, array, or iterator.`);
   }

   // Track how many plugins were invoked.
   let pluginInvokeCount = 0;
   const pluginInvokeNames = [];

   // Track if a plugin method is invoked
   let hasMethod = false;
   let hasPlugin = false;

   // Create plugin event.
   const ev = new PluginEvent(copyProps, passthruProps, extraEventData);

   if (typeof nameOrList === 'string')
   {
      const entry = pluginMap.get(nameOrList);

      if (entry instanceof PluginEntry && entry.enabled && entry.instance)
      {
         hasPlugin = true;

         if (typeof entry.instance[methodName] === 'function')
         {
            ev.eventbus = entry.eventProxy;
            ev.pluginName = entry.name;
            ev.pluginOptions = entry.data.plugin.options;

            entry.instance[methodName](ev);

            hasMethod = true;
            pluginInvokeCount++;
            pluginInvokeNames.push(entry.name);
         }
      }
   }
   else
   {
      for (const name of nameOrList)
      {
         const entry = pluginMap.get(name);

         if (entry instanceof PluginEntry && entry.enabled && entry.instance)
         {
            hasPlugin = true;

            if (typeof entry.instance[methodName] === 'function')
            {
               ev.eventbus = entry.eventProxy;
               ev.pluginName = entry.name;
               ev.pluginOptions = entry.data.plugin.options;

               entry.instance[methodName](ev);

               hasMethod = true;
               pluginInvokeCount++;
               pluginInvokeNames.push(entry.name);
            }
         }
      }
   }

   if (performErrorCheck && options.throwNoPlugin && !hasPlugin)
   {
      throw new Error(`PluginManager failed to find any target plugins.`);
   }

   if (performErrorCheck && options.throwNoMethod && !hasMethod)
   {
      throw new Error(`PluginManager failed to invoke '${methodName}'.`);
   }

   // Add meta data for plugin invoke count.
   ev.data.$$plugin_invoke_count = pluginInvokeCount;
   ev.data.$$plugin_invoke_names = pluginInvokeNames;

   return ev.data;
};

/**
 * Walks an objects inheritance tree collecting property names stopping before `Object` is reached.
 *
 * @param {object}   obj - object to walks.
 *
 * @returns {Array}
 * @ignore
 */
const s_GET_ALL_PROPERTY_NAMES = (obj) =>
{
   const props = [];

   do
   {
      Object.getOwnPropertyNames(obj).forEach((prop) => { if (props.indexOf(prop) === -1) { props.push(prop); } });
      obj = Object.getPrototypeOf(obj);
   } while (typeof obj !== 'undefined' && obj !== null && !(obj === Object.prototype));

   return props;
};
