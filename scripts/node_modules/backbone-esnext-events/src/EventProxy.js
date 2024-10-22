import TyphonEvents from './TyphonEvents.js';

/**
 * EventProxy provides a protected proxy of another TyphonEvents / eventbus instance.
 *
 * The main use case of EventProxy is to allow indirect access to an eventbus. This is handy when it comes to managing
 * the event lifecycle for a plugin system. When a plugin is added it could receive a callback, perhaps named
 * `onPluginLoaded`, which contains an EventProxy instance rather than the direct eventbus. This EventProxy instance is
 * associated in the management system controlling plugin lifecycle. When a plugin is removed / unloaded the management
 * system can automatically unregister all events for the plugin without requiring the plugin author doing it correctly
 * if they had full control. IE This allows to plugin system to guarantee no dangling listeners.
 *
 * EventProxy provides the on / off, once, and trigger methods with the same signatures as found in
 * TyphonEvents / Events. However, the proxy tracks all added event bindings which is used to proxy between the target
 * eventbus which is passed in from the constructor. All registration methods (on / off / once) proxy. In addition
 * there is a `destroy` method which will unregister all of proxied events and remove references to the managed
 * eventbus. Any further usage of a destroyed EventProxy instance results in a ReferenceError thrown.
 */
export default class EventProxy
{
   /**
    * Creates the event proxy with an existing instance of TyphonEvents.
    *
    * @param {TyphonEvents}   eventbus - The target eventbus instance.
    */
   constructor(eventbus)
   {
      if (!(eventbus instanceof TyphonEvents))
      {
         throw new TypeError(`'eventbus' is not an instance of TyphonEvents.`);
      }

      /**
       * Stores the target eventbus.
       * @type {TyphonEvents}
       * @private
       */
      this._eventbus = eventbus;

      /**
       * Stores all proxied event bindings.
       * @type {Array<{name: string, callback: function, context: *}>}
       * @private
       */
      this._events = [];
   }

   /**
    * Creates a new EventProxy from the target eventbus of this proxy.
    *
    * @returns {EventProxy}
    */
   createEventProxy()
   {
      return new EventProxy(this._eventbus);
   }

   /**
    * Unregisters all proxied events from the target eventbus and removes any local references. All subsequent calls
    * after `destroy` has been called result in a ReferenceError thrown.
    */
   destroy()
   {
      if (this._eventbus === null)
      {
         throw new ReferenceError('This EventProxy instance has been destroyed.');
      }

      for (const event of this._events) { this._eventbus.off(event.name, event.callback, event.context); }

      this._events = [];

      this._eventbus = null;
   }

   /**
    * Returns the current proxied event count.
    *
    * @returns {Number}
    */
   get eventCount() { return this._events.length; }

   /**
    * Returns the event names of proxied event listeners.
    *
    * @returns {string[]}
    */
   getEventNames()
   {
      if (!this._events) { return []; }

      const eventNames = {};

      for (const event of this._events) { eventNames[event.name] = true; }

      return Object.keys(eventNames);
   }

   /**
    * Iterates over all proxied events invoking a callback with the event data stored.
    *
    * @param {function} callback - Callback invoked for each proxied event stored.
    */
   forEachEvent(callback)
   {
      if (typeof callback !== 'function') { throw new TypeError(`'callback' is not a 'function'.`); }

      for (const event of this._events) { callback(event.name, event.callback, event.context); }
   }

   /**
    * Returns the target eventbus name.
    *
    * @returns {string|*}
    */
   getEventbusName()
   {
      if (this._eventbus === null) { throw new ReferenceError('This EventProxy instance has been destroyed.'); }

      return this._eventbus.getEventbusName();
   }

   /**
    * Remove a previously-bound proxied event binding.
    *
    * Please see {@link Events#off}.
    *
    * @param {string}   [name]     - Event name(s)
    *
    * @param {function} [callback] - Event callback function
    *
    * @param {object}   [context]  - Event context
    *
    * @returns {EventProxy}
    */
   off(name = void 0, callback = void 0, context = void 0)
   {
      if (this._eventbus === null)
      {
         throw new ReferenceError('This EventProxy instance has been destroyed.');
      }

      const hasName = typeof name !== 'undefined' && name !== null;
      const hasCallback = typeof callback !== 'undefined' && callback !== null;
      const hasContext = typeof context !== 'undefined' && context !== null;

      // Remove all events if `off()` is invoked.
      if (!hasName && !hasCallback && !hasContext)
      {
         for (const event of this._events) { this._eventbus.off(event.name, event.callback, event.context); }
         this._events = [];
      }
      else
      {
         const values = {};
         if (hasName) { values.name = name; }
         if (hasCallback) { values.callback = callback; }
         if (hasContext) { values.context = context; }

         for (let cntr = this._events.length; --cntr >= 0;)
         {
            const event = this._events[cntr];

            let foundMatch = true;

            for (const key in values)
            {
               if (event[key] !== values[key]) { foundMatch = false; break; }
            }

            if (foundMatch)
            {
               this._eventbus.off(values.name, values.callback, values.context);
               this._events.splice(cntr, 1);
            }
         }
      }

      return this;
   }

   /**
    * Bind a callback function to an object. The callback will be invoked whenever the event is fired. If you have a
    * large number of different events on a page, the convention is to use colons to namespace them: "poll:start", or
    * "change:selection".
    *
    * This is proxied through `listenTo` of an internal Events instance instead of directly modifying the target
    * eventbus.
    *
    * Please see {@link Events#on}.
    *
    * @param {string}   name     - Event name(s)
    * @param {function} callback - Event callback function
    * @param {object}   context  - Event context
    * @returns {EventProxy}
    */
   on(name, callback, context = void 0)
   {
      if (this._eventbus === null)
      {
         throw new ReferenceError('This EventProxy instance has been destroyed.');
      }

      this._eventbus.on(name, callback, context);

      this._events.push({ name, callback, context });

      return this;
   }

   /**
    * Trigger callbacks for the given event, or space-delimited list of events. Subsequent arguments to trigger will be
    * passed along to the event callbacks.
    *
    * Please see {@link Events#trigger}.
    *
    * @returns {EventProxy}
    */
   trigger()
   {
      if (this._eventbus === null) { throw new ReferenceError('This EventProxy instance has been destroyed.'); }

      this._eventbus.trigger(...arguments);

      return this;
   }

   /**
    * Provides `trigger` functionality, but collects any returned Promises from invoked targets and returns a
    * single Promise generated by `Promise.resolve` for a single value or `Promise.all` for multiple results. This is
    * a very useful mechanism to invoke asynchronous operations over an eventbus.
    *
    * Please see {@link TyphonEvents#triggerAsync}.
    *
    * @returns {Promise}
    */
   triggerAsync()
   {
      if (this._eventbus === null) { throw new ReferenceError('This EventProxy instance has been destroyed.'); }

      return this._eventbus.triggerAsync(...arguments);
   }

   /**
    * Defers invoking `trigger`. This is useful for triggering events in the next clock tick.
    *
    * Please see {@link TyphonEvents#triggerDefer}.
    *
    * @returns {EventProxy}
    */
   triggerDefer()
   {
      if (this._eventbus === null) { throw new ReferenceError('This EventProxy instance has been destroyed.'); }

      this._eventbus.triggerDefer(...arguments);

      return this;
   }

   /**
    * Provides `trigger` functionality, but collects any returned result or results from invoked targets as a single
    * value or in an array and passes it back to the callee in a synchronous manner.
    *
    * Please see {@link TyphonEvents#triggerSync}.
    *
    * @returns {*|Array.<*>}
    */
   triggerSync()
   {
      if (this._eventbus === null) { throw new ReferenceError('This EventProxy instance has been destroyed.'); }

      return this._eventbus.triggerSync(...arguments);
   }
}