/**
 * Defines a class holding the data associated with a plugin including its instance.
 */
export default class PluginEntry
{
   /**
    * Instantiates a PluginEntry.
    *
    * @param {string}      name - The plugin name.
    *
    * @param {object}      data - Data describing the plugin, manager, and optional module data.
    *
    * @param {Object}      instance - The loaded plugin instance.
    *
    * @param {EventProxy}  eventProxy - An EventProxy associated with the plugin wrapping the plugin manager eventbus.
    */
   constructor(name, data, instance, eventProxy = void 0)
   {
      /**
       * Data describing the plugin, manager, and optional module data.
       * @type {object}
       * @private
       */
      this._data = data;

      /**
       * The plugin enabled state.
       * @type {boolean}
       * @private
       */
      this._enabled = true;

      /**
       * The plugin name.
       * @type {string}
       * @private
       */
      this._name = name;

      /**
       * The loaded plugin instance.
       * @type {Object}
       * @private
       */
      this._instance = instance;

      /**
       * An EventProxy associated with the plugin wrapping the plugin manager eventbus.
       * @type {EventProxy}
       * @private
       */
      this._eventProxy = eventProxy;
   }

   /**
    * Provides a convenience method to escape file paths.
    *
    * @param {string}   value - A string to escape.
    *
    * @returns {string}
    */
   static escape(value)
   {
      if (typeof value !== 'string') { throw new TypeError(`'value' is not a 'string'`); }

      // Remove any leading relative directory paths.
      let escaped = value.replace(/^(\.\.|\.)/, '');

      // Escape any forward / reverse slashes for RegExp creation.
      escaped = escaped.replace(/[\\]/g, '\\');
      escaped = escaped.replace(/[\/]/g, '\\/');

      return escaped;
   }

   /**
    * Get plugin data.
    *
    * @returns {object}
    */
   get data() { return this._data; }

   /**
    * Get enabled.
    *
    * @returns {boolean}
    */
   get enabled() { return this._enabled; }

   /**
    * Set enabled.
    *
    * @param {boolean} enabled - New enabled state.
    */
   set enabled(enabled)
   {
      /**
       * The plugin enabled state.
       * @type {boolean}
       * @private
       */
      this._enabled = enabled;
   }

   /**
    * Get associated EventProxy.
    *
    * @returns {EventProxy}
    */
   get eventProxy() { return this._eventProxy; }

   /**
    * Get plugin instance.
    *
    * @returns {Object}
    */
   get instance() { return this._instance; }

   /**
    * Get plugin name.
    *
    * @returns {string}
    */
   get name() { return this._name; }
}
