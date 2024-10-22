import ASTWalker from './ASTWalker.js';

/**
 * Default walker instance.
 * @type {ASTWalker}
 */
const walker = new ASTWalker();

/**
 * Provides a default ASTWalker instance.
 */
export default walker;

/**
 * Wires up walker on the plugin eventbus.
 *
 * @param {PluginEvent} ev - The plugin event.
 *
 * @see https://www.npmjs.com/package/typhonjs-plugin-manager
 * @ignore
 */
export function onPluginLoad(ev)
{
   const eventbus = ev.eventbus;

   let eventPrepend = 'typhonjs';

   const options = ev.pluginOptions;

   // Apply any plugin options.
   if (typeof options === 'object')
   {
      // If `eventPrepend` is defined then it is prepended before all event bindings.
      if (typeof options.eventPrepend === 'string') { eventPrepend = `${options.eventPrepend}:`; }
   }

   eventbus.on(`${eventPrepend}:ast:walker:traverse`, walker.traverse, walker);
}
