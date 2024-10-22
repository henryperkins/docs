var code = `
/**
 * @external {PluginConfig} https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/typedef/index.html#static-typedef-PluginConfig
 */

/**
 * @external {PluginEvent} https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/class/src/PluginEvent.js~PluginEvent.html
 */

/**
 * @external {PluginData} https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/typedef/index.html#static-typedef-ManagerOptions
 */

/**
 * @external {PluginManagerOptions} https://docs.typhonjs.io/typhonjs-node-plugin/typhonjs-plugin-manager/typedef/index.html#static-typedef-ManagerOptions
 */
`;

exports.onHandleVirtual = function(ev)
{
   ev.data.code.push({ code, message: __filename });
};
