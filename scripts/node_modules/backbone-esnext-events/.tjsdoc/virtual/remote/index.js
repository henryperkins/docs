var code = `
/**
 * @external {Events} https://docs.typhonjs.io/typhonjs-backbone-esnext/backbone-esnext-events/class/src/Events.js~Events.html
 */

/**
 * @external {EventProxy} https://docs.typhonjs.io/typhonjs-backbone-esnext/backbone-esnext-events/class/src/EventProxy.js~EventProxy.html
 */

/**
 * @external {TyphonEvents} https://docs.typhonjs.io/typhonjs-backbone-esnext/backbone-esnext-events/class/src/TyphonEvents.js~TyphonEvents.html
 */
`;

exports.onHandleVirtual = function(ev)
{
   ev.data.code.push({ code, message: __filename });
};