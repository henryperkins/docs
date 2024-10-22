var code = `
/**
 * @external {ValidationEntry} https://docs.typhonjs.io/typhonjs-node-utils/typhonjs-object-util/typedef/index.html#static-typedef-ValidationEntry
 */
`;

exports.onHandleVirtual = function(ev)
{
   ev.data.code.push({ code, message: __filename });
};
