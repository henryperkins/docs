![typhonjs-ast-walker](https://i.imgur.com/V3yiAPo.png)

[![NPM](https://img.shields.io/npm/v/typhonjs-ast-walker.svg?label=npm)](https://www.npmjs.com/package/typhonjs-ast-walker)
[![Documentation](http://docs.typhonjs.io/typhonjs-node-ast/typhonjs-ast-walker/badge.svg)](http://docs.typhonjs.io/typhonjs-node-ast/typhonjs-ast-walker/)
[![Code Style](https://img.shields.io/badge/code%20style-allman-yellowgreen.svg?style=flat)](https://en.wikipedia.org/wiki/Indent_style#Allman_style)
[![License](https://img.shields.io/badge/license-MPLv2-yellowgreen.svg?style=flat)](https://github.com/typhonjs-node-ast/typhonjs-ast-walker/blob/master/LICENSE)
[![Gitter](https://img.shields.io/gitter/room/typhonjs/TyphonJS.svg)](https://gitter.im/typhonjs/TyphonJS)

[![Build Status](https://travis-ci.org/typhonjs-node-ast/typhonjs-ast-walker.svg?branch=master)](https://travis-ci.org/typhonjs-node-ast/typhonjs-ast-walker)
[![Coverage](https://img.shields.io/codecov/c/github/typhonjs-node-ast/typhonjs-ast-walker.svg)](https://codecov.io/github/typhonjs-node-ast/typhonjs-ast-walker)
[![Dependency Status](https://www.versioneye.com/user/projects/5752aa747757a0004a1dddb7/badge.svg?style=flat)](https://www.versioneye.com/user/projects/5752aa747757a0004a1dddb7)

Provides a simple Javascript AST traversal utility that traverses all nodes / children regardless of type.

A single method `traverse` takes an AST object or array of nodes and a callback object which may contain two methods `enterNode` and `exitNode` which are invoked with the current node and the parent node respectively when entering and exiting a given node during traversal.

`enterNode` may return a array of strings which provide a set of children keys to ignore or `null` to skip traversing children keys entirely.

To install `typhonjs-ast-walker` provide this entry in `package.json`:
```
{
  ...
  "dependencies": {
    "typhonjs-ast-walker": "^0.2.0"
  }
}
```  

A simple example follows:
```
import walker from 'typhonjs-ast-walker';

const ast = .... // An AST tree.

walker.traverse(ast,
{
   enterNode: (node, parent) =>
   {
      console.log(`walker - enterNode: ${node.type}`);

      // Optional return statement to ignore specific children keys.
      // return node.type === 'ClassBody' ? ['body'] : undefined;

      // Optional return statement to ignore specific children keys or skip traversal entirely.
      // return node.type === 'ClassBody' ? null : undefined;
   },
   
   exitNode: (node, parent) =>
   {
      console.log(`walker - exitNode: ${node.type}`);
   }
});
 
```

Support for [typhonjs-plugin-manager](https://www.npmjs.com/package/typhonjs-plugin-manager) is also available and
when loading `typhonjs-ast-walker` as a plugin it will automatically register event bindings on the plugin eventbus
with these event categories:

`ast:walker:traverse` - invokes `traverse` of the default walker.

Example `typhonjs-plugin-manager` usage:
```
import PluginManager    from 'typhonjs-plugin-manager';
import eventbus         from 'backbone-esnext-eventbus';

const pluginManager = new PluginManager({ eventbus });

pluginManager.add({ name: typhonjs-ast-walker });

// This will automatically wire up typhonjs-ast-walker to the eventbus. 

eventbus.trigger('ast:walker:traverse, ast, 
{
   enterNode: (node, parent) =>
   {
      console.log(`walker - enterNode: ${node.type}`);
   },
   
   exitNode: (node, parent) =>
   {
      console.log(`walker - exitNode: ${node.type}`);
   }
});
```