// scripts/css_inserter.js

const fs = require('fs');
const css = require('css');
const path = require('path');

// Read input from stdin
let inputChunks = [];
process.stdin.on('data', (chunk) => {
  inputChunks.push(chunk);
});

process.stdin.on('end', () => {
  const inputData = inputChunks.join('');

  let parsedInput;
  try {
    parsedInput = JSON.parse(inputData);
  } catch (e) {
    console.error('Error parsing input JSON:', e.message);
    process.exit(1);
  }

  const { code, documentation, language } = parsedInput;

  if (language.toLowerCase() !== 'css') {
    console.error('Unsupported language:', language);
    process.exit(1);
  }

  let ast;
  try {
    ast = css.parse(code, { source: 'input.css' });
  } catch (e) {
    console.error('Parsing error:', e.message);
    process.exit(1);
  }

  // Insert comments based on documentation
  if (documentation.rules) {
    documentation.rules.forEach(docRule => {
      const selectors = docRule.selectors;
      const docstring = docRule.docstring;

      ast.stylesheet.rules.forEach(rule => {
        if (rule.type === 'rule') {
          const ruleSelectors = rule.selectors;
          const isMatch = selectors.some(sel => ruleSelectors.includes(sel));
          if (isMatch && docstring) {
            // Insert comment before the rule
            if (!rule.comments) {
              rule.comments = [];
            }
            rule.comments.unshift(docstring);
          }
        }
      });
    });
  }

  // Stringify the modified AST
  const modifiedCSS = css.stringify(ast);
  console.log(modifiedCSS);
});
