// scripts/html_inserter.js

const fs = require('fs');
const cheerio = require('cheerio');
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

  if (language.toLowerCase() !== 'html') {
    console.error('Unsupported language:', language);
    process.exit(1);
  }

  const $ = cheerio.load(code, { xmlMode: false });

  // Traverse documentation to insert comments
  if (documentation.classes) {
    documentation.classes.forEach(cls => {
      // HTML does not have classes in the OOP sense; skip or map as needed
    });
  }

  if (documentation.functions) {
    documentation.functions.forEach(func => {
      // HTML does not have functions; skip or map as needed
    });
  }

  if (documentation.variables) {
    documentation.variables.forEach(varObj => {
      // HTML does not have variables; skip or map as needed
    });
  }

  if (documentation.constants) {
    documentation.constants.forEach(constObj => {
      // HTML does not have constants; skip or map as needed
    });
  }

  // Insert docstrings as comments before specific tags
  if (documentation.elements) {
    documentation.elements.forEach(elemDoc => {
      const tag = elemDoc.tag;
      const docstring = elemDoc.docstring;
      if (docstring) {
        $(tag).each(function(i, elem) {
          // Insert comment before the element
          $(elem).before(`<!-- ${docstring} -->\n`);
        });
      }
    });
  }

  // Generate the modified HTML
  const modifiedHTML = $.html();
  console.log(modifiedHTML);
});
