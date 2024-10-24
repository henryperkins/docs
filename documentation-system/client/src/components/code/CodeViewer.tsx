// src/components/code/CodeViewer.tsx
import React, { useState, useEffect } from 'react';
import { Documentation } from '../../types/documentation';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { docco } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import { useDocumentation } from '../../hooks/useDocumentation';

interface CodeViewerProps {
  documentation: Documentation | null;
  isDarkMode: boolean;
  selectedFile?: string | null;
}

export const CodeViewer: React.FC<CodeViewerProps> = ({ documentation, isDarkMode, selectedFile }) => {
  const { getCodeByPath } = useDocumentation({ projectId: 'your-project-id' });
  const [code, setCode] = useState<string | null>(null);
  const [loadingCode, setLoadingCode] = useState<boolean>(false);

  useEffect(() => {
    const fetchCode = async () => {
      if (selectedFile) {
        setLoadingCode(true);
        try {
          const codeContent = await getCodeByPath(selectedFile);
          setCode(codeContent);
        } catch (error) {
          console.error("Error fetching code:", error);
          setCode(null);
        } finally {
          setLoadingCode(false);
        }
      } else {
        setCode(null);
      }
    };

    fetchCode();
  }, [selectedFile, getCodeByPath]);

  if (!documentation && !selectedFile) {
    return <p>No documentation or file selected.</p>;
  }

  if (selectedFile && loadingCode) {
    return <p>Loading code...</p>;
  }

  if (selectedFile && !code) {
    return <p>Error loading code or code not found.</p>;
  }

  const displayedClasses = selectedFile ? (documentation?.classes || []) : documentation?.classes || [];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-3 space-y-6">
        {selectedFile && code ? (
          <div className="rounded-lg bg-gray-100 dark:bg-gray-900 shadow-sm p-6">
            <SyntaxHighlighter language={selectedFile.split(".").pop() || "text"} style={docco} showLineNumbers={true}>
              {code}
            </SyntaxHighlighter>
          </div>
        ) : displayedClasses.length > 0 ? (
          displayedClasses.map((cls) => (
            <div
              key={cls.name}
              id={cls.name}
              className={`rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-sm p-6`}
            >
              <h3 className="text-xl font-semibold mb-4">{cls.name}</h3>
              <pre className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 overflow-x-auto">
                <code>{cls.docstring}</code>
              </pre>
              <div className="mt-6 space-y-6">
                {cls.methods.map(method => (
                  <div
                    key={method.name}
                    id={`${cls.name}-${method.name}`}
                    className="border-t border-gray-200 dark:border-gray-700 pt-4"
                  >
                    <h4 className="text-lg font-medium mb-2">{method.name}</h4>
                    <div className="flex flex-wrap gap-2 mb-2">
                      {method.async && (
                        <span className="px-2 py-1 text-xs rounded-full bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
                          async
                        </span>
                      )}
                      <span className="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                        complexity: {method.complexity}
                      </span>
                      <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                        type: {method.type}
                      </span>
                    </div>
                    <pre className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 overflow-x-auto">
                      <code>{method.docstring}</code>
                    </pre>
                    {method.args.length > 0 && (
                      <div className="mt-2">
                        <h5 className="font-medium mb-1">Arguments</h5>
                        <ul className="list-disc list-inside space-y-1">
                          {method.args.map(arg => (
                            <li key={arg} className="text-gray-600 dark:text-gray-300">
                              {arg}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))
        ) : (
          <p>No classes found in the documentation.</p>
        )}
      </div>
    </div>
  );
};