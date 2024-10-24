import React from 'react';
import { File, Hash } from 'lucide-react';
import { Documentation } from '../../types/documentation';

interface CodeViewerProps {
  documentation: Documentation;
  isDarkMode: boolean;
}

export const CodeViewer: React.FC<CodeViewerProps> = ({ documentation, isDarkMode }) => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
      {/* Navigation Sidebar */}
      <div className={`lg:col-span-1 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-sm p-4`}>
        <h2 className="text-lg font-semibold mb-4">Navigation</h2>
        <div className="space-y-2">
          {documentation.classes.map(cls => (
            <div key={cls.name}>
              <a 
                href={`#${cls.name}`}
                className="flex items-center space-x-2 text-gray-600 dark:text-gray-300 hover:text-blue-500 dark:hover:text-blue-400"
              >
                <File className="w-4 h-4" />
                <span>{cls.name}</span>
              </a>
              <div className="ml-4 mt-1 space-y-1">
                {cls.methods.map(method => (
                  <a
                    key={method.name}
                    href={`#${cls.name}-${method.name}`}
                    className="flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400 hover:text-blue-500 dark:hover:text-blue-400"
                  >
                    <Hash className="w-3 h-3" />
                    <span>{method.name}</span>
                  </a>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="lg:col-span-3 space-y-6">
        {documentation.classes.map(cls => (
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
        ))}
      </div>
    </div>
  );
};