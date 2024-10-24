// src/components/navigation/NavigationSidebar.tsx
import React from 'react';
import { File, Hash } from 'lucide-react';
import { Documentation } from '../../types/documentation';

interface NavigationSidebarProps {
  documentation: Documentation;
  isDarkMode: boolean;
}

export const NavigationSidebar: React.FC<NavigationSidebarProps> = ({ documentation, isDarkMode }) => {
  return (
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
  );
};