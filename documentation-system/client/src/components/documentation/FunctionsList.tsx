// src/components/documentation/FunctionsList.tsx
import React from 'react';
import { Method } from '../../types/documentation';

interface FunctionsListProps {
  functions: Method[];
}

export const FunctionsList: React.FC<FunctionsListProps> = ({ functions }) => {
  return (
    <div>
      <h3>Functions</h3>
      <ul>
        {functions.map(func => (
          <li key={func.name}>
            <h4>{func.name}</h4>
            <p>{func.docstring}</p>
            <p>Arguments: {func.args.join(', ')}</p>
            <p>Complexity: {func.complexity}</p>
          </li>
        ))}
      </ul>
    </div>
  );
};