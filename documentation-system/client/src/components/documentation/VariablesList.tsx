// src/components/documentation/VariablesList.tsx
import React from 'react';
import { Variable } from '../../types/documentation';

interface VariablesListProps {
  variables: Variable[];
}

export const VariablesList: React.FC<VariablesListProps> = ({ variables }) => {
  return (
    <div>
      <h3>Variables</h3>
      <ul>
        {variables.map(variable => (
          <li key={variable.name}>
            <h4>{variable.name}</h4>
            <p>Type: {variable.type}</p>
            <p>Description: {variable.description}</p>
          </li>
        ))}
      </ul>
    </div>
  );
};