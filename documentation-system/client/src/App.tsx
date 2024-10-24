// src/App.tsx
import React from 'react';
import { DocumentationProvider } from './stores/DocumentationContext';
import DocumentationViewer from './components/DocumentationViewer';

const App: React.FC = () => {
  return (
    <DocumentationProvider>
      <DocumentationViewer />
    </DocumentationProvider>
  );
};

export default App;