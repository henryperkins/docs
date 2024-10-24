import React, { useState, useEffect } from 'react';
import { useDocumentationContext } from '../stores/DocumentationContext';
import { Header } from './documentation/Header';
import { ClassesList } from './documentation/ClassesList';
import { CodeViewer } from './code/CodeViewer';
import { MetricsOverview } from './metrics/MetricsOverview';
import { DetailedMetrics } from './metrics/DetailedMetrics';
import { LoadingSpinner, ErrorDisplay } from './common';

const DocumentationViewer: React.FC = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'code' | 'metrics'>('overview');
  const { state, dispatch } = useDocumentationContext();

  useEffect(() => {
    fetchDocumentation();
  }, []);

  const fetchDocumentation = async () => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      const response = await fetch('http://localhost:5000/api/documentation');
      const data = await response.json();
      dispatch({ type: 'SET_DOCUMENTATION', payload: data });
    } catch (err) {
      dispatch({ 
        type: 'SET_ERROR', 
        payload: err instanceof Error ? err.message : 'Failed to load documentation'
      });
    }
  };

  if (state.loading) {
    return <LoadingSpinner />;
  }

  if (state.error) {
    return <ErrorDisplay message={state.error} />;
  }

  if (!state.documentation) {
    return <ErrorDisplay message="No documentation available" />;
  }

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <div className="space-y-8">
            <MetricsOverview
              metrics={state.documentation.metrics}
              functionCount={state.documentation.functions.length}
              classCount={state.documentation.classes.length}
              isDarkMode={isDarkMode}
            />
            <ClassesList
              classes={state.documentation.classes}
              isDarkMode={isDarkMode}
            />
          </div>
        );

      case 'code':
        return (
          <CodeViewer
            documentation={state.documentation}
            isDarkMode={isDarkMode}
          />
        );

      case 'metrics':
        return (
          <DetailedMetrics
            metrics={state.documentation.metrics}
            isDarkMode={isDarkMode}
          />
        );

      default:
        return null;
    }
  };

  return (
    <div className={isDarkMode ? 'dark' : ''}>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <Header
          isDarkMode={isDarkMode}
          onDarkModeToggle={() => setIsDarkMode(!isDarkMode)}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />

        <main className="max-w-7xl mx-auto px-4 py-8">
          {renderContent()}
        </main>
      </div>
    </div>
  );
};

export default DocumentationViewer;