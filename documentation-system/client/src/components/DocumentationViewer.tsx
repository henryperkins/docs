// src/components/DocumentationViewer.tsx
import React, { useState, useEffect } from 'react';
import { useDocumentation } from '../hooks/useDocumentation';
import { MetricsOverview } from './metrics/MetricsOverview';
import { ClassesList } from './documentation/ClassesList';
import { CodeViewer } from './code/CodeViewer';
import { DetailedMetrics } from './metrics/DetailedMetrics';
import { LoadingSpinner, ErrorDisplay } from './common';
import { FileTree } from './navigation/FileTree';
import { SearchBar } from './search/SearchBar';
import { Documentation, Method, Class, CodeMetrics } from '../../types/documentation';
import { FileSummary } from './documentation/FileSummary';
import { FunctionsList } from './documentation/FunctionsList';
import { VariablesList } from './documentation/VariablesList';
import { NavigationSidebar } from './navigation/NavigationSidebar';
import { DocumentationDetails } from './documentation/DocumentationDetails';

const DocumentationViewer: React.FC = () => {
  const { documentation, loading, error, searchResults, searchQuery, setSearchQuery, getDocumentationByPath } = useDocumentation({
    projectId: 'your-project-id', // Replace with your actual project ID
  });
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [displayedDocumentation, setDisplayedDocumentation] = useState<Documentation | null>(null);
  const [codeContent, setCodeContent] = useState<string>('');

  useEffect(() => {
    if (selectedFile) {
      getDocumentationByPath(selectedFile)
        .then(data => {
          setDisplayedDocumentation(data);
          // Fetch the code content separately if needed
          // Assuming you have a function to fetch the code content
          fetchCodeContent(selectedFile).then(setCodeContent);
        })
        .catch(err => console.error("Error fetching documentation:", err));
    }
  }, [selectedFile, getDocumentationByPath]);

  useEffect(() => {
    // Update displayedDocumentation when documentation or searchQuery changes
    if (documentation && searchQuery && !selectedFile) { // Only filter if no file is selected
      const filtered = { ...documentation };
      filtered.classes = filtered.classes.filter(cls =>
        cls.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        cls.docstring.toLowerCase().includes(searchQuery.toLowerCase()) ||
        cls.methods.some(method =>
          method.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          method.docstring.toLowerCase().includes(searchQuery.toLowerCase())
        )
      );
      filtered.functions = filtered.functions.filter(func =>
        func.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        func.docstring.toLowerCase().includes(searchQuery.toLowerCase())
      );
      setDisplayedDocumentation(filtered);
    } else if (!selectedFile) { // Reset filtering if search query is empty and no file selected
      setDisplayedDocumentation(documentation);
    }
  }, [documentation, searchQuery, selectedFile]);

  const handleFileSelect = (filePath: string | null) => {
    setSelectedFile(filePath);
    if (!filePath) { // If null is passed, reset to project overview
      setDisplayedDocumentation(documentation);
    }
  };

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorDisplay message={error} />;

  // Show message if no documentation is available AND no file is selected
  if (!displayedDocumentation && !selectedFile) {
    return <ErrorDisplay message="No documentation available. Please select a file or generate documentation." />;
  }

  return (
    <div className="flex">
      <div className="w-64 bg-gray-100 dark:bg-gray-800 p-4">
        <FileTree
          documentation={documentation} // Pass the full documentation here
          onFileSelect={handleFileSelect}
        />
        <SearchBar searchQuery={searchQuery} setSearchQuery={setSearchQuery} />
      </div>
      <div className="flex-1 p-8">
        {displayedDocumentation ? ( // Conditionally render based on displayedDocumentation
          <div className="space-y-8">
            {selectedFile ? ( // Show file-specific info if a file is selected
              <div>
                <h2>{selectedFile}</h2>
                <FileSummary summary={displayedDocumentation.summary} />
                <ClassesList classes={displayedDocumentation.classes} />
                <FunctionsList functions={displayedDocumentation.functions} />
                <VariablesList variables={displayedDocumentation.variables} />
                <CodeViewer code={codeContent} /> {/* Pass the actual code content here */}
              </div>
            ) : ( // Show project overview if no file is selected
              <div>
                <div className="prose dark:prose-invert">
                  <h1>Documentation Overview</h1>
                  <p>{displayedDocumentation.summary}</p>
                </div>
                <MetricsOverview metrics={displayedDocumentation.metrics} functionCount={displayedDocumentation.functions.length} classCount={displayedDocumentation.classes.length} />
                <ClassesList classes={displayedDocumentation.classes} />
                <CodeViewer code={""} />  {/* Pass some placeholder or default code here */}
                <DetailedMetrics metrics={displayedDocumentation.metrics} />
              </div>
            )}
          </div>
        ) : selectedFile ? ( // Show loading state while fetching file documentation
          <LoadingSpinner />
        ) : null} {/* Don't render anything if no documentation and no file selected */}
      </div>
    </div>
  );
};

export default DocumentationViewer;