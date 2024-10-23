// src/hooks/useDocumentation.ts
import { useState, useEffect } from 'react';
import { documentationService, DocumentationResponse } from '../services/documentationService';

interface UseDocumentationParams {
  projectId: string;
  filePath?: string;
}

interface UseDocumentationResult {
  documentation: DocumentationResponse | null;
  loading: boolean;
  error: string | null;
  searchResults: DocumentationResponse[];
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  refreshDocumentation: () => Promise<void>;
}

export const useDocumentation = ({
  projectId,
  filePath
}: UseDocumentationParams): UseDocumentationResult => {
  const [documentation, setDocumentation] = useState<DocumentationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<DocumentationResponse[]>([]);
  const [searchQuery, setSearchQuery] = useState('');

  const fetchDocumentation = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await documentationService.getDocumentation(projectId, filePath);
      setDocumentation(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const searchDocumentation = async (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    try {
      const results = await documentationService.searchDocumentation(projectId, query);
      setSearchResults(results);
    } catch (err) {
      console.error('Search failed:', err);
      setSearchResults([]);
    }
  };

  useEffect(() => {
    fetchDocumentation();
  }, [projectId, filePath]);

  useEffect(() => {
    const debounceTimeout = setTimeout(() => {
      searchDocumentation(searchQuery);
    }, 300);

    return () => clearTimeout(debounceTimeout);
  }, [searchQuery]);

  return {
    documentation,
    loading,
    error,
    searchResults,
    searchQuery,
    setSearchQuery,
    refreshDocumentation: fetchDocumentation
  };
};