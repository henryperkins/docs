// src/stores/DocumentationContext.tsx
import React, { createContext, useContext, useReducer } from 'react';
import { Documentation, DocumentationState } from '../types/documentation';

type Action =
  | { type: 'SET_DOCUMENTATION'; payload: Documentation | null } // Allow null payload
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_SEARCH_QUERY'; payload: string }
  | { type: 'SET_SELECTED_FILE'; payload: string | null } // New action for selected file
  | { type: 'SET_DISPLAYED_DOCUMENTATION'; payload: Documentation | null } // New action for displayed documentation
  | { type: 'TOGGLE_SECTION'; payload: string };

interface DocumentationContextType {
  state: DocumentationState;
  dispatch: React.Dispatch<Action>;
  setSelectedFile: (filePath: string | null) => void; // Add setSelectedFile function
  setSearchQuery: (query: string) => void; // Add setSearchQuery function
}

const initialState: DocumentationState = {
  documentation: null,
  displayedDocumentation: null, // New state for displayed documentation
  loading: false,
  error: null,
  searchQuery: '',
  selectedFile: null, // New state for selected file
  expandedSections: new Set(),
};

const DocumentationContext = createContext<DocumentationContextType | undefined>(undefined);

function documentationReducer(state: DocumentationState, action: Action): DocumentationState {
  switch (action.type) {
    case 'SET_DOCUMENTATION':
      return { ...state, documentation: action.payload, loading: false, error: null };
    case 'SET_DISPLAYED_DOCUMENTATION':
      return { ...state, displayedDocumentation: action.payload, loading: false, error: null };
    case 'SET_LOADING':
      return { ...state, loading: action.payload, error: null };
    case 'SET_ERROR':
      return { ...state, error: action.payload, loading: false };
    case 'SET_SEARCH_QUERY':
      return { ...state, searchQuery: action.payload };
    case 'SET_SELECTED_FILE':
      return { ...state, selectedFile: action.payload };
    case 'TOGGLE_SECTION': {
      const newSections = new Set(state.expandedSections);
      if (newSections.has(action.payload)) {
        newSections.delete(action.payload);
      } else {
        newSections.add(action.payload);
      }
      return { ...state, expandedSections: newSections };
    }
    default:
      return state;
  }
}

export function DocumentationProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(documentationReducer, initialState);

  const setSelectedFile = (filePath: string | null) => {
    dispatch({ type: 'SET_SELECTED_FILE', payload: filePath });
  };

  const setSearchQuery = (query: string) => {
    dispatch({ type: 'SET_SEARCH_QUERY', payload: query });
  };

  const contextValue = {
    state,
    dispatch,
    setSelectedFile, // Add setSelectedFile to context
    setSearchQuery, // Add setSearchQuery to context
  };

  return (
    <DocumentationContext.Provider value={contextValue}>
      {children}
    </DocumentationContext.Provider>
  );
}

export function useDocumentationContext() {
  const context = useContext(DocumentationContext);
  if (context === undefined) {
    throw new Error('useDocumentationContext must be used within a DocumentationProvider');
  }
  return context;
}