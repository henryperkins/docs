import React, { createContext, useContext, useReducer } from 'react';
import { DocumentationState } from '../types/documentation';

type Action = 
  | { type: 'SET_DOCUMENTATION'; payload: Documentation }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_SEARCH_QUERY'; payload: string }
  | { type: 'TOGGLE_SECTION'; payload: string };

const initialState: DocumentationState = {
  documentation: null,
  loading: false,
  error: null,
  searchQuery: '',
  expandedSections: new Set()
};

const DocumentationContext = createContext<{
  state: DocumentationState;
  dispatch: React.Dispatch<Action>;
} | undefined>(undefined);

function documentationReducer(state: DocumentationState, action: Action): DocumentationState {
  switch (action.type) {
    case 'SET_DOCUMENTATION':
      return { ...state, documentation: action.payload, loading: false };
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload, loading: false };
    case 'SET_SEARCH_QUERY':
      return { ...state, searchQuery: action.payload };
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

  return (
    <DocumentationContext.Provider value={{ state, dispatch }}>
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