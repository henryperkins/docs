// src/App.tsx
import React from 'react';
import { RouterProvider } from 'react-router-dom';
import { createBrowserRouter, createRoutesFromElements, Route } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext'; // Import your theme provider
import { QueryClient, QueryClientProvider } from 'react-query';
import { ReactQueryDevtools } from 'react-query/devtools';
import { ToastProvider } from './components/ui/use-toast'; // Import toast provider
import ProjectOverview from './pages/ProjectOverview';
import Documentation from './pages/Documentation';
import Settings from './pages/Settings';
import NotFound from './pages/NotFound';
import ErrorBoundary from './app/ErrorBoundary';
import './index.css';

// Create a query client
const queryClient = new QueryClient();


const router = createBrowserRouter(
  createRoutesFromElements(
    <Route path="/" element={<ProjectOverview />} errorElement={<ErrorBoundary />}>
      <Route path="projects/:projectId" element={<Documentation />} />
      <Route path="settings" element={<Settings />} />
      <Route path="*" element={<NotFound />} />
    </Route>
  )
);

function App() {
  return (
    <React.StrictMode>
      <QueryClientProvider client={queryClient}> {/* Wrap with QueryClientProvider */}
        <ThemeProvider>
          <ToastProvider>
            <RouterProvider router={router} />
            <ReactQueryDevtools initialIsOpen={false} />
          </ToastProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </React.StrictMode>
  );
}

export default App;