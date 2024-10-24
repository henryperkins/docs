// src/components/documentation/FileSummary.tsx
import React from 'react';

interface FileSummaryProps {
  summary: string;
}

export const FileSummary: React.FC<FileSummaryProps> = ({ summary }) => {
  return <p>{summary}</p>;
};