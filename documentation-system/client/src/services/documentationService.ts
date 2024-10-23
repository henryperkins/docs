// src/services/documentationService.ts
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

export interface DocumentationResponse {
  project_id: string;
  file_path: string;
  version: string;
  language: string;
  summary: string;
  classes: Array<{
    name: string;
    docstring: string;
    methods: Array<{
      name: string;
      docstring: string;
      args: string[];
      async: boolean;
      complexity: number;
      type: string;
    }>;
  }>;
  functions: Array<{
    name: string;
    docstring: string;
    args: string[];
    async: boolean;
    complexity: number;
  }>;
  metrics: {
    maintainability_index: number;
    complexity: number;
    halstead: {
      volume: number;
      difficulty: number;
      effort: number;
    };
  };
}

export const documentationService = {
  async getDocumentation(projectId: string, filePath?: string): Promise<DocumentationResponse> {
    try {
      const params = new URLSearchParams();
      params.append('project_id', projectId);
      if (filePath) params.append('file_path', filePath);

      const response = await axios.get<DocumentationResponse>(
        `${API_URL}/documentation`,
        { params }
      );
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.error || 'Failed to fetch documentation');
      }
      throw error;
    }
  },

  async searchDocumentation(
    projectId: string,
    query: string
  ): Promise<DocumentationResponse[]> {
    try {
      const response = await axios.get<DocumentationResponse[]>(
        `${API_URL}/documentation/search`,
        {
          params: {
            project_id: projectId,
            query
          }
        }
      );
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.error || 'Failed to search documentation');
      }
      throw error;
    }
  }
};