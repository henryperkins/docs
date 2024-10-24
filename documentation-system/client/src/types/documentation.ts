// src/types/documentation.ts

export interface HalsteadMetrics {
  bugs: number;
  difficulty: number;
  effort: number;
  length: number;
  level: number;
  time: number;
  vocabulary: number;
  volume: number;
  operands: { [key: string]: number };
  operators: { [key: string]: number };
}

export interface CodeMetrics {
  maintainability_index: number;
  complexity: number;
  halstead: HalsteadMetrics;
  functionCount?: number; // Make optional
  classCount?: number; // Make optional
}

export interface Method {
  name: string;
  docstring: string;
  args: string[];
  async: boolean;
  complexity: number;
  type: string;
}

export interface Class {
  name: string;
  docstring: string;
  methods: Method[];
}

export interface FileDocumentation {
  file_path: string;
  summary: string;
  classes: Class[];
  functions: Method[];
  metrics: CodeMetrics;
  code?: string; // Add optional code content
}

export interface Documentation {
  summary?: string; // Make summary optional for project level
  files: FileDocumentation[];
  metrics: CodeMetrics; // Metrics for the entire project
}

// API Response types (make more generic)
export interface ApiResponse<T> {
  data?: T; // Data might be optional
  error?: string;
  status?: string; // Add status field (e.g., "success", "error")
  taskId?: string; // Add taskId for asynchronous operations
  progress?: number; // Add progress for asynchronous operations
}

// Component prop types
export interface MetricCardProps {
  title: string;
  value: string | number;
  trend?: number;
  icon: React.FC<{ className?: string }>;
  status?: string; // Add status for styling
}

export interface ClassCardProps {
  classInfo: Class;
  isDarkMode: boolean;
  onToggle: (className: string) => void;
  isExpanded: boolean;
}

export interface CodeQualityChartProps {
  metrics: CodeMetrics;
}

// State types
export interface DocumentationState {
  documentation: Documentation | null;
  displayedDocumentation: FileDocumentation | null; // Use FileDocumentation for selected file
  loading: boolean;
  error: string | null;
  searchQuery: string;
  selectedFile: string | null; // Add selectedFile
  expandedSections: Set<string>;
}