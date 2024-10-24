// Types for the documentation data structures
export interface CodeMetrics {
    maintainability_index: number;
    complexity: number;
    halstead: {
      volume: number;
      difficulty: number;
      effort: number;
    };
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
  
  export interface Documentation {
    summary: string;
    classes: Class[];
    functions: Method[];
    metrics: CodeMetrics;
  }
  
  // API Response types
  export interface ApiResponse<T> {
    data: T;
    error?: string;
  }
  
  // Component prop types
  export interface MetricCardProps {
    title: string;
    value: string | number;
    trend?: number;
    icon: React.FC<{ className?: string }>;
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
    loading: boolean;
    error: string | null;
    searchQuery: string;
    expandedSections: Set<string>;
  }