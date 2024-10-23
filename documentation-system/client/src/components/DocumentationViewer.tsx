import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { 
  Search, Moon, Sun, ChevronDown, ChevronRight, 
  Code, FileText, Activity, Book 
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, 
  PolarRadiusAxis, Radar 
} from 'recharts';
import { documentationService, Documentation } from '../services/documentationService';

interface Documentation {
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

const DocumentationViewer: React.FC = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [searchQuery, setSearchQuery] = useState('');
  const [documentation, setDocumentation] = useState<Documentation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({});

  useEffect(() => {
    fetchDocumentation();
  }, []);

  const fetchDocumentation = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5000/api/documentation');
      const data = await response.json();
      setDocumentation(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load documentation');
      console.error('Error fetching documentation:', err);
    } finally {
      setLoading(false);
    }
  };

  const toggleSection = (sectionId: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };

  const MetricsCard: React.FC<{
    title: string;
    value: string | number;
    trend?: number;
    icon: React.FC;
  }> = ({ title, value, trend, icon: Icon }) => (
    <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-sm`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Icon className="w-5 h-5 text-gray-400" />
          <span className="text-sm text-gray-500">{title}</span>
        </div>
        {trend !== undefined && (
          <span className={trend > 0 ? 'text-green-500' : 'text-red-500'}>
            {trend > 0 ? '+' : ''}{trend}%
          </span>
        )}
      </div>
      <p className="text-2xl font-bold mt-2">{value}</p>
    </div>
  );

  const CodeQualityChart: React.FC<{ metrics: Documentation['metrics'] }> = ({ metrics }) => {
    const data = [
      { name: 'Maintainability', value: metrics.maintainability_index },
      { name: 'Complexity', value: Math.min(100, metrics.complexity * 10) },
      { name: 'Volume', value: Math.min(100, metrics.halstead.volume / 100) },
      { name: 'Effort', value: Math.min(100, metrics.halstead.effort / 1000) },
    ];

    return (
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data}>
            <PolarGrid />
            <PolarAngleAxis dataKey="name" />
            <PolarRadiusAxis />
            <Radar
              name="Code Quality"
              dataKey="value"
              stroke="#4F46E5"
              fill="#4F46E5"
              fillOpacity={0.4}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const ClassViewer: React.FC<{ classInfo: Documentation['classes'][0] }> = ({ classInfo }) => (
    <div className={`mt-4 p-4 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-sm`}>
      <div 
        className="flex items-center justify-between cursor-pointer"
        onClick={() => toggleSection(classInfo.name)}
      >
        <h3 className="text-lg font-semibold">{classInfo.name}</h3>
        {expandedSections[classInfo.name] ? <ChevronDown /> : <ChevronRight />}
      </div>
      
      {expandedSections[classInfo.name] && (
        <div className="mt-4">
          <ReactMarkdown className="prose dark:prose-invert">
            {classInfo.docstring}
          </ReactMarkdown>
          
          <div className="mt-4">
            <h4 className="font-medium mb-2">Methods</h4>
            {classInfo.methods.map(method => (
              <div key={method.name} className="mt-2 pl-4 border-l-2 border-gray-200">
                <div className="flex items-center justify-between">
                  <h5 className="font-medium">{method.name}</h5>
                  <div className="flex items-center space-x-2">
                    {method.async && (
                      <span className="px-2 py-1 text-xs rounded-full bg-purple-100 text-purple-800">
                        async
                      </span>
                    )}
                    <span className="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800">
                      complexity: {method.complexity}
                    </span>
                  </div>
                </div>
                <ReactMarkdown className="mt-1 text-sm text-gray-600 dark:text-gray-300">
                  {method.docstring}
                </ReactMarkdown>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen text-red-500">
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className={isDarkMode ? 'dark' : ''}>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        {/* Header */}
        <nav className="bg-white dark:bg-gray-800 shadow-sm">
          <div className="max-w-7xl mx-auto px-4 py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Book className="w-6 h-6 mr-2" />
                <h1 className="text-xl font-bold">Documentation</h1>
              </div>
              
              <div className="flex items-center space-x-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search documentation..."
                    className="pl-10 pr-4 py-2 rounded-lg border dark:bg-gray-700 dark:border-gray-600"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                </div>
                <button
                  onClick={() => setIsDarkMode(!isDarkMode)}
                  className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                  {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                </button>
              </div>
            </div>

            {/* Navigation Tabs */}
            <div className="flex space-x-4 mt-4">
              {['overview', 'code', 'metrics'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium ${
                    activeTab === tab
                      ? 'bg-blue-500 text-white'
                      : 'text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'
                  }`}
                >
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 py-8">
          {documentation && activeTab === 'overview' && (
            <>
              {/* Metrics Overview */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <MetricsCard
                  title="Code Quality"
                  value={`${documentation.metrics.maintainability_index}%`}
                  icon={Activity}
                />
                <MetricsCard
                  title="Functions"
                  value={documentation.functions.length}
                  icon={Code}
                />
                <MetricsCard
                  title="Classes"
                  value={documentation.classes.length}
                  icon={FileText}
                />
              </div>

              {/* Code Quality Visualization */}
              <div className={`mt-6 p-6 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-sm`}>
                <h2 className="text-lg font-semibold mb-4">Code Quality Metrics</h2>
                <CodeQualityChart metrics={documentation.metrics} />
              </div>

              {/* Documentation Content */}
              <div className="mt-6">
                <h2 className="text-lg font-semibold mb-4">Classes</h2>
                {documentation.classes.map(classInfo => (
                  <ClassViewer key={classInfo.name} classInfo={classInfo} />
                ))}
              </div>
            </>
          )}

          {activeTab === 'metrics' && documentation && (
            <div className="grid gap-6">
              {/* Detailed Metrics */}
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default DocumentationViewer;
