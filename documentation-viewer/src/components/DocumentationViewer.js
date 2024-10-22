import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Moon, 
  Sun, 
  ChevronDown, 
  ChevronRight, 
  Code, 
  FileText, 
  Activity,
  AlertCircle 
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar 
} from 'recharts';

const LoadingSpinner = () => (
  <div className="flex items-center justify-center min-h-screen">
    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
  </div>
);

const ErrorMessage = ({ message }) => (
  <div className="flex items-center justify-center min-h-screen text-red-500">
    <AlertCircle className="w-6 h-6 mr-2" />
    <span>{message}</span>
  </div>
);

const DocumentationViewer = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [searchQuery, setSearchQuery] = useState('');
  const [documentation, setDocumentation] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        // Fetch documentation
        const docResponse = await fetch('http://localhost:5000/api/documentation');
        const docData = await docResponse.json();
        
        // Fetch metrics
        const metricsResponse = await fetch('http://localhost:5000/api/metrics');
        const metricsData = await metricsResponse.json();
        
        setDocumentation(docData);
        setMetrics(metricsData);
        setError(null);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('Failed to load documentation data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message={error} />;
  if (!documentation || !metrics) return <ErrorMessage message="No data available" />;

  console.log('Documentation:', documentation); // Debugging log
  console.log('Metrics:', metrics); // Debugging log

  return (
    <div className={isDarkMode ? 'dark' : ''}>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        {/* Header */}
        <nav className="bg-white dark:bg-gray-800 shadow-sm">
          <div className="max-w-7xl mx-auto px-4 py-3">
            <div className="flex items-center justify-between">
              <h1 className="text-xl font-bold">Documentation Viewer</h1>
              
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

            {/* Tabs */}
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
          {activeTab === 'overview' && documentation && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className={`p-6 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-lg`}>
                  <h3 className="text-lg font-semibold mb-2">Code Quality</h3>
                  <div className="text-3xl font-bold text-blue-500">
                    {documentation.code_quality?.overall_score || 'N/A'}%
                  </div>
                </div>
                
                <div className={`p-6 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-lg`}>
                  <h3 className="text-lg font-semibold mb-2">Coverage</h3>
                  <div className="text-3xl font-bold text-green-500">
                    {documentation.coverage?.overall || 'N/A'}%
                  </div>
                </div>
                
                <div className={`p-6 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-lg`}>
                  <h3 className="text-lg font-semibold mb-2">Maintainability</h3>
                  <div className="text-3xl font-bold text-purple-500">
                    {documentation.maintainability_index || 'N/A'}
                  </div>
                </div>
              </div>

              {/* Metrics Chart */}
              {metrics && (
                <div className={`p-6 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-lg`}>
                  <h3 className="text-lg font-semibold mb-4">Trends</h3>
                  <div className="h-64">
                    <ResponsiveContainer>
                      <LineChart data={metrics.complexity_trends}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Line type="monotone" dataKey="complexity" stroke="#8884d8" name="Complexity" />
                        <Line type="monotone" dataKey="coverage" stroke="#82ca9d" name="Coverage" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Add other tab content here */}
        </main>
      </div>
    </div>
  );
};

export default DocumentationViewer;
