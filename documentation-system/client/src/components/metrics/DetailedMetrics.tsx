// src/components/metrics/DetailedMetrics.tsx
import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { CodeMetrics } from '../../types/documentation';
import { useDocumentation } from '../../hooks/useDocumentation';
import { MetricCard } from '../common/MetricCard';

interface DetailedMetricsProps {
  metrics: CodeMetrics;
  isDarkMode: boolean;
  selectedFile?: string | null;
  projectId: string;
}

export const DetailedMetrics: React.FC<DetailedMetricsProps> = ({ metrics, isDarkMode, selectedFile, projectId }) => {
  const { getMetricsHistoryByPath } = useDocumentation({ projectId });
  const [metricsHistory, setMetricsHistory] = useState<CodeMetrics[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);

  useEffect(() => {
    const fetchMetricsHistory = async () => {
      if (selectedFile) {
        setLoadingHistory(true);
        try {
          const history = await getMetricsHistoryByPath(selectedFile);
          setMetricsHistory(history);
        } catch (error) {
          console.error("Error fetching metrics history:", error);
        } finally {
          setLoadingHistory(false);
        }
      } else {
        setMetricsHistory([]); // Clear history if no file is selected
      }
    };
    fetchMetricsHistory();
  }, [selectedFile, projectId, getMetricsHistoryByPath]);

  const formatMetricValue = (value: number | undefined) => value?.toFixed(2) || "N/A";

  const getMetricStatus = (value: number | undefined, threshold: number): string => {
    if (value === undefined || value === null) return 'text-gray-500';
    if (value >= threshold * 1.2) return 'text-red-500 dark:text-red-400';
    if (value >= threshold) return 'text-yellow-500 dark:text-yellow-400';
    return 'text-green-500 dark:text-green-400';
  };

  const metricCardsData = [
    {
      title: 'Maintainability Index',
      value: metrics.maintainability_index,
      threshold: 70,
      description: 'Indicates how maintainable the code is. Higher is better.',
      icon: Activity, // Replace with actual icon component
    },
    {
      title: 'Cyclomatic Complexity',
      value: metrics.complexity,
      threshold: 10,
      description: 'Measures code complexity based on control flow.',
      icon: Code, // Replace with actual icon component
    },
    {
      title: 'Halstead Volume',
      value: metrics.halstead.volume,
      threshold: 100,
      description: 'Represents the size of the implementation.',
      icon: FileText, // Replace with actual icon component
    },
    {
      title: 'Halstead Difficulty',
      value: metrics.halstead.difficulty,
      threshold: 20,
      description: 'Indicates how difficult the code is to understand.',
      icon: FileText, // Replace with actual icon component
    },
    {
      title: 'Halstead Effort',
      value: metrics.halstead.effort,
      threshold: 1000,
      description: 'Estimated mental effort required to develop the code.',
      icon: FileText, // Replace with actual icon component
    },
  ];

  // Function to format data for Recharts, including Halstead metrics
  const formatChartData = (history: CodeMetrics[]) => {
    return history.map((metric, index) => ({
      name: index.toString(), // Use index as X-axis (you might want to use timestamps or version numbers if available)
      maintainability_index: metric.maintainability_index,
      complexity: metric.complexity,
      ...metric.halstead, // Spread Halstead metrics directly into the data object
    }));
  };

  const chartData = formatChartData(metricsHistory);

  if (loadingHistory) {
    return <p>Loading metrics history...</p>;
  }

  return (
    <div className="space-y-6">
      {/* Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {metricCardsData.map(metric => (
          <MetricCard
            key={metric.title}
            title={metric.title}
            value={formatMetricValue(metric.value)}
            status={getMetricStatus(metric.value, metric.threshold)}
            icon={metric.icon} // Pass the icon
          />
        ))}
      </div>

      {/* Comparison Chart */}
      <div className={`rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-sm p-6`}>
        <h3 className="text-lg font-semibold mb-4">Metrics History</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" label={{ value: 'Versions/Time', position: 'insideBottomRight', offset: -5 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="maintainability_index" stroke="#4F46E5" name="Maintainability" />
              <Line type="monotone" dataKey="complexity" stroke="#EF4444" name="Complexity" />
              {/* Halstead Metrics Lines */}
              <Line type="monotone" dataKey="volume" stroke="#10B981" name="Volume" />
              <Line type="monotone" dataKey="difficulty" stroke="#F59E0B" name="Difficulty" />
              <Line type="monotone" dataKey="effort" stroke="#6366F1" name="Effort" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Halstead Metrics Details (New Section) */}
      <div className={`rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-sm p-6`}>
        <h3 className="text-lg font-semibold mb-4">Halstead Metrics Details</h3>
        {/* Display detailed Halstead metrics here */}
        <pre>{JSON.stringify(metrics.halstead, null, 2)}</pre> {/* Example: Displaying raw JSON data */}
      </div>
    </div>
  );
};