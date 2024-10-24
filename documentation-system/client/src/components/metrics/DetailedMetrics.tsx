import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { CodeMetrics } from '../../types/documentation';

interface DetailedMetricsProps {
  metrics: CodeMetrics;
  isDarkMode: boolean;
}

export const DetailedMetrics: React.FC<DetailedMetricsProps> = ({ metrics, isDarkMode }) => {
  const getMetricStatus = (value: number, threshold: number) => {
    if (value >= threshold * 1.2) return 'text-red-500 dark:text-red-400';
    if (value >= threshold) return 'text-yellow-500 dark:text-yellow-400';
    return 'text-green-500 dark:text-green-400';
  };

  const formatMetricValue = (value: number) => {
    return value.toFixed(2);
  };

  const metricCards = [
    {
      title: 'Maintainability Index',
      value: metrics.maintainability_index,
      threshold: 70,
      description: 'Indicates how maintainable the code is. Higher is better.'
    },
    {
      title: 'Cyclomatic Complexity',
      value: metrics.complexity,
      threshold: 10,
      description: 'Measures code complexity based on control flow.'
    },
    {
      title: 'Halstead Volume',
      value: metrics.halstead.volume,
      threshold: 100,
      description: 'Represents the size of the implementation.'
    },
    {
      title: 'Halstead Difficulty',
      value: metrics.halstead.difficulty,
      threshold: 20,
      description: 'Indicates how difficult the code is to understand.'
    },
    {
      title: 'Halstead Effort',
      value: metrics.halstead.effort,
      threshold: 1000,
      description: 'Estimated mental effort required to develop the code.'
    }
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {metricCards.map(metric => (
          <div 
            key={metric.title}
            className={`rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-sm p-4`}
          >
            <div className="flex justify-between items-start mb-2">
              <h3 className="font-medium">{metric.title}</h3>
              <span className={getMetricStatus(metric.value, metric.threshold)}>
                {formatMetricValue(metric.value)}
              </span>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-300">
              {metric.description}
            </p>
            {/* Simple progress bar */}
            <div className="mt-4 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${
                  getMetricStatus(metric.value, metric.threshold)
                    .replace('text', 'bg')
                }`}
                style={{ 
                  width: `${Math.min((metric.value / metric.threshold) * 100, 100)}%`
                }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Comparison Chart */}
      <div className={`rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-sm p-6`}>
        <h3 className="text-lg font-semibold mb-4">Metrics Comparison</h3>
        <div className="h-64">
          <ResponsiveContainer>
            <LineChart
              data={[
                {
                  name: 'Current',
                  maintainability: metrics.maintainability_index / 100,
                  complexity: metrics.complexity / 10,
                  volume: metrics.halstead.volume / 100,
                  difficulty: metrics.halstead.difficulty / 20,
                  effort: metrics.halstead.effort / 1000
                }
              ]}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="maintainability" 
                stroke="#4F46E5" 
                name="Maintainability"
              />
              <Line 
                type="monotone" 
                dataKey="complexity" 
                stroke="#EF4444" 
                name="Complexity"
              />
              <Line 
                type="monotone" 
                dataKey="volume" 
                stroke="#10B981" 
                name="Volume"
              />
              <Line 
                type="monotone" 
                dataKey="difficulty" 
                stroke="#F59E0B" 
                name="Difficulty"
              />
              <Line 
                type="monotone" 
                dataKey="effort" 
                stroke="#6366F1" 
                name="Effort"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};