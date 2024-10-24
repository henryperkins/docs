import React from 'react';
import { Activity, Code, FileText } from 'lucide-react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';
import { MetricCard } from '../common/MetricCard';
import { CodeMetrics } from '../../types/documentation';

interface MetricsOverviewProps {
  metrics: CodeMetrics;
  functionCount: number;
  classCount: number;
  isDarkMode: boolean;
}

export const MetricsOverview: React.FC<MetricsOverviewProps> = ({
  metrics,
  functionCount,
  classCount,
  isDarkMode
}) => {
  const getQualityData = (metrics: CodeMetrics) => [
    { metric: 'Maintainability', value: metrics.maintainability_index },
    { metric: 'Complexity', value: Math.min(100, metrics.complexity * 10) },
    { metric: 'Volume', value: Math.min(100, metrics.halstead.volume / 100) },
    { metric: 'Effort', value: Math.min(100, metrics.halstead.effort / 1000) }
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <MetricCard
          title="Code Quality"
          value={`${metrics.maintainability_index}%`}
          icon={Activity}
        />
        <MetricCard
          title="Functions"
          value={functionCount}
          icon={Code}
        />
        <MetricCard
          title="Classes"
          value={classCount}
          icon={FileText}
        />
      </div>

      <div className={`p-6 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-sm`}>
        <h2 className="text-lg font-semibold mb-4">Code Quality Metrics</h2>
        <div className="h-64">
          <ResponsiveContainer>
            <RadarChart data={getQualityData(metrics)}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" />
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
      </div>
    </div>
  );
};