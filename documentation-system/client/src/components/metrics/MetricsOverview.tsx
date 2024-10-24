// src/components/metrics/MetricsOverview.tsx
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
  previousMetrics?: CodeMetrics; // Add optional previous metrics for comparison
}

export const MetricsOverview: React.FC<MetricsOverviewProps> = ({
  metrics,
  functionCount,
  classCount,
  isDarkMode,
  previousMetrics,
}) => {
  const formatMetricValue = (value: number | undefined) => value?.toFixed(2) || "N/A";

  const calculateTrend = (currentValue: number | undefined, previousValue: number | undefined): number | undefined => {
    if (currentValue === undefined || previousValue === undefined || previousValue === 0) {
      return undefined;
    }
    return ((currentValue - previousValue) / previousValue) * 100;
  };

  const metricCardsData = [
    {
      title: 'Maintainability Index',
      value: metrics.maintainability_index,
      trend: previousMetrics ? calculateTrend(metrics.maintainability_index, previousMetrics.maintainability_index) : undefined,
      icon: Activity,
    },
    {
      title: 'Functions',
      value: functionCount,
      trend: previousMetrics ? calculateTrend(functionCount, previousMetrics.functionCount) : undefined, // Assuming functionCount exists in previousMetrics
      icon: Code,
    },
    {
      title: 'Classes',
      value: classCount,
      trend: previousMetrics ? calculateTrend(classCount, previousMetrics.classCount) : undefined, // Assuming classCount exists in previousMetrics
      icon: FileText,
    },
  ];

  const radarChartData = [
    {
      subject: 'Maintainability',
      A: metrics.maintainability_index,
      B: previousMetrics?.maintainability_index || 0, // Provide a default value if previousMetrics is undefined
      fullMark: 100,
    },
    {
      subject: 'Complexity',
      A: metrics.complexity,
      B: previousMetrics?.complexity || 0,
      fullMark: 100, // Adjust fullMark as needed
    },
    {
      subject: 'Volume',
      A: metrics.halstead.volume,
      B: previousMetrics?.halstead?.volume || 0,
      fullMark: 1000, // Adjust fullMark as needed
    },
    {
      subject: 'Effort',
      A: metrics.halstead.effort,
      B: previousMetrics?.halstead?.effort || 0,
      fullMark: 1000, // Adjust fullMark as needed
    },
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {metricCardsData.map((metric) => (
          <MetricCard
            key={metric.title}
            title={metric.title}
            value={formatMetricValue(metric.value)}
            trend={metric.trend}
            icon={metric.icon}
          />
        ))}
      </div>

      <div className={`p-6 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-sm`}>
        <h2 className="text-lg font-semibold mb-4">Code Quality Metrics</h2>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarChartData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" />
              <PolarRadiusAxis />
              <Radar name="Current" dataKey="A" stroke="#4F46E5" fill="#4F46E5" fillOpacity={0.6} />
              {previousMetrics && ( // Conditionally render previous metrics radar
                <Radar name="Previous" dataKey="B" stroke="#EF4444" fill="#EF4444" fillOpacity={0.6} />
              )}
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};