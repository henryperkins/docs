// src/components/common/MetricCard.tsx
import React from 'react';
import { MetricCardProps } from '../../types/documentation';

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  trend,
  icon: Icon,
  status,
}) => (
  <div className={`p-4 rounded-lg shadow-sm ${status ? status.replace('text-', 'bg-') + ' ' + status : 'bg-white dark:bg-gray-800'}`}>
    <div className="flex items-center justify-between">
      <div className="flex items-center space-x-2">
        {Icon && <Icon className="w-5 h-5 text-gray-400" />} {/* Conditionally render icon */}
        <span className="text-sm font-medium text-gray-600 dark:text-gray-300">{title}</span>
      </div>
      {trend !== undefined && (
        <span className={trend > 0 ? 'text-green-500' : 'text-red-500'}>
          {trend > 0 ? '+' : ''}{trend.toFixed(1)}% {/* Format trend to one decimal place */}
        </span>
      )}
    </div>
    <p className="text-2xl font-bold mt-2 text-gray-800 dark:text-gray-100">{value}</p> {/* Improved text color */}
  </div>
);