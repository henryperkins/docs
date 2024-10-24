import { MetricCardProps } from '../../types/documentation';

export const MetricCard: React.FC<MetricCardProps> = ({ 
  title, 
  value, 
  trend, 
  icon: Icon 
}) => (
  <div className="p-4 rounded-lg bg-white dark:bg-gray-800 shadow-sm">
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