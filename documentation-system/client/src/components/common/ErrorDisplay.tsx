export const ErrorDisplay: React.FC<{ message: string }> = ({ message }) => (
    <div className="flex items-center justify-center min-h-screen text-red-500">
      <span>Error: {message}</span>
    </div>
  );