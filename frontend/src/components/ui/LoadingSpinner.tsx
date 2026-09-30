import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  message?: string;
  className?: string;
}

export function LoadingSpinner({ message, className }: LoadingSpinnerProps) {
  return (
    <div className={`flex flex-col items-center justify-center py-12 gap-4 ${className ?? ''}`}>
      <Loader2 className="w-8 h-8 text-brand animate-spin" />
      {message && (
        <span className="text-body text-text-secondary">{message}</span>
      )}
    </div>
  );
}
