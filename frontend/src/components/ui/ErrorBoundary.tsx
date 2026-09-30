import { Component, type ReactNode, type ErrorInfo } from 'react';
import { AlertTriangle } from 'lucide-react';
import { t } from '@/i18n/t';

interface ErrorBoundaryProps {
  fallback?: ReactNode;
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('ErrorBoundary caught:', error, info.componentStack);
  }

  render(): ReactNode {
    if (!this.state.hasError) {
      return this.props.children;
    }

    if (this.props.fallback) {
      return this.props.fallback;
    }

    return (
      <div className="flex flex-col items-center justify-center gap-4 p-8 text-center">
        <AlertTriangle className="w-10 h-10 text-warning" />
        <h2 className="text-h3 text-text-primary">
          {t('error.title', 'Something went wrong')}
        </h2>
        {this.state.error && (
          <p className="text-body text-text-secondary max-w-md">
            {this.state.error.message}
          </p>
        )}
        <button
          onClick={() => this.setState({ hasError: false, error: null })}
          className="bg-brand text-white rounded-[var(--radius-btn)] px-4 py-2 text-body font-medium hover:bg-brand-light transition-colors"
        >
          {t('error.tryAgain', 'Try again')}
        </button>
      </div>
    );
  }
}
