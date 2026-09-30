import { cn } from '@/design/cn';
import { motion } from 'motion/react';
import { springGentle } from '@/design/animations';

interface ProbabilityBarProps {
  value: number;
  threshold?: number;
  label?: string;
  className?: string;
}

function getFillColor(value: number, threshold?: number): string {
  const cutoff = threshold ?? 0.5;
  if (value >= cutoff) return 'bg-success';
  if (value >= 0.3) return 'bg-warning';
  return 'bg-danger';
}

export function ProbabilityBar({ value, threshold, label, className }: ProbabilityBarProps) {
  const safeValue = value ?? 0;
  const percent = Math.min(Math.max(safeValue * 100, 0), 100);
  const fillColor = getFillColor(safeValue, threshold);

  return (
    <div className={cn('flex items-center gap-3', className)}>
      <div className="relative flex-1 h-1.5 bg-surface-2 rounded-full overflow-hidden">
        <motion.div
          className={cn('absolute inset-y-0 left-0 rounded-full', fillColor)}
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          transition={springGentle}
        />
        {threshold !== undefined && (
          <div
            className="absolute top-[-2px] bottom-[-2px] w-px bg-text-primary"
            style={{ left: `${threshold * 100}%` }}
          />
        )}
      </div>
      {label !== undefined ? (
        <span className="text-data font-mono">{label}</span>
      ) : (
        <span className="text-data font-mono">{safeValue.toFixed(2)}</span>
      )}
    </div>
  );
}
