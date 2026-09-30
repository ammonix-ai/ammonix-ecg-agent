import { Info } from 'lucide-react';
import { cn } from '@/design/cn';
import { scoreKindInfo, placementInfo, isApproximate } from '@/utils/scoreKind';

/**
 * The label that travels with every probability on this page.
 *
 * `ensemble_5fold` and `oof_single_fold` are different quantities. Rendering
 * one without saying which it is would let a reader treat them as the same
 * number, which they are not.
 */
export function ScoreKindTag({
  kind,
  className,
}: {
  kind: string | null | undefined;
  className?: string;
}) {
  const info = scoreKindInfo(kind);
  return (
    <span
      title={info.meaning}
      className={cn(
        'inline-flex items-center gap-1 rounded-[var(--radius-pill)] border px-2 py-0.5',
        'text-caption font-semibold tracking-tight whitespace-nowrap',
        'border-[var(--border-default)] bg-surface-1 text-text-secondary',
        className,
      )}
    >
      <Info className="w-3 h-3 shrink-0" aria-hidden />
      {info.label}
      <span className="font-mono font-normal text-text-muted">{kind ?? '—'}</span>
    </span>
  );
}

/**
 * How a point got its coordinate. An approximated placement is coloured and
 * worded as an approximation — never presented as an exact coordinate.
 */
export function PlacementTag({
  placement,
  className,
}: {
  placement: string | null | undefined;
  className?: string;
}) {
  const info = placementInfo(placement);
  const approx = isApproximate(placement);
  return (
    <span
      title={info.meaning}
      className={cn(
        'inline-flex items-center gap-1 rounded-[var(--radius-pill)] border px-2 py-0.5',
        'text-caption font-semibold tracking-tight whitespace-nowrap',
        approx
          ? 'border-warning/40 bg-warning/10 text-[color:var(--color-status-medium)]'
          : 'border-[var(--border-default)] bg-surface-1 text-text-secondary',
        className,
      )}
    >
      {info.label}
    </span>
  );
}

/** A plain prose caveat block. Used for the training-overlap warning. */
export function CaveatNote({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <p
      className={cn(
        'text-caption leading-relaxed text-text-secondary',
        'border-l-2 border-warning/60 bg-warning/5 pl-3 py-2 rounded-r-[var(--radius-sm)]',
        className,
      )}
    >
      {children}
    </p>
  );
}
