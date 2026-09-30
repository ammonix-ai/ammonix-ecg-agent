import { Link } from 'react-router-dom';
import { cn } from '@/design/cn';
import { PlacementTag } from './ScoreKindTag';
import { isApproximate, placementInfo } from '@/utils/scoreKind';
import type { AnalyzeResult } from '@/types/analyze';

/**
 * Where this recording lands in the fixed 63,256-point universe, and the six
 * nearest rows to it in probability space.
 *
 * The embedding is never refit — the point is placed into it. Each projection
 * carries how it was placed, because they are not placed the same way: PCA is
 * linear and recoverable, t-SNE has no out-of-sample operator at all.
 */
export function UniversePlacement({
  result,
  selfDisplayId,
}: {
  result: AnalyzeResult;
  selfDisplayId?: string | null;
}) {
  const { projection = {}, neighbors = [] } = result;

  const axes: { key: 'pca' | 'umap' | 'tsne'; label: string; placement?: string }[] = [
    { key: 'pca', label: 'PCA', placement: result.pcaPlacement },
    { key: 'umap', label: 'UMAP', placement: result.umapPlacement },
    { key: 'tsne', label: 't-SNE', placement: result.tsnePlacement },
  ];

  const anyApprox = axes.some((a) => isApproximate(a.placement));

  return (
    <section className="rounded-[var(--radius-card)] border border-[var(--border-default)] bg-surface-0 p-4">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-h3 text-text-primary">Position in the universe</h2>
        <Link
          to="/universe"
          className="text-caption font-semibold text-brand hover:underline"
        >
          Open the universe
        </Link>
      </header>

      <ul className="mt-3 space-y-2">
        {axes.map(({ key, label, placement }) => {
          const coord = projection[key];
          return (
            <li key={key} className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <span className="w-14 shrink-0 text-caption font-semibold uppercase tracking-wider text-text-muted">
                {label}
              </span>
              <span className="font-mono tabular-nums text-body text-text-primary">
                {coord && coord.length > 0
                  ? coord.map((v) => (Number.isFinite(v) ? v.toFixed(3) : '—')).join(', ')
                  : 'not placed'}
              </span>
              <PlacementTag placement={placement} />
            </li>
          );
        })}
      </ul>

      {anyApprox && (
        <p className="mt-3 border-l-2 border-warning/60 bg-warning/5 py-2 pl-3 text-caption leading-relaxed text-text-secondary rounded-r-[var(--radius-sm)]">
          {placementInfo('knn_approx').meaning}
          {result.umapNote ? ` ${result.umapNote}` : ''}
        </p>
      )}

      {result.pcaResidualRms != null && (
        <p className="mt-2 font-mono text-caption text-text-muted">
          PCA re-derivation residual RMS {result.pcaResidualRms.toExponential(1)}
        </p>
      )}

      <h3 className="mt-4 text-caption font-semibold uppercase tracking-wider text-text-muted">
        {neighbors.length} nearest neighbours · probability space
      </h3>
      {neighbors.length === 0 ? (
        <p className="mt-1 text-caption text-text-muted">
          The backend returned no neighbours for this point.
        </p>
      ) : (
        <ul className="mt-2 divide-y divide-[var(--border-subtle)]">
          {neighbors.map((n, i) => {
            const isSelf = Boolean(selfDisplayId && n.displayId === selfDisplayId);
            return (
              <li
                key={`${n.displayId}-${i}`}
                className={cn(
                  'flex items-baseline justify-between gap-2 py-1.5',
                  isSelf && 'bg-brand-50/60 -mx-1 px-1 rounded-[var(--radius-sm)]',
                )}
              >
                <span className="flex min-w-0 items-baseline gap-2">
                  <span className="font-mono text-caption font-semibold text-text-primary">
                    {n.displayId}
                  </span>
                  {isSelf && (
                    <span
                      className="shrink-0 text-caption text-brand"
                      title="This recording is already a point in the published universe, so it matches itself at distance 0."
                    >
                      this recording
                    </span>
                  )}
                  <span className="truncate text-caption text-text-secondary">
                    {n.primary ?? '—'}
                  </span>
                </span>
                <span className="shrink-0 font-mono tabular-nums text-caption text-text-muted">
                  {n.distance.toFixed(4)}
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
