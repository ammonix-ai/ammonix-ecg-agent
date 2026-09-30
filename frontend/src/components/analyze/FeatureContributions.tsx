import { useMemo } from 'react';
import type { AnalyzeResult, FeatureContribution } from '@/types/analyze';

/**
 * Top contributing features — mean SHAP over the five folds, in log-odds.
 *
 * Signed: a bar to the right pushed the score up, a bar to the left pushed it
 * down. The scale is shared across rows so the lengths are comparable.
 *
 * These are QPSI feature names, not clinical findings. The panel says which
 * diagnosis they were computed for, and says so only when the backend
 * reported it.
 */
export function FeatureContributions({ result }: { result: AnalyzeResult }) {
  const features = result.topFeatures ?? [];

  const scale = useMemo(() => {
    const max = Math.max(...features.map((f) => Math.abs(f.contribution ?? 0)), 0);
    return max > 0 ? max : 1;
  }, [features]);

  const target =
    result.topPrediction ?? result.predictions?.[0] ?? null;

  return (
    <section className="rounded-[var(--radius-card)] border border-[var(--border-default)] bg-surface-0 p-4">
      <h2 className="text-h3 text-text-primary">Top contributing features</h2>
      <p className="mt-1 text-caption text-text-secondary">
        {target ? (
          <>
            Mean SHAP over the five folds, in log-odds, for{' '}
            <span className="font-semibold text-text-primary">{target}</span>.
          </>
        ) : (
          <>
            Mean SHAP over the five folds, in log-odds. The backend did not report which
            diagnosis these were computed for.
          </>
        )}
      </p>

      {features.length === 0 ? (
        <p className="mt-3 text-caption text-text-muted">
          The backend returned no feature contributions for this recording.
        </p>
      ) : (
        <ul className="mt-3 space-y-2">
          {features.map((f, i) => (
            <FeatureRow key={`${f.name}-${i}`} feature={f} scale={scale} />
          ))}
        </ul>
      )}
    </section>
  );
}

function FeatureRow({ feature, scale }: { feature: FeatureContribution; scale: number }) {
  const contribution = feature.contribution ?? 0;
  const width = (Math.abs(contribution) / scale) * 50; // % of the full-width axis
  const positive = contribution >= 0;

  return (
    <li>
      <div className="flex items-baseline justify-between gap-2">
        <span
          className="truncate font-mono text-caption text-text-primary"
          title={feature.name}
        >
          {feature.name}
        </span>
        <span className="shrink-0 font-mono tabular-nums text-caption text-text-secondary">
          {Number.isFinite(feature.value) ? feature.value.toPrecision(4) : '—'}
        </span>
      </div>
      <div className="relative mt-1 h-2 rounded-full bg-surface-2">
        <div className="absolute inset-y-0 left-1/2 w-px bg-[var(--border-default)]" />
        <div
          className={positive ? 'absolute inset-y-0 rounded-full bg-success' : 'absolute inset-y-0 rounded-full bg-danger'}
          style={
            positive
              ? { left: '50%', width: `${width}%` }
              : { right: '50%', width: `${width}%` }
          }
        />
      </div>
      <div className="mt-0.5 text-right font-mono text-caption text-text-muted tabular-nums">
        {contribution >= 0 ? '+' : ''}
        {contribution.toFixed(4)} log-odds
      </div>
    </li>
  );
}
