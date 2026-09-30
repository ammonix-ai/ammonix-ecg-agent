import { useMemo, useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { cn } from '@/design/cn';
import { ProbabilityBar } from '@/components/ui/ProbabilityBar';
import { ScoreKindTag, CaveatNote } from './ScoreKindTag';
import { foldSpread, scoreKindInfo, trainingOverlapCaveat } from '@/utils/scoreKind';
import type { AnalyzeResult } from '@/types/analyze';

/**
 * What the frozen classifier said.
 *
 * Two rules this panel exists to keep:
 *  1. Every probability is rendered next to the kind of score it is.
 *  2. The live score and the stored universe score are shown as two separate
 *     rows with two separate labels — never averaged, never swapped.
 */
export function ClassificationPanel({
  result,
  isShippedRecord,
}: {
  result: AnalyzeResult;
  isShippedRecord: boolean;
}) {
  const [showAll, setShowAll] = useState(false);

  const ranked = useMemo(
    () =>
      Object.entries(result.probabilities ?? {}).sort(
        (a, b) => (b[1] ?? 0) - (a[1] ?? 0),
      ),
    [result.probabilities],
  );

  const predicted = new Set(result.predictions ?? []);
  const caveat = trainingOverlapCaveat(result.scoreKind, isShippedRecord);
  const stored = result.stored;
  const storedKindDiffers =
    stored?.scoreKind != null && stored.scoreKind !== result.scoreKind;

  const rows = showAll ? ranked : ranked.slice(0, Math.max(predicted.size, 6));

  return (
    <section className="rounded-[var(--radius-card)] border border-[var(--border-default)] bg-surface-0 p-4">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-h3 text-text-primary">Classification</h2>
        <ScoreKindTag kind={result.scoreKind} />
      </header>

      <p className="mt-1 text-caption text-text-secondary">
        {scoreKindInfo(result.scoreKind).meaning}{' '}
        {result.featureCount.toLocaleString()} features
        {result.thresholdKind ? ` · thresholds: ${result.thresholdKind}` : ''}
        {result.modelVersion ? ` · ${result.modelVersion}` : ''}
      </p>

      {caveat && <CaveatNote className="mt-3">{caveat}</CaveatNote>}

      {/* Stored universe score — a different quantity, kept on its own row. */}
      {stored && (stored.topPrediction || stored.maxScore != null) && (
        <div className="mt-3 rounded-[var(--radius-sm)] border border-[var(--border-default)] bg-surface-1 p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-caption font-semibold uppercase tracking-wider text-text-muted">
              Stored in the universe
            </span>
            <ScoreKindTag kind={stored.scoreKind} />
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-body text-text-primary">{stored.topPrediction ?? '—'}</span>
            <span className="font-mono tabular-nums text-body font-semibold text-text-primary">
              {stored.maxScore != null ? stored.maxScore.toFixed(3) : '—'}
            </span>
          </div>
          <p className="mt-1 text-caption text-text-secondary">
            {storedKindDiffers
              ? 'This is a different quantity from the live score above. The two are not interchangeable and are never combined.'
              : 'Recomputed live above from the same frozen analysis; both are reported as the same score kind.'}
          </p>
        </div>
      )}

      {/* Ranked probabilities. The list is not the prediction list: rows that
          cleared their threshold are tagged, the rest are shown for context. */}
      <div className="mt-4">
        <h3 className="text-caption font-semibold uppercase tracking-wider text-text-muted">
          Ranked probabilities · {result.predictions?.length ?? 0} above threshold
        </h3>
        {(result.predictions?.length ?? 0) === 0 && (
          <p className="mt-1 text-caption text-text-secondary">
            No diagnosis cleared its threshold. The ranked probabilities below are still the full
            output — an empty prediction list is a result, not a failure.
          </p>
        )}
      </div>

      <ul className="mt-2 space-y-2">
        {rows.map(([dx, prob]) => (
          <DiagnosisRow
            key={dx}
            diagnosis={dx}
            probability={prob ?? 0}
            threshold={result.thresholds?.[dx]}
            folds={result.foldProbabilities?.[dx]}
            predicted={predicted.has(dx)}
          />
        ))}
      </ul>

      {ranked.length > rows.length || showAll ? (
        <button
          type="button"
          onClick={() => setShowAll((v) => !v)}
          className="mt-3 flex items-center gap-1 text-caption font-semibold text-brand hover:underline"
        >
          {showAll ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          {showAll ? 'Show fewer' : `Show all ${ranked.length} diagnoses`}
        </button>
      ) : null}
    </section>
  );
}

function DiagnosisRow({
  diagnosis,
  probability,
  threshold,
  folds,
  predicted,
}: {
  diagnosis: string;
  probability: number;
  threshold?: number;
  folds?: number[];
  predicted: boolean;
}) {
  const spread = foldSpread(folds);
  return (
    <li>
      <div className="flex items-baseline justify-between gap-2">
        <span className="flex min-w-0 items-baseline gap-1.5">
          <span
            className={cn(
              'truncate text-body',
              predicted ? 'font-semibold text-text-primary' : 'text-text-secondary',
            )}
            title={diagnosis}
          >
            {diagnosis}
          </span>
          {predicted && (
            <span className="shrink-0 rounded-[var(--radius-pill)] bg-success/15 px-1.5 py-px text-caption font-semibold text-[color:var(--color-status-normal)]">
              above threshold
            </span>
          )}
        </span>
        <span className="shrink-0 font-mono tabular-nums text-body text-text-primary">
          {probability.toFixed(3)}
        </span>
      </div>
      <ProbabilityBar
        value={probability}
        threshold={threshold}
        label={threshold !== undefined ? (probability >= threshold ? '✓' : '·') : ''}
        className="mt-1"
      />
      <div className="mt-0.5 flex flex-wrap items-center gap-x-3 text-caption text-text-muted">
        {threshold !== undefined && (
          <span className="font-mono">threshold {threshold.toFixed(3)}</span>
        )}
        {spread && (
          <span
            className="font-mono"
            title={`Per-fold probabilities: ${spread.values.map((v) => v.toFixed(3)).join(', ')}`}
          >
            folds {spread.min.toFixed(3)}–{spread.max.toFixed(3)}{' '}
            {spread.range > 0.25 && (
              <span className="text-[color:var(--color-status-medium)]">
                (wide spread — the folds disagree)
              </span>
            )}
          </span>
        )}
      </div>
    </li>
  );
}
