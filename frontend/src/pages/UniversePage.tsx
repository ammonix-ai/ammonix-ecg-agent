import { useState, useEffect, useRef, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Minimize2 } from 'lucide-react';
import { t } from '@/i18n/t';
import { useVisualisationStore } from '@/stores/visualisationStore';
import { installedData, MISSING_DATA_NOTE, useStatusStore } from '@/stores/statusStore';
import type { CohortLatticeStatus, SourceSafeProjectionMetadata } from '@/types/projection';
import {
  UniverseScene,
  UniverseControls,
  SelectionPanel,
  SettingsPanel,
  ColorSettingsPanel,
  UniverseLegendBar,
  LassoOverlay,
  MiniMap,
} from '@/components/universe';
import { KeyboardShortcutsOverlay } from '@/components/universe/KeyboardShortcutsOverlay';
import { ROCCurveChart } from '@/components/universe/ROCCurveChart';
import { getSourceSafeROC, type SourceSafeROC } from '@/services/projectionService';

function formatMetric(value?: number): string {
  return typeof value === 'number' ? value.toFixed(3) : '---';
}

function aurocColorClass(value?: number): string {
  if (typeof value !== 'number') return 'text-slate-400';
  if (value >= 0.9) return 'text-emerald-600';
  if (value >= 0.8) return 'text-amber-600';
  return 'text-rose-600';
}

function SourceSafeStatsPanel({
  metadata,
  onClose,
}: {
  metadata: SourceSafeProjectionMetadata
  onClose: () => void
}) {
  const cohortCounts = Object.entries(metadata.cohortCounts ?? {});
  const sourceCounts = Object.entries(metadata.sourceCounts ?? {}).slice(0, 8);
  // Per-class AUROC, strongest first. metadata.diagnoses is loosely typed
  // (Record<string, any>); each accepted diagnosis carries final_full_auc
  // (final full model) and primary_auc (held-out primary) + n_total_pos.
  const perClassAuroc = [...(metadata.diagnoses ?? [])]
    .filter((d) => d && (typeof d.final_full_auc === 'number' || typeof d.primary_auc === 'number'))
    .sort(
      (a, b) =>
        Number(b.final_full_auc ?? b.primary_auc ?? 0) - Number(a.final_full_auc ?? a.primary_auc ?? 0),
    );

  // Click a per-class row → focus that diagnosis (show only cases that HAVE it,
  // matching the ROC's positive class) and fetch its ROC on the displayed universe.
  const [rocDx, setRocDx] = useState<string | null>(null);
  const [roc, setRoc] = useState<SourceSafeROC | null>(null);
  const [rocLoading, setRocLoading] = useState(false);
  const [rocError, setRocError] = useState<string | null>(null);

  useEffect(() => {
    if (!rocDx) {
      setRoc(null);
      setRocError(null);
      return;
    }
    let cancelled = false;
    setRocLoading(true);
    setRocError(null);
    setRoc(null);
    getSourceSafeROC(rocDx)
      .then((d) => { if (!cancelled) setRoc(d); })
      .catch((e) => { if (!cancelled) setRocError(e instanceof Error ? e.message : 'Failed to load ROC'); })
      .finally(() => { if (!cancelled) setRocLoading(false); });
    return () => { cancelled = true; };
  }, [rocDx]);

  const focusDiagnosis = (dx: string) => {
    if (rocDx === dx) {
      setRocDx(null);
      useVisualisationStore.getState().clearDiagnosisFilters();
      return;
    }
    setRocDx(dx);
    useVisualisationStore.setState({
      highlightedDiagnoses: new Set([dx]),
      filledDiagnoses: new Set<string>(),
      postFillHighlighted: new Set<string>(),
      diagnosisOperators: new Map<string, 'and' | 'or'>(),
    });
  };

  const clearFocus = () => {
    setRocDx(null);
    useVisualisationStore.getState().clearDiagnosisFilters();
  };

  return (
    <div>
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
        <h3 className="text-sm font-semibold text-gray-900">Model Stats</h3>
        <button
          onClick={onClose}
          className="p-1 rounded hover:bg-gray-100 transition-colors"
          title="Close model stats"
        >
          <span className="text-gray-500 text-lg leading-none">&times;</span>
        </button>
      </div>
      <div className="p-4 space-y-4 text-xs text-slate-700">
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-center">
            <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold mb-1">Macro AUC</div>
            <div className="text-xl font-bold text-slate-800 tabular-nums">{formatMetric(metadata.macroFinalAuc)}</div>
          </div>
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-center">
            <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold mb-1">Classes</div>
            <div className="text-xl font-bold text-slate-800 tabular-nums">{metadata.nClasses ?? '---'}</div>
          </div>
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-center">
            <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold mb-1">Features</div>
            <div className="text-xl font-bold text-slate-800 tabular-nums">{metadata.featureCount?.toLocaleString() ?? '---'}</div>
          </div>
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-center">
            <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold mb-1">Primary AUC</div>
            <div className="text-xl font-bold text-slate-800 tabular-nums">{formatMetric(metadata.macroPrimaryAuc)}</div>
          </div>
        </div>

        {rocDx && (
          <div className="rounded-lg border border-brand/30 bg-brand/5 p-3">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="text-xs font-semibold text-slate-800 truncate" title={rocDx}>{rocDx}</div>
                <div className="text-[10px] text-slate-500">ROC — this universe</div>
              </div>
              <button
                onClick={clearFocus}
                title="Clear focus"
                className="p-1 rounded hover:bg-white/70 transition-colors shrink-0"
              >
                <span className="text-slate-400 text-base leading-none">&times;</span>
              </button>
            </div>

            {rocLoading && (
              <div className="text-[11px] text-slate-400 py-6 text-center animate-pulse">Computing ROC…</div>
            )}
            {rocError && <div className="text-[11px] text-rose-500 py-3">{rocError}</div>}

            {roc && !rocLoading && (
              <>
                <div className="mt-1 flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-[10px]">
                  <span className="font-mono">
                    <span className="text-slate-400">area </span>
                    <span className={`font-bold ${aurocColorClass(roc.auroc)}`}>{formatMetric(roc.auroc)}</span>
                  </span>
                  <span className="font-mono text-slate-400">
                    held-out {formatMetric(roc.heldOutFinalAuroc ?? undefined)} final · {formatMetric(roc.heldOutPrimaryAuroc ?? undefined)} primary
                  </span>
                </div>
                <div className="text-[10px] text-slate-400 font-mono mt-0.5">
                  n+ {roc.nPositive.toLocaleString()} · n− {roc.nNegative.toLocaleString()} in universe · op t={roc.threshold.toFixed(3)}
                </div>
                <ROCCurveChart
                  curve={{
                    diagnosis: roc.diagnosis,
                    points: roc.points,
                    auroc: roc.auroc,
                    threshold: roc.threshold,
                    operatingFpr: roc.operatingFpr,
                    operatingTpr: roc.operatingTpr,
                  }}
                  compact
                />
                <div className="mt-1 text-[10px] text-slate-400 leading-snug">
                  Area is on the displayed universe (includes training cases) and differs from the held-out numbers.
                </div>
              </>
            )}
          </div>
        )}

        {perClassAuroc.length > 0 && (
          <div>
            <div className="flex items-baseline justify-between mb-1">
              <span className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">
                Per-class AUROC ({perClassAuroc.length})
              </span>
              <span className="text-[10px] text-slate-300 font-mono">click→ROC</span>
            </div>
            <div className="space-y-0.5 max-h-56 overflow-y-auto pr-1 -mr-1 [scrollbar-gutter:stable]">
              {/* Column header lives inside the scroll box and mirrors the row box model
                  (justify-between, gap-2, py-0.5 px-1 -mx-1, same column widths) so the
                  labels sit over their columns regardless of scrollbar width. */}
              {/* w-full matters: the rows are w-full with -mx-1, so their width stays 100%
                  of the content box. A width:auto header with the same -mx-1 would instead
                  grow by 8px and sit one gutter to the right of every column. */}
              <div className="sticky top-0 z-10 bg-white w-full flex items-center justify-between gap-2 py-0.5 px-1 -mx-1 text-[10px] font-mono text-slate-400 border-b border-slate-100">
                <span className="truncate">diagnosis</span>
                <span className="flex items-center gap-2 shrink-0 tabular-nums">
                  <span className="w-9 text-right">final</span>
                  <span className="w-9 text-right">primary</span>
                  <span className="w-10 text-right">n+</span>
                </span>
              </div>
              {perClassAuroc.map((d, i) => {
                const dx = (d.diagnosis as string) ?? 'unknown';
                const active = rocDx === dx;
                return (
                  <button
                    type="button"
                    key={dx ?? i}
                    onClick={() => focusDiagnosis(dx)}
                    title={`${dx} — click for ROC curve & focus`}
                    className={`w-full flex items-center justify-between gap-2 py-0.5 px-1 -mx-1 rounded text-left transition-colors ${active ? 'bg-brand/10 ring-1 ring-brand/30' : 'hover:bg-slate-100'}`}
                  >
                    <span className="truncate" title={dx}>{dx}</span>
                    <span className="flex items-center gap-2 shrink-0 font-mono tabular-nums">
                      <span className={`w-9 text-right font-semibold ${aurocColorClass(d.final_full_auc as number)}`}>
                        {formatMetric(d.final_full_auc as number)}
                      </span>
                      <span className="text-slate-400 text-[10px] w-9 text-right">
                        {formatMetric(d.primary_auc as number)}
                      </span>
                      {/* always rendered so the column never collapses and unsets the alignment */}
                      <span className="text-slate-300 text-[10px] w-10 text-right">
                        {typeof d.n_total_pos === 'number' ? (d.n_total_pos as number).toLocaleString() : ''}
                      </span>
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <div>
          <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold mb-1">Package</div>
          <div className="font-mono text-[11px] text-slate-600 break-all">{metadata.modelVersion ?? 'source-safe model'}</div>
          {metadata.modelFamily && (
            <div className="font-mono text-[11px] text-slate-400 break-all mt-0.5">{metadata.modelFamily}</div>
          )}
        </div>

        {metadata.modelHeadline && (
          <p className="text-[11px] leading-relaxed text-slate-500">{metadata.modelHeadline}</p>
        )}

        {cohortCounts.length > 0 && (
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold mb-1">Universe Cohorts</div>
            <div className="space-y-1">
              {cohortCounts.map(([name, count]) => (
                <div key={name} className="flex items-center justify-between gap-2">
                  <span className="truncate">{name}</span>
                  <span className="font-mono text-slate-500">{count.toLocaleString()}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {sourceCounts.length > 0 && (
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold mb-1">Top Sources</div>
            <div className="space-y-1">
              {sourceCounts.map(([name, count]) => (
                <div key={name} className="flex items-center justify-between gap-2">
                  <span className="truncate">{name}</span>
                  <span className="font-mono text-slate-500">{count.toLocaleString()}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * UniversePage — the fixed 63,256-point diagnosis embedding.
 *
 * t-SNE / UMAP / PCA over the frozen classifier's probability space. The
 * embedding never refits: these coordinates are read from the published
 * payload, identical for every viewer.
 */

export default function UniversePage() {
  const universeInstalled = installedData(useStatusStore((s) => s.probe)).universe;
  const isLoading = useVisualisationStore((s) => s.isLoading);
  const error = useVisualisationStore((s) => s.error);
  const points = useVisualisationStore((s) => s.points);
  const projectionData = useVisualisationStore((s) => s.projectionData);
  const method = useVisualisationStore((s) => s.method);
  const fetchProjection = useVisualisationStore((s) => s.fetchProjection);

  // Cohort lattice state
  const primaryCohortId = useVisualisationStore((s) => s.primaryCohortId);
  const modelCohortId = useVisualisationStore((s) => s.modelCohortId);
  const cohortStatuses = useVisualisationStore((s) => s.cohortStatuses);
  const cohortStatusesLoading = useVisualisationStore((s) => s.cohortStatusesLoading);
  const fetchCohortStatuses = useVisualisationStore((s) => s.fetchCohortStatuses);
  const setPrimaryCohort = useVisualisationStore((s) => s.setPrimaryCohort);
  const setModelCohort = useVisualisationStore((s) => s.setModelCohort);
  const fetchCohortProjection = useVisualisationStore((s) => s.fetchCohortProjection);

  // Panel visibility
  const showSettings = useVisualisationStore((s) => s.showSettings);
  const showStats = useVisualisationStore((s) => s.showStats);
  const showColorSettings = useVisualisationStore((s) => s.showColorSettings);
  const showLegend = useVisualisationStore((s) => s.showLegend);
  const isFullscreen = useVisualisationStore((s) => s.isFullscreen);
  const setFullscreen = useVisualisationStore((s) => s.setFullscreen);
  const setShowStats = useVisualisationStore((s) => s.setShowStats);
  const panelsCollapsed = useVisualisationStore((s) => s.panelsCollapsed);
  const selectedPatients = useVisualisationStore((s) => s.selectedPatients);

  // Local UI state
  const [showShortcuts, setShowShortcuts] = useState(false);
  const screenPositionsRef = useRef<Map<string, { x: number; y: number }>>(new Map());

  // Deep-link entry: /universe?primary=<cohort>&highlight=<cluster id>
  const [searchParams] = useSearchParams();
  const primaryParam = searchParams.get('primary');
  const highlightParam = searchParams.get('highlight'); // cluster id; passed for future scene highlight

  // Derived: primary cohort info
  const primaryCohort = useMemo(
    () => cohortStatuses.find((c) => c.cohort_id === primaryCohortId),
    [cohortStatuses, primaryCohortId],
  );

  // Derived: trained cohorts (for model selector)
  const trainedCohorts = useMemo(
    () => cohortStatuses.filter((c) => c.has_training && c.cohort_id !== primaryCohortId),
    [cohortStatuses, primaryCohortId],
  );

  // Fetch cohort statuses on mount
  useEffect(() => {
    fetchCohortStatuses();
  }, [fetchCohortStatuses]);

  // Auto-select cohort: incoming ?primary=NAME wins; otherwise first trained
  useEffect(() => {
    if (primaryCohortId || cohortStatuses.length === 0) return;
    if (primaryParam) {
      const matchByName = cohortStatuses.find(
        (c) => c.cohort_name === primaryParam || String(c.cohort_id) === primaryParam,
      );
      if (matchByName) {
        setPrimaryCohort(matchByName.cohort_id);
        return;
      }
    }
    const firstTrained = cohortStatuses.find((c) => c.has_training);
    if (firstTrained) {
      setPrimaryCohort(firstTrained.cohort_id);
    }
  }, [cohortStatuses, primaryCohortId, primaryParam, setPrimaryCohort]);

  // Surface incoming highlight cluster id to the user; full scene highlight is Phase B work.
  useEffect(() => {
    if (highlightParam) {
      console.info(`[UniversePage] Incoming highlight cluster: ${highlightParam}`);
    }
  }, [highlightParam]);

  // Escape key exits fullscreen, ? toggles shortcuts overlay
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) return;
      if (e.key === 'Escape' && isFullscreen) setFullscreen(false);
      if (e.key === '?') { e.preventDefault(); setShowShortcuts((prev) => !prev); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isFullscreen, setFullscreen]);

  // Open build: the inline CV-training driver (startCVPipeline +
  // createCVEventSource against /api/admin/cv/start and /api/admin/cv/stream)
  // was removed together with the "Train..." option that invoked it. The open
  // universe ships pre-trained; there is nothing to kick off from the UI.

  // --- Status badge ---
  const statusBadge = (c: CohortLatticeStatus) => {
    if (c.has_training) return '\u2713';
    if (c.has_qpsi) return '\u25CB';
    return '\u2014';
  };

  const showLeftPanel = selectedPatients.size > 0 && !panelsCollapsed;
  const showRightOverlay = (showSettings || showStats || showColorSettings) && !panelsCollapsed;

  return (
    <div className="h-full flex flex-col">
      {/* Cohort source bar */}
      {!isFullscreen && (
        <div className="flex flex-col gap-1 px-3 py-1.5 border-b border-[var(--border-subtle)] bg-surface-0/80 backdrop-blur-sm">
          {/* Row 1: Cohort selector + overlay chips + model selector */}
          <div className="flex items-center gap-2 flex-wrap">
            {/* DD1: Primary cohort */}
            <select
              value={primaryCohortId ?? ''}
              onChange={(e) => setPrimaryCohort(e.target.value ? Number(e.target.value) : null)}
              disabled={cohortStatusesLoading}
              className="px-2 py-1 rounded-[var(--radius-btn)] border border-[var(--border-default)] bg-surface-0 text-caption text-text-primary min-w-[180px]"
            >
              <option value="">Select cohort...</option>
              {cohortStatuses.map((c) => (
                <option key={c.cohort_id} value={c.cohort_id}>
                  {statusBadge(c)} {c.cohort_name} ({c.record_count.toLocaleString()} rec)
                </option>
              ))}
            </select>

            {/* Open build: overlay chips and the [+ add] overlay picker are
                gone. Overlaying a second cohort posts to
                /api/admin/cv/projection/{method}/overlay, which needs the
                private per-fold model pickles and raw feature matrices — it is
                not part of the open backend. */}

            <div className="w-px h-5 bg-gray-200" />

            {/* DD2: Model selector */}
            <label className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Model</label>
            <select
              value={modelCohortId ?? 'own'}
              onChange={(e) => {
                const val = e.target.value;
                if (val === 'own') setModelCohort(null);
                else setModelCohort(Number(val));
              }}
              className="px-2 py-1 rounded-[var(--radius-btn)] border border-[var(--border-default)] bg-surface-0 text-caption text-text-primary"
            >
              {primaryCohort?.has_training && (
                <option value="own">Own Model (CV OOF)</option>
              )}
              {trainedCohorts.map((c) => (
                <option key={c.cohort_id} value={c.cohort_id}>
                  {c.cohort_name}&apos;s Model
                </option>
              ))}
              {/* Open build: the "Train..." option is gone — it drove
                  POST /api/admin/cv/start + the /api/admin/cv/stream SSE,
                  neither of which the open backend serves. */}
            </select>
          </div>

          {/* Row 2: Stats + training progress */}
          <div className="flex items-center gap-3 text-caption text-text-muted">
            <span>
              {points.length.toLocaleString()} patients
              {primaryCohort?.training?.n_classes ? ` \u00B7 ${primaryCohort.training.n_classes} classes` : ''}
              {` \u00B7 ${method.toUpperCase()}`}
            </span>
            {primaryCohort?.source_safe && (
              <span className="inline-flex items-center gap-1 text-emerald-600">
                source-safe
                {primaryCohort.training?.version ? ` \u00B7 ${primaryCohort.training.version}` : ''}
                {primaryCohort.cache_built ? ' \u00B7 scored cache ready' : ' \u00B7 scoring cache needed'}
              </span>
            )}
            {isLoading && <span className="text-brand animate-pulse">Computing...</span>}
          </div>
        </div>
      )}

      {/* Top toolbar */}
      {!isFullscreen && <UniverseControls />}

      {/* Main content — full-width canvas with floating overlays */}
      <div className="flex-1 relative min-h-0">

        {/* Left sidebar: Selection panel (absolute overlay) */}
        {showLeftPanel && (
          <div className="absolute left-0 top-0 bottom-0 w-72 z-20 bg-white/80 backdrop-blur-sm border-r border-gray-200 overflow-y-auto shadow-lg rounded-r-[var(--radius-card)]">
            <SelectionPanel />
          </div>
        )}

        {/* Center: 3D canvas (full size, base layer) */}
        {!universeInstalled ? (
          <div className="flex flex-col items-center justify-center h-full gap-2 px-6 text-center">
            <p className="text-body font-semibold text-text-primary">
              The universe package is not installed
            </p>
            <p className="max-w-md text-caption text-text-secondary">
              This page shows the 63,256 recordings of the paper's universe once the package is
              in place. {MISSING_DATA_NOTE}
            </p>
          </div>
        ) : isLoading && points.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-3">
            <div className="w-8 h-8 border-2 border-brand border-t-transparent rounded-full animate-spin" />
            <p className="text-sm text-text-secondary">
              {t('lattice.computing', `Computing ${method.toUpperCase()} projection...`)}
            </p>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-full gap-3">
            <p className="text-sm text-red-500">
              {t('lattice.error', 'Failed to load projection')}
            </p>
            <p className="text-caption text-text-muted">{error}</p>
            <button
              onClick={() => primaryCohortId ? fetchCohortProjection() : fetchProjection()}
              className="px-4 py-1.5 bg-brand text-white text-sm rounded-lg hover:bg-brand/90"
            >
              {t('lattice.retry', 'Retry')}
            </button>
          </div>
        ) : (
          <>
            <UniverseScene screenPositionsRef={screenPositionsRef} />
            <LassoOverlay screenPositions={screenPositionsRef.current} />
            <MiniMap />
          </>
        )}

        {/* Floating exit-fullscreen button */}
        {isFullscreen && (
          <button
            onClick={() => setFullscreen(false)}
            className="absolute top-3 right-3 z-30 p-2 rounded-lg bg-white/80 backdrop-blur-sm border border-gray-200 shadow-sm hover:bg-white text-slate-600 transition-colors"
            title={t('lattice.exitFullscreen', 'Exit fullscreen (Esc)')}
          >
            <Minimize2 className="w-4 h-4" />
          </button>
        )}

        {/* Loading overlay when recomputing */}
        {isLoading && points.length > 0 && (
          <div className="absolute inset-0 bg-white/60 backdrop-blur-sm flex items-center justify-center z-10">
            <div className="flex items-center gap-2 bg-white/80 backdrop-blur-sm px-4 py-2 rounded-lg shadow-lg border">
              <div className="w-4 h-4 border-2 border-brand border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-text-secondary">
                {t('lattice.recomputing', 'Recomputing...')}
              </span>
            </div>
          </div>
        )}

        {/* Right overlay panels — float on top of 3D canvas */}
        {showRightOverlay && (
          <div className="absolute right-2 top-2 bottom-2 w-[340px] z-20 flex flex-col gap-2 overflow-y-auto pointer-events-none">
            {showSettings && (
              <div className="pointer-events-auto bg-white/80 backdrop-blur-sm rounded-[var(--radius-card)] shadow-lg">
                <SettingsPanel />
              </div>
            )}
            {showColorSettings && (
              <div className="pointer-events-auto bg-white/80 backdrop-blur-sm rounded-[var(--radius-card)] shadow-lg">
                <ColorSettingsPanel />
              </div>
            )}
            {showStats && (
              <div className="pointer-events-auto bg-white/80 backdrop-blur-sm rounded-[var(--radius-card)] border border-gray-200 shadow-lg overflow-hidden">
                {projectionData?.sourceSafe ? (
                  <SourceSafeStatsPanel
                    metadata={projectionData.sourceSafe}
                    onClose={() => setShowStats(false)}
                  />
                ) : (
                  <div className="px-4 py-3">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-semibold text-gray-900">Model Stats</h3>
                      <button
                        onClick={() => setShowStats(false)}
                        className="p-1 rounded hover:bg-gray-100 transition-colors"
                        title="Close model stats"
                      >
                        <span className="text-gray-500 text-lg leading-none">&times;</span>
                      </button>
                    </div>
                    <p className="mt-2 text-xs leading-relaxed text-slate-500">
                      This projection carries no model metadata, so there are no statistics to
                      show. The published source-safe universe ships its own; nothing is
                      substituted for it here.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Bottom legend bar */}
      {showLegend && !isFullscreen && (
        <div className="flex-shrink-0 border-t border-gray-200 bg-surface-0/80 backdrop-blur-sm">
          <UniverseLegendBar />
        </div>
      )}

      {/* Keyboard shortcuts overlay */}
      <KeyboardShortcutsOverlay open={showShortcuts} onClose={() => setShowShortcuts(false)} />
    </div>
  );
}
