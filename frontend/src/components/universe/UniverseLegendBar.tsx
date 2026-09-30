import { Fragment, useEffect, useMemo, useRef } from 'react'
import { X } from 'lucide-react'
import { useVisualisationStore, operatorKey } from '@/stores/visualisationStore'
import {
  useDiagnosisColorMap,
  useCohortColorMap,
  useScoredClasses,
} from '@/stores/visualisationSelectors'
import { cn } from '@/design/cn'
import {
  classifyErrorHighlight,
  DIAGNOSIS_FAMILIES,
  ERROR_HIGHLIGHT_COLORS,
  FALLBACK_COLOR,
  FAMILY_COLORS,
  FAMILY_UNMAPPED_LABEL,
  type ErrorHighlightBucket,
} from '@/utils/colorUtils'
import type { EnrichedProjectionPoint } from '@/types/projection'

interface DxEntry {
  name: string
  count: number
  color: string
}

/** One read-only key entry: a swatch, a label, a count. */
interface LegendEntry {
  key: string
  label: string
  count: number
  color: string
  /** Hover text — where a colour's definition needs more than its label. */
  title?: string
}

function PillButton({
  dx,
  isHighlighted,
  isFilled,
  hasSelection,
  onClick,
}: {
  dx: DxEntry
  isHighlighted: boolean
  isFilled: boolean
  hasSelection: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'flex items-center gap-1 px-2 py-1 rounded-full text-[11px] font-medium whitespace-nowrap transition-all shrink-0',
        isFilled
          ? 'shadow-sm'
          : isHighlighted
            ? 'ring-2 bg-white shadow-sm'
            : hasSelection
              ? 'opacity-40 bg-white/80 hover:opacity-70'
              : 'bg-white/80 hover:bg-white hover:shadow-sm',
      )}
      style={
        isFilled
          ? { backgroundColor: dx.color, color: '#fff' }
          : isHighlighted
            ? { '--tw-ring-color': dx.color } as React.CSSProperties
            : undefined
      }
    >
      {!isFilled && (
        <span
          className="w-2 h-2 rounded-full shrink-0"
          style={{ backgroundColor: dx.color }}
        />
      )}
      <span className={isFilled ? 'text-white' : 'text-gray-700'}>{dx.name}</span>
      <span className={isFilled ? 'text-white/70 text-[10px]' : 'text-gray-400 text-[10px]'}>({dx.count})</span>
    </button>
  )
}

/**
 * Read-only key: an uppercase mode caption, then one pill per colour with its
 * live count from the current projection.
 *
 * Every grading mode except `diagnosis` uses this. A legend that explains
 * colours the points are not using is worse than no legend at all, so each
 * mode gets its own entries rather than borrowing the diagnosis pills. The
 * pills carry no click behaviour: outside `diagnosis` these are keys, not
 * controls.
 */
function ReadOnlyLegend({
  caption,
  entries,
  captionTitle,
}: {
  caption: string
  entries: LegendEntry[]
  captionTitle?: string
}) {
  if (entries.length === 0) return null

  return (
    <div
      className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-raised/90 backdrop-blur-sm border border-surface-border overflow-x-auto scrollbar-thin"
      onPointerDown={(e) => e.stopPropagation()}
    >
      <span
        className="text-[10px] uppercase tracking-wide text-gray-400 shrink-0 pr-1"
        title={captionTitle}
      >
        {caption}
      </span>
      {entries.map((e) => (
        <span
          key={e.key}
          className="flex items-center gap-1.5 px-2 py-0.5 rounded-full border border-surface-border shrink-0"
          title={e.title ?? `${e.label} — ${e.count.toLocaleString()} recordings`}
        >
          <span
            className="w-2.5 h-2.5 rounded-full shrink-0"
            style={{ backgroundColor: e.color }}
          />
          <span className="text-[11px] text-gray-700 whitespace-nowrap">{e.label}</span>
          <span className="text-gray-400 text-[10px]">({e.count.toLocaleString()})</span>
        </span>
      ))}
    </div>
  )
}

const NO_POINTS: EnrichedProjectionPoint[] = []

/** Full, unfiltered points of the current projection — the counting basis for
 *  every read-only legend, so a diagnosis filter does not move the key. */
function useProjectionPoints(): EnrichedProjectionPoint[] {
  const projectionData = useVisualisationStore((s) => s.projectionData)
  return (projectionData?.points as EnrichedProjectionPoint[] | undefined) ?? NO_POINTS
}

function CohortLegend() {
  const points = useProjectionPoints()
  const cohortColorMap = useCohortColorMap()

  const entries = useMemo(() => {
    const counts = new Map<string, number>()
    for (const p of points) {
      if (p.cohort) counts.set(p.cohort, (counts.get(p.cohort) || 0) + 1)
    }
    return Array.from(counts.entries())
      .map(([name, count]) => ({
        key: name,
        label: name,
        count,
        color: cohortColorMap.get(name) || '#94A3B8',
      }))
      .sort((a, b) => b.count - a.count)
  }, [points, cohortColorMap])

  return <ReadOnlyLegend caption="Cohort" entries={entries} />
}

const ERROR_HIGHLIGHT_ENTRIES: {
  key: ErrorHighlightBucket
  label: string
  title: string
}[] = [
  // Ordered best to worst, left to right: a clean read, then the two single
  // failure modes, then both at once. The colours follow the same ramp —
  // light green, dark green, orange, red.
  {
    key: 'match',
    label: 'Exact match',
    title: 'The predicted set equals the gold labels the model has classes for.',
  },
  {
    key: 'over',
    label: 'Over-called',
    title: 'False positives only: every scored gold label was called, plus extras. Coloured green because the reference labels are incomplete — an extra call is not necessarily wrong.',
  },
  {
    key: 'missed',
    label: 'Missed',
    title: 'False negatives only: no extra calls, but at least one scored gold label was not called. The clinically worse failure, hence the warm colour.',
  },
  {
    key: 'both',
    label: 'Over-called + missed',
    title: 'At least one prediction the gold set does not have AND at least one scored gold label the model did not call.',
  },
  {
    key: 'na',
    label: 'Not graded',
    title:
      'No predictions, or no gold label among the 32 scored classes — a normal ' +
      'ECG labelled only "sinus rhythm" / "normal ecg" / "lvef >=55% control" has ' +
      'nothing the model could be right or wrong about, so it is not graded ' +
      'rather than being painted as an error.',
  },
]

function ErrorHighlightLegend() {
  const points = useProjectionPoints()
  const scoredClasses = useScoredClasses()

  const counts = useMemo(() => {
    const c: Record<ErrorHighlightBucket, number> = {
      both: 0, over: 0, missed: 0, match: 0, na: 0,
    }
    for (const p of points) c[classifyErrorHighlight(p, scoredClasses)]++
    return c
  }, [points, scoredClasses])

  const entries = ERROR_HIGHLIGHT_ENTRIES
    .filter((e) => counts[e.key] > 0)
    .map((e) => ({
      key: e.key,
      label: e.label,
      count: counts[e.key],
      color: ERROR_HIGHLIGHT_COLORS[e.key],
      title: `${e.label} — ${counts[e.key].toLocaleString()} recordings. ${e.title}`,
    }))

  return (
    <ReadOnlyLegend
      caption="Errors"
      captionTitle={
        scoredClasses.size > 0
          ? `Predicted vs gold, compared only over the ${scoredClasses.size} classes the model scores.`
          : 'This projection ships no threshold metadata, so the scored class list is unknown and nothing can be graded.'
      }
      entries={entries}
    />
  )
}

/** One entry per clinical family present in the data, plus grey for primaries
 *  the family map does not cover ('unknown' — honest, and left grey). */
function FamilyLegend() {
  const points = useProjectionPoints()

  const entries = useMemo(() => {
    const counts = new Map<string, number>()
    for (const p of points) {
      const family = DIAGNOSIS_FAMILIES[(p.primaryDiagnosis ?? '').toLowerCase()]
        ?? FAMILY_UNMAPPED_LABEL
      counts.set(family, (counts.get(family) || 0) + 1)
    }
    return Array.from(counts.entries())
      .map(([family, count]) => ({
        key: family,
        label: family,
        count,
        color: FAMILY_COLORS[family] ?? FALLBACK_COLOR,
        title:
          family === FAMILY_UNMAPPED_LABEL
            ? `No clinical family for this primary — ${count.toLocaleString()} recordings.`
            : `${family} — ${count.toLocaleString()} recordings, by primary diagnosis.`,
      }))
      .sort((a, b) => {
        // Unmapped last regardless of size; everything else by count desc.
        const aUnmapped = a.key === FAMILY_UNMAPPED_LABEL ? 1 : 0
        const bUnmapped = b.key === FAMILY_UNMAPPED_LABEL ? 1 : 0
        if (aUnmapped !== bUnmapped) return aUnmapped - bUnmapped
        return b.count - a.count
      })
  }, [points])

  return (
    <ReadOnlyLegend
      caption="Family"
      captionTitle="Clinical family of each record's primary diagnosis."
      entries={entries}
    />
  )
}

/**
 * Key for the predicted-diagnosis blend.
 *
 * Each point is the average of the colours of its predicted classes, so this
 * is a key to the ingredients rather than to the mixed colour on screen. Only
 * classes the diagnosis colour map resolves are listed — those are the only
 * ones that contribute to the blend.
 */
function PredictedDiagnosisLegend() {
  const points = useProjectionPoints()
  const diagnosisColorMap = useDiagnosisColorMap()

  const entries = useMemo(() => {
    const counts = new Map<string, number>()
    for (const p of points) {
      for (const dx of p.xgbPredictions ?? []) {
        if (!diagnosisColorMap.has(dx)) continue
        counts.set(dx, (counts.get(dx) || 0) + 1)
      }
    }
    return Array.from(counts.entries())
      .map(([name, count]) => ({
        key: name,
        label: name,
        count,
        color: diagnosisColorMap.get(name.toLowerCase())!,
      }))
      .sort((a, b) => b.count - a.count)
  }, [points, diagnosisColorMap])

  return (
    <ReadOnlyLegend
      caption="Predicted"
      captionTitle="Each point is the blend of the colours of the classes the model called for it."
      entries={entries}
    />
  )
}

export function UniverseLegendBar() {
  const colorBy = useVisualisationStore((s) => s.colorBy)

  const highlightedDiagnoses = useVisualisationStore((s) => s.highlightedDiagnoses)
  const filledDiagnoses = useVisualisationStore((s) => s.filledDiagnoses)
  const diagnosisOperators = useVisualisationStore((s) => s.diagnosisOperators)
  const cycleDiagnosisFilter = useVisualisationStore((s) => s.cycleDiagnosisFilter)
  const clearDiagnosisFilters = useVisualisationStore((s) => s.clearDiagnosisFilters)
  const toggleDiagnosisOperator = useVisualisationStore((s) => s.toggleDiagnosisOperator)
  const projectionData = useVisualisationStore((s) => s.projectionData)

  const diagnosisColorMap = useDiagnosisColorMap()

  // Count from ALL diagnoses[] per point (not just primaryDiagnosis)
  // Use full unfiltered points so counts don't change when a label is selected
  const diagnosisCounts = useMemo(() => {
    if (!projectionData) return []
    // Count PRIMARIES, not every label.
    //
    // The legend is the colour key, so it has to list exactly what colours the
    // points and nothing else. Points are coloured by primaryDiagnosis, but
    // this counted every label on every point — 98 entries against 39 actual
    // colours, so 59 of them explained nothing on screen and several were not
    // diagnoses at all ('normal ecg', 'abnormal ecg', raw SNOMED codes).
    // Counting primaries makes the key and the plot the same set by
    // construction, and the count beside each name is now the number of points
    // wearing that colour, which is what a key should say.
    const counts = new Map<string, number>()
    for (const p of projectionData.points as EnrichedProjectionPoint[]) {
      const dx = p.primaryDiagnosis
      if (dx) counts.set(dx, (counts.get(dx) || 0) + 1)
    }
    return Array.from(counts.entries())
      .map(([name, count]) => ({ name, count, color: diagnosisColorMap.get(name.toLowerCase()) || '#94A3B8' }))
      .sort((a, b) => {
        // Filled first, then highlighted, then unselected; within each group by count desc
        const aScore = filledDiagnoses.has(a.name) ? 2 : highlightedDiagnoses.has(a.name) ? 1 : 0
        const bScore = filledDiagnoses.has(b.name) ? 2 : highlightedDiagnoses.has(b.name) ? 1 : 0
        if (aScore !== bScore) return bScore - aScore
        return b.count - a.count
      })
  }, [projectionData, diagnosisColorMap, highlightedDiagnoses, filledDiagnoses])

  // Auto-scroll to the left when selection changes so selected pills are visible
  const scrollRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    scrollRef.current?.scrollTo({ left: 0, behavior: 'smooth' })
  }, [highlightedDiagnoses, filledDiagnoses])

  // The legend must describe the colours actually on screen. Only `diagnosis`
  // uses the interactive pills below — every other mode gets a read-only key
  // for the colours it is actually painting.
  if (colorBy === 'cohort') return <CohortLegend />
  if (colorBy === 'error_highlight') return <ErrorHighlightLegend />
  if (colorBy === 'family') return <FamilyLegend />
  if (colorBy === 'diagnosis_predicted') return <PredictedDiagnosisLegend />

  if (diagnosisCounts.length === 0) return null

  const hasSelection = highlightedDiagnoses.size > 0 || filledDiagnoses.size > 0

  // Split into selected and unselected
  const selectedPills = diagnosisCounts.filter(
    (dx) => highlightedDiagnoses.has(dx.name) || filledDiagnoses.has(dx.name),
  )
  const unselectedPills = diagnosisCounts.filter(
    (dx) => !highlightedDiagnoses.has(dx.name) && !filledDiagnoses.has(dx.name),
  )

  return (
    <div
      ref={scrollRef}
      className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-raised/90 backdrop-blur-sm border border-surface-border overflow-x-auto scrollbar-thin"
      onPointerDown={(e) => e.stopPropagation()}
    >
      {/* Clear button */}
      {hasSelection && (
        <button
          onClick={clearDiagnosisFilters}
          className="flex items-center justify-center w-5 h-5 rounded-full bg-slate-200 hover:bg-slate-300 text-slate-500 transition-colors shrink-0"
          title="Clear diagnosis filter"
        >
          <X className="w-3 h-3" />
        </button>
      )}

      {/* Selected pills with operators between them */}
      {selectedPills.map((dx, i) => (
        <Fragment key={dx.name}>
          <PillButton
            dx={dx}
            isHighlighted={highlightedDiagnoses.has(dx.name)}
            isFilled={filledDiagnoses.has(dx.name)}
            hasSelection={hasSelection}
            onClick={() => cycleDiagnosisFilter(dx.name)}
          />
          {i < selectedPills.length - 1 && (
            <button
              onClick={() => toggleDiagnosisOperator(dx.name, selectedPills[i + 1]!.name)}
              className={cn(
                'px-1.5 py-0.5 rounded text-[10px] font-bold shrink-0 transition-colors',
                (diagnosisOperators.get(operatorKey(dx.name, selectedPills[i + 1]!.name)) === 'and')
                  ? 'bg-blue-100 text-blue-600'
                  : 'bg-slate-100 text-slate-400 hover:bg-slate-200',
              )}
              title="Toggle AND/OR"
            >
              {(diagnosisOperators.get(operatorKey(dx.name, selectedPills[i + 1]!.name)) === 'and') ? '&' : '/'}
            </button>
          )}
        </Fragment>
      ))}

      {/* Unselected pills */}
      {unselectedPills.map((dx) => (
        <PillButton
          key={dx.name}
          dx={dx}
          isHighlighted={false}
          isFilled={false}
          hasSelection={hasSelection}
          onClick={() => cycleDiagnosisFilter(dx.name)}
        />
      ))}
    </div>
  )
}
