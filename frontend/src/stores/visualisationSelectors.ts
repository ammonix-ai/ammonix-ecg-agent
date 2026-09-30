import { useMemo } from 'react'
import { useVisualisationStore, operatorKey } from './visualisationStore'
import type { EnrichedProjectionPoint } from '@/types/projection'
import { COHORT_COLOR_PALETTE, DIAGNOSIS_COLOR_PALETTE } from '@/types/projection'
import { getPointColor } from '@/utils/colorUtils'

/**
 * Sort selected diagnoses in the same order as the legend bar:
 * filled (2) > highlighted (1) > unselected (0), then by count desc.
 */
export function sortSelectedDiagnoses(
  selected: string[],
  highlightedDiagnoses: Set<string>,
  filledDiagnoses: Set<string>,
  countMap: Map<string, number>,
): string[] {
  return [...selected].sort((a, b) => {
    const aScore = filledDiagnoses.has(a) ? 2 : highlightedDiagnoses.has(a) ? 1 : 0
    const bScore = filledDiagnoses.has(b) ? 2 : highlightedDiagnoses.has(b) ? 1 : 0
    if (aScore !== bScore) return bScore - aScore
    return (countMap.get(b) || 0) - (countMap.get(a) || 0)
  })
}

/**
 * Compute uniform scale factor to fit all points within a 10-unit cube.
 * Uses the full (unfiltered) dataset.
 */
export function useScaleFactor(): number {
  const projectionData = useVisualisationStore((s) => s.projectionData)

  return useMemo(() => {
    if (!projectionData || projectionData.points.length === 0) return 1

    let maxCoord = 0
    for (const p of projectionData.points) {
      maxCoord = Math.max(maxCoord, Math.abs(p.x), Math.abs(p.y), Math.abs(p.z))
    }

    return maxCoord > 0 ? 10 / maxCoord : 1
  }, [projectionData])
}

/**
 * Filter projection points by all active filters:
 * 1. visibleCohorts
 * 2. selectedTribes (empty = show all)
 * 3. highlightedDiagnoses (inclusive/union) + filledDiagnoses (exclusive)
 * 4. searchQuery
 * 5. External point filter
 * 6. showSupplements (overlay cohort patients)
 */
export function useFilteredPoints(): EnrichedProjectionPoint[] {
  const projectionData = useVisualisationStore((s) => s.projectionData)
  const visibleCohorts = useVisualisationStore((s) => s.visibleCohorts)
  const selectedTribes = useVisualisationStore((s) => s.selectedTribes)
  const showNoTribes = useVisualisationStore((s) => s.showNoTribes)
  const highlightedDiagnoses = useVisualisationStore((s) => s.highlightedDiagnoses)
  const filledDiagnoses = useVisualisationStore((s) => s.filledDiagnoses)
  const diagnosisOperators = useVisualisationStore((s) => s.diagnosisOperators)
  const pointFilter = useVisualisationStore((s) => s.pointFilter)
  const usingCVData = useVisualisationStore((s) => s._usingCVData)
  const showSupplements = useVisualisationStore((s) => s.showSupplements)

  return useMemo(() => {
    if (!projectionData) return []

    let points = projectionData.points

    // 1. Filter by cohort (skip when using CV data — CV points use cohort 'oof')
    if (!usingCVData && visibleCohorts.size > 0) {
      points = points.filter((p) => visibleCohorts.has(p.cohort))
    }

    // 2. Filter by tribes (empty selectedTribes = no tribe filter)
    if (selectedTribes.size > 0) {
      points = points.filter((p) => {
        const tribes = p.tribes ?? []
        if (tribes.length === 0) return showNoTribes
        return tribes.some((t) => selectedTribes.has(t))
      })
    }

    // 3. Filter by diagnoses
    const totalSelected = highlightedDiagnoses.size + filledDiagnoses.size
    if (totalSelected === 1) {
      // Single pill: original highlight/fill behavior
      points = points.filter((p) => {
        if (p.diagnoses.some((d) => highlightedDiagnoses.has(d))) return true
        if (filledDiagnoses.size > 0 && p.diagnoses.every((d) => filledDiagnoses.has(d))) return true
        return false
      })
    } else if (totalSelected >= 2) {
      // Multi-pill: operator-aware grouping
      // Build count map for sorting (use pre-filtered points for consistent order)
      const countMap = new Map<string, number>()
      for (const p of points) {
        for (const dx of p.diagnoses) {
          countMap.set(dx, (countMap.get(dx) || 0) + 1)
        }
      }

      // Get all selected diagnoses in display order
      const allSelected = [
        ...Array.from(filledDiagnoses),
        ...Array.from(highlightedDiagnoses),
      ]
      const ordered = sortSelectedDiagnoses(allSelected, highlightedDiagnoses, filledDiagnoses, countMap)

      // Group into AND-groups: walk adjacent pairs, split at '/' (or) boundaries
      // Track whether each group is exclusive (all filled) or inclusive (any highlighted)
      const andGroups: { diagnoses: string[]; exclusive: boolean }[] = []
      let currentGroup: string[] = [ordered[0]!]
      for (let i = 1; i < ordered.length; i++) {
        const key = operatorKey(ordered[i - 1]!, ordered[i]!)
        const op = diagnosisOperators.get(key) || 'or'
        if (op === 'and') {
          currentGroup.push(ordered[i]!)
        } else {
          andGroups.push({
            diagnoses: currentGroup,
            exclusive: currentGroup.every((d) => filledDiagnoses.has(d)),
          })
          currentGroup = [ordered[i]!]
        }
      }
      andGroups.push({
        diagnoses: currentGroup,
        exclusive: currentGroup.every((d) => filledDiagnoses.has(d)),
      })

      // Filter: patient matches if ANY AND-group is satisfied
      // Exclusive groups: patient has ONLY these diagnoses
      // Inclusive groups: patient has all these diagnoses (can have others)
      points = points.filter((p) => {
        const patientDx = new Set(p.diagnoses)
        return andGroups.some((group) => {
          const hasAll = group.diagnoses.every((dx) => patientDx.has(dx))
          if (!hasAll) return false
          if (group.exclusive) return patientDx.size === group.diagnoses.length
          return true
        })
      })
    }

    // 4. Search query — handled by search dropdown in UniverseControls only.
    // Does NOT filter the 3D scene (points stay visible while typing).

    // 5. External point filter
    if (pointFilter) {
      points = points.filter(pointFilter)
    }

    // 6. Supplement visibility (overlay cohort patients)
    if (!showSupplements) {
      points = points.filter((p) => !p.isSupplement)
    }

    return points
  }, [projectionData, visibleCohorts, selectedTribes, showNoTribes, highlightedDiagnoses, filledDiagnoses, diagnosisOperators, pointFilter, usingCVData, showSupplements])
}

/**
 * Find N nearest neighbors for the first selected patient.
 * Uses Euclidean distance in 3D over the full (unfiltered) dataset.
 */
export function useNearestNeighbors(): EnrichedProjectionPoint[] {
  const projectionData = useVisualisationStore((s) => s.projectionData)
  const selectedPatients = useVisualisationStore((s) => s.selectedPatients)
  const showNeighbors = useVisualisationStore((s) => s.showNeighbors)
  const neighborCount = useVisualisationStore((s) => s.neighborCount)
  const neighborsRespectFilter = useVisualisationStore((s) => s.neighborsRespectFilter)
  const filteredPoints = useFilteredPoints()

  return useMemo(() => {
    if (!showNeighbors || !projectionData || selectedPatients.size === 0) return []

    const targetId = selectedPatients.values().next().value
    if (!targetId) return []

    const target = projectionData.points.find((p) => p.patientId === targetId)
    if (!target) return []

    // Search pool: filtered points when respecting filter, all points otherwise
    const pool = neighborsRespectFilter ? filteredPoints : projectionData.points

    return pool
      .filter((p) => p.patientId !== targetId)
      .map((p) => ({
        point: p,
        distance: Math.sqrt(
          (p.x - target.x) ** 2 +
          (p.y - target.y) ** 2 +
          (p.z - target.z) ** 2,
        ),
      }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, neighborCount)
      .map((entry) => entry.point)
  }, [projectionData, selectedPatients, showNeighbors, neighborCount, neighborsRespectFilter, filteredPoints])
}

/**
 * Extract unique tribe IDs from points, filtered by visible cohorts.
 */
export function useAvailableTribes(): string[] {
  const projectionData = useVisualisationStore((s) => s.projectionData)
  const visibleCohorts = useVisualisationStore((s) => s.visibleCohorts)

  return useMemo(() => {
    if (!projectionData) return []

    const tribes = new Set<string>()
    for (const p of projectionData.points) {
      if (visibleCohorts.size > 0 && !visibleCohorts.has(p.cohort)) continue
      for (const t of p.tribes ?? []) {
        tribes.add(t)
      }
    }

    return Array.from(tribes).sort()
  }, [projectionData, visibleCohorts])
}

/**
 * Resolve selectedPatients Set to full EnrichedProjectionPoint objects.
 */
export function useSelectedPatientObjects(): EnrichedProjectionPoint[] {
  const projectionData = useVisualisationStore((s) => s.projectionData)
  const selectedPatients = useVisualisationStore((s) => s.selectedPatients)

  return useMemo(() => {
    if (!projectionData || selectedPatients.size === 0) return []

    const lookup = new Map<string, EnrichedProjectionPoint>()
    for (const p of projectionData.points) {
      if (selectedPatients.has(p.patientId)) {
        lookup.set(p.patientId, p)
      }
    }

    const result: EnrichedProjectionPoint[] = []
    for (const id of selectedPatients) {
      const point = lookup.get(id)
      if (point) result.push(point)
    }

    return result
  }, [projectionData, selectedPatients])
}

/**
 * Build alphabetically-sorted diagnosis -> color mapping.
 */
export function useDiagnosisColorMap(): Map<string, string> {
  const projectionData = useVisualisationStore((s) => s.projectionData)
  const diagnosisColorOverrides = useVisualisationStore((s) => s.viewerSettings.diagnosisColorOverrides)

  return useMemo(() => {
    const colorMap = new Map<string, string>()
    if (!projectionData) return colorMap

    // Keys are LOWERCASED, and the lookup lowercases too.
    //
    // The map is built from primaryDiagnosis but read with the point's own
    // labels, and those two do not agree on case: get_primary_diagnosis()
    // lowercases its input, so a record labelled
    // 'lvef <40% expanded MIMIC EF>=55 controls' has the primary
    // 'lvef <40% expanded mimic ef>=55 controls'. The label never matched the
    // key, so 6,140 records — every LVEF EF<40 case — rendered grey despite
    // carrying a perfectly good diagnosis. 'STEMI' vs 'stemi' is the same trap.
    // Primaries first, then every other label.
    //
    // Points are coloured by their primary, so those keys decide what is on
    // screen and are assigned first — that keeps a diagnosis's colour stable
    // no matter what else appears in the data. The remaining labels are
    // assigned afterwards purely so the legend can show a swatch for them:
    // the legend lists all 96 labels because it doubles as the filter, and
    // without this 58 of them rendered grey against a fully-coloured plot.
    const primaries = new Set<string>()
    const others = new Set<string>()
    for (const p of projectionData.points) {
      primaries.add(p.primaryDiagnosis.toLowerCase())
    }
    for (const p of projectionData.points) {
      for (const dx of p.diagnoses) {
        const key = dx.toLowerCase()
        if (!primaries.has(key)) others.add(key)
      }
    }
    const sorted = [...Array.from(primaries).sort(), ...Array.from(others).sort()]

    // Backend colours and user overrides are keyed on the original casing, so
    // index them case-insensitively rather than requiring an exact hit.
    const backendColors: Record<string, string> = {}
    for (const [k, v] of Object.entries(projectionData.diagnosisColors || {})) {
      backendColors[k.toLowerCase()] = v
    }
    const overrides: Record<string, string> = {}
    for (const [k, v] of Object.entries(diagnosisColorOverrides)) {
      if (v) overrides[k.toLowerCase()] = v
    }

    let paletteIdx = 0
    for (const dx of sorted) {
      // Priority: user overrides > backend colors > palette
      if (overrides[dx]) {
        colorMap.set(dx, overrides[dx]!)
      } else if (backendColors[dx]) {
        colorMap.set(dx, backendColors[dx]!)
      } else {
        colorMap.set(dx, DIAGNOSIS_COLOR_PALETTE[paletteIdx % DIAGNOSIS_COLOR_PALETTE.length]!)
        paletteIdx++
      }
    }

    return colorMap
  }, [projectionData, diagnosisColorOverrides])
}

/**
 * Build cohort -> color mapping for the 'cohort' grading mode.
 *
 * Cohort names come from the data (the published universe ships four:
 * Apr28 multi-source, MIMIC SR normals, LVEF EF<40 cases, LVEF EF>=55
 * controls). Sorting before assigning keeps a cohort's colour stable across
 * renders and across projection methods.
 */
export function useCohortColorMap(): Map<string, string> {
  const projectionData = useVisualisationStore((s) => s.projectionData)

  return useMemo(() => {
    const colorMap = new Map<string, string>()
    if (!projectionData) return colorMap

    const cohorts = new Set<string>()
    for (const p of projectionData.points) {
      if (p.cohort) cohorts.add(p.cohort)
    }

    Array.from(cohorts)
      .sort()
      .forEach((cohort, i) => {
        colorMap.set(cohort, COHORT_COLOR_PALETTE[i % COHORT_COLOR_PALETTE.length]!)
      })

    return colorMap
  }, [projectionData])
}

/**
 * The classes the model actually scores, lowercased.
 *
 * The keys of `sourceSafe.thresholds` are exactly that set (32 classes in the
 * published universe) — the backend already puts them on the wire with the
 * projection, so nothing extra is fetched. Needed by the error-highlight
 * colouring, which must not count a gold label the model has no class for as
 * a missed diagnosis.
 *
 * Lowercased because the wire mixes cases ('STEMI' in labels and thresholds,
 * 'stemi' in primary). Empty when the projection ships no threshold metadata.
 */
export function useScoredClasses(): Set<string> {
  const projectionData = useVisualisationStore((s) => s.projectionData)

  return useMemo(() => {
    const thresholds = projectionData?.sourceSafe?.thresholds
    if (!thresholds) return new Set<string>()
    return new Set(Object.keys(thresholds).map((k) => k.toLowerCase()))
  }, [projectionData])
}

/**
 * Point statistics: total, filtered, and selected counts.
 */
export function usePointStats(): { total: number; filtered: number; selected: number } {
  const projectionData = useVisualisationStore((s) => s.projectionData)
  const selectedPatients = useVisualisationStore((s) => s.selectedPatients)
  const filteredPoints = useFilteredPoints()

  return useMemo(() => ({
    total: projectionData?.points.length ?? 0,
    filtered: filteredPoints.length,
    selected: selectedPatients.size,
  }), [projectionData, filteredPoints, selectedPatients])
}

/**
 * Pre-compute colors for all filtered points based on the active color mode.
 * Returns Map<patientId, hexColor> so the 3D scene does a simple lookup per point.
 */
export function usePointColors(): Map<string, string> {
  const colorMode = useVisualisationStore((s) => s.colorBy)
  const viewerSettings = useVisualisationStore((s) => s.viewerSettings)
  const filteredPoints = useFilteredPoints()
  const diagnosisColorMap = useDiagnosisColorMap()
  const cohortColorMap = useCohortColorMap()
  const scoredClasses = useScoredClasses()

  return useMemo(() => {
    const colorMap = new Map<string, string>()
    if (filteredPoints.length === 0) return colorMap

    for (const point of filteredPoints) {
      colorMap.set(
        point.patientId,
        getPointColor(point, colorMode, viewerSettings, diagnosisColorMap, cohortColorMap, scoredClasses),
      )
    }

    return colorMap
  }, [filteredPoints, colorMode, viewerSettings, diagnosisColorMap, cohortColorMap, scoredClasses])
}
