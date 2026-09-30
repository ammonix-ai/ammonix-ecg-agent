import { create } from 'zustand'
import type {
  EnrichedProjectionPoint,
  EnrichedProjectionResponse,
  ProjectionMethod,
  ColorByMode,
  ViewerSettings,
  CohortLatticeStatus,
} from '@/types/projection'
import { DEFAULT_VIEWER_SETTINGS } from '@/types/projection'
import { getEnrichedProjection, getCVProjection } from '@/services/projectionService'
import {
  getLatticeStatus,
  getCohortProjection,
} from '@/services/cvService'

interface VisualisationState {
  // Data
  projectionData: EnrichedProjectionResponse | null
  points: EnrichedProjectionPoint[]
  isLoading: boolean
  error: string | null

  // Projection settings
  method: ProjectionMethod
  tsnePerplexity: number
  umapNeighbors: number

  // Display
  colorBy: ColorByMode
  viewerSettings: ViewerSettings

  // Filters
  visibleCohorts: Set<string>
  selectedTribes: Set<string>
  showNoTribes: boolean
  highlightedDiagnoses: Set<string>
  filledDiagnoses: Set<string>
  postFillHighlighted: Set<string>
  diagnosisOperators: Map<string, 'and' | 'or'>

  /**
   * True once the scene is filled from a CV/cohort projection endpoint rather
   * than the legacy cohort-code one. Those payloads do not use the Ar/Br/Tr
   * cohort codes, so `visibleCohorts` must not filter them.
   */
  _usingCVData: boolean

  // Selection
  selectedPatientId: string | null
  selectedPatients: Set<string>
  hoveredPatient: string | null
  patientLabels: Map<string, string>

  // Neighbors
  showNeighbors: boolean
  neighborCount: number
  neighborsRespectFilter: boolean

  // Search
  searchQuery: string
  searchedPatientId: string | null

  // Panels
  showSettings: boolean
  showStats: boolean
  showLegend: boolean
  showTribeFilter: boolean
  showColorSettings: boolean

  // Legacy compat
  totalPatients: number
  cohortsIncluded: string[]
  diagnosisColors: Record<string, string>
  cached: boolean

  // Upload (for beacon animation)
  uploadedPatientIds: Set<string>

  // Fly-to camera
  flyToTarget: { x: number; y: number; z: number } | null

  // Export
  screenshotRequested: boolean
  exportFormat: 'png' | 'csv' | 'json' | null

  // Mini-map
  showMiniMap: boolean

  // Layout
  isFullscreen: boolean
  isOrthographic: boolean
  panelsCollapsed: boolean

  // External point filter (e.g. demo mode)
  pointFilter: ((point: EnrichedProjectionPoint) => boolean) | null

  // Cohort-first lattice
  primaryCohortId: number | null
  overlayCohortIds: number[]
  modelCohortId: number | null
  cohortStatuses: CohortLatticeStatus[]
  cohortStatusesLoading: boolean
  showSupplements: boolean

  // === Actions ===

  // Data
  fetchProjection: () => Promise<void>
  clearError: () => void

  // Projection settings
  setMethod: (m: ProjectionMethod) => void
  setTsnePerplexity: (p: number) => void
  setUmapNeighbors: (n: number) => void

  // Display
  setColorBy: (mode: ColorByMode) => void
  updateViewerSettings: (partial: Partial<ViewerSettings>) => void

  // Filters
  toggleCohort: (cohort: string) => void
  setCohorts: (cohorts: Set<string>) => void
  toggleTribe: (tribe: string) => void
  setTribes: (tribes: Set<string>) => void
  setShowNoTribes: (show: boolean) => void
  cycleDiagnosisFilter: (dx: string) => void
  clearDiagnosisFilters: () => void
  toggleDiagnosisOperator: (dx1: string, dx2: string) => void

  // Selection
  setSelectedPatient: (id: string | null) => void
  selectPatient: (id: string) => void
  deselectPatient: (id: string) => void
  dismissCard: () => void
  togglePatient: (id: string) => void
  clearSelection: () => void
  setHoveredPatient: (id: string | null) => void
  setPatientLabel: (id: string, label: string) => void

  // Neighbors
  setShowNeighbors: (v: boolean) => void
  setNeighborCount: (n: number) => void
  setNeighborsRespectFilter: (v: boolean) => void

  // Search
  setSearchQuery: (query: string) => void
  clearSearch: () => void

  // Panels
  setShowSettings: (show: boolean) => void
  setShowStats: (show: boolean) => void
  setShowLegend: (v: boolean) => void
  setShowTribeFilter: (v: boolean) => void
  setShowColorSettings: (v: boolean) => void

  // Bulk selection (for lasso)
  setSelectedPatients: (ids: Set<string>) => void

  // Fly-to
  setFlyToTarget: (target: { x: number; y: number; z: number } | null) => void
  clearFlyToTarget: () => void

  // Export
  requestScreenshot: () => void
  clearScreenshotRequest: () => void
  requestExport: (format: 'png' | 'csv' | 'json') => void
  clearExportRequest: () => void

  // Mini-map
  setShowMiniMap: (v: boolean) => void

  // Layout
  setFullscreen: (v: boolean) => void
  setOrthographic: (v: boolean) => void
  setPanelsCollapsed: (v: boolean) => void

  // External filter
  setPointFilter: (fn: ((point: EnrichedProjectionPoint) => boolean) | null) => void

  // Cohort lattice actions
  fetchCohortStatuses: () => Promise<void>
  setPrimaryCohort: (id: number | null) => void
  setModelCohort: (id: number | null) => void
  setShowSupplements: (v: boolean) => void
  fetchCohortProjection: () => Promise<void>

  // Reset
  resetFilters: () => void
  resetAll: () => void
  clearProjectionCache: () => void
}

const DEFAULT_FILTERS = {
  visibleCohorts: new Set(['Ar', 'Br', 'Tr']),
  selectedTribes: new Set<string>(),
  showNoTribes: true,
  highlightedDiagnoses: new Set<string>(),
  filledDiagnoses: new Set<string>(),
  postFillHighlighted: new Set<string>(),
  diagnosisOperators: new Map<string, 'and' | 'or'>(),
}

/** Canonical key for a diagnosis operator pair (alphabetical order). */
export function operatorKey(a: string, b: string): string {
  return a < b ? `${a}|||${b}` : `${b}|||${a}`
}

export const useVisualisationStore = create<VisualisationState>((set, get) => ({
  // Data
  projectionData: null,
  points: [],
  isLoading: false,
  error: null,

  // Projection settings
  method: 'tsne',
  tsnePerplexity: 100,
  umapNeighbors: 15,

  // Display. Diagnosis is the default view: it is what the universe is for,
  // and it is the only mode whose legend is interactive.
  colorBy: 'diagnosis',
  viewerSettings: DEFAULT_VIEWER_SETTINGS,

  // Filters
  visibleCohorts: new Set(['Ar', 'Br', 'Tr']),
  selectedTribes: new Set(),
  showNoTribes: true,
  highlightedDiagnoses: new Set<string>(),
  filledDiagnoses: new Set<string>(),
  postFillHighlighted: new Set<string>(),
  diagnosisOperators: new Map<string, 'and' | 'or'>(),

  _usingCVData: false,

  // Selection
  selectedPatientId: null,
  selectedPatients: new Set<string>(),
  hoveredPatient: null,
  patientLabels: new Map(),

  // Neighbors
  showNeighbors: false,
  neighborCount: 6,
  neighborsRespectFilter: false,

  // Search
  searchQuery: '',
  searchedPatientId: null,

  // Panels
  showSettings: true,
  showStats: false,
  showLegend: true,
  showTribeFilter: false,
  showColorSettings: false,

  // Legacy compat
  totalPatients: 0,
  cohortsIncluded: [],
  diagnosisColors: {},
  cached: false,

  // Upload
  uploadedPatientIds: new Set<string>(),

  // Fly-to
  flyToTarget: null,

  // Export
  screenshotRequested: false,
  exportFormat: null,

  // Mini-map
  showMiniMap: true,

  // Layout
  isFullscreen: false,
  isOrthographic: false,
  panelsCollapsed: false,

  // External point filter
  pointFilter: null,

  // Cohort-first lattice
  primaryCohortId: null,
  overlayCohortIds: [],
  modelCohortId: null,
  cohortStatuses: [],
  cohortStatusesLoading: false,
  showSupplements: true,

  // =========================================================================
  // Actions — Data
  // =========================================================================

  fetchProjection: async () => {
    // Delegate to cohort-aware path if a primary cohort is selected
    if (get().primaryCohortId) {
      return get().fetchCohortProjection()
    }

    const { method, visibleCohorts, tsnePerplexity, umapNeighbors, _usingCVData } = get()
    set({ isLoading: true, error: null })

    try {
      if (_usingCVData) {
        // CV endpoint: every patient in one payload
        const resp = await getCVProjection({
          method,
          foldMode: 'all',
          perplexity: method === 'tsne' ? tsnePerplexity : undefined,
          n_neighbors: method === 'umap' ? umapNeighbors : undefined,
        })
        set({
          projectionData: resp,
          points: [...resp.points],
          totalPatients: resp.totalPatients,
          cohortsIncluded: [],
          diagnosisColors: resp.diagnosisColors,
          cached: resp.cached,
          isLoading: false,
          _usingCVData: true,
        })
      } else {
        // Legacy endpoint: cohort-based projections
        const resp = await getEnrichedProjection(method, {
          cohorts: Array.from(visibleCohorts).join(','),
          perplexity: method === 'tsne' ? tsnePerplexity : undefined,
          n_neighbors: method === 'umap' ? umapNeighbors : undefined,
        })
        set({
          projectionData: resp,
          points: [...resp.points],
          totalPatients: resp.totalPatients,
          cohortsIncluded: [...resp.cohortsIncluded],
          diagnosisColors: resp.diagnosisColors,
          cached: resp.cached,
          isLoading: false,
          _usingCVData: false,
        })
      }
    } catch (e) {
      set({
        error: e instanceof Error ? e.message : 'Failed to load projection',
        isLoading: false,
      })
    }
  },

  clearError: () => set({ error: null }),

  // =========================================================================
  // Actions — Projection settings
  // =========================================================================

  setMethod: (m) => {
    set({ method: m })
    get().fetchProjection()
  },

  setTsnePerplexity: (p) => {
    set({ tsnePerplexity: p })
    get().fetchProjection()
  },
  setUmapNeighbors: (n) => {
    set({ umapNeighbors: n })
    get().fetchProjection()
  },

  // =========================================================================
  // Actions — Display
  // =========================================================================

  // The 'fold' colour mode used to force a CV refetch here. It is gone in the
  // open build, so setting the mode is a pure display change.
  setColorBy: (mode) => set({ colorBy: mode }),

  updateViewerSettings: (partial) => {
    const current = get().viewerSettings
    set({
      viewerSettings: {
        colors: partial.colors ? { ...current.colors, ...partial.colors } : current.colors,
        display: partial.display ? { ...current.display, ...partial.display } : current.display,
        diagnosisColorOverrides: partial.diagnosisColorOverrides ?? current.diagnosisColorOverrides,
      },
    })
  },

  // =========================================================================
  // Actions — Filters
  // =========================================================================

  toggleCohort: (cohort) => {
    const cohorts = new Set(get().visibleCohorts)
    if (cohorts.has(cohort)) cohorts.delete(cohort)
    else cohorts.add(cohort)
    set({ visibleCohorts: cohorts })
  },

  setCohorts: (cohorts) => set({ visibleCohorts: new Set(cohorts) }),

  toggleTribe: (tribe) => {
    const tribes = new Set(get().selectedTribes)
    if (tribes.has(tribe)) tribes.delete(tribe)
    else tribes.add(tribe)
    set({ selectedTribes: tribes })
  },

  setTribes: (tribes) => set({ selectedTribes: new Set(tribes) }),
  setShowNoTribes: (v) => set({ showNoTribes: v }),
  cycleDiagnosisFilter: (dx) => set((s) => {
    const hl = new Set(s.highlightedDiagnoses)
    const fl = new Set(s.filledDiagnoses)
    const pfh = new Set(s.postFillHighlighted)

    if (fl.has(dx)) {
      // filled -> highlighted (post-fill)
      fl.delete(dx)
      hl.add(dx)
      pfh.add(dx)
    } else if (hl.has(dx) && pfh.has(dx)) {
      // highlighted (post-fill) -> unselected — clean up operators
      hl.delete(dx)
      pfh.delete(dx)
      const ops = new Map(s.diagnosisOperators)
      for (const key of ops.keys()) {
        if (key.includes(dx)) ops.delete(key)
      }
      return { highlightedDiagnoses: hl, filledDiagnoses: fl, postFillHighlighted: pfh, diagnosisOperators: ops }
    } else if (hl.has(dx)) {
      // highlighted (first) -> filled
      hl.delete(dx)
      fl.add(dx)
    } else {
      // unselected -> highlighted
      hl.add(dx)
    }
    return { highlightedDiagnoses: hl, filledDiagnoses: fl, postFillHighlighted: pfh }
  }),
  clearDiagnosisFilters: () => set({
    highlightedDiagnoses: new Set<string>(),
    filledDiagnoses: new Set<string>(),
    postFillHighlighted: new Set<string>(),
    diagnosisOperators: new Map<string, 'and' | 'or'>(),
  }),

  toggleDiagnosisOperator: (dx1, dx2) => set((s) => {
    const ops = new Map(s.diagnosisOperators)
    const key = operatorKey(dx1, dx2)
    ops.set(key, ops.get(key) === 'and' ? 'or' : 'and')
    return { diagnosisOperators: ops }
  }),

  // =========================================================================
  // Actions — Selection
  // =========================================================================

  setSelectedPatient: (id) => {
    if (id) {
      set({
        selectedPatientId: id,
        selectedPatients: new Set([id]),
      })
    } else {
      set({
        selectedPatientId: null,
        selectedPatients: new Set<string>(),
      })
    }
  },

  selectPatient: (id) => {
    const patients = new Set(get().selectedPatients)
    patients.add(id)
    set({ selectedPatients: patients, selectedPatientId: id })
  },

  deselectPatient: (id) => {
    const patients = new Set(get().selectedPatients)
    patients.delete(id)
    set({ selectedPatients: patients, selectedPatientId: null })
  },

  dismissCard: () => set({ selectedPatientId: null }),

  togglePatient: (id) => {
    const patients = new Set(get().selectedPatients)
    let nextSelected: string | null
    if (patients.has(id)) {
      patients.delete(id)
      nextSelected = null
    } else {
      patients.add(id)
      nextSelected = id
    }
    set({ selectedPatients: patients, selectedPatientId: nextSelected })
  },

  clearSelection: () =>
    set({
      selectedPatientId: null,
      selectedPatients: new Set<string>(),
    }),

  setHoveredPatient: (id) => set({ hoveredPatient: id }),

  setPatientLabel: (id, label) =>
    set((s) => {
      const next = new Map(s.patientLabels)
      if (label) next.set(id, label)
      else next.delete(id)
      return { patientLabels: next }
    }),

  // =========================================================================
  // Actions — Neighbors
  // =========================================================================

  setShowNeighbors: (v) => set({ showNeighbors: v }),
  setNeighborCount: (n) => set({ neighborCount: n }),
  setNeighborsRespectFilter: (v) => set({ neighborsRespectFilter: v }),

  // =========================================================================
  // Actions — Search
  // =========================================================================

  setSearchQuery: (query) => {
    const data = get().projectionData
    let searchedPatientId: string | null = null

    if (query.trim() && data) {
      const q = query.toLowerCase()
      const match = data.points.find(
        (p) =>
          p.displayId?.toLowerCase().includes(q) ||
          p.patientId.toLowerCase().includes(q) ||
          p.primaryDiagnosis.toLowerCase().includes(q),
      )
      if (match) searchedPatientId = match.patientId
    }

    set({ searchQuery: query, searchedPatientId })
  },

  clearSearch: () => set({ searchQuery: '', searchedPatientId: null }),

  // =========================================================================
  // Actions — Panels
  // =========================================================================

  setShowSettings: (show) => set({ showSettings: show }),
  setShowStats: (show) => set({ showStats: show }),
  setShowLegend: (v) => set({ showLegend: v }),
  setShowTribeFilter: (v) => set({ showTribeFilter: v }),
  setShowColorSettings: (v) => set({ showColorSettings: v }),

  // =========================================================================
  // Actions — Bulk selection (lasso)
  // =========================================================================

  setSelectedPatients: (ids) => {
    set({ selectedPatients: new Set(ids), selectedPatientId: null })
  },

  // =========================================================================
  // Actions — Fly-to
  // =========================================================================

  setFlyToTarget: (target) => set({ flyToTarget: target }),
  clearFlyToTarget: () => set({ flyToTarget: null }),

  // =========================================================================
  // Actions — Export
  // =========================================================================

  requestScreenshot: () => set({ screenshotRequested: true }),
  clearScreenshotRequest: () => set({ screenshotRequested: false }),
  requestExport: (format) => set({ exportFormat: format }),
  clearExportRequest: () => set({ exportFormat: null }),

  // =========================================================================
  // Actions — Mini-map
  // =========================================================================

  setShowMiniMap: (v) => set({ showMiniMap: v }),

  // =========================================================================
  // Actions — Layout
  // =========================================================================

  setFullscreen: (v) => set({ isFullscreen: v }),
  setOrthographic: (v) => set({ isOrthographic: v }),
  setPanelsCollapsed: (v) => set({ panelsCollapsed: v }),

  // =========================================================================
  // Actions — External filter
  // =========================================================================

  setPointFilter: (fn) => set({ pointFilter: fn }),

  // =========================================================================
  // Actions — Cohort lattice
  // =========================================================================

  fetchCohortStatuses: async () => {
    set({ cohortStatusesLoading: true })
    try {
      const { cohorts } = await getLatticeStatus()
      set({ cohortStatuses: cohorts, cohortStatusesLoading: false })
    } catch {
      set({ cohortStatusesLoading: false })
    }
  },

  setPrimaryCohort: (id) => {
    set({ primaryCohortId: id, modelCohortId: null })
    if (id) get().fetchCohortProjection()
  },

  // Open build: addOverlayCohort / removeOverlayCohort are gone with the
  // overlay endpoint they drove. `overlayCohortIds` stays in state (always []),
  // because the supplement-count pill in UniverseControls reads it and would
  // light up again if overlay support is re-added.

  setModelCohort: (id) => {
    set({ modelCohortId: id })
    if (get().primaryCohortId) get().fetchCohortProjection()
  },

  setShowSupplements: (v) => set({ showSupplements: v }),

  fetchCohortProjection: async () => {
    const { primaryCohortId, method, tsnePerplexity, umapNeighbors } = get()
    if (!primaryCohortId) return

    set({ isLoading: true, error: null })
    try {
      const params = {
        // tsnePerplexity is frozen at 100 (no UI control in the open build) and
        // umapNeighbors is constrained to the pre-baked {10,15,50}. Both must be
        // sent — omitting them makes the backend recompute the projection.
        perplexity: method === 'tsne' ? tsnePerplexity : undefined,
        n_neighbors: method === 'umap' ? umapNeighbors : undefined,
      }

      const resp = await getCohortProjection(method, primaryCohortId, params)

      set({
        projectionData: resp,
        points: resp.points ?? [],
        totalPatients: resp.totalPatients ?? 0,
        diagnosisColors: resp.diagnosisColors ?? {},
        isLoading: false,
        _usingCVData: true,
      })
    } catch (e) {
      set({
        error: e instanceof Error ? e.message : 'Failed to load cohort projection',
        isLoading: false,
      })
    }
  },

  // =========================================================================
  // Actions — Reset
  // =========================================================================

  resetFilters: () =>
    set({
      ...DEFAULT_FILTERS,
      visibleCohorts: new Set(DEFAULT_FILTERS.visibleCohorts),
      selectedTribes: new Set(DEFAULT_FILTERS.selectedTribes),
    }),

  resetAll: () =>
    set({
      method: 'tsne',
      tsnePerplexity: 100,
      umapNeighbors: 15,
      colorBy: 'diagnosis',
      viewerSettings: DEFAULT_VIEWER_SETTINGS,
      visibleCohorts: new Set(['Ar', 'Br', 'Tr']),
      selectedTribes: new Set<string>(),
      showNoTribes: true,
      highlightedDiagnoses: new Set<string>(),
      filledDiagnoses: new Set<string>(),
      postFillHighlighted: new Set<string>(),
      diagnosisOperators: new Map<string, 'and' | 'or'>(),
      _usingCVData: false,
      selectedPatientId: null,
      selectedPatients: new Set<string>(),
      hoveredPatient: null,
      patientLabels: new Map<string, string>(),
      showNeighbors: false,
      neighborCount: 6,
      neighborsRespectFilter: false,
      searchQuery: '',
      searchedPatientId: null,
      showSettings: false,
      showStats: false,
      showLegend: false,
      showTribeFilter: false,
      showColorSettings: false,
      flyToTarget: null,
      screenshotRequested: false,
      showMiniMap: false,
      isFullscreen: false,
      isOrthographic: false,
      panelsCollapsed: false,
      pointFilter: null,
    }),

  clearProjectionCache: () =>
    set({ projectionData: null, points: [], isLoading: false, error: null, cached: false }),
}))
