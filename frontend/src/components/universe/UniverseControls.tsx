import { useState, useRef, useMemo, useCallback, useEffect } from 'react'
import { Loader2, Settings, Search, X, Download, Image, FileSpreadsheet, FileJson, Maximize2, Minimize2, PanelLeftClose, BarChart3, Info } from 'lucide-react'
import { useVisualisationStore } from '@/stores/visualisationStore'
import { useFilteredPoints, useScaleFactor } from '@/stores/visualisationSelectors'
import type { ProjectionMethod, ColorByMode } from '@/types/projection'
import { cn } from '@/design/cn'
import { pointIdLabel } from './recordingLink'

const METHODS: { value: ProjectionMethod; label: string }[] = [
  { value: 'tsne', label: 't-SNE' },
  { value: 'umap', label: 'UMAP' },
  { value: 'pca', label: 'PCA' },
]

/**
 * UMAP n_neighbors values that ship pre-computed with the universe
 * (projections/umap_neighbors_{10,15,50}.npy). Any other value makes the
 * backend fit UMAP on the full 63,256 x 32 probability matrix on the request
 * thread — minutes of wall clock. Keep this list in sync with what ships.
 */
const UMAP_NEIGHBORS = [10, 15, 50]

const GRADING_INFO: Record<string, string> = {
  diagnosis: 'Each point colored by its primary gold-standard diagnosis. One color per diagnosis class.',
  // Not a train/test split: the published universe carries no fold membership,
  // and every row is scored with the mean of all five folds. Cohort is the real
  // distinction between these records.
  cohort: 'One color per source cohort the record came from.',
  // Compared only over the classes the model scores — a gold label it has no
  // class for cannot be "missed".
  error_highlight: 'Dark red = over-called AND missed. Orange = over-called only. Amber = missed only. Dark gray = exact match. Light gray = not graded (no scored gold label). Compared only over the scored classes.',
  family: '8 clinical families: Rhythm (blue), Atrial (pink), Ventricular (red), Conduction (amber), Ischemia (dark red), Structural (emerald), Repolarization (violet), Controls & Normal (slate). Unmapped primaries stay gray.',
}

export function UniverseControls() {
  const method = useVisualisationStore((s) => s.method)
  const colorBy = useVisualisationStore((s) => s.colorBy)
  const isLoading = useVisualisationStore((s) => s.isLoading)
  const showSettings = useVisualisationStore((s) => s.showSettings)
  const showLegend = useVisualisationStore((s) => s.showLegend)
  const searchQuery = useVisualisationStore((s) => s.searchQuery)

  const setMethod = useVisualisationStore((s) => s.setMethod)
  const setColorBy = useVisualisationStore((s) => s.setColorBy)
  const setShowSettings = useVisualisationStore((s) => s.setShowSettings)
  const showStats = useVisualisationStore((s) => s.showStats)
  const setShowStats = useVisualisationStore((s) => s.setShowStats)
  const setShowColorSettings = useVisualisationStore((s) => s.setShowColorSettings)
  const setShowLegend = useVisualisationStore((s) => s.setShowLegend)
  const setSearchQuery = useVisualisationStore((s) => s.setSearchQuery)
  const selectPatient = useVisualisationStore((s) => s.selectPatient)
  const setFlyToTarget = useVisualisationStore((s) => s.setFlyToTarget)
  const requestScreenshot = useVisualisationStore((s) => s.requestScreenshot)
  const requestExport = useVisualisationStore((s) => s.requestExport)
  const [showExportMenu, setShowExportMenu] = useState(false)
  const isFullscreen = useVisualisationStore((s) => s.isFullscreen)
  const panelsCollapsed = useVisualisationStore((s) => s.panelsCollapsed)
  const setFullscreen = useVisualisationStore((s) => s.setFullscreen)
  const setPanelsCollapsed = useVisualisationStore((s) => s.setPanelsCollapsed)

  // t-SNE perplexity has no control in the open build: it is frozen at the
  // store default (100), the only large-perplexity projection that ships
  // pre-computed. The store field itself must stay — dropping it would omit
  // `?perplexity=` from the request and make the backend re-fit openTSNE.
  const umapNeighbors = useVisualisationStore((s) => s.umapNeighbors)
  const setUmapNeighbors = useVisualisationStore((s) => s.setUmapNeighbors)

  const overlayCohortIds = useVisualisationStore((s) => s.overlayCohortIds)
  const showSupplements = useVisualisationStore((s) => s.showSupplements)
  const setShowSupplements = useVisualisationStore((s) => s.setShowSupplements)

  const filteredPoints = useFilteredPoints()
  const scaleFactor = useScaleFactor()

  const openModelStats = () => {
    const next = !showStats
    setPanelsCollapsed(false)
    setShowSettings(false)
    setShowColorSettings(false)
    setShowStats(next)
  }

  const openViewerSettings = () => {
    const next = !showSettings
    setPanelsCollapsed(false)
    setShowStats(false)
    setShowColorSettings(false)
    setShowSettings(next)
  }

  const [showDropdown, setShowDropdown] = useState(false)
  const searchRef = useRef<HTMLDivElement>(null)

  // Search suggestions — search ALL loaded points, not just filtered ones
  const allPoints = useVisualisationStore((s) => s.points)
  const suggestions = useMemo(() => {
    if (searchQuery.length < 2) return []
    const q = searchQuery.toLowerCase()
    const matches = (value?: string) => value?.toLowerCase().includes(q) ?? false
    return allPoints
      .filter(
        (p) =>
          matches(p.displayId) ||
          matches(p.patientId) ||
          matches(p.primaryDiagnosis) ||
          matches(p.sourceCohort) ||
          matches(p.source) ||
          matches(p.subject) ||
          p.diagnoses?.some(matches) ||
          p.xgbPredictions?.some(matches),
      )
      .slice(0, 8)
  }, [searchQuery, allPoints])

  const handleSearchSelect = useCallback(
    (patientId: string) => {
      const point = allPoints.find((p) => p.patientId === patientId)
      if (point) {
        selectPatient(patientId)
        setFlyToTarget({
          x: point.x * scaleFactor,
          y: point.y * scaleFactor,
          z: point.z * scaleFactor,
        })
      }
      setShowDropdown(false)
    },
    [allPoints, selectPatient, setFlyToTarget, scaleFactor],
  )

  const handleSearchKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && suggestions.length > 0) {
        handleSearchSelect(suggestions[0]!.patientId)
      } else if (e.key === 'Escape') {
        setShowDropdown(false)
      }
    },
    [suggestions, handleSearchSelect],
  )

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setShowDropdown(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  return (
    <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 rounded-lg bg-surface-raised border border-surface-border">
      <div className="flex items-center gap-3 min-w-0">
        {/* Method toggle */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Method</span>
          <div className="flex rounded-lg border border-surface-border overflow-hidden">
            {METHODS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => setMethod(opt.value)}
                disabled={isLoading}
                className={cn(
                  'px-3 py-1.5 text-xs font-medium transition-colors',
                  opt.value === method
                    ? 'bg-brand text-white'
                    : 'bg-surface-raised text-slate-500 hover:text-slate-900 hover:bg-gray-100',
                  isLoading && 'opacity-50 cursor-not-allowed',
                )}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {/* Inline hyperparameter pills.
            The t-SNE "Perp" row was removed for the open build — perplexity is
            pinned to the pre-computed 100. */}
        {method === 'umap' && (
          <>
            <div className="w-px h-5 bg-gray-200" />
            <div className="flex items-center gap-1">
              <span className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold mr-1">Neighbors</span>
              {UMAP_NEIGHBORS.map((n) => (
                <button
                  key={n}
                  onClick={() => setUmapNeighbors(n)}
                  disabled={isLoading}
                  className={cn(
                    'px-2 py-1 text-[11px] font-medium rounded-md transition-colors',
                    n === umapNeighbors
                      ? 'bg-brand text-white'
                      : 'text-slate-500 hover:bg-gray-100',
                    isLoading && 'opacity-50 cursor-not-allowed',
                  )}
                >
                  {n}
                </button>
              ))}
            </div>
          </>
        )}

        {/* The CV "Fold" row that used to sit here is gone in the open build:
            the shipped universe carries no fold membership at all, so the
            buttons and their two count pills had nothing to filter. The
            supplement toggle below was nested inside that row and was
            re-homed. */}

        {/* Supplement toggle — visible only when overlay cohorts are loaded */}
        {overlayCohortIds.length > 0 && (
          <>
            <div className="w-px h-5 bg-gray-200" />
            <button
              onClick={() => setShowSupplements(!showSupplements)}
              className={cn(
                'px-2 py-1.5 text-xs font-medium tabular-nums rounded-lg border transition-colors',
                showSupplements
                  ? 'bg-emerald-500 text-white border-emerald-500'
                  : 'bg-emerald-50 text-emerald-300 border-emerald-100',
              )}
              title="Toggle overlay cohort patients"
            >
              S {filteredPoints.filter((p) => p.isSupplement).length.toLocaleString()}
            </button>
          </>
        )}

        <div className="w-px h-5 bg-gray-200" />

        {/* Grading mode dropdown + legend toggle */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Grading</span>
          <select
            value={colorBy}
            onChange={(e) => setColorBy(e.target.value as ColorByMode)}
            className="px-2 py-1.5 text-xs border border-surface-border rounded-lg bg-surface-raised text-slate-700 focus:outline-none focus:ring-1 focus:ring-brand-500"
          >
            <option value="diagnosis">Diagnosis</option>
            <option value="cohort">Cohort</option>
            <option value="family">Family</option>
            <option value="error_highlight">Error Highlight</option>
          </select>
          <button
            onClick={() => setShowLegend(!showLegend)}
            className={cn(
              'p-1 rounded-md border transition-colors',
              showLegend
                ? 'border-brand-300 bg-brand-50 text-brand-700'
                : 'border-gray-200 hover:bg-gray-50 text-slate-400 hover:text-slate-600',
            )}
            title={GRADING_INFO[colorBy] || 'Toggle legend'}
          >
            <Info className="w-3 h-3" />
          </button>
        </div>

        <div className="w-px h-5 bg-gray-200" />

        {/* Search */}
        <div className="relative" ref={searchRef}>
          <div className="flex items-center gap-1.5">
            <Search className="w-3.5 h-3.5 text-slate-400" />
            <input
              type="text"
              placeholder="Search patient..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value)
                setShowDropdown(e.target.value.length >= 2)
              }}
              onFocus={() => searchQuery.length >= 2 && setShowDropdown(true)}
              onKeyDown={handleSearchKeyDown}
              className="w-36 px-2 py-1 text-xs border border-surface-border rounded-lg bg-surface-raised text-slate-700 focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
            {searchQuery && (
              <button
                onClick={() => {
                  setSearchQuery('')
                  setShowDropdown(false)
                }}
                className="p-0.5 rounded hover:bg-gray-100"
              >
                <X className="w-3 h-3 text-gray-400" />
              </button>
            )}
          </div>

          {/* Dropdown suggestions */}
          {showDropdown && suggestions.length > 0 && (
            <div className="absolute top-full left-0 mt-1 w-64 bg-white border border-gray-200 rounded-lg shadow-lg z-50 max-h-64 overflow-y-auto">
              {suggestions.map((p) => (
                <button
                  key={p.patientId}
                  onClick={() => handleSearchSelect(p.patientId)}
                  className="w-full text-left px-3 py-2 text-xs hover:bg-gray-50 transition-colors border-b border-gray-50 last:border-0"
                >
                  <span className="font-mono font-medium text-gray-900">{pointIdLabel(p)}</span>
                  <span className="text-gray-400 ml-2">{p.primaryDiagnosis}</span>
                  {(p.source || p.topPrediction) && (
                    <span className="block text-[10px] text-gray-400 truncate">
                      {p.source ?? p.sourceCohort}
                      {p.topPrediction ? ` · top: ${p.topPrediction}` : ''}
                    </span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="flex items-center gap-3">
        {isLoading && <Loader2 className="w-3.5 h-3.5 animate-spin text-brand" />}

        <div className="w-px h-5 bg-gray-200" />

        {/* Panel toggles + actions */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setFullscreen(!isFullscreen)}
            className={cn(
              'p-1.5 rounded-lg border transition-colors',
              isFullscreen
                ? 'border-brand-300 bg-brand-50 text-brand-700'
                : 'border-gray-200 hover:bg-gray-50 text-slate-500',
            )}
            title={isFullscreen ? 'Exit fullscreen (Esc)' : 'Fullscreen'}
          >
            {isFullscreen
              ? <Minimize2 className="w-3.5 h-3.5" />
              : <Maximize2 className="w-3.5 h-3.5" />
            }
          </button>
          <button
            onClick={() => setPanelsCollapsed(!panelsCollapsed)}
            className={cn(
              'p-1.5 rounded-lg border transition-colors',
              panelsCollapsed
                ? 'border-brand-300 bg-brand-50 text-brand-700'
                : 'border-gray-200 hover:bg-gray-50 text-slate-500',
            )}
            title={panelsCollapsed ? 'Show panels' : 'Hide panels (Esc)'}
          >
            <PanelLeftClose className="w-3.5 h-3.5" />
          </button>

          <div className="w-px h-5 bg-gray-200" />

          <button
            onClick={openModelStats}
            className={cn(
              'p-1.5 rounded-lg border transition-colors',
              showStats
                ? 'border-brand-300 bg-brand-50 text-brand-700'
                : 'border-gray-200 hover:bg-gray-50 text-slate-500',
            )}
            title="Model Stats"
          >
            <BarChart3 className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={openViewerSettings}
            className={cn(
              'p-1.5 rounded-lg border transition-colors',
              showSettings
                ? 'border-brand-300 bg-brand-50 text-brand-700'
                : 'border-gray-200 hover:bg-gray-50 text-slate-500',
            )}
            title="Viewer Settings"
          >
            <Settings className="w-3.5 h-3.5" />
          </button>
          <div className="relative">
            <button
              onClick={() => setShowExportMenu(!showExportMenu)}
              className="p-1.5 rounded-lg border border-gray-200 hover:bg-gray-50 text-slate-500 transition-colors"
              title="Export"
            >
              <Download className="w-3.5 h-3.5" />
            </button>
            {showExportMenu && (
              <div className="absolute right-0 top-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg py-1 z-50 min-w-[160px]">
                <button
                  onClick={() => { requestScreenshot(); setShowExportMenu(false) }}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-slate-600 hover:bg-gray-50"
                >
                  <Image className="w-3.5 h-3.5" /> Screenshot (PNG)
                </button>
                <button
                  onClick={() => { requestExport('csv'); setShowExportMenu(false) }}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-slate-600 hover:bg-gray-50"
                >
                  <FileSpreadsheet className="w-3.5 h-3.5" /> Data (CSV)
                </button>
                <button
                  onClick={() => { requestExport('json'); setShowExportMenu(false) }}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-slate-600 hover:bg-gray-50"
                >
                  <FileJson className="w-3.5 h-3.5" /> Data (JSON)
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
