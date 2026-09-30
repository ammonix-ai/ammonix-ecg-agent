import { useVisualisationStore } from '@/stores/visualisationStore'
import { useFilteredPoints } from '@/stores/visualisationSelectors'
import { cn } from '@/design/cn'
import { X } from 'lucide-react'
import { TribeFilterPanel } from './TribeFilterPanel'

function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      role="switch"
      aria-checked={value}
      onClick={() => onChange(!value)}
      className={cn(
        'relative inline-flex h-5 w-9 items-center rounded-full transition-colors flex-shrink-0',
        value ? 'bg-brand-600' : 'bg-gray-300',
      )}
    >
      <span
        className={cn(
          'inline-block h-3.5 w-3.5 rounded-full bg-white transition-transform',
          value ? 'translate-x-4' : 'translate-x-0.5',
        )}
      />
    </button>
  )
}

function Slider({
  value, min, max, step, onChange,
}: {
  value: number; min: number; max: number; step: number; onChange: (v: number) => void
}) {
  return (
    <div className="flex items-center gap-2">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-brand-600"
      />
      <span className="text-[10px] text-gray-400 w-8 text-right tabular-nums">{value}</span>
    </div>
  )
}

interface SettingsPanelProps {
  className?: string
}

export function SettingsPanel({ className }: SettingsPanelProps) {
  const viewerSettings = useVisualisationStore((s) => s.viewerSettings)
  const setShowSettings = useVisualisationStore((s) => s.setShowSettings)
  const updateViewerSettings = useVisualisationStore((s) => s.updateViewerSettings)
  const method = useVisualisationStore((s) => s.method)
  const colorBy = useVisualisationStore((s) => s.colorBy)
  const filteredPoints = useFilteredPoints()
  const { display } = viewerSettings

  // Count unique diagnoses
  const diagnosisCount = new Set(filteredPoints.map((p) => p.primaryDiagnosis)).size

  const updateDisplay = (partial: Record<string, unknown>) => {
    updateViewerSettings({ display: { ...display, ...partial } as typeof display })
  }

  return (
    <div className={cn('bg-white rounded-xl border border-gray-200 shadow-lg', className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
        <h3 className="text-sm font-semibold text-gray-900">Viewer Settings</h3>
        <button
          onClick={() => setShowSettings(false)}
          className="p-1 rounded hover:bg-gray-100 transition-colors"
        >
          <X className="w-4 h-4 text-gray-500" />
        </button>
      </div>

      <div className="p-4 space-y-5 max-h-[calc(100vh-200px)] overflow-y-auto">
        {/* View Stats (compact) */}
        <div className="rounded-lg bg-slate-50 border border-slate-200 p-3">
          <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold mb-2">View</div>
          <div className="grid grid-cols-2 gap-y-1 gap-x-4 text-xs">
            <span className="text-slate-500">Points</span>
            <span className="text-right font-medium text-slate-800 tabular-nums">{filteredPoints.length.toLocaleString()}</span>
            <span className="text-slate-500">Diagnoses</span>
            <span className="text-right font-medium text-slate-800 tabular-nums">{diagnosisCount}</span>
            <span className="text-slate-500">Method</span>
            <span className="text-right font-medium text-slate-800 uppercase">{method}</span>
          </div>
        </div>

        {/* Point Display */}
        <div>
          <div className="text-xs font-medium text-gray-500 mb-2">Point Display</div>
          <div className="space-y-3">
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-gray-600">Point Size</span>
              </div>
              <Slider
                value={display.pointSize}
                min={0.02}
                max={0.2}
                step={0.01}
                onChange={(v) => updateDisplay({ pointSize: v })}
              />
            </div>
          </div>
        </div>

        {/* Toggles */}
        <div>
          <div className="text-xs font-medium text-gray-500 mb-2">Display Options</div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-600">Edge Outlines</span>
              <Toggle
                value={display.showEdgeOutlines}
                onChange={(v) => updateDisplay({ showEdgeOutlines: v })}
              />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-600">Grid</span>
              <Toggle
                value={display.showGrid}
                onChange={(v) => updateDisplay({ showGrid: v })}
              />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-600">Diagnosis Labels</span>
              <Toggle
                value={display.showDiagnosisLabels}
                onChange={(v) => updateDisplay({ showDiagnosisLabels: v })}
              />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-600">Pinned Labels</span>
              <Toggle
                value={display.showPinnedLabels}
                onChange={(v) => updateDisplay({ showPinnedLabels: v })}
              />
            </div>
          </div>
        </div>

        {/* Tribe Visualization */}
        <div>
          <div className="text-xs font-medium text-gray-500 mb-2">Tribe Visualization</div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-600">Show Tribe Boundaries</span>
              <Toggle
                value={display.showTribeBoundaries}
                onChange={(v) => updateDisplay({ showTribeBoundaries: v })}
              />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-600">Show Tribe Centroids</span>
              <Toggle
                value={display.showCentroids}
                onChange={(v) => updateDisplay({ showCentroids: v })}
              />
            </div>
          </div>
          <div className="space-y-3 mt-3">
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-gray-600">Boundary Opacity</span>
              </div>
              <Slider
                value={display.boundaryOpacity}
                min={0.01}
                max={0.3}
                step={0.01}
                onChange={(v) => updateDisplay({ boundaryOpacity: v })}
              />
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-gray-600">Centroid Size</span>
              </div>
              <Slider
                value={display.centroidSize}
                min={0.3}
                max={2.0}
                step={0.1}
                onChange={(v) => updateDisplay({ centroidSize: v })}
              />
            </div>
          </div>
        </div>

        {/* Tribe Filter — inline (no close button, embedded in settings) */}
        <TribeFilterPanel className="border-0 shadow-none rounded-none p-0" />

        {/* Color Settings */}
        <div>
          <div className="text-xs font-medium text-gray-500 mb-2">Color Settings</div>
          <p className="text-xs text-gray-600 mb-2">
            Mode: <span className="font-medium capitalize">{colorBy.replace('_', ' ')}</span>
          </p>
          <button
            onClick={() => useVisualisationStore.getState().setShowColorSettings(
              !useVisualisationStore.getState().showColorSettings
            )}
            className="w-full px-3 py-1.5 text-xs font-medium text-brand-700 bg-brand-50 border border-brand-200 rounded-lg hover:bg-brand-100 transition-colors"
          >
            Customize Colors
          </button>
        </div>

      </div>
    </div>
  )
}
