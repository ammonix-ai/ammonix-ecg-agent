import { useMemo } from 'react'
import { useVisualisationStore } from '@/stores/visualisationStore'
import { useFilteredPoints } from '@/stores/visualisationSelectors'
import type { ColorByMode } from '@/types/projection'
import { cn } from '@/design/cn'
import { X } from 'lucide-react'
import { DiagnosisColorWheel } from './DiagnosisColorWheel'

const COLOR_MODE_LABELS: Record<ColorByMode, string> = {
  cohort: 'Cohort',
  diagnosis: 'Diagnosis (Gold)',
  diagnosis_predicted: 'Diagnosis (Predicted)',
  error_highlight: 'Error Highlight',
  family: 'Diagnosis Family',
}

interface ColorSettingsPanelProps {
  className?: string
}

export function ColorSettingsPanel({ className }: ColorSettingsPanelProps) {
  const colorBy = useVisualisationStore((s) => s.colorBy)
  const setColorBy = useVisualisationStore((s) => s.setColorBy)
  const setShowColorSettings = useVisualisationStore((s) => s.setShowColorSettings)
  const viewerSettings = useVisualisationStore((s) => s.viewerSettings)
  const updateViewerSettings = useVisualisationStore((s) => s.updateViewerSettings)
  const filteredPoints = useFilteredPoints()

  const activeDiagnoses = useMemo(() => {
    const set = new Set<string>()
    for (const p of filteredPoints) {
      set.add(p.primaryDiagnosis)
    }
    return Array.from(set)
  }, [filteredPoints])

  const showWheel = colorBy === 'diagnosis' || colorBy === 'diagnosis_predicted'

  return (
    <div className={cn('bg-white rounded-xl border border-gray-200 shadow-lg', className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
        <h3 className="text-sm font-semibold text-gray-900">Grading Settings</h3>
        <button
          onClick={() => setShowColorSettings(false)}
          className="p-1 rounded hover:bg-gray-100 transition-colors"
        >
          <X className="w-4 h-4 text-gray-500" />
        </button>
      </div>

      <div className="p-4 space-y-3">
        <label className="block text-xs font-medium text-gray-500 mb-1">Color Mode</label>
        <div className="space-y-1">
          {(Object.entries(COLOR_MODE_LABELS) as [ColorByMode, string][]).map(([mode, label]) => (
            <label
              key={mode}
              className={cn(
                'flex items-center gap-2.5 px-3 py-2 rounded-lg cursor-pointer transition-colors',
                colorBy === mode ? 'bg-brand-50 border border-brand-200' : 'hover:bg-gray-50',
              )}
            >
              <input
                type="radio"
                name="colorMode"
                value={mode}
                checked={colorBy === mode}
                onChange={() => setColorBy(mode)}
                className="w-3.5 h-3.5 text-brand-600 focus:ring-brand-500"
              />
              <span className={cn(
                'text-sm',
                colorBy === mode ? 'text-brand-700 font-medium' : 'text-gray-600',
              )}>
                {label}
              </span>
            </label>
          ))}
        </div>

        <p className="text-[10px] text-gray-400 pt-2">
          Choose how points are colored in the 3D visualization.
        </p>

        {/* Color wheel for diagnosis modes */}
        {showWheel && activeDiagnoses.length > 0 && (
          <div className="pt-3 border-t border-gray-100">
            <DiagnosisColorWheel
              colorOverrides={viewerSettings.diagnosisColorOverrides}
              activeDiagnoses={activeDiagnoses}
              onChange={(overrides) => updateViewerSettings({ diagnosisColorOverrides: overrides })}
            />
          </div>
        )}
      </div>
    </div>
  )
}
