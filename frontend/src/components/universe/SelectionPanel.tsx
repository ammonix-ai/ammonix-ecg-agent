import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useVisualisationStore } from '@/stores/visualisationStore'
import { useAnnotationStore } from '@/stores/annotationStore'
import { useSelectedPatientObjects } from '@/stores/visualisationSelectors'
import { cn } from '@/design/cn'
import { X, MousePointerClick, Tag, ExternalLink, ChevronDown, Bookmark } from 'lucide-react'
import type { EnrichedProjectionPoint } from '@/types/projection'
import { NO_RECORDING_NA, analyzeHref, pointIdsOf, recordingIdOf } from './recordingLink'

interface SelectionPanelProps {
  className?: string
}

// =============================================================================
// Toggle sub-component
// =============================================================================

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

// =============================================================================
// Badge sub-component
// =============================================================================

function Badge({ children, color = 'gray' }: { children: React.ReactNode; color?: 'gray' | 'blue' | 'amber' | 'green' | 'red' | 'purple' }) {
  const colors = {
    gray: 'bg-gray-100 text-gray-700',
    blue: 'bg-blue-100 text-blue-700',
    amber: 'bg-amber-100 text-amber-700',
    green: 'bg-green-100 text-green-700',
    red: 'bg-red-100 text-red-700',
    purple: 'bg-purple-100 text-purple-700',
  }
  return (
    <span className={cn('px-1.5 py-0.5 text-[10px] font-medium rounded', colors[color])}>
      {children}
    </span>
  )
}

// =============================================================================
// LabelEditor sub-component
// =============================================================================

function LabelEditor({ patientId, label }: { patientId: string; label?: string }) {
  const setPatientLabel = useVisualisationStore((s) => s.setPatientLabel)
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState(label || '')

  const handleSave = () => {
    setPatientLabel(patientId, value)
    setEditing(false)
  }

  if (editing) {
    return (
      <div className="flex items-center gap-1">
        <Tag className="w-3 h-3 text-gray-400" />
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onBlur={handleSave}
          onKeyDown={(e) => e.key === 'Enter' && handleSave()}
          className="flex-1 px-1.5 py-0.5 text-xs border border-brand-300 rounded focus:outline-none focus:ring-1 focus:ring-brand-500"
          autoFocus
          placeholder="Add label..."
        />
      </div>
    )
  }

  return (
    <button
      onClick={() => { setValue(label || ''); setEditing(true) }}
      className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 transition-colors"
    >
      <Tag className="w-3 h-3" />
      {label || 'Add label...'}
    </button>
  )
}

// =============================================================================
// DetailRow helper
// =============================================================================

function DetailRow({ label, value, stacked = false }: { label: string; value: React.ReactNode; stacked?: boolean }) {
  if (stacked) {
    return (
      <div className="space-y-1">
        <span className="text-xs text-gray-500">{label}</span>
        <div>{value}</div>
      </div>
    )
  }
  return (
    <div className="flex items-start justify-between gap-2">
      <span className="text-xs text-gray-500 flex-shrink-0">{label}</span>
      <div className="text-left">{value}</div>
    </div>
  )
}

// =============================================================================
// PatientDetailContent -- reusable detail fields for a patient
// =============================================================================

/**
 * A record's ids. Both are shown when it has two — the PhysioNet recording id
 * first, since that is what a user searches and what names the shipped trace
 * file. Rows with no recording id show their display id alone.
 */
function PatientIdValue({ patient }: { patient: EnrichedProjectionPoint }) {
  const { recordingId, displayId } = pointIdsOf(patient)

  if (!recordingId || recordingId === displayId) {
    return <span className="font-mono text-xs">{displayId}</span>
  }

  return (
    <span
      className="font-mono text-xs"
      title={`Same record: PhysioNet recording ${recordingId}, universe display id ${displayId}`}
    >
      {recordingId}
      <span className="text-gray-400"> · </span>
      <span className="text-gray-500">{displayId}</span>
    </span>
  )
}

function PatientDetailContent({ patient }: { patient: EnrichedProjectionPoint }) {
  const patientLabels = useVisualisationStore((s) => s.patientLabels)

  return (
    <div className="space-y-3">
      <LabelEditor patientId={patient.patientId} label={patientLabels.get(patient.patientId)} />

      <div className="space-y-2">
        <DetailRow label="Patient ID" value={<PatientIdValue patient={patient} />} />

        {/* `!= null`, not `!== undefined`: the universe publishes no
            demographics and sends JSON null, which used to render "?, ?". */}
        {(patient.sex != null || patient.age != null) && (
          <DetailRow
            label="Sex / Age"
            value={<span className="text-xs text-gray-700">{patient.sex || '?'}, {patient.age ?? '?'}</span>}
          />
        )}
        {patient.xgbCorrect !== undefined && (
          <DetailRow
            label="Label match"
            value={
              patient.xgbCorrect === null
                ? <Badge>Unknown</Badge>
                : patient.xgbCorrect
                  ? <Badge color="green">Correct</Badge>
                  : <Badge color="red">Incorrect</Badge>
            }
          />
        )}
        {patient.xgbPredictions && patient.xgbPredictions.length > 0 && (
          <DetailRow
            label="Predictions"
            stacked
            value={
              <div className="flex flex-wrap gap-1">
                {patient.xgbPredictions.map((p) => (
                  <Badge key={p}>{p}</Badge>
                ))}
              </div>
            }
          />
        )}
        {patient.diagnoses && (
          <DetailRow
            label="Diagnoses"
            stacked
            value={
              <div className="flex flex-wrap gap-1">
                {patient.diagnoses.map((dx) => (
                  <Badge key={dx}>{dx}</Badge>
                ))}
                {patient.diagnoses.length === 0 && (
                  <span className="text-xs text-gray-400 italic">None</span>
                )}
              </div>
            }
          />
        )}
        {(patient.source || patient.sourceCohort) && (
          <DetailRow
            label="Source"
            value={<span className="text-xs text-gray-700">{patient.source ?? patient.sourceCohort}</span>}
          />
        )}
        {patient.subject && (
          <DetailRow
            label="Subject"
            value={<span className="font-mono text-xs text-gray-700">{patient.subject}</span>}
          />
        )}
        {patient.topPrediction && (
          <DetailRow
            label="Top score"
            value={
              <span className="text-xs text-gray-700">
                {patient.topPrediction}
                {typeof patient.maxScore === 'number' ? ` (${patient.maxScore.toFixed(3)})` : ''}
              </span>
            }
          />
        )}
      </div>

      {patient.tribes && (
        <div>
          <div className="text-xs font-medium text-gray-500 mb-1">Tribes</div>
          <div className="flex flex-wrap gap-1">
            {patient.tribes.map((t) => (
              <Badge key={t}>{t}</Badge>
            ))}
            {patient.multiTribe && <Badge color="purple">Multi-Tribe</Badge>}
            {patient.tribes.length === 0 && (
              <span className="text-xs text-gray-400 italic">None</span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// =============================================================================
// SinglePatientView
// =============================================================================

/** Send a universe point to the Analyze page, where its trace can be read. */
function OpenInAnalyzeButton({
  patient,
  compact = false,
}: {
  patient: EnrichedProjectionPoint;
  compact?: boolean;
}) {
  const navigate = useNavigate();
  const recordingId = recordingIdOf(patient);
  const disabled = recordingId === null;

  return (
    <>
      <button
        onClick={() => {
          if (recordingId) navigate(analyzeHref(recordingId));
        }}
        disabled={disabled}
        title={disabled ? NO_RECORDING_NA : undefined}
        className={`${compact ? 'flex-1 px-3 py-1.5 text-xs' : 'w-full px-3 py-2 text-sm'} flex items-center justify-center gap-2 font-medium text-white bg-brand-600 rounded-lg transition-colors ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-brand-700'}`}
      >
        Open in Analyze
        <ExternalLink className={compact ? 'w-3 h-3' : 'w-3.5 h-3.5'} />
      </button>
      {disabled && !compact && (
        <p className="text-[10px] text-gray-400 mt-1 leading-snug">{NO_RECORDING_NA}</p>
      )}
    </>
  );
}

function SinglePatientView({ patient }: { patient: EnrichedProjectionPoint }) {
  const clearSelection = useVisualisationStore((s) => s.clearSelection)
  const isPinned = useAnnotationStore((s) => s.isPinned(patient.patientId))
  const togglePinned = useAnnotationStore((s) => s.togglePin)
  const showNeighbors = useVisualisationStore((s) => s.showNeighbors)
  const neighborCount = useVisualisationStore((s) => s.neighborCount)
  const neighborsRespectFilter = useVisualisationStore((s) => s.neighborsRespectFilter)
  const setShowNeighbors = useVisualisationStore((s) => s.setShowNeighbors)
  const setNeighborCount = useVisualisationStore((s) => s.setNeighborCount)
  const setNeighborsRespectFilter = useVisualisationStore((s) => s.setNeighborsRespectFilter)

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
        <h3 className="text-sm font-semibold text-gray-900">Selected Patient</h3>
        <div className="flex items-center gap-1">
          <button
            onClick={() => togglePinned(patient.patientId)}
            className={`p-1 rounded transition-colors ${isPinned ? 'text-blue-500' : 'text-gray-400 hover:text-gray-600'}`}
            title={isPinned ? 'Unpin from hotbar' : 'Pin to hotbar'}
          >
            <Bookmark className="w-4 h-4" fill={isPinned ? 'currentColor' : 'none'} />
          </button>
          <button onClick={clearSelection} className="p-1 rounded hover:bg-gray-100 transition-colors">
            <X className="w-4 h-4 text-gray-500" />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <PatientDetailContent patient={patient} />

        {/* Nearest Neighbors */}
        <div className="border-t border-gray-100 pt-3">
          <div className="text-xs font-medium text-gray-500 mb-2">Nearest Neighbors</div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-600">Show</span>
            <Toggle value={showNeighbors} onChange={setShowNeighbors} />
          </div>
          {showNeighbors && (
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-gray-600">Count</span>
                <span className="text-[10px] text-gray-400">{neighborCount}</span>
              </div>
              <input
                type="range"
                min={1}
                max={50}
                step={1}
                value={neighborCount}
                onChange={(e) => setNeighborCount(Number(e.target.value))}
                className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-brand-600"
              />
              <div className="flex items-center justify-between mt-2">
                <span className="text-xs text-gray-600">Respect filters</span>
                <Toggle value={neighborsRespectFilter} onChange={setNeighborsRespectFilter} />
              </div>
              <p className="text-[10px] text-gray-400 mt-1">
                {neighborsRespectFilter
                  ? `Showing ${neighborCount} nearest among filtered points`
                  : `Showing ${neighborCount} nearest from all points`}
              </p>
            </div>
          )}
        </div>

        <OpenInAnalyzeButton patient={patient} />
      </div>
    </div>
  )
}

function MultiPatientExpanded({ patient }: { patient: EnrichedProjectionPoint }) {
  const isPinned = useAnnotationStore((s) => s.isPinned(patient.patientId))
  const togglePinned = useAnnotationStore((s) => s.togglePin)

  return (
    <div className="px-2.5 pb-3">
      <PatientDetailContent patient={patient} />
      <div className="flex items-center gap-2 mt-3">
        <button
          onClick={() => togglePinned(patient.patientId)}
          className={`p-1.5 rounded-lg border transition-colors ${isPinned ? 'border-blue-300 bg-blue-50 text-blue-500' : 'border-gray-200 text-gray-400 hover:text-gray-600'}`}
          title={isPinned ? 'Unpin from hotbar' : 'Pin to hotbar'}
        >
          <Bookmark className="w-3.5 h-3.5" fill={isPinned ? 'currentColor' : 'none'} />
        </button>
        <OpenInAnalyzeButton patient={patient} compact />
      </div>
    </div>
  )
}

// =============================================================================
// MultiPatientView
// =============================================================================

function MultiPatientView({ patients }: { patients: EnrichedProjectionPoint[] }) {
  const clearSelection = useVisualisationStore((s) => s.clearSelection)
  const [expandedPatientId, setExpandedPatientId] = useState<string | null>(null)

  /** How many of the selected records came from each source cohort. */
  const cohortCounts = useMemo(() => {
    const counts = new Map<string, number>()
    for (const p of patients) {
      const cohort = p.cohort || 'unknown'
      counts.set(cohort, (counts.get(cohort) ?? 0) + 1)
    }
    return Array.from(counts.entries()).sort((a, b) => b[1] - a[1])
  }, [patients])

  const handleCardClick = (patientId: string) => {
    setExpandedPatientId((prev) => (prev === patientId ? null : patientId))
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
        <h3 className="text-sm font-semibold text-gray-900">
          Selected Patients ({patients.length})
        </h3>
        <button onClick={clearSelection} className="p-1 rounded hover:bg-gray-100 transition-colors">
          <X className="w-4 h-4 text-gray-500" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <div className="flex flex-wrap items-center gap-1.5 text-xs mb-3">
          {cohortCounts.map(([cohort, count]) => (
            <Badge key={cohort}>{count} {cohort}</Badge>
          ))}
        </div>

        <div className="space-y-1.5 max-h-[500px] overflow-y-auto">
          {patients.map((patient) => {
            const isExpanded = expandedPatientId === patient.patientId
            return (
              <div key={patient.patientId} className="border rounded-lg overflow-hidden">
                {/* Collapsed header -- always visible */}
                <button
                  onClick={() => handleCardClick(patient.patientId)}
                  className={cn(
                    'w-full flex items-center justify-between p-2.5 text-left transition-colors',
                    isExpanded ? 'bg-gray-50 border-b border-gray-100' : 'hover:bg-gray-50',
                  )}
                >
                  <span className="text-xs font-medium text-gray-900 font-mono truncate">
                    <PatientIdValue patient={patient} />
                  </span>
                  <div className="flex items-center gap-2">
                    <ChevronDown
                      className={cn(
                        'w-3.5 h-3.5 text-gray-400 transition-transform',
                        isExpanded && 'rotate-180',
                      )}
                    />
                  </div>
                </button>

                {/* Expanded detail */}
                {isExpanded && (
                  <MultiPatientExpanded patient={patient} />
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// Main SelectionPanel
// =============================================================================

export function SelectionPanel({ className }: SelectionPanelProps) {
  const selectedPatientObjects = useSelectedPatientObjects()

  if (selectedPatientObjects.length === 0) {
    return (
      <div className={cn('flex flex-col items-center justify-center h-full text-gray-400 px-6', className)}>
        <MousePointerClick className="w-12 h-12 mb-4 text-gray-300" />
        <p className="font-medium text-gray-500">No Selection</p>
        <p className="text-sm mt-1 text-center">Click a point to select</p>
        <p className="text-xs mt-0.5 text-center">Ctrl+Click for multi-select</p>
      </div>
    )
  }

  if (selectedPatientObjects.length === 1) {
    return <SinglePatientView patient={selectedPatientObjects[0]!} />
  }

  return <MultiPatientView patients={selectedPatientObjects} />
}
