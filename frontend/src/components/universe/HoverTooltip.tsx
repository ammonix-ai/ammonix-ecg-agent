import { Html } from '@react-three/drei'
import type { EnrichedProjectionPoint } from '@/types/projection'
import { ShieldCheck, ShieldAlert } from 'lucide-react'
import { pointIdLabel } from './recordingLink'

interface HoverTooltipProps {
  point: EnrichedProjectionPoint | undefined
}

export function HoverTooltip({ point }: HoverTooltipProps) {
  if (!point) return null

  return (
    <Html position={[point.x, point.y + 0.5, point.z]} center zIndexRange={[100, 0]}>
      <div className="bg-surface-0/95 backdrop-blur-md border border-border shadow-xl rounded-lg p-3 text-sm min-w-[200px] pointer-events-none">
        {/* Record ids + source cohort. Both ids show when the record has two
            (HR07445 · SS-00936) — they name the same trace. */}
        <div className="flex items-center justify-between gap-2 mb-1 border-b border-border/50 pb-1">
          <span className="font-mono text-brand font-bold truncate">{pointIdLabel(point)}</span>
          {point.cohort && (
            <span className="text-[10px] px-1.5 py-0.5 rounded font-mono font-medium shrink-0 bg-surface-raised text-text-secondary">
              {point.cohort}
            </span>
          )}
        </div>

        <div className="flex flex-col gap-1 mt-2 text-text-secondary">
          <span className="font-semibold text-text-primary capitalize">{point.primaryDiagnosis}</span>

          {/* Pipeline correctness */}
          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs">Label match:</span>
            {point.pipelineCorrect === true ? (
              <span className="flex items-center text-status-normal font-medium text-xs"><ShieldCheck className="w-3 h-3 mr-1" /> Correct</span>
            ) : point.pipelineCorrect === false ? (
              <span className="flex items-center text-status-critical font-medium text-xs"><ShieldAlert className="w-3 h-3 mr-1" /> Error</span>
            ) : (
              <span className="text-text-muted text-xs">—</span>
            )}
          </div>

          {/* Tribes */}
          {point.tribes && point.tribes.length > 0 && (
            <div className="mt-2 pt-2 border-t border-border/50">
              <span className="text-[10px] text-text-muted uppercase tracking-wider block mb-1">Tribes</span>
              <div className="flex flex-wrap gap-1">
                {point.tribes.map(t => (
                  <span key={t} className="text-[10px] bg-brand/10 border border-brand/20 text-brand px-1.5 py-0.5 rounded">{t}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </Html>
  )
}
