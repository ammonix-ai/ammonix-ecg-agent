import { useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ReferenceLine,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  CartesianGrid,
} from 'recharts'
import { X, Maximize2 } from 'lucide-react'

// =============================================================================
// Types
//
// Declared here rather than imported from a generated OpenAPI schema: the open
// build ships one ROC producer, GET /api/admin/cv/source-safe-roc, and these
// are the fields it returns.
// =============================================================================

/** Single point on an ROC curve. */
export interface ROCPoint {
  fpr: number
  tpr: number
}

/** One diagnosis's ROC curve plus its operating point. */
export interface DiagnosisROCCurve {
  diagnosis: string
  points: ROCPoint[]
  threshold: number
  auroc?: number | null
  operatingFpr?: number | null
  operatingTpr?: number | null
}

// =============================================================================
// Compact chart (inline in diagnosis row)
// =============================================================================

function CompactROCChart({ curve, onClickExpand }: { curve: DiagnosisROCCurve; onClickExpand?: () => void }) {
  return (
    <div
      className="relative mt-2 cursor-pointer group"
      onClick={onClickExpand}
      title="Click to expand"
    >
      <div className="h-[150px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={curve.points}
            margin={{ top: 8, right: 8, bottom: 4, left: -10 }}
          >
            <XAxis
              dataKey="fpr"
              type="number"
              domain={[0, 1]}
              ticks={[0, 0.5, 1]}
              tick={{ fontSize: 9, fill: '#94a3b8' }}
              axisLine={{ stroke: '#e2e8f0' }}
              tickLine={false}
            />
            <YAxis
              dataKey="tpr"
              type="number"
              domain={[0, 1]}
              ticks={[0, 0.5, 1]}
              tick={{ fontSize: 9, fill: '#94a3b8' }}
              axisLine={{ stroke: '#e2e8f0' }}
              tickLine={false}
            />
            <ReferenceLine
              segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
              stroke="#cbd5e1"
              strokeDasharray="4 3"
              strokeWidth={1}
            />
            <Line
              dataKey="tpr"
              stroke="#3b82f6"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            {curve.operatingFpr != null && curve.operatingTpr != null && (
              <ReferenceDot
                x={curve.operatingFpr}
                y={curve.operatingTpr}
                r={4}
                fill="#ef4444"
                stroke="#ffffff"
                strokeWidth={1.5}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="absolute top-2 right-2 text-xs font-mono text-slate-500 bg-white/80 px-1 rounded">
        AUC {curve.auroc != null ? curve.auroc.toFixed(3) : 'N/A'}
      </div>
      <div className="absolute bottom-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <span className="inline-flex items-center gap-1 text-[10px] text-blue-500 bg-white/90 px-1.5 py-0.5 rounded">
          <Maximize2 className="w-2.5 h-2.5" />
          Expand
        </span>
      </div>
    </div>
  )
}

// =============================================================================
// Full-size modal chart
// =============================================================================

function FullROCChart({ curve, onClose }: { curve: DiagnosisROCCurve; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className="bg-white rounded-xl shadow-2xl w-[520px] max-w-[95vw]" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-200">
          <div>
            <h3 className="text-sm font-semibold text-slate-800">{curve.diagnosis}</h3>
            <span className="text-xs text-slate-400 font-mono">
              AUROC: {curve.auroc != null ? curve.auroc.toFixed(4) : 'N/A'}
              {' \u00b7 '}Threshold: {curve.threshold.toFixed(3)}
            </span>
          </div>
          <button onClick={onClose} className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="px-5 py-4">
          <div className="h-[340px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={curve.points}
                margin={{ top: 12, right: 20, bottom: 36, left: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis
                  dataKey="fpr"
                  type="number"
                  domain={[0, 1]}
                  ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
                  tick={{ fontSize: 11, fill: '#64748b' }}
                  axisLine={{ stroke: '#cbd5e1' }}
                  tickLine={{ stroke: '#cbd5e1' }}
                  label={{
                    value: 'False Positive Rate',
                    position: 'insideBottom',
                    offset: -20,
                    style: { fontSize: 12, fill: '#64748b' },
                  }}
                />
                <YAxis
                  dataKey="tpr"
                  type="number"
                  domain={[0, 1]}
                  ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
                  tick={{ fontSize: 11, fill: '#64748b' }}
                  axisLine={{ stroke: '#cbd5e1' }}
                  tickLine={{ stroke: '#cbd5e1' }}
                  label={{
                    value: 'True Positive Rate',
                    angle: -90,
                    position: 'insideLeft',
                    offset: 0,
                    style: { fontSize: 12, fill: '#64748b', textAnchor: 'middle' },
                  }}
                />
                <ReferenceLine
                  segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                  stroke="#94a3b8"
                  strokeDasharray="6 4"
                  strokeWidth={1}
                />
                <Tooltip
                  formatter={(value: unknown) => [(value as number).toFixed(3)]}
                  labelFormatter={(fpr: unknown) => `FPR: ${Number(fpr).toFixed(3)}`}
                  contentStyle={{
                    fontSize: 12,
                    borderRadius: 8,
                    border: '1px solid #e2e8f0',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                  }}
                />
                <Line
                  dataKey="tpr"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 3, fill: '#3b82f6', strokeWidth: 0 }}
                  isAnimationActive={false}
                  name="TPR"
                />
                {curve.operatingFpr != null && curve.operatingTpr != null && (
                  <ReferenceDot
                    x={curve.operatingFpr}
                    y={curve.operatingTpr}
                    r={5}
                    fill="#ef4444"
                    stroke="#ffffff"
                    strokeWidth={2}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        {curve.operatingFpr != null && curve.operatingTpr != null && (
          <div className="px-5 pb-4 flex items-center gap-4 text-xs text-slate-500">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500 inline-block" />
              <span>Operating point (t={curve.threshold.toFixed(3)})</span>
            </div>
            <span className="font-mono">
              FPR={curve.operatingFpr.toFixed(3)} &middot; TPR={curve.operatingTpr.toFixed(3)}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

// =============================================================================
// Exports
// =============================================================================

export function ROCCurveChart({ curve, compact = false, onClickExpand }: {
  curve: DiagnosisROCCurve
  compact?: boolean
  onClickExpand?: () => void
}) {
  const [modalOpen, setModalOpen] = useState(false)

  if (curve.points.length < 3) {
    return <div className="mt-2 text-xs text-slate-400 italic">Insufficient data for ROC curve</div>
  }

  if (compact) {
    return (
      <>
        <CompactROCChart curve={curve} onClickExpand={onClickExpand ?? (() => setModalOpen(true))} />
        {modalOpen && <FullROCChart curve={curve} onClose={() => setModalOpen(false)} />}
      </>
    )
  }

  return <FullROCChart curve={curve} onClose={() => {}} />
}
