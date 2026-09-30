import { useState, useEffect, useCallback } from 'react'
import { AlertCircle, Layers } from 'lucide-react'
import type { ECGData, LeadName, GainSetting, PaperSpeed } from '@/types/ecg'
import { getPixelsPerSecond, getPixelsPerMv } from '@/types/ecg'
import { getTemplateForDiagnosis } from '@/services/diagnosticsService'

// ── Diagnosis color palette (matches DiagnosisCard severity scheme) ──

const DIAGNOSIS_COLORS: Record<string, string> = {
  // Critical (red family)
  stemi: '#EF4444',
  'ventricular tachycardia': '#DC2626',
  'ventricular fibrillation': '#B91C1C',
  '3rd degree av block': '#991B1B',
  // Normal (green family)
  'sinus rhythm': '#22C55E',
  'normal ecg': '#16A34A',
  // Abnormal (amber/blue/purple cycling for visibility)
  'atrial fibrillation': '#F59E0B',
  'atrial flutter': '#D97706',
  'sinus tachycardia': '#8B5CF6',
  'sinus bradycardia': '#6366F1',
  'left bundle branch block': '#0EA5E9',
  'right bundle branch block': '#06B6D4',
  'st deviation': '#EC4899',
  'myocardial infarction': '#F43F5E',
  'left ventricular hypertrophy': '#A855F7',
  'right ventricular hypertrophy': '#7C3AED',
}

const FALLBACK_COLORS = [
  '#3B82F6', '#10B981', '#F97316', '#8B5CF6', '#EC4899',
  '#14B8A6', '#F59E0B', '#6366F1', '#EF4444', '#84CC16',
]

function getColorForDiagnosis(dx: string, index: number): string {
  return DIAGNOSIS_COLORS[dx.toLowerCase()] ?? FALLBACK_COLORS[index % FALLBACK_COLORS.length] ?? '#3B82F6'
}

// ── Template data types ──

interface TemplateData {
  diagnosis: string
  mean_trace?: number[]
  std_trace?: number[]
  n_patients?: number
  vector_length?: number
  gaussian_params?: Record<string, unknown>
}

interface LoadedTemplate {
  diagnosis: string
  color: string
  data: TemplateData
}

// ── Props ──

interface TemplateOverlayProps {
  ecgData: ECGData
  leadName: LeadName
  activeDiagnoses: string[]
  onToggleDiagnosis?: (dx: string) => void
  svgRef: React.RefObject<SVGSVGElement>
  width: number
  height: number
  gain: GainSetting
  paperSpeed: PaperSpeed
  className?: string
}

/**
 * TemplateOverlay — Renders diagnosis waveform templates over the ECG trace.
 *
 * For each active diagnosis, fetches the template from the API and renders:
 * - Mean trace as a colored semi-transparent SVG path
 * - +/-1 sigma band as a filled semi-transparent region
 * - +/-2 sigma band as a lighter filled region
 * - Legend with diagnosis names and colors
 *
 * When template data is not available, shows a placeholder message with
 * instructions for generating templates.
 *
 * H52 / Q-A2-11 status (2026-04-30): wire-ready overlay; ZERO current V3
 * consumers. Pairs with TemplateEditor (see ecg/TemplateEditor.tsx). Deferred
 * to Tier 2 V3 surfacing — natural host is the ECGViewer overlay layer when
 * a TemplatesPage or ClassifierPage sub-tab activates "show templates over
 * ECG". Do NOT delete; the overlay is wire-ready.
 */
export function TemplateOverlay({
  ecgData,
  leadName: _leadName,
  activeDiagnoses,
  onToggleDiagnosis,
  svgRef: _svgRef,
  width,
  height,
  gain,
  paperSpeed,
  className,
}: TemplateOverlayProps) {
  const [templates, setTemplates] = useState<LoadedTemplate[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // ── Fetch template data for each active diagnosis ──

  const fetchTemplates = useCallback(async () => {
    if (activeDiagnoses.length === 0) {
      setTemplates([])
      return
    }

    setLoading(true)
    setError(null)

    const results: LoadedTemplate[] = []

    for (let i = 0; i < activeDiagnoses.length; i++) {
      const dx = activeDiagnoses[i]
      try {
        const data = await getTemplateForDiagnosis(dx!) as unknown as TemplateData
        results.push({
          diagnosis: dx!,
          color: getColorForDiagnosis(dx!, i),
          data,
        })
      } catch {
        // Template not available for this diagnosis — skip silently
        console.warn(`[TemplateOverlay] No template data for "${dx}"`)
      }
    }

    setTemplates(results)
    setLoading(false)

    if (results.length === 0 && activeDiagnoses.length > 0) {
      setError('no_data')
    }
  }, [activeDiagnoses])

  useEffect(() => {
    fetchTemplates()
  }, [fetchTemplates])

  // ── Scale conversion ──

  const pixelsPerSecond = getPixelsPerSecond(paperSpeed)
  const pixelsPerMv = getPixelsPerMv(gain)
  const samplingRate = ecgData.samplingRate || 500

  /**
   * Convert a template trace (array of mV values at sampling rate) to an SVG path.
   *
   * TODO: Implement proper Gaussian-to-SVG-path conversion:
   *   1. The template trace is stored as Gaussian component parameters
   *      (amplitude, center, width per P/Q/R/S/T wave).
   *   2. Reconstruct the continuous waveform by summing Gaussian components:
   *      trace(t) = sum_i( amplitude_i * exp( -(t - center_i)^2 / (2 * width_i^2) ) )
   *   3. Map time axis to x pixels: x = (sampleIndex / samplingRate) * pixelsPerSecond
   *   4. Map voltage axis to y pixels: y = centerY - (mV * pixelsPerMv)
   *   5. Generate SVG path string: M x0,y0 L x1,y1 L x2,y2 ...
   *
   * For now, we render a placeholder path from the mean_trace array if available.
   */
  function traceToSvgPath(trace: number[]): string {
    if (trace.length === 0) return ''

    const centerY = height / 2
    const points = trace.map((mV, i) => {
      const x = (i / samplingRate) * pixelsPerSecond
      const y = centerY - mV * pixelsPerMv
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })

    return `M ${points.join(' L ')}`
  }

  /**
   * Convert a trace pair (upper + lower bounds) to a closed SVG area path.
   *
   * TODO: Implement proper sigma band rendering:
   *   1. Compute upper = mean + N*std, lower = mean - N*std per sample
   *   2. Create a closed polygon: forward along upper, backward along lower
   *   3. Handle clipping to the visible viewport bounds
   *   4. Consider using <clipPath> to constrain to the lead frame area
   */
  function bandToSvgPath(upper: number[], lower: number[]): string {
    if (upper.length === 0 || lower.length === 0) return ''

    const centerY = height / 2
    const forwardPoints = upper.map((mV, i) => {
      const x = (i / samplingRate) * pixelsPerSecond
      const y = centerY - mV * pixelsPerMv
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })

    const backwardPoints = [...lower].reverse().map((mV, i) => {
      const revIndex = lower.length - 1 - i
      const x = (revIndex / samplingRate) * pixelsPerSecond
      const y = centerY - mV * pixelsPerMv
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })

    return `M ${forwardPoints.join(' L ')} L ${backwardPoints.join(' L ')} Z`
  }

  // ── Render: No data placeholder ──

  if (error === 'no_data' || (activeDiagnoses.length > 0 && templates.length === 0 && !loading)) {
    return (
      <div
        className={`absolute inset-0 flex items-center justify-center pointer-events-none ${className ?? ''}`}
      >
        <div className="bg-slate-800/80 text-white rounded-lg px-4 py-3 max-w-md text-center pointer-events-auto">
          <AlertCircle className="w-5 h-5 mx-auto mb-2 text-amber-400" />
          <p className="text-sm font-medium mb-1">Template data not generated yet</p>
          <p className="text-xs text-slate-300">
            Run: <code className="bg-slate-700 px-1.5 py-0.5 rounded text-amber-300">
              python scripts/build_waveform_templates.py
            </code>
          </p>
        </div>
      </div>
    )
  }

  // ── Render: Loading state ──

  if (loading) {
    return (
      <div
        className={`absolute inset-0 flex items-center justify-center pointer-events-none ${className ?? ''}`}
      >
        <div className="bg-slate-800/60 text-white rounded px-3 py-2 text-sm animate-pulse">
          Loading templates...
        </div>
      </div>
    )
  }

  // ── Render: No active diagnoses ──

  if (activeDiagnoses.length === 0) return null

  // ── Render: Template overlays ──

  return (
    <div className={`absolute inset-0 pointer-events-none ${className ?? ''}`}>
      {/* SVG overlay for template traces */}
      <svg
        width={width}
        height={height}
        className="absolute inset-0"
        style={{ overflow: 'visible' }}
      >
        {templates.map((tmpl) => {
          const { data, color, diagnosis } = tmpl
          const meanTrace = data.mean_trace ?? []
          const stdTrace = data.std_trace ?? []

          if (meanTrace.length === 0) return null

          // Compute sigma bands
          // TODO: Pull sigma multipliers from domain config or user preference
          const upper1 = meanTrace.map((v, i) => v + (stdTrace[i] ?? 0))
          const lower1 = meanTrace.map((v, i) => v - (stdTrace[i] ?? 0))
          const upper2 = meanTrace.map((v, i) => v + 2 * (stdTrace[i] ?? 0))
          const lower2 = meanTrace.map((v, i) => v - 2 * (stdTrace[i] ?? 0))

          return (
            <g key={diagnosis} data-diagnosis={diagnosis}>
              {/* +/-2 sigma band (lightest) */}
              {stdTrace.length > 0 && (
                <path
                  d={bandToSvgPath(upper2, lower2)}
                  fill={color}
                  fillOpacity={0.06}
                  stroke="none"
                />
              )}

              {/* +/-1 sigma band */}
              {stdTrace.length > 0 && (
                <path
                  d={bandToSvgPath(upper1, lower1)}
                  fill={color}
                  fillOpacity={0.12}
                  stroke="none"
                />
              )}

              {/* Mean trace */}
              <path
                d={traceToSvgPath(meanTrace)}
                fill="none"
                stroke={color}
                strokeWidth={2}
                strokeOpacity={0.7}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </g>
          )
        })}
      </svg>

      {/* Legend */}
      {templates.length > 0 && (
        <div className="absolute top-2 right-2 pointer-events-auto">
          <div className="bg-white/90 backdrop-blur-sm rounded-lg shadow-sm border border-slate-200 px-3 py-2">
            <div className="flex items-center gap-1.5 mb-1.5">
              <Layers className="w-3.5 h-3.5 text-slate-500" />
              <span className="text-xs font-medium text-slate-600">Templates</span>
            </div>
            <div className="space-y-1">
              {templates.map((tmpl) => (
                <button
                  key={tmpl.diagnosis}
                  className="flex items-center gap-2 text-xs text-slate-700 hover:text-slate-900 w-full text-left"
                  onClick={() => onToggleDiagnosis?.(tmpl.diagnosis)}
                  title={`${tmpl.data.n_patients ?? '?'} patients`}
                >
                  <span
                    className="w-3 h-0.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: tmpl.color }}
                  />
                  <span className="truncate max-w-[140px]">{tmpl.diagnosis}</span>
                  {tmpl.data.n_patients != null && (
                    <span className="text-slate-400 ml-auto">n={tmpl.data.n_patients}</span>
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
