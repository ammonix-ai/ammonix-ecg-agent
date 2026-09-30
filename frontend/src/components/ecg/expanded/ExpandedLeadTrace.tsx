import { useRef, useEffect, useMemo, useState } from 'react'
import { ECG_GRID_COLORS, LARGE_BOX_PX } from '@/types/ecg'
import type { LeadName, GainSetting, SampleArray } from '@/types/ecg'

interface ExpandedLeadTraceProps {
  leadName: LeadName
  // C0.1: SampleArray = number[] | Float32Array. Accepts both paths.
  samples: SampleArray
  samplingRate: number
  totalDuration: number
  /** Base pixels/second (before zoom), computed by parent */
  basePixelsPerSecond: number
  gain: GainSetting
  zoom: number
  panOffset: number
  boxSize?: number
  className?: string
}

/**
 * ExpandedLeadTrace — High-res ECG trace with zoom/pan + variable gain
 *
 * Always shows fully drawn trace (no animation).
 * Supports variable gain (5/10/20/50 mm/mV).
 * Handles zoom and pan for finance-chart navigation.
 */
export function ExpandedLeadTrace({
  leadName: _leadName,
  samples, samplingRate, totalDuration,
  basePixelsPerSecond, gain, zoom, panOffset,
  boxSize, className,
}: ExpandedLeadTraceProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [dims, setDims] = useState({ width: 0, height: 0 })

  useEffect(() => {
    if (!ref.current) return
    const update = () => {
      if (ref.current) setDims({ width: ref.current.clientWidth, height: ref.current.clientHeight })
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])

  const boxPx = boxSize ?? LARGE_BOX_PX
  const pixelsPerMv = boxPx * gain / 5
  const pxPerSec = basePixelsPerSecond * zoom
  const visDuration = dims.width > 0 ? dims.width / pxPerSec : 0
  const startTime = panOffset
  const endTime = Math.min(panOffset + visDuration, totalDuration)

  const visSamples = useMemo(() => {
    const s0 = Math.floor(startTime * samplingRate)
    const s1 = Math.ceil(endTime * samplingRate)
    return samples.slice(Math.max(0, s0), Math.min(samples.length, s1))
  }, [samples, samplingRate, startTime, endTime])

  const pathData = useMemo(() => {
    if (!visSamples.length || dims.width === 0 || dims.height === 0) return ''
    const { width, height } = dims
    const duration = endTime - startTime
    if (duration <= 0) return ''

    const centerY = height / 2
    const parts: string[] = []
    visSamples.forEach((sample, i) => {
      const t = startTime + i / samplingRate
      const x = ((t - startTime) / duration) * width
      const y = Math.max(0, Math.min(height, centerY - sample * pixelsPerMv))
      parts.push(i === 0 ? `M ${x.toFixed(2)} ${y.toFixed(2)}` : `L ${x.toFixed(2)} ${y.toFixed(2)}`)
    })
    return parts.join(' ')
  }, [visSamples, dims, startTime, endTime, samplingRate, pixelsPerMv])

  return (
    <div ref={ref} className={`w-full h-full relative ${className ?? ''}`}>
      {dims.width > 0 && dims.height > 0 && (
        <svg width={dims.width} height={dims.height} className="block absolute inset-0" style={{ pointerEvents: 'none' }}>
          <line x1={0} y1={dims.height / 2} x2={dims.width} y2={dims.height / 2}
            stroke="#CBD5E1" strokeWidth={0.5} strokeDasharray="4,4" opacity={0.5} />
          <path d={pathData} fill="none" stroke={ECG_GRID_COLORS.trace}
            strokeWidth={ECG_GRID_COLORS.traceWidth} strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )}
    </div>
  )
}
