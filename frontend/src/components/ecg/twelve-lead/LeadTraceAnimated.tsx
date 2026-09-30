import { useRef, useEffect, useMemo, useState } from 'react'
import { ECG_GRID_COLORS, LARGE_BOX_PX } from '@/types/ecg'
import type { LeadName, SampleArray } from '@/types/ecg'

interface LeadTraceAnimatedProps {
  leadName: LeadName
  // C0.1: SampleArray = number[] | Float32Array. Accepts both paths.
  samples: SampleArray
  samplingRate: number
  startTime: number
  endTime: number
  /** 0-1: fraction of trace to draw (for sweep animation) */
  animationProgress?: number
  /**
   * Major-gridline size of the paper this trace sits on. The amplitude
   * mapping is 1 bold box = 0.5mV, so the trace must use the same box size
   * as the ECGGridPaper behind it or the mV labels lie.
   */
  boxPx?: number
  className?: string
}

/**
 * LeadTraceAnimated — SVG trace with progressive draw (sweep line)
 *
 * Maps ECG samples to SVG coordinates:
 * - X: time mapped linearly across container width
 * - Y: amplitude in mV, 0mV at center, 1 bold box = 0.5mV (10mm/mV)
 *
 * animationProgress=1 draws the full trace. Values < 1 draw a
 * partial trace for the animated sweep effect.
 */
export function LeadTraceAnimated({
  leadName: _leadName,
  samples,
  samplingRate,
  startTime,
  endTime,
  animationProgress = 1,
  boxPx = LARGE_BOX_PX,
  className,
}: LeadTraceAnimatedProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })

  useEffect(() => {
    if (!containerRef.current) return
    const updateSize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        })
      }
    }
    updateSize()
    const ro = new ResizeObserver(updateSize)
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [])

  const windowSamples = useMemo(() => {
    const s0 = Math.floor(startTime * samplingRate)
    const s1 = Math.ceil(endTime * samplingRate)
    return samples.slice(Math.max(0, s0), Math.min(samples.length, s1))
  }, [samples, samplingRate, startTime, endTime])

  const pathData = useMemo(() => {
    if (!windowSamples.length || dimensions.width === 0 || dimensions.height === 0) return ''

    const { width, height } = dimensions
    const duration = endTime - startTime
    if (duration <= 0) return ''

    const count = Math.floor(windowSamples.length * animationProgress)
    const toDraw = windowSamples.slice(0, count)
    if (!toDraw.length) return ''

    const centerY = height / 2
    const pxPerMv = 2 * boxPx // 10mm/mV = 2 bold boxes per mV

    const parts: string[] = []
    toDraw.forEach((sample, i) => {
      const x = animationProgress >= 1 && i === toDraw.length - 1
        ? width
        : ((startTime + i / samplingRate - startTime) / duration) * width
      const y = Math.max(-height, Math.min(2 * height, centerY - sample * pxPerMv))

      parts.push(i === 0
        ? `M ${x.toFixed(2)} ${y.toFixed(2)}`
        : `L ${x.toFixed(2)} ${y.toFixed(2)}`)
    })

    return parts.join(' ')
  }, [windowSamples, dimensions, startTime, endTime, samplingRate, animationProgress, boxPx])

  return (
    <div ref={containerRef} className={`w-full h-full relative ${className ?? ''}`}>
      {dimensions.width > 0 && dimensions.height > 0 && (
        <svg
          width={dimensions.width}
          height={dimensions.height}
          className="block absolute inset-0"
          style={{ pointerEvents: 'none' }}
        >
          <path
            d={pathData}
            fill="none"
            stroke={ECG_GRID_COLORS.trace}
            strokeWidth={ECG_GRID_COLORS.traceWidth}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      )}
    </div>
  )
}

/** Convenience wrapper for static (non-animated) traces */
export function LeadTraceStatic(
  props: Omit<LeadTraceAnimatedProps, 'animationProgress'>
) {
  return <LeadTraceAnimated {...props} animationProgress={1} />
}
