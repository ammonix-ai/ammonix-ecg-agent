import { useMemo } from 'react'

interface ExpandedXAxisProps {
  startTime: number
  endTime: number
  pixelsPerSecond: number
  containerWidth: number
  height?: number
  className?: string
}

/**
 * ExpandedXAxis — Time axis with progressive detail on zoom
 *
 * Labels always at round time values. Adapts label density to zoom level.
 */
export function ExpandedXAxis({
  startTime, endTime, pixelsPerSecond, containerWidth,
  height = 28, className,
}: ExpandedXAxisProps) {
  const duration = endTime - startTime

  const ticks = useMemo(() => {
    if (duration <= 0 || containerWidth <= 0) return []

    // Pick tick interval aiming for ~8-12 labels
    const candidates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    const targetCount = 10
    const rawInterval = duration / targetCount
    const interval = candidates.find(c => c >= rawInterval) ?? candidates[candidates.length - 1]

    const result: { time: number; xPx: number }[] = []
    const firstIdx = Math.ceil(startTime / interval!)
    const lastIdx = Math.floor(endTime / interval!)

    for (let i = firstIdx; i <= lastIdx; i++) {
      const time = Math.round(i * interval! * 10000) / 10000
      const xPx = (time - startTime) * pixelsPerSecond
      if (xPx >= 4 && xPx <= containerWidth - 4) {
        result.push({ time, xPx })
      }
    }
    return result
  }, [startTime, endTime, pixelsPerSecond, containerWidth, duration])

  const formatTime = (s: number) => {
    if (Math.abs(s) < 0.0001) return '0s'
    if (s < 0.1) return `${(s * 1000).toFixed(0)}ms`
    if (s < 1) { const ms = s * 1000; return Math.abs(ms - Math.round(ms)) < 0.1 ? `${Math.round(ms)}ms` : `${s.toFixed(2)}s` }
    return s % 1 !== 0 ? `${(Math.round(s * 10) / 10).toFixed(1)}s` : `${s.toFixed(0)}s`
  }

  return (
    <div className={`relative ${className ?? ''}`} style={{ height }}>
      <div className="absolute top-0 left-0 right-0 h-px bg-gray-300" />

      {ticks.map(({ time, xPx }) => (
        <div key={time} className="absolute top-0 flex flex-col items-center" style={{ left: xPx, transform: 'translateX(-50%)' }}>
          <div className="w-px h-2 bg-gray-400" />
          <span className="text-[10px] text-gray-600 mt-0.5 font-medium">{formatTime(time)}</span>
        </div>
      ))}

      {/* Edge labels */}
      <div className="absolute top-0 left-0 flex flex-col items-start">
        <div className="w-px h-2 bg-gray-500" />
        <span className="text-[10px] text-gray-700 mt-0.5 font-semibold">{formatTime(startTime)}</span>
      </div>
      <div className="absolute top-0 right-0 flex flex-col items-end">
        <div className="w-px h-2 bg-gray-500" />
        <span className="text-[10px] text-gray-700 mt-0.5 font-semibold">{formatTime(endTime)}</span>
      </div>

      <div className="absolute bottom-1 right-0 text-[9px] text-gray-400 font-medium">s</div>
    </div>
  )
}
