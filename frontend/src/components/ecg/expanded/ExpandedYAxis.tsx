import { useMemo } from 'react'
import { LARGE_BOX_PX } from '@/types/ecg'
import type { GainSetting } from '@/types/ecg'

interface ExpandedYAxisProps {
  gain: GainSetting
  boxSize?: number
  width?: number
  containerHeight: number
  className?: string
}

/**
 * ExpandedYAxis — mV labels at major-gridline intervals
 *
 * Dynamic tick interval adapts to gain setting.
 * Labels always land on major gridlines.
 */
export function ExpandedYAxis({
  gain, boxSize, width = 48, containerHeight, className,
}: ExpandedYAxisProps) {
  const boxPx = boxSize ?? LARGE_BOX_PX
  const pixelsPerMv = boxPx * gain / 5
  const centerPx = containerHeight / 2

  const { tickInterval, labelRange } = useMemo(() => {
    const mvPerSquare = 5 / gain
    const multipliers = [1, 2, 3, 5, 10]
    const candidates = multipliers.map(m => mvPerSquare * m)
    const tick = candidates.find(c => c * pixelsPerMv >= 45) ?? candidates[candidates.length - 1]
    const range = (containerHeight / 2) / pixelsPerMv
    return { tickInterval: tick, labelRange: range }
  }, [pixelsPerMv, containerHeight, gain])

  const ticks = useMemo(() => {
    const result: number[] = []
    for (let v = Math.floor(labelRange / tickInterval!) * tickInterval!; v >= -labelRange; v -= tickInterval!) {
      result.push(Math.round(v * 1000) / 1000)
    }
    return result
  }, [labelRange, tickInterval])

  return (
    <div className={`relative flex flex-col overflow-hidden ${className ?? ''}`} style={{ width }}>
      {ticks.map((tick) => {
        const topPx = centerPx - tick * pixelsPerMv
        const isZero = Math.abs(tick) < 0.001
        if (topPx < -8 || topPx > containerHeight + 8) return null

        return (
          <div key={tick} className="absolute right-0 flex items-center w-full"
            style={{ top: topPx, transform: 'translateY(-50%)' }}>
            <span className={`flex-1 text-right pr-1 text-[10px] ${isZero ? 'font-semibold text-gray-700' : 'text-gray-500'}`}>
              {tick > 0 ? `+${tick.toFixed(2)}` : tick.toFixed(2)}
            </span>
            <div className={`h-px flex-shrink-0 ${isZero ? 'w-3 bg-gray-600' : 'w-2 bg-gray-300'}`} />
          </div>
        )
      })}
      <div className="absolute top-1 right-0 left-0 text-right pr-1 text-[9px] text-gray-400 font-medium">mV</div>
    </div>
  )
}
