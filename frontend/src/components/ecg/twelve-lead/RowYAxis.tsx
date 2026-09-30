import { LARGE_BOX_PX } from '@/types/ecg'

interface RowYAxisProps {
  yMin?: number
  yMax?: number
  width?: number
  /** Row height in px — enables grid-aligned positioning */
  rowHeightPx?: number
  /** Major-gridline size of the paper the labels annotate (1 bold box = 0.5mV) */
  boxPx?: number
  className?: string
}

/**
 * RowYAxis — mV labels at bold-gridline intervals
 *
 * 0mV at center, ticks every 0.5mV (1 bold box).
 */
export function RowYAxis({
  yMin = -1.5,
  yMax = 1.5,
  width = 36,
  rowHeightPx,
  boxPx = LARGE_BOX_PX,
  className,
}: RowYAxisProps) {
  const interval = 0.5
  const ticks: number[] = []
  for (let v = yMax; v >= yMin; v -= interval) {
    ticks.push(Math.round(v * 10) / 10)
  }

  const centerPx = rowHeightPx ? rowHeightPx / 2 : undefined

  return (
    <div className={`relative h-full ${className ?? ''}`} style={{ width }}>
      {ticks.map((tick) => {
        const isZero = tick === 0

        if (rowHeightPx && centerPx !== undefined) {
          const topPx = centerPx - (tick / 0.5) * boxPx
          const overflow = boxPx / 2
          if (topPx < -overflow || topPx > rowHeightPx + overflow) return null

          return (
            <div
              key={tick}
              className="absolute right-0 flex items-center"
              style={{ top: topPx, transform: 'translateY(-50%)' }}
            >
              <span className={`text-[9px] text-right pr-1 ${isZero ? 'font-medium text-gray-700' : 'text-gray-500'}`}>
                {tick > 0 ? `+${tick.toFixed(1)}` : tick.toFixed(1)}
              </span>
              <div className={`h-px ${isZero ? 'w-2 bg-gray-500' : 'w-1 bg-gray-300'}`} />
            </div>
          )
        }

        // Fallback: percentage-based
        const range = yMax - yMin
        const position = ((yMax - tick) / range) * 100

        return (
          <div
            key={tick}
            className="absolute right-0 flex items-center"
            style={{ top: `${position}%`, transform: 'translateY(-50%)' }}
          >
            <span className={`text-[9px] text-right pr-1 ${isZero ? 'font-medium text-gray-700' : 'text-gray-500'}`}>
              {tick > 0 ? `+${tick.toFixed(1)}` : tick.toFixed(1)}
            </span>
            <div className={`h-px ${isZero ? 'w-2 bg-gray-500' : 'w-1 bg-gray-300'}`} />
          </div>
        )
      })}

      <div className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1">
        <span className="text-[8px] text-gray-400 -rotate-90 block whitespace-nowrap">mV</span>
      </div>
    </div>
  )
}
