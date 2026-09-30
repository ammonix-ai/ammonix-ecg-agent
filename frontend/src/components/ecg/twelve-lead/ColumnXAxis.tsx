import { useMemo } from 'react'

interface ColumnXAxisProps {
  startTime: number
  endTime: number
  height?: number
  className?: string
}

/**
 * ColumnXAxis — Time axis for a lead column
 *
 * Shows ~5 ticks with time labels in seconds.
 */
export function ColumnXAxis({
  startTime,
  endTime,
  height = 20,
  className,
}: ColumnXAxisProps) {
  const duration = endTime - startTime

  const ticks = useMemo(() => {
    const count = 5
    const interval = duration / count
    return Array.from({ length: count + 1 }, (_, i) => ({
      time: startTime + i * interval,
      position: i / count,
    }))
  }, [startTime, duration])

  return (
    <div className={`relative ${className ?? ''}`} style={{ height }}>
      {ticks.map(({ time, position }, i) => (
        <div
          key={i}
          className="absolute flex flex-col items-center"
          style={{ left: `${position * 100}%`, transform: 'translateX(-50%)' }}
        >
          <div className="w-px h-1 bg-gray-400" />
          <span className="text-[9px] text-gray-500 mt-0.5">
            {time.toFixed(0)}s
          </span>
        </div>
      ))}
    </div>
  )
}
