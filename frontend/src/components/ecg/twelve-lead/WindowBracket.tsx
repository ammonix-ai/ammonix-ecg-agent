import { useRef, useState, useCallback } from 'react'

interface WindowBracketProps {
  totalDuration: number
  windowSize: number
  windowPosition: number
  onPositionChange: (position: number) => void
  interactive?: boolean
  height?: number
  className?: string
  style?: React.CSSProperties
}

/**
 * WindowBracket — Draggable time window indicator on the rhythm strip
 *
 * Shows the full recording timeline with a bracket indicating the
 * 10-second window displayed in the 12-lead grid.
 */
export function WindowBracket({
  totalDuration,
  windowSize,
  windowPosition,
  onPositionChange,
  interactive = true,
  height = 24,
  className,
  style,
}: WindowBracketProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isDragging, setIsDragging] = useState(false)

  const isFullCoverage = windowSize >= totalDuration
  const startPct = (windowPosition / totalDuration) * 100
  const widthPct = Math.min((windowSize / totalDuration) * 100, 100)

  const pointerToTime = useCallback((clientX: number): number => {
    if (!containerRef.current) return windowPosition
    const rect = containerRef.current.getBoundingClientRect()
    const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    const centered = pct * totalDuration - windowSize / 2
    const max = Math.max(0, totalDuration - windowSize)
    return Math.max(0, Math.min(max, centered))
  }, [totalDuration, windowSize, windowPosition])

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    if (!interactive || isFullCoverage) return
    e.preventDefault()
    setIsDragging(true)
    containerRef.current?.setPointerCapture(e.pointerId)
    onPositionChange(pointerToTime(e.clientX))
  }, [interactive, isFullCoverage, pointerToTime, onPositionChange])

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (!isDragging) return
    onPositionChange(pointerToTime(e.clientX))
  }, [isDragging, pointerToTime, onPositionChange])

  const handlePointerUp = useCallback((e: React.PointerEvent) => {
    setIsDragging(false)
    containerRef.current?.releasePointerCapture(e.pointerId)
  }, [])

  // Time ticks every second
  const ticks: number[] = []
  for (let t = 0; t <= totalDuration; t += 1) ticks.push(t)

  return (
    <div
      ref={containerRef}
      className={`relative select-none ${interactive && !isFullCoverage ? 'cursor-grab' : ''} ${isDragging ? 'cursor-grabbing' : ''} ${className ?? ''}`}
      style={{ height, touchAction: 'none', ...style }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
    >
      {/* Time axis line */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gray-300" />

      {/* Ticks */}
      {ticks.map((t) => {
        const pos = (t / totalDuration) * 100
        const isEdge = t === 0 || t >= totalDuration
        const align = t === 0 ? 'translate-x-0' : t >= totalDuration ? '-translate-x-full' : '-translate-x-1/2'
        return (
          <div key={t} className="absolute top-0 flex flex-col" style={{ left: `${pos}%` }}>
            <div className="w-px h-2 bg-gray-400" style={{ transform: 'translateX(-50%)' }} />
            <span className={`text-[9px] mt-0.5 ${align} ${isEdge ? 'font-medium text-gray-600' : 'text-gray-500'}`}>
              {t}s
            </span>
          </div>
        )
      })}

      {/* Bracket */}
      {!isFullCoverage && (
        <div
          className={`absolute top-0 border-x-2 border-t-2 rounded-t-sm transition-colors pointer-events-none ${
            isDragging ? 'border-indigo-600 bg-indigo-100/50' : 'border-indigo-500 bg-indigo-50/30'
          }`}
          style={{ left: `${startPct}%`, width: `${widthPct}%`, height: 6 }}
        >
          {interactive && (
            <div className="absolute inset-x-0 top-0 flex justify-center pointer-events-none">
              <div className={`flex gap-px px-1 py-px rounded-b ${isDragging ? 'bg-indigo-500' : 'bg-indigo-400'}`}>
                <div className="w-px h-1 bg-white/80 rounded-full" />
                <div className="w-px h-1 bg-white/80 rounded-full" />
                <div className="w-px h-1 bg-white/80 rounded-full" />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Position labels while dragging */}
      {isDragging && !isFullCoverage && (
        <>
          <div className="absolute top-3 text-[9px] font-medium text-indigo-600 -translate-x-1/2"
            style={{ left: `${startPct}%` }}>
            {windowPosition.toFixed(1)}s
          </div>
          <div className="absolute top-3 text-[9px] font-medium text-indigo-600 -translate-x-1/2"
            style={{ left: `${startPct + widthPct}%` }}>
            {(windowPosition + windowSize).toFixed(1)}s
          </div>
        </>
      )}
    </div>
  )
}
