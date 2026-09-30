import { useState, useEffect, useRef } from 'react'

interface HRCalculatorOverlayProps {
  isActive: boolean
  pixelsPerSecond: number
  className?: string
}

/**
 * HRCalculatorOverlay — Click R-peaks to compute heart rate
 *
 * Click to place R-peak markers. After 2+ points, shows running average HR.
 * Double-click or Enter finalizes. Wipes when tool is deactivated.
 * Right-click/Escape to clear.
 */
export function HRCalculatorOverlay({ isActive, pixelsPerSecond, className }: HRCalculatorOverlayProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [points, setPoints] = useState<{ x: number; y: number }[]>([])
  const [finalized, setFinalized] = useState(false)

  useEffect(() => {
    if (!isActive) { setPoints([]); setFinalized(false) }
  }, [isActive])

  const sorted = [...points].sort((a, b) => a.x - b.x)

  const computeStats = (pts: { x: number; y: number }[]) => {
    const s = [...pts].sort((a, b) => a.x - b.x)
    const hrs: number[] = []
    for (let i = 1; i < s.length; i++) {
      const ms = ((s[i]!.x - s[i - 1]!.x) / pixelsPerSecond) * 1000
      if (ms > 0) hrs.push(60000 / ms)
    }
    const avg = hrs.length ? Math.round(hrs.reduce((a, b) => a + b, 0) / hrs.length) : 0
    return { avg, min: hrs.length ? Math.round(Math.min(...hrs)) : 0, max: hrs.length ? Math.round(Math.max(...hrs)) : 0, beats: pts.length }
  }

  const handleClick = (e: React.MouseEvent) => {
    if (!isActive || finalized) return
    const rect = ref.current?.getBoundingClientRect()
    if (!rect) return
    setPoints(prev => [...prev, { x: e.clientX - rect.left, y: e.clientY - rect.top }])
  }

  const handleDoubleClick = (e: React.MouseEvent) => {
    if (!isActive) return; e.stopPropagation()
    if (points.length >= 2) setFinalized(true)
  }

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault(); setPoints([]); setFinalized(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && points.length >= 2) setFinalized(true)
    if (e.key === 'Escape') { setPoints([]); setFinalized(false) }
  }

  const stats = points.length >= 2 ? computeStats(points) : null
  const lastPt = sorted.length ? sorted[sorted.length - 1] : null
  const midY = sorted.length ? sorted.reduce((s, p) => s + p.y, 0) / sorted.length : 0

  return (
    <div ref={ref} tabIndex={isActive ? 0 : -1}
      className={`absolute inset-0 outline-none ${isActive ? 'pointer-events-auto cursor-crosshair' : 'pointer-events-none'} ${className ?? ''}`}
      onClick={handleClick} onDoubleClick={handleDoubleClick}
      onContextMenu={handleContextMenu} onKeyDown={handleKeyDown}>

      {/* R-peak markers */}
      {sorted.map((pt, i) => (
        <div key={i} className={`absolute w-3 h-3 rounded-full border-2 border-white shadow-lg -translate-x-1/2 -translate-y-1/2 pointer-events-none ${finalized ? 'bg-rose-600' : 'bg-rose-500'}`}
          style={{ left: pt.x, top: pt.y }} />
      ))}

      {/* Connecting lines */}
      {sorted.length >= 2 && (
        <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ overflow: 'visible' }}>
          {sorted.slice(1).map((pt, i) => (
            <line key={i} x1={sorted[i]!.x} y1={sorted[i]!.y} x2={pt.x} y2={pt.y}
              stroke={finalized ? '#E11D48' : '#F43F5E'} strokeWidth={1.5}
              strokeDasharray={finalized ? undefined : '4,2'} />
          ))}
        </svg>
      )}

      {/* Result */}
      {stats && lastPt && (
        <div className={`absolute px-2 py-1 text-white text-xs font-bold rounded whitespace-nowrap pointer-events-none shadow-md ${finalized ? 'bg-rose-600' : 'bg-rose-500'}`}
          style={{ left: lastPt.x + 12, top: midY - 12 }}>
          avg {stats.avg} bpm ({stats.beats} beats, {stats.min}–{stats.max})
        </div>
      )}
    </div>
  )
}
