import { useState, useRef } from 'react'
import { X } from 'lucide-react'
import type { RulerMeasurement } from '@/types/ecgTools'
import { calculateAngle, calculateDistance } from '@/types/ecgTools'

interface RulerOverlayProps {
  isActive: boolean
  onAddMeasurement: (m: RulerMeasurement) => void
  className?: string
}

/**
 * RulerOverlay — Two-click angle/distance measurement
 *
 * Click first point, move to see preview, click second point.
 * Shows angle relative to horizontal + distance.
 * Right-click cancels.
 */
export function RulerOverlay({ isActive, onAddMeasurement, className }: RulerOverlayProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [first, setFirst] = useState<{ x: number; y: number } | null>(null)
  const [current, setCurrent] = useState<{ x: number; y: number } | null>(null)

  const getPos = (e: React.MouseEvent) => {
    const rect = ref.current?.getBoundingClientRect()
    return rect ? { x: e.clientX - rect.left, y: e.clientY - rect.top } : null
  }

  const handleClick = (e: React.MouseEvent) => {
    if (!isActive) return
    const pos = getPos(e)
    if (!pos) return

    if (!first) {
      setFirst(pos)
    } else {
      const dist = calculateDistance(first.x, first.y, pos.x, pos.y)
      if (dist >= 10) {
        onAddMeasurement({
          id: `ruler-${Date.now()}`,
          startX: first.x, startY: first.y,
          endX: pos.x, endY: pos.y,
          angleDegrees: calculateAngle(first.x, first.y, pos.x, pos.y),
          distancePx: dist,
        })
      }
      setFirst(null); setCurrent(null)
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isActive || !first) return
    const pos = getPos(e)
    if (pos) setCurrent(pos)
  }

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault()
    setFirst(null); setCurrent(null)
  }

  const preview = first && current ? {
    startX: first.x, startY: first.y, endX: current.x, endY: current.y,
    angleDegrees: calculateAngle(first.x, first.y, current.x, current.y),
    distancePx: calculateDistance(first.x, first.y, current.x, current.y),
  } : null

  return (
    <div ref={ref}
      className={`absolute inset-0 ${isActive ? 'pointer-events-auto cursor-crosshair' : 'pointer-events-none'} ${className ?? ''}`}
      onClick={handleClick} onMouseMove={handleMouseMove} onContextMenu={handleContextMenu}>

      {first && (
        <div className="absolute w-3 h-3 bg-emerald-500 rounded-full border-2 border-white shadow-lg -translate-x-1/2 -translate-y-1/2 pointer-events-none"
          style={{ left: first.x, top: first.y }} />
      )}

      {preview && <RulerDisplay measurement={{ id: 'preview', ...preview }} isPreview />}
    </div>
  )
}

/** Render a completed ruler measurement */
export function RulerDisplay({ measurement, isPreview = false, onRemove }: {
  measurement: RulerMeasurement; isPreview?: boolean; onRemove?: () => void
}) {
  const { startX, startY, endX, endY, angleDegrees, distancePx } = measurement
  const midX = (startX + endX) / 2
  const midY = (startY + endY) / 2
  const angle = Math.atan2(endY - startY, endX - startX)
  const labelX = midX + Math.sin(angle) * 20
  const labelY = midY - Math.cos(angle) * 20
  const color = isPreview ? '#10B981' : '#059669'

  return (
    <div className="absolute inset-0 pointer-events-none">
      <svg className="absolute inset-0 w-full h-full overflow-visible">
        <line x1={startX} y1={startY} x2={endX} y2={endY} stroke={color} strokeWidth={2}
          strokeDasharray={isPreview ? '4,2' : undefined} />
        <circle cx={startX} cy={startY} r={4} fill={color} stroke="white" strokeWidth={2} />
        <circle cx={endX} cy={endY} r={4} fill={color} stroke="white" strokeWidth={2} />
        {distancePx > 30 && (
          <line x1={startX} y1={startY} x2={startX + 25} y2={startY}
            stroke={color} strokeWidth={1} strokeDasharray="3,3" opacity={0.5} />
        )}
      </svg>
      <div className={`absolute px-2 py-1 rounded text-xs font-bold whitespace-nowrap shadow-md flex items-center gap-2
        ${isPreview ? 'bg-emerald-500 text-white' : 'bg-emerald-600 text-white pointer-events-auto'}`}
        style={{ left: labelX, top: labelY, transform: 'translate(-50%, -50%)' }}>
        <span>{angleDegrees}°</span>
        {!isPreview && onRemove && (
          <button onClick={(e) => { e.stopPropagation(); onRemove() }} className="p-0.5 hover:bg-emerald-700 rounded">
            <X className="w-3 h-3" />
          </button>
        )}
      </div>
    </div>
  )
}
