import { useState, useRef } from 'react'
import { X } from 'lucide-react'
import type { CaliperMeasurement, CaliperAxis } from '@/types/ecgTools'

interface CaliperOverlayProps {
  isActive: boolean
  pixelsPerSecond: number
  onAddMeasurement: (m: CaliperMeasurement) => void
  caliperAxis?: CaliperAxis
  containerHeight?: number
  yRangeMv?: number
  className?: string
}

/**
 * CaliperOverlay — Click-drag to measure time intervals (ms) or amplitude (mV)
 *
 * X-axis: measures time in ms between two vertical lines
 * Y-axis: measures amplitude in mV between two horizontal lines
 */
export function CaliperOverlay({
  isActive,
  pixelsPerSecond,
  onAddMeasurement,
  caliperAxis = 'x',
  containerHeight = 300,
  yRangeMv = 3.0,
  className,
}: CaliperOverlayProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null)
  const [dragEnd, setDragEnd] = useState<{ x: number; y: number } | null>(null)

  const getPos = (e: React.MouseEvent) => {
    const rect = ref.current?.getBoundingClientRect()
    if (!rect) return null
    return { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!isActive) return
    const pos = getPos(e)
    if (!pos) return
    setIsDragging(true)
    setDragStart(pos)
    setDragEnd(pos)
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || !dragStart) return
    const pos = getPos(e)
    if (pos) setDragEnd(pos)
  }

  const handleMouseUp = () => {
    if (!isDragging || !dragStart || !dragEnd) { setIsDragging(false); return }

    if (caliperAxis === 'x' && Math.abs(dragEnd.x - dragStart.x) >= 10) {
      const ms = Math.round(Math.abs(dragEnd.x - dragStart.x) / pixelsPerSecond * 1000)
      onAddMeasurement({
        id: `caliper-${Date.now()}`,
        axis: 'x',
        startX: Math.min(dragStart.x, dragEnd.x),
        endX: Math.max(dragStart.x, dragEnd.x),
        y: (dragStart.y + dragEnd.y) / 2,
        milliseconds: ms,
      })
    } else if (caliperAxis === 'y' && Math.abs(dragEnd.y - dragStart.y) >= 10) {
      const mv = Math.round(Math.abs(dragEnd.y - dragStart.y) / containerHeight * yRangeMv * 100) / 100
      onAddMeasurement({
        id: `caliper-${Date.now()}`,
        axis: 'y',
        startX: (dragStart.x + dragEnd.x) / 2,
        endX: (dragStart.x + dragEnd.x) / 2,
        y: Math.min(dragStart.y, dragEnd.y),
        milliseconds: 0,
        startY: Math.min(dragStart.y, dragEnd.y),
        endY: Math.max(dragStart.y, dragEnd.y),
        millivolts: mv,
      })
    }

    setIsDragging(false)
    setDragStart(null)
    setDragEnd(null)
  }

  // Preview while dragging
  const preview = isDragging && dragStart && dragEnd ? (() => {
    if (caliperAxis === 'x') {
      const ms = Math.round(Math.abs(dragEnd.x - dragStart.x) / pixelsPerSecond * 1000)
      return { axis: 'x' as const, x1: Math.min(dragStart.x, dragEnd.x), x2: Math.max(dragStart.x, dragEnd.x), y: (dragStart.y + dragEnd.y) / 2, label: `${ms} ms` }
    } else {
      const mv = Math.round(Math.abs(dragEnd.y - dragStart.y) / containerHeight * yRangeMv * 100) / 100
      return { axis: 'y' as const, x1: (dragStart.x + dragEnd.x) / 2, y1: Math.min(dragStart.y, dragEnd.y), y2: Math.max(dragStart.y, dragEnd.y), label: `${mv} mV` }
    }
  })() : null

  return (
    <div
      ref={ref}
      className={`absolute inset-0 ${isActive ? 'pointer-events-auto cursor-crosshair' : 'pointer-events-none'} ${className ?? ''}`}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {preview && preview.axis === 'x' && (
        <CaliperPreviewX x1={preview.x1} x2={preview.x2} y={preview.y} label={preview.label} />
      )}
      {preview && preview.axis === 'y' && (
        <CaliperPreviewY x={preview.x1} y1={preview.y1!} y2={preview.y2!} label={preview.label} />
      )}
    </div>
  )
}

function CaliperPreviewX({ x1, x2, y, label }: { x1: number; x2: number; y: number; label: string }) {
  const w = x2 - x1
  return (
    <div className="absolute pointer-events-none" style={{ left: x1, top: y - 30, width: w }}>
      <svg style={{ width: w, height: 60, overflow: 'visible' }}>
        <line x1={0} y1={0} x2={0} y2={60} stroke="#3B82F6" strokeWidth={2} strokeDasharray="4,2" />
        <line x1={w} y1={0} x2={w} y2={60} stroke="#3B82F6" strokeWidth={2} strokeDasharray="4,2" />
        <line x1={0} y1={30} x2={w} y2={30} stroke="#3B82F6" strokeWidth={2} strokeDasharray="4,2" />
        <polygon points={`0,30 8,26 8,34`} fill="#3B82F6" />
        <polygon points={`${w},30 ${w - 8},26 ${w - 8},34`} fill="#3B82F6" />
      </svg>
      <div className="absolute -top-8 left-1/2 -translate-x-1/2 px-2 py-1 rounded text-xs font-bold whitespace-nowrap bg-blue-500 text-white">
        {label}
      </div>
    </div>
  )
}

function CaliperPreviewY({ x, y1, y2, label }: { x: number; y1: number; y2: number; label: string }) {
  const h = y2 - y1
  return (
    <div className="absolute pointer-events-none" style={{ left: x - 30, top: y1, width: 60, height: h }}>
      <svg style={{ width: 60, height: h, overflow: 'visible' }}>
        <line x1={0} y1={0} x2={60} y2={0} stroke="#3B82F6" strokeWidth={2} strokeDasharray="4,2" />
        <line x1={0} y1={h} x2={60} y2={h} stroke="#3B82F6" strokeWidth={2} strokeDasharray="4,2" />
        <line x1={30} y1={0} x2={30} y2={h} stroke="#3B82F6" strokeWidth={2} strokeDasharray="4,2" />
        <polygon points="30,0 26,8 34,8" fill="#3B82F6" />
        <polygon points={`30,${h} 26,${h - 8} 34,${h - 8}`} fill="#3B82F6" />
      </svg>
      <div className="absolute left-16 top-1/2 -translate-y-1/2 px-2 py-1 rounded text-xs font-bold whitespace-nowrap bg-blue-500 text-white">
        {label}
      </div>
    </div>
  )
}

/** Render a completed caliper measurement (used by MeasurementsDisplayLayer) */
export function CaliperDisplay({ measurement, onRemove }: { measurement: CaliperMeasurement; onRemove?: () => void }) {
  if (measurement.axis === 'y') {
    const h = (measurement.endY ?? 0) - (measurement.startY ?? 0)
    return (
      <div className="absolute pointer-events-none" style={{ left: measurement.startX - 30, top: measurement.startY, width: 60, height: h }}>
        <svg style={{ width: 60, height: h, overflow: 'visible' }}>
          <line x1={0} y1={0} x2={60} y2={0} stroke="#8B5CF6" strokeWidth={2} />
          <line x1={0} y1={h} x2={60} y2={h} stroke="#8B5CF6" strokeWidth={2} />
          <line x1={30} y1={0} x2={30} y2={h} stroke="#8B5CF6" strokeWidth={2} />
        </svg>
        <div className="absolute left-16 top-1/2 -translate-y-1/2 px-2 py-1 rounded text-xs font-bold whitespace-nowrap bg-violet-500 text-white pointer-events-auto flex items-center gap-1">
          {measurement.millivolts} mV
          {onRemove && <button onClick={onRemove} className="p-0.5 hover:bg-violet-600 rounded"><X className="w-3 h-3" /></button>}
        </div>
      </div>
    )
  }
  const w = measurement.endX - measurement.startX
  return (
    <div className="absolute pointer-events-none" style={{ left: measurement.startX, top: measurement.y - 30, width: w }}>
      <svg style={{ width: w, height: 60, overflow: 'visible' }}>
        <line x1={0} y1={0} x2={0} y2={60} stroke="#8B5CF6" strokeWidth={2} />
        <line x1={w} y1={0} x2={w} y2={60} stroke="#8B5CF6" strokeWidth={2} />
        <line x1={0} y1={30} x2={w} y2={30} stroke="#8B5CF6" strokeWidth={2} />
        <polygon points="0,30 8,26 8,34" fill="#8B5CF6" />
        <polygon points={`${w},30 ${w - 8},26 ${w - 8},34`} fill="#8B5CF6" />
      </svg>
      <div className="absolute -top-8 left-1/2 -translate-x-1/2 px-2 py-1 rounded text-xs font-bold whitespace-nowrap bg-violet-500 text-white pointer-events-auto flex items-center gap-1">
        {measurement.milliseconds} ms
        {onRemove && <button onClick={onRemove} className="p-0.5 hover:bg-violet-600 rounded"><X className="w-3 h-3" /></button>}
      </div>
    </div>
  )
}
