import { useState, useRef, useCallback } from 'react'
import { X, Pencil } from 'lucide-react'
import type { Annotation } from '@/types/ecgTools'
import { ANNOTATION_PRESETS } from '@/types/ecgTools'

interface DrawOverlayProps {
  isActive: boolean
  onAddAnnotation: (a: Annotation) => void
  className?: string
}

const MIN_DRAG = 5
const MIN_SPACING = 2

const PRESETS = [...ANNOTATION_PRESETS, { label: 'Custom...', color: '#374151' }] as const

/**
 * DrawOverlay — Freehand annotation with preset/custom labels
 *
 * Click: place a point marker with label picker
 * Drag: draw a freehand stroke, then pick a label
 * Right-click: cancel
 *
 * H53 / Q-A9-8 status (2026-04-30): wired in V3 via
 * `expanded/ExpandedLeadView.tsx` (consumer) + `overlays/MeasurementsDisplayLayer.tsx`
 * (renders placed AnnotationMarkers). Q-A9-8's "underscore-prefixed unused
 * props" was resolved by a prior session — `isActive`, `onAddAnnotation`,
 * and `className` are all live props. Annotations live in component state
 * only; bridging `onAddAnnotation` to `services/annotationService.ts` for
 * backend persistence is a deferred Tier 2 V3-surfacing task.
 */
export function DrawOverlay({ isActive, onAddAnnotation, className }: DrawOverlayProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [pendingClick, setPendingClick] = useState<{ x: number; y: number } | null>(null)
  const [pendingStroke, setPendingStroke] = useState<{ x: number; y: number }[] | null>(null)
  const [customLabel, setCustomLabel] = useState('')
  const [isDrawing, setIsDrawing] = useState(false)
  const [currentStroke, setCurrentStroke] = useState<{ x: number; y: number }[]>([])
  const dragStartRef = useRef<{ x: number; y: number } | null>(null)

  const getPos = useCallback((e: React.MouseEvent) => {
    const rect = ref.current?.getBoundingClientRect()
    return rect ? { x: e.clientX - rect.left, y: e.clientY - rect.top } : null
  }, [])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (!isActive || e.button !== 0) return
    const t = e.target as HTMLElement
    if (t.closest('[data-picker]')) return
    const pos = getPos(e)
    if (!pos) return
    dragStartRef.current = pos
    setCurrentStroke([pos])
    setIsDrawing(true)
    setPendingClick(null)
    setPendingStroke(null)
    setCustomLabel('')
  }, [isActive, getPos])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDrawing) return
    const pos = getPos(e)
    if (!pos) return
    setCurrentStroke(prev => {
      const last = prev[prev.length - 1]
      if (!last) return [pos]
      if (Math.sqrt((pos.x - last.x) ** 2 + (pos.y - last.y) ** 2) < MIN_SPACING) return prev
      return [...prev, pos]
    })
  }, [isDrawing, getPos])

  const handleMouseUp = useCallback((e: React.MouseEvent) => {
    if (!isDrawing || !dragStartRef.current) return
    const pos = getPos(e)
    const start = dragStartRef.current
    setIsDrawing(false)
    dragStartRef.current = null
    if (!pos || !start) { setCurrentStroke([]); return }
    const dist = Math.sqrt((pos.x - start.x) ** 2 + (pos.y - start.y) ** 2)
    if (dist <= MIN_DRAG) {
      setCurrentStroke([])
      setPendingClick(start)
    } else {
      setPendingStroke(currentStroke.length > 1 ? currentStroke : [start, pos])
      setCurrentStroke([])
    }
  }, [isDrawing, getPos, currentStroke])

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    if (!isActive) return
    e.preventDefault()
    setPendingClick(null); setPendingStroke(null); setCurrentStroke([]); setIsDrawing(false); setCustomLabel('')
    dragStartRef.current = null
  }, [isActive])

  const handleSelectLabel = useCallback((preset: { label: string; color: string }) => {
    if (preset.label === 'Custom...') { setCustomLabel(' '); return }
    const point = pendingClick ?? (pendingStroke ? pendingStroke[0] : null)
    if (!point) return
    onAddAnnotation({
      id: `ann-${Date.now()}`, x: point.x, y: point.y,
      type: pendingStroke ? 'stroke' : 'marker',
      label: preset.label, color: preset.color,
      strokePoints: pendingStroke ?? undefined,
    })
    setPendingClick(null); setPendingStroke(null); setCustomLabel('')
  }, [pendingClick, pendingStroke, onAddAnnotation])

  const handleCustomSubmit = useCallback(() => {
    if (!customLabel.trim()) return
    const point = pendingClick ?? (pendingStroke ? pendingStroke[0] : null)
    if (!point) return
    onAddAnnotation({
      id: `ann-${Date.now()}`, x: point.x, y: point.y,
      type: pendingStroke ? 'stroke' : 'label',
      label: customLabel.trim(), color: '#374151',
      strokePoints: pendingStroke ?? undefined,
    })
    setPendingClick(null); setPendingStroke(null); setCustomLabel('')
  }, [pendingClick, pendingStroke, customLabel, onAddAnnotation])

  const pickerPos = pendingClick ?? (pendingStroke?.length ? pendingStroke[pendingStroke.length - 1] : null)

  return (
    <div ref={ref}
      className={`absolute inset-0 ${isActive ? 'pointer-events-auto cursor-crosshair' : 'pointer-events-none'} ${className ?? ''}`}
      onMouseDown={handleMouseDown} onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp} onContextMenu={handleContextMenu}>

      {/* Live stroke */}
      {isDrawing && currentStroke.length > 1 && (
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          <polyline points={currentStroke.map(p => `${p.x},${p.y}`).join(' ')}
            fill="none" stroke="#6644ff" strokeWidth="2" strokeOpacity="0.6"
            strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )}

      {/* Pending stroke preview */}
      {pendingStroke && pendingStroke.length > 1 && (
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          <polyline points={pendingStroke.map(p => `${p.x},${p.y}`).join(' ')}
            fill="none" stroke="#6644ff" strokeWidth="2" strokeOpacity="0.8"
            strokeLinecap="round" strokeLinejoin="round" strokeDasharray="4 2" />
        </svg>
      )}

      {/* Pending click dot */}
      {pendingClick && (
        <div className="absolute w-4 h-4 rounded-full bg-indigo-500 border-2 border-white shadow-lg -translate-x-1/2 -translate-y-1/2 pointer-events-none"
          style={{ left: pendingClick.x, top: pendingClick.y }} />
      )}

      {/* Label picker */}
      {pickerPos && (
        <div className="absolute z-30" style={{ left: pickerPos.x + 8, top: pickerPos.y + 8 }} data-picker="label">
          <div className="bg-white rounded-lg shadow-xl border border-gray-200 p-2 min-w-[160px]">
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide px-2 py-1">Add Label</div>
            <div className="space-y-0.5 max-h-48 overflow-y-auto">
              {PRESETS.map((p) => (
                <button key={p.label} onClick={(e) => { e.stopPropagation(); handleSelectLabel(p as { label: string; color: string }) }}
                  className="w-full flex items-center gap-2 px-2 py-1.5 text-sm text-left rounded hover:bg-gray-100 transition-colors">
                  <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: p.color }} />
                  <span>{p.label}</span>
                </button>
              ))}
            </div>
            {customLabel && (
              <div className="mt-2 pt-2 border-t border-gray-200">
                <input type="text" value={customLabel.trim() ? customLabel : ''} onChange={(e) => setCustomLabel(e.target.value)}
                  placeholder="Custom label..." className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  autoFocus onKeyDown={(e) => { if (e.key === 'Enter') handleCustomSubmit(); if (e.key === 'Escape') { setPendingClick(null); setPendingStroke(null); setCustomLabel('') } }}
                  onClick={(e) => e.stopPropagation()} />
                <div className="flex gap-1 mt-1">
                  <button onClick={(e) => { e.stopPropagation(); handleCustomSubmit() }} disabled={!customLabel.trim()}
                    className="flex-1 px-2 py-1 text-xs font-medium text-white bg-indigo-600 rounded hover:bg-indigo-700 disabled:opacity-50 transition-colors">Add</button>
                  <button onClick={(e) => { e.stopPropagation(); setPendingClick(null); setPendingStroke(null); setCustomLabel('') }}
                    className="px-2 py-1 text-xs font-medium text-gray-600 bg-gray-100 rounded hover:bg-gray-200 transition-colors">Cancel</button>
                </div>
              </div>
            )}
            {!customLabel && (
              <button onClick={(e) => { e.stopPropagation(); setPendingClick(null); setPendingStroke(null) }}
                className="w-full mt-1 px-2 py-1.5 text-xs font-medium text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded transition-colors">Cancel</button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

/** Render a completed annotation marker (used by MeasurementsDisplayLayer) */
export function AnnotationMarker({ annotation, onRemove }: { annotation: Annotation; onRemove: () => void }) {
  if (annotation.type === 'stroke' && annotation.strokePoints && annotation.strokePoints.length > 1) {
    const last = annotation.strokePoints[annotation.strokePoints.length - 1]
    return (
      <div className="absolute inset-0 z-10 pointer-events-none">
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          <polyline points={annotation.strokePoints.map(p => `${p.x},${p.y}`).join(' ')}
            fill="none" stroke={annotation.color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <div className="absolute pointer-events-auto" style={{ left: last!.x + 4, top: last!.y - 20 }}>
          <div className="flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-medium text-white shadow-md" style={{ backgroundColor: annotation.color }}>
            <Pencil className="w-3 h-3" /><span>{annotation.label}</span>
            <button onClick={onRemove} className="p-0.5 hover:bg-white/20 rounded"><X className="w-3 h-3" /></button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="absolute z-10 pointer-events-auto" style={{ left: annotation.x, top: annotation.y }}>
      <div className="absolute -translate-x-1/2 -translate-y-1/2 w-3 h-3 rounded-full border-2 border-white shadow" style={{ backgroundColor: annotation.color }} />
      <div className="absolute left-1 -top-6 flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-medium text-white whitespace-nowrap shadow-md" style={{ backgroundColor: annotation.color }}>
        <span>{annotation.label}</span>
        <button onClick={onRemove} className="p-0.5 hover:bg-white/20 rounded"><X className="w-3 h-3" /></button>
      </div>
      <div className="absolute left-0 -top-5 w-px h-4 -translate-x-1/2" style={{ backgroundColor: annotation.color }} />
    </div>
  )
}
