import { useRef, useCallback, useState, useEffect } from 'react'

interface NavigateOverlayProps {
  isActive: boolean
  zoom: number
  panOffset: number
  visibleDuration: number
  totalDuration: number
  onZoomChange: (zoom: number) => void
  onPanChange: (offset: number) => void
  minZoom?: number
  maxZoom?: number
  className?: string
}

/**
 * NavigateOverlay — Pan/zoom with scroll-wheel and drag
 *
 * Scroll wheel: zoom centered on cursor position
 * Click-drag: pan viewport
 * Double-click: reset zoom to 1x
 */
export function NavigateOverlay({
  isActive, zoom, panOffset, visibleDuration, totalDuration,
  onZoomChange, onPanChange, minZoom = 1, maxZoom = 10, className,
}: NavigateOverlayProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState<{ x: number; panOffset: number } | null>(null)

  const zoomedDuration = visibleDuration / zoom
  const canPan = totalDuration > zoomedDuration
  const maxPan = Math.max(0, totalDuration - zoomedDuration)

  const handleWheel = useCallback((e: WheelEvent) => {
    if (!isActive) return
    e.preventDefault()
    const rect = ref.current?.getBoundingClientRect()
    if (!rect) return

    const delta = e.deltaY > 0 ? -0.15 : 0.15
    const newZoom = Math.max(minZoom, Math.min(maxZoom, zoom * (1 + delta)))
    const cursorX = (e.clientX - rect.left) / rect.width
    const cursorTime = panOffset + cursorX * zoomedDuration
    const newDur = visibleDuration / newZoom
    const newPan = cursorTime - cursorX * newDur
    const newMax = Math.max(0, totalDuration - newDur)

    onZoomChange(newZoom)
    onPanChange(Math.max(0, Math.min(newMax, newPan)))
  }, [isActive, zoom, panOffset, zoomedDuration, visibleDuration, totalDuration, minZoom, maxZoom, onZoomChange, onPanChange])

  useEffect(() => {
    const el = ref.current
    if (!el || !isActive) return
    el.addEventListener('wheel', handleWheel, { passive: false })
    return () => el.removeEventListener('wheel', handleWheel)
  }, [isActive, handleWheel])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (!isActive || !canPan) return
    e.preventDefault()
    setIsDragging(true)
    setDragStart({ x: e.clientX, panOffset })
  }, [isActive, canPan, panOffset])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging || !dragStart || !ref.current) return
    const rect = ref.current.getBoundingClientRect()
    const pxPerSec = rect.width / zoomedDuration
    const dt = -(e.clientX - dragStart.x) / pxPerSec
    onPanChange(Math.max(0, Math.min(maxPan, dragStart.panOffset + dt)))
  }, [isDragging, dragStart, zoomedDuration, maxPan, onPanChange])

  const handleMouseUp = useCallback(() => { setIsDragging(false); setDragStart(null) }, [])

  const handleDoubleClick = useCallback(() => {
    if (!isActive) return
    onZoomChange(1); onPanChange(0)
  }, [isActive, onZoomChange, onPanChange])

  const cursor = !isActive ? 'default' : isDragging ? 'grabbing' : canPan ? 'grab' : 'zoom-in'

  return (
    <div ref={ref}
      className={`absolute inset-0 ${isActive ? 'pointer-events-auto' : 'pointer-events-none'} ${className ?? ''}`}
      style={{ cursor }}
      onMouseDown={handleMouseDown} onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}
      onDoubleClick={handleDoubleClick}>
      {isActive && zoom > 1 && (
        <div className="absolute top-2 right-2 px-2 py-1 bg-black/70 text-white text-xs font-medium rounded-full pointer-events-none">
          {Math.round(zoom * 100)}%
        </div>
      )}
      {isActive && canPan && (
        <div className="absolute top-2 left-2 px-2 py-1 bg-black/70 text-white text-xs font-medium rounded-full pointer-events-none">
          {panOffset.toFixed(1)}s — {(panOffset + zoomedDuration).toFixed(1)}s
        </div>
      )}
    </div>
  )
}
