import { useState, useRef, useCallback, useEffect } from 'react'
import { useVisualisationStore } from '@/stores/visualisationStore'

// ---------------------------------------------------------------------------
// Geometry helper — ray-casting point-in-polygon
// ---------------------------------------------------------------------------

function pointInPolygon(
  x: number,
  y: number,
  polygon: { x: number; y: number }[],
): boolean {
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i]!.x,
      yi = polygon[i]!.y
    const xj = polygon[j]!.x,
      yj = polygon[j]!.y
    const intersect =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi
    if (intersect) inside = !inside
  }
  return inside
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface LassoOverlayProps {
  /** Pre-computed screen positions of filtered points: Map<patientId, {x, y}> */
  screenPositions: Map<string, { x: number; y: number }>
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function LassoOverlay({ screenPositions }: LassoOverlayProps) {
  const [isDrawing, setIsDrawing] = useState(false)
  const [shiftHeld, setShiftHeld] = useState(false)
  const [path, setPath] = useState<{ x: number; y: number }[]>([])
  const containerRef = useRef<HTMLDivElement>(null)

  const setSelectedPatients = useVisualisationStore(
    (s) => s.setSelectedPatients,
  )

  // -----------------------------------------------------------------------
  // Shift key tracking
  // -----------------------------------------------------------------------

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'Shift') setShiftHeld(true)
    }
    const up = (e: KeyboardEvent) => {
      if (e.key === 'Shift') setShiftHeld(false)
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [])

  // -----------------------------------------------------------------------
  // Relative position helper
  // -----------------------------------------------------------------------

  const getRelativePos = useCallback((e: React.MouseEvent) => {
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect) return { x: 0, y: 0 }
    return { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }, [])

  // -----------------------------------------------------------------------
  // Mouse handlers
  // -----------------------------------------------------------------------

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!e.shiftKey) return
      e.preventDefault()
      e.stopPropagation()
      const pos = getRelativePos(e)
      setIsDrawing(true)
      setPath([pos])
    },
    [getRelativePos],
  )

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isDrawing) return
      e.preventDefault()
      e.stopPropagation()
      const pos = getRelativePos(e)
      setPath((prev) => [...prev, pos])
    },
    [isDrawing, getRelativePos],
  )

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      if (!isDrawing) return
      e.preventDefault()
      e.stopPropagation()
      setIsDrawing(false)

      // Need at least 3 points to form a polygon
      if (path.length < 3) {
        setPath([])
        return
      }

      // Test every screen-projected point against the lasso polygon
      const selected = new Set<string>()
      for (const [patientId, pos] of screenPositions) {
        if (pointInPolygon(pos.x, pos.y, path)) {
          selected.add(patientId)
        }
      }

      setSelectedPatients(selected)
      setPath([])
    },
    [isDrawing, path, screenPositions, setSelectedPatients],
  )

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  const isActive = shiftHeld || isDrawing

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 z-20"
      style={{
        pointerEvents: isActive ? 'auto' : 'none',
        cursor: shiftHeld ? 'crosshair' : 'default',
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      {path.length > 1 && (
        <svg className="absolute inset-0 h-full w-full pointer-events-none">
          <polyline
            points={path.map((p) => `${p.x},${p.y}`).join(' ')}
            fill="rgba(39, 67, 184, 0.1)"
            stroke="#2743B8"
            strokeWidth={2}
            strokeDasharray="4,4"
          />
        </svg>
      )}
    </div>
  )
}
