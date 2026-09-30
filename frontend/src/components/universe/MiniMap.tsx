import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { useVisualisationStore } from '@/stores/visualisationStore'
import { useFilteredPoints, usePointColors, useScaleFactor } from '@/stores/visualisationSelectors'

type Slice = 'XZ' | 'XY' | 'YZ'

interface SliceConfig {
  /** Map a 3D point to [horizontal, vertical] in the mini-map. */
  project: (x: number, y: number, z: number) => [number, number]
  hLabel: string
  vLabel: string
}

const SLICE_CONFIGS: Record<Slice, SliceConfig> = {
  XZ: {
    project: (x, _y, z) => [x, z],
    hLabel: 'X',
    vLabel: 'Z',
  },
  XY: {
    project: (x, y, _z) => [x, y],
    hLabel: 'X',
    vLabel: 'Y',
  },
  YZ: {
    project: (_x, y, z) => [y, z],
    hLabel: 'Y',
    vLabel: 'Z',
  },
}

const SMALL_W = 160
const SMALL_H = 120
const EXPANDED_W = 400
const EXPANDED_H = 300

export function MiniMap() {
  const showMiniMap = useVisualisationStore((s) => s.showMiniMap)
  const filteredPoints = useFilteredPoints()
  const pointColors = usePointColors()
  const scaleFactor = useScaleFactor()
  const setFlyToTarget = useVisualisationStore((s) => s.setFlyToTarget)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const [expanded, setExpanded] = useState(false)
  const [slice, setSlice] = useState<Slice>('XZ')
  const [isDragging, setIsDragging] = useState(false)

  const w = expanded ? EXPANDED_W : SMALL_W
  const h = expanded ? EXPANDED_H : SMALL_H
  const config = SLICE_CONFIGS[slice]

  // Compute 2D bounds for the current projection slice
  const bounds = useMemo(() => {
    if (filteredPoints.length === 0) return { minH: -10, maxH: 10, minV: -10, maxV: 10 }
    let minH = Infinity, maxH = -Infinity, minV = Infinity, maxV = -Infinity
    for (const p of filteredPoints) {
      const [ph, pv] = config.project(p.x * scaleFactor, p.y * scaleFactor, p.z * scaleFactor)
      if (ph < minH) minH = ph
      if (ph > maxH) maxH = ph
      if (pv < minV) minV = pv
      if (pv > maxV) maxV = pv
    }
    const padH = (maxH - minH) * 0.1 || 1
    const padV = (maxV - minV) * 0.1 || 1
    return { minH: minH - padH, maxH: maxH + padH, minV: minV - padV, maxV: maxV + padV }
  }, [filteredPoints, scaleFactor, config])

  // Draw points onto the canvas
  useEffect(() => {
    if (!showMiniMap) return
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const { minH, maxH, minV, maxV } = bounds
    const rangeH = maxH - minH || 1
    const rangeV = maxV - minV || 1

    // Clear
    ctx.fillStyle = '#FFFFFF'
    ctx.fillRect(0, 0, w, h)

    // Draw points
    const dotRadius = expanded ? 2 : 1.5
    for (const p of filteredPoints) {
      const [ph, pv] = config.project(p.x * scaleFactor, p.y * scaleFactor, p.z * scaleFactor)
      const px = ((ph - minH) / rangeH) * w
      const py = ((pv - minV) / rangeV) * h
      const color = pointColors.get(p.patientId) || '#94A3B8'
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.arc(px, py, dotRadius, 0, Math.PI * 2)
      ctx.fill()
    }

    // Draw border
    ctx.strokeStyle = '#CBD5E1'
    ctx.lineWidth = 1
    ctx.strokeRect(0, 0, w, h)

    // Axis labels (only when expanded)
    if (expanded) {
      ctx.fillStyle = '#64748B'
      ctx.font = '11px "Hanken Grotesk Variable", system-ui, sans-serif'
      ctx.textAlign = 'center'
      // Bottom-center: horizontal axis label
      ctx.fillText(config.hLabel, w / 2, h - 4)
      // Left-center: vertical axis label (rotated)
      ctx.save()
      ctx.translate(12, h / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.fillText(config.vLabel, 0, 0)
      ctx.restore()
    }
  }, [showMiniMap, filteredPoints, pointColors, scaleFactor, bounds, w, h, config, expanded])

  // Convert canvas pixel position to 3D world coordinates and fly there
  const flyToCanvasPosition = useCallback(
    (canvasX: number, canvasY: number) => {
      const { minH, maxH, minV, maxV } = bounds
      const rangeH = maxH - minH || 1
      const rangeV = maxV - minV || 1

      const worldH = minH + (canvasX / w) * rangeH
      const worldV = minV + (canvasY / h) * rangeV

      // Reconstruct 3D target from the 2D slice position (set missing axis to 0)
      let target: { x: number; y: number; z: number }
      if (slice === 'XZ') target = { x: worldH, y: 0, z: worldV }
      else if (slice === 'XY') target = { x: worldH, y: worldV, z: 0 }
      else target = { x: 0, y: worldH, z: worldV }

      setFlyToTarget(target)
    },
    [bounds, w, h, slice, setFlyToTarget],
  )

  const handleCanvasPointerDown = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!expanded) return
      e.stopPropagation()
      setIsDragging(true)
      const rect = canvasRef.current!.getBoundingClientRect()
      flyToCanvasPosition(e.clientX - rect.left, e.clientY - rect.top)
    },
    [expanded, flyToCanvasPosition],
  )

  const handleCanvasPointerMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!isDragging || !expanded) return
      e.stopPropagation()
      const rect = canvasRef.current!.getBoundingClientRect()
      flyToCanvasPosition(e.clientX - rect.left, e.clientY - rect.top)
    },
    [isDragging, expanded, flyToCanvasPosition],
  )

  const handleCanvasPointerUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      // Only toggle expansion if clicking the container (not buttons), and not currently expanded
      if (!expanded) {
        e.stopPropagation()
        setExpanded(true)
      }
    },
    [expanded],
  )

  if (!showMiniMap) return null

  return (
    <div
      className="absolute bottom-3 left-3 z-10"
      onClick={handleClick}
    >
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={w}
          height={h}
          className={`rounded-lg border border-gray-300 bg-white/80 backdrop-blur-sm shadow-sm ${
            expanded ? 'cursor-crosshair' : 'cursor-pointer'
          }`}
          onPointerDown={handleCanvasPointerDown}
          onPointerMove={handleCanvasPointerMove}
          onPointerUp={handleCanvasPointerUp}
          onPointerLeave={handleCanvasPointerUp}
        />

        {/* Expanded controls overlay */}
        {expanded && (
          <>
            {/* Close button */}
            <button
              onClick={(e) => {
                e.stopPropagation()
                setExpanded(false)
              }}
              className="absolute top-1.5 right-1.5 w-5 h-5 flex items-center justify-center rounded bg-gray-800/60 text-white hover:bg-gray-800/80 transition-colors"
              title="Collapse mini-map"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            {/* Slice toggle buttons */}
            <div className="absolute top-1.5 left-1.5 flex gap-1">
              {(['XZ', 'XY', 'YZ'] as Slice[]).map((s) => (
                <button
                  key={s}
                  onClick={(e) => {
                    e.stopPropagation()
                    setSlice(s)
                  }}
                  className={`px-1.5 py-0.5 rounded text-[10px] font-semibold transition-colors ${
                    slice === s
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800/50 text-gray-200 hover:bg-gray-800/70'
                  }`}
                  title={
                    s === 'XZ' ? 'Top-down view'
                    : s === 'XY' ? 'Front view'
                    : 'Side view'
                  }
                >
                  {s}
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
