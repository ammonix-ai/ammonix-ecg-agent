import { useRef, useState, useCallback, useEffect, useMemo } from 'react'
import { Lock, Unlock } from 'lucide-react'
import { cn } from '@/design/cn'

/**
 * Diagnoses ordered by clinical category (11 groups), so related
 * diagnoses sit adjacent on the HSL wheel.
 */
const ORDERED_DIAGNOSES = [
  // Rhythm
  'sinus rhythm', 'sinus tachycardia', 'sinus bradycardia',
  'atrial fibrillation', 'atrial flutter',
  // Axis
  'left axis deviation', 'right axis deviation',
  // AV Block
  '1st degree av block', '2nd degree av block', '3rd degree av block',
  // Bundle Block
  'left bundle branch block', 'right bundle branch block',
  'left anterior fascicular block', 'left posterior fascicular block',
  // ST Segment
  'st elevation', 'st depression', 'st deviation',
  // T Wave
  't wave abnormal', 't wave inversion',
  // QT Interval
  'prolonged qt', 'short qt',
  // Ectopy
  'premature atrial complex', 'premature ventricular complex',
  // Cardiac Function
  'low qrs voltages', 'pacing rhythm',
  // Hypertrophy
  'left ventricular hypertrophy', 'right ventricular hypertrophy',
  // Infarction
  'myocardial infarction', 'STEMI',
]

interface DiagnosisColorWheelProps {
  /** Current color overrides */
  colorOverrides: Record<string, string>
  /** The diagnoses actually present in the data */
  activeDiagnoses: string[]
  /** Called when colors change (partial update to diagnosisColorOverrides) */
  onChange: (overrides: Record<string, string>) => void
}

function hslToHex(h: number, s: number, l: number): string {
  s /= 100
  l /= 100
  const a = s * Math.min(l, 1 - l)
  const f = (n: number) => {
    const k = (n + h / 30) % 12
    const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1)
    return Math.round(255 * color).toString(16).padStart(2, '0')
  }
  return `#${f(0)}${f(8)}${f(4)}`
}


const WHEEL_RADIUS = 85
const DOT_RADIUS = 7
const CENTER = 100

export function DiagnosisColorWheel({ colorOverrides: _colorOverrides, activeDiagnoses, onChange }: DiagnosisColorWheelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [locked, setLocked] = useState(true)
  const [dragging, setDragging] = useState<number | null>(null)
  const frameRef = useRef<number>(0)

  // Filter to only diagnoses present in the data, but keep the ordering
  const diagnoses = useMemo(() => {
    const activeSet = new Set(activeDiagnoses.map(d => d.toLowerCase()))
    return ORDERED_DIAGNOSES.filter(d => activeSet.has(d.toLowerCase()))
  }, [activeDiagnoses])

  // Angle state: evenly distributed initially, with optional rotation offset
  const [rotationOffset, setRotationOffset] = useState(0)
  const [freeAngles, setFreeAngles] = useState<number[]>([])

  // Initialize free angles when diagnoses change
  useEffect(() => {
    if (diagnoses.length === 0) return
    const step = 360 / diagnoses.length
    setFreeAngles(diagnoses.map((_, i) => i * step))
    setRotationOffset(0)
  }, [diagnoses.length]) // only re-init when count changes

  const getAngle = useCallback((idx: number): number => {
    if (locked) {
      const step = 360 / diagnoses.length
      return (idx * step + rotationOffset) % 360
    }
    return freeAngles[idx] ?? 0
  }, [locked, rotationOffset, freeAngles, diagnoses.length])

  const getColor = useCallback((angle: number): string => {
    return hslToHex(angle, 70, 50)
  }, [])

  // Compute current colors map
  const currentColors = useMemo(() => {
    const colors: Record<string, string> = {}
    diagnoses.forEach((dx, i) => {
      colors[dx] = getColor(getAngle(i))
    })
    return colors
  }, [diagnoses, getAngle, getColor])

  // Draw the wheel
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = 200 * dpr
    canvas.height = 200 * dpr
    ctx.scale(dpr, dpr)

    // Clear
    ctx.clearRect(0, 0, 200, 200)

    // Draw HSL ring background
    for (let deg = 0; deg < 360; deg++) {
      const rad = (deg - 90) * Math.PI / 180
      ctx.beginPath()
      ctx.arc(CENTER, CENTER, WHEEL_RADIUS, rad, rad + Math.PI / 180 + 0.01)
      ctx.lineWidth = 8
      ctx.strokeStyle = hslToHex(deg, 70, 50)
      ctx.stroke()
    }

    // Draw dots
    diagnoses.forEach((_dx, idx) => {
      const angle = getAngle(idx)
      const rad = (angle - 90) * Math.PI / 180
      const x = CENTER + WHEEL_RADIUS * Math.cos(rad)
      const y = CENTER + WHEEL_RADIUS * Math.sin(rad)
      const color = getColor(angle)

      // Dot
      ctx.beginPath()
      ctx.arc(x, y, DOT_RADIUS, 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.fill()
      ctx.strokeStyle = dragging === idx ? '#2743B8' : '#FFFFFF'
      ctx.lineWidth = dragging === idx ? 2.5 : 1.5
      ctx.stroke()
    })
  }, [diagnoses, getAngle, getColor, dragging])

  const getMouseAngle = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return 0
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left - CENTER
    const y = e.clientY - rect.top - CENTER
    return ((Math.atan2(y, x) * 180 / Math.PI) + 90 + 360) % 360
  }, [])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top

    // Find closest dot
    let closest = -1
    let closestDist = Infinity

    diagnoses.forEach((_, i) => {
      const angle = getAngle(i)
      const rad = (angle - 90) * Math.PI / 180
      const x = CENTER + WHEEL_RADIUS * Math.cos(rad)
      const y = CENTER + WHEEL_RADIUS * Math.sin(rad)
      const dist = Math.sqrt((mx - x) ** 2 + (my - y) ** 2)
      if (dist < closestDist) {
        closestDist = dist
        closest = i
      }
    })

    if (closestDist < DOT_RADIUS * 2.5 && closest >= 0) {
      setDragging(closest)
    }
  }, [diagnoses, getAngle])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (dragging === null) return

    if (frameRef.current) cancelAnimationFrame(frameRef.current)
    frameRef.current = requestAnimationFrame(() => {
      const angle = getMouseAngle(e)

      if (locked) {
        // In locked mode, compute the offset needed for this dot to be at the mouse angle
        const step = 360 / diagnoses.length
        const baseAngle = dragging * step
        setRotationOffset((angle - baseAngle + 360) % 360)
      } else {
        setFreeAngles(prev => {
          const next = [...prev]
          next[dragging] = angle
          return next
        })
      }
    })
  }, [dragging, locked, diagnoses.length, getMouseAngle])

  const handleMouseUp = useCallback(() => {
    if (dragging !== null) {
      // Emit color changes
      const colors: Record<string, string> = {}
      diagnoses.forEach((dx, i) => {
        colors[dx] = getColor(getAngle(i))
      })
      onChange(colors)
      setDragging(null)
    }
  }, [dragging, diagnoses, getAngle, getColor, onChange])

  if (diagnoses.length === 0) return null

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-gray-500">Color Wheel</span>
        <button
          onClick={() => setLocked(!locked)}
          className={cn(
            'flex items-center gap-1 px-2 py-0.5 text-[10px] rounded-md border transition-colors',
            locked
              ? 'border-brand-300 bg-brand-50 text-brand-700'
              : 'border-gray-200 text-gray-500 hover:bg-gray-50',
          )}
        >
          {locked ? <Lock className="w-3 h-3" /> : <Unlock className="w-3 h-3" />}
          {locked ? 'Locked' : 'Free'}
        </button>
      </div>

      <div className="flex justify-center">
        <canvas
          ref={canvasRef}
          width={200}
          height={200}
          className="w-[200px] h-[200px] cursor-grab active:cursor-grabbing"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        />
      </div>

      {/* Legend showing current assignments */}
      <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 max-h-[140px] overflow-y-auto">
        {diagnoses.map((dx) => (
          <div key={dx} className="flex items-center gap-1 text-[10px] text-gray-600">
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: currentColors[dx] }}
            />
            <span className="truncate">{dx}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
