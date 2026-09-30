import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import type { ThreeEvent } from '@react-three/fiber'
import type { EnrichedProjectionPoint } from '@/types/projection'
import { useVisualisationStore } from '@/stores/visualisationStore'
import { useDiagnosisColorMap } from '@/stores/visualisationSelectors'

interface DiagnosisLabels3DProps {
  points: EnrichedProjectionPoint[]
  scaleFactor: number
  visible: boolean
}

interface LabelData {
  diagnosis: string
  x: number
  y: number
  z: number
  count: number
}

// Shared geometry for invisible click targets (small sphere at each centroid)
const clickTargetGeo = new THREE.SphereGeometry(0.35, 8, 8)
const clickTargetMat = new THREE.MeshBasicMaterial({
  transparent: true,
  opacity: 0,
  depthWrite: false,
})


export function DiagnosisLabels3D({ points, scaleFactor, visible }: DiagnosisLabels3DProps) {
  const diagnosisColorMap = useDiagnosisColorMap()
  const isOrthographic = useVisualisationStore((s) => s.isOrthographic)
  const highlightedDiagnoses = useVisualisationStore((s) => s.highlightedDiagnoses)
  const filledDiagnoses = useVisualisationStore((s) => s.filledDiagnoses)
  const cycleDiagnosisFilter = useVisualisationStore((s) => s.cycleDiagnosisFilter)

  const labels = useMemo<LabelData[]>(() => {
    if (!visible || points.length === 0) return []

    // Compute centroid per primaryDiagnosis
    const accum = new Map<string, { sx: number; sy: number; sz: number; count: number }>()

    for (const p of points) {
      const dx = p.primaryDiagnosis
      const entry = accum.get(dx)
      if (entry) {
        entry.sx += p.x
        entry.sy += p.y
        entry.sz += p.z
        entry.count++
      } else {
        accum.set(dx, { sx: p.x, sy: p.y, sz: p.z, count: 1 })
      }
    }

    return Array.from(accum.entries()).map(([diagnosis, { sx, sy, sz, count }]) => ({
      diagnosis,
      x: (sx / count) * scaleFactor,
      y: (sy / count) * scaleFactor + 0.3,
      z: (sz / count) * scaleFactor,
      count,
    }))
  }, [points, scaleFactor, visible])

  if (!visible || labels.length === 0) return null

  const hasSelection = highlightedDiagnoses.size > 0 || filledDiagnoses.size > 0

  return (
    <group>
      {labels.map((label) => {
        const color = diagnosisColorMap.get(label.diagnosis.toLowerCase()) || '#94A3B8'
        const isHighlighted = highlightedDiagnoses.has(label.diagnosis)
        const isFilled = filledDiagnoses.has(label.diagnosis)
        const isActive = isHighlighted || isFilled
        const isDimmed = hasSelection && !isActive
        return (
          <group key={label.diagnosis}>
            {/* Display label — clickable, rendered above points */}
            <Html
              position={[label.x, label.y, label.z]}
              center
              distanceFactor={isOrthographic ? undefined : 12}
              zIndexRange={[5, 5]}
              style={{ pointerEvents: 'auto' }}
            >
              <div
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-full shadow-sm border whitespace-nowrap select-none transition-opacity duration-150 cursor-pointer"
                style={{
                  opacity: isDimmed ? 0.35 : 1,
                  backgroundColor: isFilled ? color : 'rgba(255,255,255,0.7)',
                  borderColor: isActive ? color : 'rgba(255,255,255,0.5)',
                  borderWidth: isFilled ? 2 : 1,
                }}
                onClick={() => cycleDiagnosisFilter(label.diagnosis)}
              >
                <span
                  className="text-[11px] font-semibold leading-none"
                  style={{ color: isFilled ? '#fff' : color }}
                >
                  {label.diagnosis}
                </span>
                <span
                  className="text-[10px] leading-none"
                  style={{ color: isFilled ? 'rgba(255,255,255,0.7)' : '#9CA3AF' }}
                >
                  {label.count.toLocaleString()}
                </span>
              </div>
            </Html>

            {/* Invisible click target — lives in Three.js scene, not DOM */}
            <mesh
              position={[label.x, label.y, label.z]}
              geometry={clickTargetGeo}
              material={clickTargetMat}
              onClick={(e: ThreeEvent<MouseEvent>) => {
                e.stopPropagation()
                cycleDiagnosisFilter(label.diagnosis)
              }}
            />
          </group>
        )
      })}
    </group>
  )
}
