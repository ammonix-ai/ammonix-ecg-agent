import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js'
import type { EnrichedProjectionPoint, ViewerSettings } from '@/types/projection'
import { DIAGNOSIS_COLOR_PALETTE } from '@/types/projection'

// =============================================================================
// Types
// =============================================================================

interface TribeVisualizationProps {
  points: EnrichedProjectionPoint[]
  scaleFactor: number
  settings: ViewerSettings
  diagnosisColorMap: Map<string, string>
}

interface TribeData {
  tribeId: string
  positions: THREE.Vector3[]
  centroid: THREE.Vector3
  count: number
  color: string
}

interface TribeBoundaryProps {
  tribe: TribeData
  boundaryOpacity: number
}

interface TribeCentroidProps {
  tribe: TribeData
  centroidSize: number
  centroidOpacity: number
}

// =============================================================================
// Helpers
// =============================================================================

/** Simple hash of a tribe ID string to pick a fallback color index. */
function tribeColorIndex(tribeId: string): number {
  let hash = 0
  for (let i = 0; i < tribeId.length; i++) {
    hash = ((hash << 5) - hash + tribeId.charCodeAt(i)) | 0
  }
  return Math.abs(hash) % DIAGNOSIS_COLOR_PALETTE.length
}

// =============================================================================
// TribeBoundary — ConvexGeometry hull with semi-transparent fill + wireframe
// =============================================================================

function TribeBoundary({ tribe, boundaryOpacity }: TribeBoundaryProps) {
  const geometry = useMemo(() => {
    // ConvexGeometry requires at least 4 non-coplanar points
    if (tribe.positions.length < 4) return null

    // Add tiny jitter to avoid coplanar failures
    const jittered = tribe.positions.map(
      (p) =>
        new THREE.Vector3(
          p.x + Math.random() * 0.01,
          p.y + Math.random() * 0.01,
          p.z + Math.random() * 0.01,
        ),
    )

    try {
      return new ConvexGeometry(jittered)
    } catch {
      return null
    }
  }, [tribe.positions])

  if (!geometry) return null

  const wireframeOpacity = Math.min(boundaryOpacity * 2, 0.6)

  return (
    <group>
      {/* Filled semi-transparent hull */}
      <mesh geometry={geometry}>
        <meshBasicMaterial
          color={tribe.color}
          transparent
          opacity={boundaryOpacity}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      {/* Wireframe overlay */}
      <mesh geometry={geometry}>
        <meshBasicMaterial
          color={tribe.color}
          transparent
          opacity={wireframeOpacity}
          wireframe
        />
      </mesh>
    </group>
  )
}

// =============================================================================
// TribeCentroid — wireframe octahedron + Html label
// =============================================================================

function TribeCentroid({ tribe, centroidSize, centroidOpacity }: TribeCentroidProps) {
  return (
    <group position={[tribe.centroid.x, tribe.centroid.y, tribe.centroid.z]}>
      {/* Wireframe octahedron marker */}
      <mesh>
        <octahedronGeometry args={[centroidSize]} />
        <meshBasicMaterial
          color={tribe.color}
          transparent
          opacity={centroidOpacity}
          wireframe
        />
      </mesh>

      {/* Label floating above the centroid */}
      <Html
        position={[0, centroidSize + 0.3, 0]}
        center
        style={{ pointerEvents: 'none' }}
      >
        <div
          className="px-2 py-1 text-[10px] font-semibold text-white rounded shadow-md whitespace-nowrap"
          style={{ backgroundColor: tribe.color }}
        >
          {tribe.tribeId} ({tribe.count})
        </div>
      </Html>
    </group>
  )
}

// =============================================================================
// TribeVisualization — main component
// =============================================================================

export function TribeVisualization({
  points,
  scaleFactor,
  settings,
  diagnosisColorMap,
}: TribeVisualizationProps) {
  const { display } = settings

  const tribeDataList = useMemo<TribeData[]>(() => {
    // Group points by tribe
    const tribeMap = new Map<string, EnrichedProjectionPoint[]>()

    for (const point of points) {
      if (!point.tribes || point.tribes.length === 0) continue
      for (const tribeId of point.tribes) {
        let members = tribeMap.get(tribeId)
        if (!members) {
          members = []
          tribeMap.set(tribeId, members)
        }
        members.push(point)
      }
    }

    const result: TribeData[] = []

    for (const [tribeId, members] of tribeMap) {
      // Collect scaled positions
      const positions = members.map(
        (p) => new THREE.Vector3(p.x * scaleFactor, p.y * scaleFactor, p.z * scaleFactor),
      )

      // Compute centroid (average of positions)
      const centroid = new THREE.Vector3()
      for (const pos of positions) {
        centroid.add(pos)
      }
      centroid.divideScalar(positions.length)

      // Determine color: use first member's primaryDiagnosis color, fallback to palette hash
      const firstDiagnosis = members[0]?.primaryDiagnosis
      let color = firstDiagnosis ? diagnosisColorMap.get(firstDiagnosis.toLowerCase()) : undefined
      if (!color) {
        color = DIAGNOSIS_COLOR_PALETTE[tribeColorIndex(tribeId)] ?? '#78909C'
      }

      result.push({
        tribeId,
        positions,
        centroid,
        count: members.length,
        color,
      })
    }

    return result
  }, [points, scaleFactor, diagnosisColorMap])

  // Render guard: nothing to show if both features are off
  if (!display.showCentroids && !display.showTribeBoundaries) return null

  return (
    <group>
      {tribeDataList.map((tribe) => (
        <group key={tribe.tribeId}>
          {display.showTribeBoundaries && (
            <TribeBoundary
              tribe={tribe}
              boundaryOpacity={display.boundaryOpacity}
            />
          )}
          {display.showCentroids && (
            <TribeCentroid
              tribe={tribe}
              centroidSize={display.centroidSize}
              centroidOpacity={display.centroidOpacity}
            />
          )}
        </group>
      ))}
    </group>
  )
}
