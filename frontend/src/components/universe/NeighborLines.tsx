import * as THREE from 'three'
import { Line } from '@react-three/drei'
import type { EnrichedProjectionPoint } from '@/types/projection'

interface NeighborLinesProps {
  selectedPoint: EnrichedProjectionPoint
  neighbors: EnrichedProjectionPoint[]
  scaleFactor: number
  color?: string
}

export function NeighborLines({
  selectedPoint,
  neighbors,
  scaleFactor,
  color = '#F59E0B',
}: NeighborLinesProps) {
  const selectedPos: [number, number, number] = [
    selectedPoint.x * scaleFactor,
    selectedPoint.y * scaleFactor,
    selectedPoint.z * scaleFactor,
  ]

  return (
    <group>
      {neighbors.map((neighbor) => {
        const neighborPos: [number, number, number] = [
          neighbor.x * scaleFactor,
          neighbor.y * scaleFactor,
          neighbor.z * scaleFactor,
        ]

        return (
          <group key={neighbor.patientId}>
            <Line
              points={[selectedPos, neighborPos]}
              color={color}
              lineWidth={1}
              dashed
              dashScale={3}
              dashSize={0.3}
              gapSize={0.15}
            />
            <mesh position={neighborPos} rotation={[Math.PI / 2, 0, 0]}>
              <ringGeometry args={[0.15, 0.18, 32]} />
              <meshBasicMaterial
                color={color}
                transparent
                opacity={0.7}
                side={THREE.DoubleSide}
              />
            </mesh>
          </group>
        )
      })}
    </group>
  )
}
