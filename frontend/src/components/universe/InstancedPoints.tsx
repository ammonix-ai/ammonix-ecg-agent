import { useRef, useLayoutEffect, useMemo } from 'react'
import * as THREE from 'three'
import type { EnrichedProjectionPoint, ViewerSettings } from '@/types/projection'

// Disable color management for saturated colors
THREE.ColorManagement.enabled = false

// =============================================================================
// Geometry (module-level singletons)
//
// Every record is drawn as the same sphere. The published universe carries no
// fold membership, so there is no held-out set to mark with a second shape.
// =============================================================================

const POINT_GEOMETRY = new THREE.SphereGeometry(1, 12, 12)
const POINT_EDGE_GEOMETRY = new THREE.EdgesGeometry(POINT_GEOMETRY)

const OUTLINE_COLORS = {
  hovered: '#2743B8',
  selected: '#2743B8',
  neighbor: '#D97706',
}

// =============================================================================
// Types
// =============================================================================

interface InstancedPointsProps {
  points: EnrichedProjectionPoint[]
  pointColors: Map<string, string>
  scaleFactor: number
  selectedPatients: Set<string>
  hoveredPatient: string | null
  neighbors: EnrichedProjectionPoint[]
  settings: ViewerSettings
}

// =============================================================================
// PointMesh — one InstancedMesh for the whole scene
// =============================================================================

interface PointMeshProps {
  points: EnrichedProjectionPoint[]
  pointColors: Map<string, string>
  scaleFactor: number
  selectedPatients: Set<string>
  neighbors: Set<string>
  settings: ViewerSettings
}

const FALLBACK_COLOR = '#A0B4C2'
const dummy = new THREE.Object3D()
const tempColor = new THREE.Color()

function PointMesh({
  points,
  pointColors,
  scaleFactor,
  selectedPatients,
  neighbors,
  settings,
}: PointMeshProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const count = points.length
  const { display } = settings

  useLayoutEffect(() => {
    const mesh = meshRef.current
    if (!mesh || count === 0) return

    // Ensure instanceColor buffer exists with correct size
    if (!mesh.instanceColor || mesh.instanceColor.count !== count) {
      mesh.instanceColor = new THREE.InstancedBufferAttribute(
        new Float32Array(count * 3),
        3,
      )
    }

    const colorArray = mesh.instanceColor.array as Float32Array

    for (let i = 0; i < count; i++) {
      const p = points[i]!

      // Position
      dummy.position.set(
        p.x * scaleFactor,
        p.y * scaleFactor,
        p.z * scaleFactor,
      )

      // Scale based on state
      let size = display.pointSize
      if (selectedPatients.has(p.patientId)) {
        size = display.selectedPointSize
      } else if (neighbors.has(p.patientId)) {
        size = display.neighborPointSize
      }
      dummy.scale.setScalar(size)

      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)

      // Color
      tempColor.set(pointColors.get(p.patientId) || FALLBACK_COLOR)
      colorArray[i * 3] = tempColor.r
      colorArray[i * 3 + 1] = tempColor.g
      colorArray[i * 3 + 2] = tempColor.b
    }

    mesh.instanceMatrix.needsUpdate = true
    mesh.instanceColor.needsUpdate = true

    // Store patient IDs for manual raycaster hover detection
    mesh.userData.patientIds = points.map(p => p.patientId)

    // Recompute bounding sphere so raycaster uses current instance positions
    mesh.computeBoundingSphere()

    if (mesh.material && !Array.isArray(mesh.material)) {
      ;(mesh.material as THREE.Material).needsUpdate = true
    }
  }, [points, pointColors, scaleFactor, selectedPatients, neighbors, display, count])

  if (count === 0) return null

  return (
    <instancedMesh
      ref={meshRef}
      args={[POINT_GEOMETRY, undefined, count]}
      frustumCulled={false}
    >
      <meshBasicMaterial
        transparent
        opacity={display.opacity}
        toneMapped={false}
        color="white"
      />
    </instancedMesh>
  )
}

// =============================================================================
// EdgeOutline
// =============================================================================

interface EdgeOutlineProps {
  point: EnrichedProjectionPoint
  scaleFactor: number
  size: number
  color: string
}

function EdgeOutline({ point, scaleFactor, size, color }: EdgeOutlineProps) {
  return (
    <lineSegments
      position={[
        point.x * scaleFactor,
        point.y * scaleFactor,
        point.z * scaleFactor,
      ]}
      scale={[size, size, size]}
    >
      <primitive object={POINT_EDGE_GEOMETRY} attach="geometry" />
      <lineBasicMaterial color={color} />
    </lineSegments>
  )
}

// =============================================================================
// Main InstancedPoints component
// =============================================================================

export function InstancedPoints({
  points,
  pointColors,
  scaleFactor,
  selectedPatients,
  hoveredPatient,
  neighbors,
  settings,
}: InstancedPointsProps) {
  const { display } = settings

  const neighborIds = useMemo(
    () => new Set(neighbors.map((n) => n.patientId)),
    [neighbors],
  )

  // Collect outlined points (selected, hovered, neighbors)
  const outlinePoints = useMemo(() => {
    if (!display.showEdgeOutlines) return []

    const outlines: { point: EnrichedProjectionPoint; color: string; size: number }[] = []

    for (const n of neighbors) {
      outlines.push({
        point: n,
        color: OUTLINE_COLORS.neighbor,
        size: display.neighborPointSize,
      })
    }

    for (const point of points) {
      if (selectedPatients.has(point.patientId)) {
        outlines.push({
          point,
          color: OUTLINE_COLORS.selected,
          size: display.selectedPointSize,
        })
      }
    }

    if (hoveredPatient) {
      const hoveredPoint = points.find((p) => p.patientId === hoveredPatient)
      if (hoveredPoint) {
        outlines.push({
          point: hoveredPoint,
          color: OUTLINE_COLORS.hovered,
          size: display.selectedPointSize,
        })
      }
    }

    return outlines
  }, [points, selectedPatients, hoveredPatient, neighbors, display])

  return (
    <group>
      <PointMesh
        points={points}
        pointColors={pointColors}
        scaleFactor={scaleFactor}
        selectedPatients={selectedPatients}
        neighbors={neighborIds}
        settings={settings}
      />

      {outlinePoints.map((o) => (
        <EdgeOutline
          key={`outline-${o.point.patientId}-${o.color}`}
          point={o.point}
          scaleFactor={scaleFactor}
          size={o.size}
          color={o.color}
        />
      ))}
    </group>
  )
}
