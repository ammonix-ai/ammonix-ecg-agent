import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import type { EnrichedProjectionPoint } from '@/types/projection'

interface UploadBeaconsProps {
  points: EnrichedProjectionPoint[]
  uploadedPatientIds: Set<string>
  scaleFactor: number
}

interface BeaconRingProps {
  position: [number, number, number]
}

function BeaconRing({ position }: BeaconRingProps) {
  const meshRef = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    if (!meshRef.current) return
    const scale = 1 + Math.sin(clock.elapsedTime * 3) * 0.3
    meshRef.current.scale.setScalar(scale)
    const material = meshRef.current.material as THREE.MeshBasicMaterial
    material.opacity = 0.3 + Math.sin(clock.elapsedTime * 3) * 0.2
  })

  return (
    <mesh ref={meshRef} position={position} rotation={[Math.PI / 2, 0, 0]}>
      <ringGeometry args={[0.2, 0.28, 32]} />
      <meshBasicMaterial color="#F97316" transparent opacity={0.5} side={THREE.DoubleSide} />
    </mesh>
  )
}

export function UploadBeacons({ points, uploadedPatientIds, scaleFactor }: UploadBeaconsProps) {
  const beaconPoints = useMemo(() => {
    return points.filter(
      (p) => p.patientId.startsWith('upload_') || uploadedPatientIds.has(p.patientId)
    )
  }, [points, uploadedPatientIds])

  if (beaconPoints.length === 0) return null

  return (
    <>
      {beaconPoints.map((point) => (
        <BeaconRing
          key={point.patientId}
          position={[point.x * scaleFactor, point.y * scaleFactor, point.z * scaleFactor]}
        />
      ))}
    </>
  )
}
