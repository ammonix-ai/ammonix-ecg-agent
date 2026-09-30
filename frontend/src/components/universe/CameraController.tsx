import { useEffect, useRef, useCallback } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { useVisualisationStore } from '@/stores/visualisationStore'

interface CameraControllerProps {
  controlsRef: React.RefObject<any>
}

const ROTATION_STEP = Math.PI / 12 // 15 degrees
const ZOOM_STEP = 0.8
const PAN_STEP = 2

const DEFAULT_FOV = 50
const DEFAULT_DIST = Math.sqrt(17.5 * 17.5 * 3) // ~30.3

const PRESETS = {
  default: new THREE.Vector3(17.5, 17.5, 17.5),
  top: new THREE.Vector3(0, DEFAULT_DIST, 0.001),
  front: new THREE.Vector3(0, 0, DEFAULT_DIST),
  side: new THREE.Vector3(DEFAULT_DIST, 0, 0),
}

const TARGET = new THREE.Vector3(0, 0, 0)

export function CameraController({ controlsRef }: CameraControllerProps) {
  const { camera } = useThree()
  const isIsometric = useRef(false)

  const getSpherical = useCallback(() => {
    const controls = controlsRef.current
    if (!controls) return null

    const offset = camera.position.clone().sub(controls.target)
    const spherical = new THREE.Spherical().setFromVector3(offset)
    return { spherical, target: controls.target.clone() }
  }, [camera, controlsRef])

  const setFromSpherical = useCallback(
    (spherical: THREE.Spherical, target: THREE.Vector3) => {
      const offset = new THREE.Vector3().setFromSpherical(spherical)
      camera.position.copy(target).add(offset)
      camera.lookAt(target)
      controlsRef.current?.update()
    },
    [camera, controlsRef],
  )

  const animateTo = useCallback(
    (targetPosition: THREE.Vector3, opts?: { fov?: number; zoom?: number; duration?: number }) => {
      const controls = controlsRef.current
      if (!controls) return

      const duration = opts?.duration ?? 300
      const startPosition = camera.position.clone()
      const startTarget = controls.target.clone()
      const startTime = performance.now()

      const isPerspective = camera instanceof THREE.PerspectiveCamera
      const startFov = isPerspective ? (camera as THREE.PerspectiveCamera).fov : 50
      const startZoom = camera.zoom

      const animate = () => {
        const elapsed = performance.now() - startTime
        const progress = Math.min(elapsed / duration, 1)
        const eased = 1 - Math.pow(1 - progress, 3) // easeOutCubic

        camera.position.lerpVectors(startPosition, targetPosition, eased)
        controls.target.lerpVectors(startTarget, TARGET, eased)

        if (isPerspective && opts?.fov !== undefined) {
          ;(camera as THREE.PerspectiveCamera).fov = startFov + (opts.fov - startFov) * eased
          camera.updateProjectionMatrix()
        }

        if (opts?.zoom !== undefined) {
          camera.zoom = startZoom + (opts.zoom - startZoom) * eased
          camera.updateProjectionMatrix()
        }

        controls.update()

        if (progress < 1) {
          requestAnimationFrame(animate)
        }
      }

      animate()
    },
    [camera, controlsRef],
  )

  const pan = useCallback(
    (dx: number, dy: number) => {
      const controls = controlsRef.current
      if (!controls) return

      // Build right and up vectors relative to the camera
      const right = new THREE.Vector3()
      const up = new THREE.Vector3()
      camera.getWorldDirection(new THREE.Vector3())
      right.setFromMatrixColumn(camera.matrixWorld, 0) // camera-local X
      up.setFromMatrixColumn(camera.matrixWorld, 1)     // camera-local Y

      const offset = right.multiplyScalar(dx).add(up.multiplyScalar(dy))

      camera.position.add(offset)
      controls.target.add(offset)
      controls.update()
    },
    [camera, controlsRef],
  )

  const resetView = useCallback(() => {
    isIsometric.current = false
    // Switch to perspective camera
    useVisualisationStore.getState().setOrthographic(false)
    // Small delay to let R3F swap camera, then animate
    requestAnimationFrame(() => {
      animateTo(PRESETS.default, { fov: DEFAULT_FOV })
    })
  }, [animateTo])

  const toggleIsometric = useCallback(() => {
    isIsometric.current = !isIsometric.current
    if (isIsometric.current) {
      // Switch to orthographic camera (true isometric — no perspective distortion)
      useVisualisationStore.getState().setOrthographic(true)
    } else {
      // Switch back to perspective
      useVisualisationStore.getState().setOrthographic(false)
      requestAnimationFrame(() => {
        animateTo(PRESETS.default, { fov: DEFAULT_FOV, duration: 400 })
      })
    }
  }, [animateTo])

  const topView = useCallback(() => {
    isIsometric.current = false
    animateTo(PRESETS.top)
  }, [animateTo])

  const frontView = useCallback(() => {
    isIsometric.current = false
    animateTo(PRESETS.front)
  }, [animateTo])

  const rotateHorizontal = useCallback(
    (direction: 1 | -1) => {
      const result = getSpherical()
      if (!result) return
      result.spherical.theta += ROTATION_STEP * direction
      setFromSpherical(result.spherical, result.target)
    },
    [getSpherical, setFromSpherical],
  )

  const rotateVertical = useCallback(
    (direction: 1 | -1) => {
      const result = getSpherical()
      if (!result) return
      result.spherical.phi = Math.max(
        0.1,
        Math.min(Math.PI - 0.1, result.spherical.phi + ROTATION_STEP * direction),
      )
      setFromSpherical(result.spherical, result.target)
    },
    [getSpherical, setFromSpherical],
  )

  const zoom = useCallback(
    (direction: 'in' | 'out') => {
      const result = getSpherical()
      if (!result) return
      if (direction === 'in') {
        result.spherical.radius = Math.max(5, result.spherical.radius * ZOOM_STEP)
      } else {
        result.spherical.radius = Math.min(500, result.spherical.radius / ZOOM_STEP)
      }
      setFromSpherical(result.spherical, result.target)
    },
    [getSpherical, setFromSpherical],
  )

  // Fly-to target from store
  const flyToTarget = useVisualisationStore((s) => s.flyToTarget)
  const clearFlyToTarget = useVisualisationStore((s) => s.clearFlyToTarget)

  useEffect(() => {
    if (!flyToTarget) return
    const controls = controlsRef.current
    if (!controls) return

    const targetPos = new THREE.Vector3(flyToTarget.x, flyToTarget.y, flyToTarget.z)
    const startTarget = controls.target.clone()
    const startPosition = camera.position.clone()

    // Compute a camera position offset from the fly-to target
    const offset = camera.position.clone().sub(controls.target)
    const cameraTarget = targetPos.clone().add(offset.normalize().multiplyScalar(8))

    const duration = 500
    const startTime = performance.now()

    const animate = () => {
      const elapsed = performance.now() - startTime
      const progress = Math.min(elapsed / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)

      camera.position.lerpVectors(startPosition, cameraTarget, eased)
      controls.target.lerpVectors(startTarget, targetPos, eased)
      controls.update()

      if (progress < 1) {
        requestAnimationFrame(animate)
      } else {
        clearFlyToTarget()
      }
    }

    animate()
  }, [flyToTarget, camera, controlsRef, clearFlyToTarget])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement ||
        event.target instanceof HTMLSelectElement
      ) {
        return
      }

      switch (event.key.toLowerCase()) {
        case 'r':
          event.preventDefault()
          resetView()
          break
        case 'i':
          event.preventDefault()
          toggleIsometric()
          break
        case 'arrowleft':
          event.preventDefault()
          if (event.shiftKey) {
            pan(-PAN_STEP, 0)
          } else {
            rotateHorizontal(-1)
          }
          break
        case 'arrowright':
          event.preventDefault()
          if (event.shiftKey) {
            pan(PAN_STEP, 0)
          } else {
            rotateHorizontal(1)
          }
          break
        case 'arrowup':
          event.preventDefault()
          if (event.shiftKey) {
            pan(0, PAN_STEP)
          } else {
            rotateVertical(-1)
          }
          break
        case 'arrowdown':
          event.preventDefault()
          if (event.shiftKey) {
            pan(0, -PAN_STEP)
          } else {
            rotateVertical(1)
          }
          break
        case '+':
        case '=':
          event.preventDefault()
          zoom('in')
          break
        case '-':
        case '_':
          event.preventDefault()
          zoom('out')
          break
        case 'home':
          event.preventDefault()
          topView()
          break
        case 'end':
          event.preventDefault()
          frontView()
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [resetView, toggleIsometric, rotateHorizontal, rotateVertical, zoom, topView, frontView, pan])

  return null
}
