import { useRef, useEffect, useMemo } from 'react'
import { Canvas, useThree, useFrame } from '@react-three/fiber'
import { OrbitControls, Html, PerspectiveCamera, OrthographicCamera } from '@react-three/drei'
import * as THREE from 'three'
import { cn } from '@/design/cn'
import { useVisualisationStore } from '@/stores/visualisationStore'
import { useAnnotationStore } from '@/stores/annotationStore'
import {
  useFilteredPoints,
  usePointColors,
  useScaleFactor,
  useNearestNeighbors,
  useDiagnosisColorMap,
} from '@/stores/visualisationSelectors'
import { InstancedPoints } from './InstancedPoints'
import { CameraController } from './CameraController'
import { TribeVisualization } from './TribeVisualization'
import { NeighborLines } from './NeighborLines'
import { UploadBeacons } from './UploadBeacons'
import { DiagnosisLabels3D } from './DiagnosisLabels3D'
import { NO_RECORDING_NA, analyzeHref, pointIdLabel, recordingIdOf } from './recordingLink'
import type { EnrichedProjectionPoint } from '@/types/projection'

// =============================================================================
// UniverseScene — main exported component (wraps Canvas)
// =============================================================================

interface UniverseSceneProps {
  className?: string
  screenPositionsRef?: React.RefObject<Map<string, { x: number; y: number }>>
}

export function UniverseScene({ className, screenPositionsRef }: UniverseSceneProps) {
  return (
    <div className={cn('w-full h-full bg-slate-50', className)}>
      <Canvas
        gl={{ antialias: true, preserveDrawingBuffer: true }}
      >
        <ClearColor />
        <ambientLight intensity={0.6} />
        <pointLight position={[10, 10, 10]} intensity={0.8} />
        <pointLight position={[-10, -10, -10]} intensity={0.4} />

        <UniverseContent screenPositionsRef={screenPositionsRef} />
      </Canvas>
    </div>
  )
}

/** Set the WebGL clear color to match the light theme background */
function ClearColor() {
  const { gl } = useThree()
  useEffect(() => {
    gl.setClearColor('#F8FAFC')
  }, [gl])
  return null
}

// =============================================================================
// PulsatingRing — animated selection ring for a single patient
// =============================================================================

function PulsatingRing({ position, color }: { position: [number, number, number]; color: string }) {
  const meshRef = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    if (!meshRef.current) return
    const t = clock.elapsedTime * 2
    meshRef.current.scale.setScalar(1 + Math.sin(t) * 0.12)
    ;(meshRef.current.material as THREE.MeshBasicMaterial).opacity = 0.65 + Math.sin(t) * 0.15
  })

  return (
    <mesh ref={meshRef} position={position} rotation={[Math.PI / 2, 0, 0]}>
      <ringGeometry args={[0.2, 0.25, 32]} />
      <meshBasicMaterial color={color} transparent opacity={0.8} side={THREE.DoubleSide} />
    </mesh>
  )
}

// =============================================================================
// HoverDetector — manual raycaster on window events
// Bypasses R3F event system so hover + clicks work regardless of DOM overlays
// =============================================================================

function HoverDetector() {
  const { camera, scene, gl } = useThree()
  const setHoveredPatient = useVisualisationStore((s) => s.setHoveredPatient)
  const togglePatient = useVisualisationStore((s) => s.togglePatient)
  const raycaster = useMemo(() => new THREE.Raycaster(), [])
  const pointer = useMemo(() => new THREE.Vector2(), [])

  useEffect(() => {
    const container = gl.domElement.parentElement!

    // Shared helper: raycast and find the first InstancedMesh hit with patientIds
    const raycastPatient = (e: MouseEvent): string | null => {
      // Ignore events that originated outside the canvas (e.g. overlay panels)
      if (!container.contains(e.target as Node)) return null

      const rect = container.getBoundingClientRect()
      if (
        e.clientX < rect.left || e.clientX > rect.right ||
        e.clientY < rect.top || e.clientY > rect.bottom
      ) return null

      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1

      raycaster.setFromCamera(pointer, camera)
      const intersects = raycaster.intersectObjects(scene.children, true)

      for (const hit of intersects) {
        if (
          hit.object instanceof THREE.InstancedMesh &&
          hit.instanceId !== undefined
        ) {
          const ids = hit.object.userData.patientIds as string[] | undefined
          if (ids && hit.instanceId < ids.length) return ids[hit.instanceId] ?? null
        }
      }
      return null
    }

    const onPointerMove = (e: PointerEvent) => {
      setHoveredPatient(raycastPatient(e))
    }

    const onClick = (e: MouseEvent) => {
      const patientId = raycastPatient(e)
      if (patientId) {
        togglePatient(patientId)
      }
    }

    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('click', onClick)
    return () => {
      window.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('click', onClick)
    }
  }, [camera, scene, gl, setHoveredPatient, togglePatient, raycaster, pointer])

  return null
}

// =============================================================================
// SelectedPatientCard — floating card near selected point in 3D space
// =============================================================================


function SelectedPatientCard({
  point,
  scaleFactor,
  onClose,
}: {
  point: EnrichedProjectionPoint
  scaleFactor: number
  onClose: () => void
}) {
  const isPinned = useAnnotationStore((s) => s.isPinned(point.patientId))
  const togglePinned = useAnnotationStore((s) => s.togglePin)

  const { age, sex } = point

  // Open build: the hover card used to fetch GET /api/patients/{id} for a
  // clinical-history blurb. Universe points are de-identified recordings that
  // are not in any patient store, so that call 404'd on every hover and the
  // result was always swallowed. Both the fetch and the disclosure it fed are
  // gone.

  return (
    <Html
      position={[
        point.x * scaleFactor + 2.5,
        point.y * scaleFactor + 2.0,
        point.z * scaleFactor,
      ]}
      zIndexRange={[10, 10]}
      style={{ pointerEvents: 'auto' }}
    >
      <div
        className="w-[240px] bg-white rounded-lg shadow-xl border border-gray-200 text-xs"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2 border-b border-gray-100">
          <span className="font-mono font-semibold text-slate-800 truncate" title={pointIdLabel(point)}>
            {pointIdLabel(point)}
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => togglePinned(point.patientId)}
              className={cn(
                'p-0.5 rounded transition-colors',
                isPinned ? 'text-blue-500' : 'text-slate-400 hover:text-slate-600',
              )}
              title={isPinned ? 'Unpin from hotbar' : 'Pin to hotbar'}
            >
              <svg className="w-3.5 h-3.5" fill={isPinned ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
              </svg>
            </button>
            <button
              onClick={onClose}
              className="p-0.5 text-slate-400 hover:text-slate-600 rounded"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div className="px-3 py-2 space-y-2">
          {/* Badges */}
          <div className="flex flex-wrap items-center gap-1.5">
            {point.cohort && (
              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-600">
                {point.cohort}
              </span>
            )}
            {age != null && <span className="text-slate-500">{age}y</span>}
            {sex && <span className="text-slate-500">{sex}</span>}
          </div>

          {/* Correctness indicators */}
          <div className="flex items-center gap-2">
            {point.xgbCorrect != null && (
              <span className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-medium ${point.xgbCorrect ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700'}`}>
                {point.xgbCorrect ? '✓' : '✗'} Raw score
              </span>
            )}
            {point.pipelineCorrect != null && (
              <span className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-medium ${point.pipelineCorrect ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700'}`}>
                {point.pipelineCorrect ? '✓' : '✗'} After rules
              </span>
            )}
          </div>

          {/* Gold labels */}
          {point.diagnoses.length > 0 && (
            <div>
              <div className="text-[10px] text-slate-400 mb-0.5">Gold Labels</div>
              <div className="flex flex-wrap gap-1">
                {point.diagnoses.map((dx) => (
                  <span key={dx} className="px-1.5 py-0.5 rounded bg-slate-100 text-[10px] text-slate-700">{dx}</span>
                ))}
              </div>
            </div>
          )}

          {/* Open in Analyze — only rows that ship a waveform can be opened.
              This card renders inside the R3F canvas, which is its own React
              root, so react-router's context does not reach it: navigate with
              a plain location assignment rather than useNavigate(). */}
          <button
            onClick={() => {
              const rid = recordingIdOf(point)
              if (rid) window.location.assign(analyzeHref(rid))
            }}
            disabled={recordingIdOf(point) === null}
            title={recordingIdOf(point) === null ? NO_RECORDING_NA : undefined}
            className={`block w-full text-center px-2 py-1.5 rounded bg-blue-600 text-white text-[10px] font-medium transition-colors ${recordingIdOf(point) === null ? 'opacity-50 cursor-not-allowed' : 'hover:bg-blue-700'}`}
          >
            Open in Analyze
          </button>
        </div>
      </div>
    </Html>
  )
}

// =============================================================================
// AxisLabels3D — floating labels along axes for PCA / clinical projections
// =============================================================================

function AxisLabels3D({
  axisLabels,
  scaleFactor,
  points,
}: {
  axisLabels: string[]
  scaleFactor: number
  points: EnrichedProjectionPoint[]
}) {
  // Compute the max extent per axis so labels sit at the edge of the point cloud
  const extents = useMemo(() => {
    let maxX = 1, maxY = 1, maxZ = 1
    for (const p of points) {
      maxX = Math.max(maxX, Math.abs(p.x))
      maxY = Math.max(maxY, Math.abs(p.y))
      maxZ = Math.max(maxZ, Math.abs(p.z))
    }
    return {
      x: (maxX + maxX * 0.15) * scaleFactor,
      y: (maxY + maxY * 0.15) * scaleFactor,
      z: (maxZ + maxZ * 0.15) * scaleFactor,
    }
  }, [points, scaleFactor])

  const labelStyle: React.CSSProperties = {
    pointerEvents: 'none',
    whiteSpace: 'nowrap',
    maxWidth: 260,
  }

  return (
    <group>
      {axisLabels[0] && (
        <Html position={[extents.x, 0, 0]} center zIndexRange={[5, 5]} style={labelStyle}>
          <div className="px-2 py-1 rounded bg-gray-900/70 backdrop-blur-sm text-white text-[10px] leading-tight">
            {axisLabels[0]}
          </div>
        </Html>
      )}
      {axisLabels[1] && (
        <Html position={[0, extents.y, 0]} center zIndexRange={[5, 5]} style={labelStyle}>
          <div className="px-2 py-1 rounded bg-gray-900/70 backdrop-blur-sm text-white text-[10px] leading-tight">
            {axisLabels[1]}
          </div>
        </Html>
      )}
      {axisLabels[2] && (
        <Html position={[0, 0, extents.z]} center zIndexRange={[5, 5]} style={labelStyle}>
          <div className="px-2 py-1 rounded bg-gray-900/70 backdrop-blur-sm text-white text-[10px] leading-tight">
            {axisLabels[2]}
          </div>
        </Html>
      )}
    </group>
  )
}

// =============================================================================
// UniverseContent — inside Canvas, reads from store
// =============================================================================

function UniverseContent({
  screenPositionsRef,
}: {
  screenPositionsRef?: React.RefObject<Map<string, { x: number; y: number }>>
}) {
  const { gl, scene, camera, size } = useThree()
  const controlsRef = useRef<any>(null)

  // Store state
  const selectedPatients = useVisualisationStore((s) => s.selectedPatients)
  const selectedPatientId = useVisualisationStore((s) => s.selectedPatientId)
  const hoveredPatient = useVisualisationStore((s) => s.hoveredPatient)
  const patientLabels = useVisualisationStore((s) => s.patientLabels)
  const viewerSettings = useVisualisationStore((s) => s.viewerSettings)
  const showNeighbors = useVisualisationStore((s) => s.showNeighbors)
  const uploadedPatientIds = useVisualisationStore((s) => s.uploadedPatientIds)
  const screenshotRequested = useVisualisationStore((s) => s.screenshotRequested)
  const exportFormat = useVisualisationStore((s) => s.exportFormat)
  const clearExportRequest = useVisualisationStore((s) => s.clearExportRequest)
  const projectionMethod = useVisualisationStore((s) => s.method)
  const projectionData = useVisualisationStore((s) => s.projectionData)
  const isOrthographic = useVisualisationStore((s) => s.isOrthographic)

  // Store actions
  const selectPatient = useVisualisationStore((s) => s.selectPatient)
  const clearSelection = useVisualisationStore((s) => s.clearSelection)
  const deselectPatient = useVisualisationStore((s) => s.deselectPatient)
  const dismissCard = useVisualisationStore((s) => s.dismissCard)
  const clearScreenshotRequest = useVisualisationStore((s) => s.clearScreenshotRequest)

  // Selectors
  const filteredPoints = useFilteredPoints()
  const pointColors = usePointColors()
  const scaleFactor = useScaleFactor()
  const neighbors = useNearestNeighbors()
  const diagnosisColorMap = useDiagnosisColorMap()

  // Update screen positions for lasso overlay
  useEffect(() => {
    if (!screenPositionsRef?.current) return
    const map = screenPositionsRef.current
    map.clear()
    const vec = new THREE.Vector3()
    for (const p of filteredPoints) {
      vec.set(p.x * scaleFactor, p.y * scaleFactor, p.z * scaleFactor)
      vec.project(camera)
      const x = (vec.x * 0.5 + 0.5) * size.width
      const y = (-vec.y * 0.5 + 0.5) * size.height
      map.set(p.patientId, { x, y })
    }
  }, [filteredPoints, scaleFactor, camera, size, screenPositionsRef])

  // Screenshot handler
  useEffect(() => {
    if (!screenshotRequested) return
    gl.render(scene, camera)
    const dataUrl = gl.domElement.toDataURL('image/png')
    const link = document.createElement('a')
    link.download = `universe-screenshot-${Date.now()}.png`
    link.href = dataUrl
    link.click()
    clearScreenshotRequest()
  }, [screenshotRequested, gl, scene, camera, clearScreenshotRequest])

  // Data export handler (CSV / JSON)
  useEffect(() => {
    if (!exportFormat || exportFormat === 'png') return
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, '')
    const rows = filteredPoints.map((p) => ({
      recordingId: recordingIdOf(p) ?? '',
      displayId: p.displayId ?? p.patientId,
      x: p.x,
      y: p.y,
      z: p.z,
      primaryDiagnosis: p.primaryDiagnosis,
      diagnoses: (p.diagnoses ?? []).join(';'),
      cohort: p.cohort,
      source: p.source ?? '',
      xgbCorrect: p.xgbCorrect ?? '',
      age: p.age ?? '',
      sex: p.sex ?? '',
    }))

    let blob: Blob
    let ext: string
    if (exportFormat === 'csv') {
      const headers = Object.keys(rows[0] || {}).join(',')
      const lines = rows.map((r) =>
        Object.values(r)
          .map((v) => (typeof v === 'string' && v.includes(',') ? `"${v}"` : v))
          .join(','),
      )
      blob = new Blob([headers + '\n' + lines.join('\n')], { type: 'text/csv' })
      ext = 'csv'
    } else {
      blob = new Blob([JSON.stringify(rows, null, 2)], { type: 'application/json' })
      ext = 'json'
    }

    const link = document.createElement('a')
    link.download = `universe-export-${timestamp}.${ext}`
    link.href = URL.createObjectURL(blob)
    link.click()
    URL.revokeObjectURL(link.href)
    clearExportRequest()
  }, [exportFormat, filteredPoints, clearExportRequest])

  // Cursor style for hover
  useEffect(() => {
    gl.domElement.style.cursor = hoveredPatient ? 'pointer' : 'default'
  }, [hoveredPatient, gl])

  // Escape key to clear selection
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        clearSelection()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [clearSelection])

  // Hovered point data for tooltip
  const hoveredPoint = useMemo<EnrichedProjectionPoint | null>(() => {
    if (!hoveredPatient) return null
    return filteredPoints.find((p) => p.patientId === hoveredPatient) ?? null
  }, [hoveredPatient, filteredPoints])

  // All selected points for rings, plus primary for neighbor lines
  const selectedPoints = useMemo<EnrichedProjectionPoint[]>(() => {
    if (selectedPatients.size === 0) return []
    return filteredPoints.filter((p) => selectedPatients.has(p.patientId))
  }, [selectedPatients, filteredPoints])

  const selectedPoint = selectedPatientId
    ? selectedPoints.find((p) => p.patientId === selectedPatientId) ?? null
    : null

  const { display } = viewerSettings

  // Both cameras start at equal x=y=z = 17.5 (true isometric corner)
  const isoPos: [number, number, number] = [17.5, 17.5, 17.5]

  return (
    <group>
      {/* Dual cameras — only one is active at a time via makeDefault */}
      <PerspectiveCamera
        makeDefault={!isOrthographic}
        position={isoPos}
        fov={50}
        near={0.1}
        far={2000}
      />
      <OrthographicCamera
        makeDefault={isOrthographic}
        position={isoPos}
        left={-20}
        right={20}
        top={15}
        bottom={-15}
        zoom={1}
        near={-1000}
        far={2000}
      />

      <OrbitControls
        ref={controlsRef}
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={isOrthographic ? undefined : 5}
        maxDistance={isOrthographic ? undefined : 500}
        minZoom={isOrthographic ? 0.2 : undefined}
        maxZoom={isOrthographic ? 10 : undefined}
        autoRotate={false}
      />

      <CameraController controlsRef={controlsRef} />
      <HoverDetector />

      {/* Grid (toggled via settings) */}
      {display.showGrid && (
        <gridHelper args={[30, 30, '#CBD5E1', '#E2E8F0']} position={[0, -8, 0]} />
      )}

      <InstancedPoints
        points={filteredPoints}
        pointColors={pointColors}
        scaleFactor={scaleFactor}
        selectedPatients={selectedPatients}
        hoveredPatient={hoveredPatient}
        neighbors={neighbors}
        settings={viewerSettings}
      />

      {/* Diagnosis labels at cluster centroids */}
      <DiagnosisLabels3D
        points={filteredPoints}
        scaleFactor={scaleFactor}
        visible={display.showDiagnosisLabels}
      />

      {/* Tribe visualization (convex hulls + centroids) */}
      <TribeVisualization
        points={filteredPoints}
        scaleFactor={scaleFactor}
        settings={viewerSettings}
        diagnosisColorMap={diagnosisColorMap}
      />

      {/* Neighbor connection lines */}
      {showNeighbors && selectedPoint && neighbors.length > 0 && (
        <NeighborLines
          selectedPoint={selectedPoint}
          neighbors={neighbors}
          scaleFactor={scaleFactor}
        />
      )}

      {/* Upload patient beacons */}
      <UploadBeacons
        points={filteredPoints}
        uploadedPatientIds={uploadedPatientIds}
        scaleFactor={scaleFactor}
      />

      {/* Open build: the density-cloud isosurface layer is gone. It fetched
          GET /api/projections/cv/isosurfaces, which needs the CV runtime
          artifacts the open backend does not carry. */}

      {/* Pulsating selection rings + floating labels for all selected patients */}
      {selectedPoints.map((pt) => {
        const pos: [number, number, number] = [pt.x * scaleFactor, pt.y * scaleFactor, pt.z * scaleFactor]
        const customLabel = patientLabels.get(pt.patientId)
        return (
          <group key={pt.patientId}>
            <PulsatingRing position={pos} color="#2743B8" />
            {display.showPinnedLabels && (
              <Html
                position={[pos[0], pos[1] + 0.55, pos[2]]}
                center
                distanceFactor={isOrthographic ? undefined : 14}
                zIndexRange={[7, 7]}
                style={{ pointerEvents: 'auto', cursor: 'pointer' }}
              >
                <div
                  onClick={() => selectPatient(pt.patientId)}
                  className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-[#2743B8]/90 backdrop-blur-sm shadow-md whitespace-nowrap cursor-pointer hover:bg-[#2743B8] transition-colors"
                >
                  <span className="text-[10px] font-semibold text-white tracking-wide">
                    {customLabel || pointIdLabel(pt)}
                  </span>
                  {customLabel && (
                    <span className="text-[9px] text-white/60 font-mono">{pointIdLabel(pt)}</span>
                  )}
                </div>
              </Html>
            )}
          </group>
        )
      })}

      {/* Hover tooltip — only when no patient selected or hovering a different patient */}
      {hoveredPoint && (!selectedPoint || hoveredPoint.patientId !== selectedPoint.patientId) && (
        <Html
          position={[
            hoveredPoint.x * scaleFactor,
            hoveredPoint.y * scaleFactor + 0.8,
            hoveredPoint.z * scaleFactor,
          ]}
          center
          zIndexRange={[10, 10]}
          style={{ pointerEvents: 'none' }}
        >
          <div className="px-3 py-2 bg-white text-gray-900 text-xs rounded-lg shadow-lg border border-gray-200 whitespace-nowrap max-w-[280px]">
            <div className="flex items-center gap-1.5">
              {/* Both ids when the record has two — HR07445 · SS-00936 — so a
                  search hit never looks like a jump to another patient. */}
              <span className="font-semibold font-mono">{pointIdLabel(hoveredPoint)}</span>
              {(hoveredPoint.age != null || hoveredPoint.sex) && (
                <span className="text-gray-400">
                  ({hoveredPoint.age ?? '?'} / {hoveredPoint.sex ?? '?'})
                </span>
              )}
            </div>
            <div className="mt-0.5 text-gray-500 whitespace-normal">
              <span className="text-gray-400">Gold: </span>
              {hoveredPoint.diagnoses.join(', ') || 'None'}
            </div>
          </div>
        </Html>
      )}

      {/* Selected patient card — anchored near the point */}
      {selectedPatientId && selectedPoint && (
        <SelectedPatientCard
          point={selectedPoint}
          scaleFactor={scaleFactor}
          onClose={() => {
            const pinned = useAnnotationStore.getState().isPinned(selectedPoint.patientId)
            if (pinned) dismissCard()
            else deselectPatient(selectedPoint.patientId)
          }}
        />
      )}

      {/* PCA interpretable axis labels */}
      {projectionMethod === 'pca' &&
        projectionData?.axisLabels &&
        projectionData.axisLabels.length >= 3 && (
          <AxisLabels3D
            axisLabels={projectionData.axisLabels}
            scaleFactor={scaleFactor}
            points={filteredPoints}
          />
        )}
    </group>
  )
}
