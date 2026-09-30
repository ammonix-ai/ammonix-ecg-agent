import { useState, useEffect, useRef } from 'react'
import { RhythmStripSection } from './RhythmStripSection'
import { LeadGrid } from './LeadGrid'
import { CalibrationPulse, getCalibrationDimensions } from '../CalibrationPulse'
import type { LeadName, GridMode, ECGData } from '@/types/ecg'

interface TwelveLeadViewProps {
  ecgData: ECGData
  onExpandLead: (leadName: LeadName) => void
  skipAnimation?: boolean
  windowPosition?: number
  onWindowPositionChange?: (position: number) => void
  onAnimationComplete?: () => void
  gridMode?: GridMode
  onGridModeChange?: (mode: GridMode) => void
  rhythmStripLead?: LeadName
  onRhythmStripLeadChange?: (lead: LeadName) => void
  className?: string
}

const LOADING_DURATION = 1000
const ANIMATION_DURATION = 10000

/**
 * TwelveLeadView — Full 12-lead display with rhythm strip
 *
 * Phases: loading -> playing (sweep animation) -> complete (interactive)
 * Rhythm strip shows full recording with draggable window bracket.
 * Lead grid shows the 10-second window.
 */
export function TwelveLeadView({
  ecgData,
  onExpandLead,
  skipAnimation = false,
  windowPosition: controlledPos,
  onWindowPositionChange,
  onAnimationComplete,
  gridMode: controlledGrid,
  onGridModeChange,
  rhythmStripLead,
  onRhythmStripLeadChange,
  className,
}: TwelveLeadViewProps) {
  const [localGrid, setLocalGrid] = useState<GridMode>('full')
  const gridMode = controlledGrid ?? localGrid
  // @ts-expect-error TODO: wire setGridMode to grid mode toggle UI
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const setGridMode = onGridModeChange ?? setLocalGrid

  const [phase, setPhase] = useState<'loading' | 'playing' | 'complete'>(
    skipAnimation ? 'complete' : 'loading'
  )
  const [progress, setProgress] = useState(skipAnimation ? 1 : 0)
  const animRef = useRef<number | null>(null)
  const notifiedRef = useRef(skipAnimation)

  const [internalPos, setInternalPos] = useState(0)
  const windowPosition = controlledPos ?? internalPos
  const setWindowPosition = onWindowPositionChange ?? setInternalPos

  const totalDuration = ecgData.numSamples / ecgData.samplingRate
  const windowSize = 10
  const windowStart = windowPosition
  const windowEnd = Math.min(windowPosition + windowSize, totalDuration)

  const currentRhythmLead = rhythmStripLead ?? 'II'
  const rhythmSamples = ecgData.leads[currentRhythmLead] || []

  // Notify parent on animation complete
  useEffect(() => {
    if (phase === 'complete' && !notifiedRef.current) {
      notifiedRef.current = true
      onAnimationComplete?.()
    }
  }, [phase, onAnimationComplete])

  // Run animation
  useEffect(() => {
    if (skipAnimation) {
      setPhase('complete')
      setProgress(1)
      return
    }

    notifiedRef.current = false
    setPhase('loading')
    setProgress(0)

    const timer = setTimeout(() => {
      setPhase('playing')
      const t0 = performance.now()

      const animate = (now: number) => {
        const p = Math.min((now - t0) / ANIMATION_DURATION, 1)
        setProgress(p)
        if (p < 1) {
          animRef.current = requestAnimationFrame(animate)
        } else {
          setPhase('complete')
        }
      }
      animRef.current = requestAnimationFrame(animate)
    }, LOADING_DURATION)

    return () => {
      clearTimeout(timer)
      if (animRef.current) cancelAnimationFrame(animRef.current)
    }
  }, [ecgData.patientId, skipAnimation])

  const isInteractive = phase === 'complete'

  if (phase === 'loading') {
    return (
      <div className={`flex-1 flex items-center justify-center ${className ?? ''}`}>
        <div className="flex flex-col items-center gap-3 text-gray-600">
          <div className="w-8 h-8 border-[3px] border-indigo-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm font-medium">Preparing ECG...</span>
        </div>
      </div>
    )
  }

  const calibDims = getCalibrationDimensions()

  return (
    <div className={`flex flex-col bg-white border border-gray-200 overflow-hidden ${className ?? ''}`}>
      {/* IEC 60601-2-25 calibration reference */}
      <div className="flex items-center gap-2 px-3 py-1 border-b border-[#FFCCCC] bg-[#FFF5F5] shrink-0">
        <CalibrationPulse heightPx={calibDims.height} widthPx={calibDims.width} showLabel />
        <span className="text-[10px] text-gray-400 font-medium tracking-wide">IEC 60601-2-25</span>
      </div>

      {/* Rhythm strip */}
      <RhythmStripSection
        samples={rhythmSamples}
        samplingRate={ecgData.samplingRate}
        leadName={currentRhythmLead}
        onLeadChange={onRhythmStripLeadChange}
        totalDuration={totalDuration}
        animationProgress={progress}
        windowSize={windowSize}
        windowPosition={windowPosition}
        onWindowPositionChange={setWindowPosition}
        interactive={isInteractive}
        gridMode={gridMode}
        traceHeight={120}
      />

      {/* 12-lead grid */}
      <LeadGrid
        ecgData={ecgData}
        windowStart={windowStart}
        windowEnd={windowEnd}
        animationProgress={progress}
        onLeadClick={onExpandLead}
        interactive={isInteractive}
        gridMode={gridMode}
        activeRhythmLead={currentRhythmLead}
        className="flex-1 min-h-0"
      />
    </div>
  )
}
