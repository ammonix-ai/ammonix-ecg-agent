import { useCallback, useRef, useState, useEffect } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { ECGGridPaper } from '../ECGGridPaper'
import { LeadTraceAnimated } from './LeadTraceAnimated'
import { RowYAxis } from './RowYAxis'
import { WindowBracket } from './WindowBracket'
import { LEAD_GROUPS, LARGE_BOX_PX, Y_AXIS_WIDTH } from '@/types/ecg'
import type { LeadName, GridMode, SampleArray } from '@/types/ecg'

interface RhythmStripSectionProps {
  // C0.1: SampleArray = number[] | Float32Array. Accepts both paths.
  samples: SampleArray
  samplingRate: number
  leadName?: LeadName
  totalDuration: number
  animationProgress?: number
  windowSize?: number
  windowPosition?: number
  onWindowPositionChange?: (position: number) => void
  onLeadChange?: (lead: LeadName) => void
  interactive?: boolean
  gridMode?: GridMode
  traceHeight?: number
  className?: string
}

/**
 * RhythmStripSection — Continuous lead strip with draggable window bracket
 *
 * Shows the full recording for one lead (default Lead II) with a time axis
 * and a bracket indicating the window displayed in the 12-lead grid.
 */
export function RhythmStripSection({
  samples,
  samplingRate,
  leadName = 'II',
  totalDuration,
  animationProgress = 1,
  windowSize = 10,
  windowPosition = 0,
  onWindowPositionChange,
  onLeadChange,
  interactive = true,
  gridMode = 'full',
  traceHeight = 80,
  className,
}: RhythmStripSectionProps) {
  const bracketInteractive = interactive && animationProgress >= 1

  const measureRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(0)

  useEffect(() => {
    if (!measureRef.current) return
    const el = measureRef.current
    const update = () => setContainerWidth(el.clientWidth)
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // The strip stretches `totalDuration` seconds across its width, and the
  // WindowBracket below draws a tick every second as a fraction of that
  // width — so one second must be a whole number of bold boxes or the ticks
  // drift off the gridlines. Pick an integer boxes-per-second near the
  // nominal box size and derive the box size so the boxes fill the width
  // exactly (the strip then starts and ends on a bold line).
  let boxPx = 0
  let contentWidth: number | undefined
  if (containerWidth > 0 && totalDuration > 0) {
    const boxesPerSecond = Math.max(1, Math.round(containerWidth / totalDuration / LARGE_BOX_PX))
    const totalBoxes = Math.max(1, Math.round(totalDuration * boxesPerSecond))
    boxPx = containerWidth / totalBoxes
    contentWidth = totalBoxes * boxPx
  }

  // Quantize to even bold boxes so 0mV sits on a gridline
  const heightBoxPx = boxPx > 0 ? boxPx : LARGE_BOX_PX
  const boldBoxes = Math.floor(traceHeight / heightBoxPx)
  const evenBoxes = boldBoxes - (boldBoxes % 2)
  const quantizedHeight = Math.max(2, evenBoxes) * heightBoxPx

  const handlePrev = useCallback(() => {
    if (!onLeadChange) return
    const all = LEAD_GROUPS.all
    const idx = all.indexOf(leadName)
    onLeadChange(all[idx <= 0 ? all.length - 1 : idx - 1]!)
  }, [leadName, onLeadChange])

  const handleNext = useCallback(() => {
    if (!onLeadChange) return
    const all = LEAD_GROUPS.all
    const idx = all.indexOf(leadName)
    onLeadChange(all[idx >= all.length - 1 ? 0 : idx + 1]!)
  }, [leadName, onLeadChange])

  return (
    <div className={`flex flex-col pt-2 ${className ?? ''}`}>
      {/* Lead selector */}
      <div className="flex justify-center items-center gap-2 py-1 mb-1">
        <button
          onClick={handlePrev}
          disabled={!onLeadChange}
          className="p-0 text-slate-400 hover:text-slate-600 transition-colors disabled:opacity-30"
        >
          <ChevronLeft className="w-3.5 h-3.5" />
        </button>
        <span className="text-[10px] font-bold text-indigo-700 min-w-[24px] text-center">
          {leadName}
        </span>
        <button
          onClick={handleNext}
          disabled={!onLeadChange}
          className="p-0 text-slate-400 hover:text-slate-600 transition-colors disabled:opacity-30"
        >
          <ChevronRight className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Y-axis + Trace */}
      <div className="flex">
        <div className="flex-shrink-0 border-r border-gray-200"
          style={{ width: Y_AXIS_WIDTH, height: quantizedHeight }}>
          <RowYAxis rowHeightPx={quantizedHeight} boxPx={boxPx > 0 ? boxPx : undefined} className="h-full" />
        </div>

        <div ref={measureRef} className="flex-1 min-w-0">
          <ECGGridPaper gridMode={gridMode} boxSize={boxPx > 0 ? boxPx : undefined}
            style={{ width: contentWidth, height: quantizedHeight }}>
            <LeadTraceAnimated
              leadName={leadName}
              samples={samples}
              samplingRate={samplingRate}
              startTime={0}
              endTime={totalDuration}
              animationProgress={animationProgress}
              boxPx={boxPx > 0 ? boxPx : undefined}
            />
          </ECGGridPaper>
        </div>

        <div className="flex-shrink-0" style={{ width: Y_AXIS_WIDTH }} />
      </div>

      {/* Window bracket */}
      <div className="flex pb-1">
        <div className="flex-shrink-0" style={{ width: Y_AXIS_WIDTH }} />
        {/* shrink-0: the bracket is a flex item, the paper above is not.
            When a scrollbar lands after first measure, a shrinking bracket
            compresses its percent-placed second ticks off the paper's
            gridlines while the paper keeps its px width. */}
        <WindowBracket
          totalDuration={totalDuration}
          windowSize={windowSize}
          windowPosition={windowPosition}
          onPositionChange={onWindowPositionChange || (() => {})}
          interactive={bracketInteractive}
          height={28}
          className="shrink-0"
          style={{ width: contentWidth }}
        />
        <div className="flex-shrink-0" style={{ width: Y_AXIS_WIDTH }} />
      </div>
    </div>
  )
}
