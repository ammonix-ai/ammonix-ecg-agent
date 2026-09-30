import { useRef, useState, useEffect } from 'react'
import { ECGGridPaper } from '../ECGGridPaper'
import { LeadFrame } from '../LeadFrame'
import { LeadTraceAnimated } from './LeadTraceAnimated'
import { RowYAxis } from './RowYAxis'
import { ColumnXAxis } from './ColumnXAxis'
import { LEAD_GRID_2x6, LARGE_BOX_PX, FRAME_GAP_PX, Y_AXIS_WIDTH } from '@/types/ecg'
import type { LeadName, GridMode, ECGData } from '@/types/ecg'

interface LeadGridProps {
  ecgData: ECGData
  windowStart: number
  windowEnd: number
  animationProgress?: number
  onLeadClick?: (leadName: LeadName) => void
  /**
   * Accepted for call-site compatibility but no longer gates clicking — see
   * canClick below. Kept so TwelveLeadView does not need changing and so the
   * intent is recorded rather than silently dropped.
   */
  interactive?: boolean
  gridMode?: GridMode
  activeRhythmLead?: LeadName
  className?: string
}

/**
 * LeadGrid — 2x6 lead arrangement on clinical grid paper
 *
 * Quantizes column widths and row heights to bold-gridline multiples
 * so the grid paper aligns correctly. Y-axes on the left, X-axes below.
 */
export function LeadGrid({
  ecgData,
  windowStart,
  windowEnd,
  animationProgress = 1,
  onLeadClick,
  interactive: _interactive = true,
  gridMode = 'full',
  activeRhythmLead,
  className,
}: LeadGridProps) {
  const layout = LEAD_GRID_2x6
  const numRows = layout.length
  const numCols = layout[0]!.length

  const measureRef = useRef<HTMLDivElement>(null)
  const [dims, setDims] = useState({ width: 0, height: 0 })

  useEffect(() => {
    if (!measureRef.current) return
    const el = measureRef.current
    const update = () => setDims({ width: el.clientWidth, height: el.clientHeight })
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // The grid paper is the time base: one second must be a whole number of
  // bold boxes, or the second ticks under the columns drift off the
  // gridlines (each column shows `duration` seconds stretched to its width,
  // and the ticks are laid out as fractions of that width). So instead of
  // quantizing the column to whole boxes of a FIXED size, pick an integer
  // boxes-per-second near the nominal box size and derive the box size so
  // the columns (plus one-box gaps) fill the available width exactly.
  const gridAvailW = dims.width - 2 * Y_AXIS_WIDTH
  const duration = windowEnd - windowStart
  let boxPx = 0
  let qColW = 0
  if (gridAvailW > 0 && duration > 0) {
    const rawColW = (gridAvailW - (numCols - 1) * FRAME_GAP_PX) / numCols
    const boxesPerSecond = Math.max(1, Math.round(rawColW / duration / LARGE_BOX_PX))
    const boxesPerCol = Math.max(1, Math.round(duration * boxesPerSecond))
    boxPx = gridAvailW / (numCols * boxesPerCol + (numCols - 1))
    qColW = boxesPerCol * boxPx
  }
  const gapPx = boxPx > 0 ? boxPx : FRAME_GAP_PX
  const colTpl = qColW > 0 ? `repeat(${numCols}, ${qColW}px)` : `repeat(${numCols}, 1fr)`

  // Quantize rows to even bold-gridline multiples (0mV on a gridline).
  //
  // The floor-to-even has a trap: below roughly 345px of container the row
  // height quantized to ZERO, so all twelve leads rendered at zero height and
  // the grid silently showed nothing — an empty strip under the rhythm lead,
  // with no error anywhere. Two big boxes per row (±1 mV at 10 mm/mV) is the
  // smallest row that is still a readable ECG, so clamp to that and let the
  // grid scroll rather than disappear.
  const X_AXIS_OVERHEAD = 24
  const MIN_ROW_BOXES = 2
  let qRowH = 0
  if (dims.height > 0 && boxPx > 0) {
    const avail = dims.height - X_AXIS_OVERHEAD
    const perRow = (avail - (numRows - 1) * gapPx) / numRows
    const boxes = Math.floor(perRow / boxPx)
    const even = boxes - (boxes % 2)
    qRowH = Math.max(MIN_ROW_BOXES, even) * boxPx
  }
  const rowTpl = qRowH > 0 ? `repeat(${numRows}, ${qRowH}px)` : `repeat(${numRows}, 1fr)`

  const contentH = qRowH > 0 ? numRows * qRowH + (numRows - 1) * gapPx : undefined
  const contentW = qColW > 0 ? numCols * qColW + (numCols - 1) * gapPx : undefined

  // Clicking a lead to expand it is NOT gated on the draw animation.
  //
  // It used to be `interactive && animationProgress >= 1`, so a lead was
  // unclickable until the sweep finished. That fails whenever the browser
  // throttles animation frames — a background tab, reduced-motion settings,
  // low-power mode — and the frame then sits at cursor-not-allowed forever
  // with no way for the reader to tell why. The animation is decorative; the
  // samples are present from the first render, and the expanded view reads the
  // same data, so there is nothing to wait for.
  const canClick = Boolean(onLeadClick)

  return (
    // overflow-auto so a short container scrolls to the remaining leads instead
    // of clipping them away unseen. measureRef still reports the container's
    // own clientHeight, so the quantisation above is unaffected.
    <div ref={measureRef} className={`flex flex-col overflow-auto ${className ?? ''}`}>
      <div className="flex">
        {/* Y-axes */}
        <div className="flex-shrink-0" style={{ width: Y_AXIS_WIDTH }}>
          {layout.map((_, ri) => (
            <div key={ri} className="border-r border-gray-200"
              style={{ height: qRowH || undefined, marginBottom: ri < numRows - 1 ? gapPx : 0 }}>
              <RowYAxis yMin={-1.0} yMax={1.0} rowHeightPx={qRowH || undefined}
                boxPx={boxPx > 0 ? boxPx : undefined} className="h-full" />
            </div>
          ))}
        </div>

        {/* Grid paper + leads */}
        <ECGGridPaper gridMode={gridMode} boxSize={boxPx > 0 ? boxPx : undefined}
          style={{ width: contentW, height: contentH }}>
          <div className="grid" style={{ gridTemplateRows: rowTpl, gridTemplateColumns: colTpl, gap: gapPx }}>
            {layout.map((row, ri) =>
              row.map((leadName, ci) => {
                const samples = ecgData.leads[leadName]
                if (!samples?.length) {
                  return (
                    <LeadFrame key={`${ri}-${ci}`} leadName={leadName}
                      isActiveRhythm={leadName === activeRhythmLead}
                      className="flex items-center justify-center">
                      <span className="text-xs text-gray-400">No data</span>
                    </LeadFrame>
                  )
                }
                return (
                  <LeadFrame key={`${ri}-${ci}`} leadName={leadName}
                    isActiveRhythm={leadName === activeRhythmLead}
                    onClick={canClick && onLeadClick ? () => onLeadClick(leadName) : undefined}
                    disabled={!canClick}>
                    <LeadTraceAnimated
                      leadName={leadName}
                      samples={samples}
                      samplingRate={ecgData.samplingRate}
                      startTime={windowStart}
                      endTime={windowEnd}
                      animationProgress={animationProgress}
                      boxPx={boxPx > 0 ? boxPx : undefined}
                    />
                  </LeadFrame>
                )
              })
            )}
          </div>
        </ECGGridPaper>

        {/* Right padding */}
        <div className="flex-shrink-0" style={{ width: Y_AXIS_WIDTH }} />
      </div>

      {/* X-axes */}
      <div className="flex" style={{ marginTop: 4 }}>
        <div className="flex-shrink-0" style={{ width: Y_AXIS_WIDTH }} />
        <div className="grid" style={{ width: contentW, gridTemplateColumns: colTpl, gap: gapPx }}>
          {Array.from({ length: numCols }).map((_, ci) => (
            <ColumnXAxis key={ci} startTime={windowStart} endTime={windowEnd} />
          ))}
        </div>
        <div className="flex-shrink-0" style={{ width: Y_AXIS_WIDTH }} />
      </div>
    </div>
  )
}
