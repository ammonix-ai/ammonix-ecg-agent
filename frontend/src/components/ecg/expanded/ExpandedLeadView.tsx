import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { ArrowLeft, ChevronLeft, ChevronRight, Layers } from 'lucide-react'
import { ECGGridPaper } from '../ECGGridPaper'
import { CaliperOverlay } from '../overlays/CaliperOverlay'
import { DrawOverlay } from '../overlays/DrawOverlay'
import { CrosshairOverlay } from '../overlays/CrosshairOverlay'
import { RulerOverlay } from '../overlays/RulerOverlay'
import { HRCalculatorOverlay } from '../overlays/HRCalculatorOverlay'
import { MeasurementsDisplayLayer } from '../overlays/MeasurementsDisplayLayer'
import { TemplateOverlay } from '../overlays/TemplateOverlay'
import { ExpandedToolbar, type ExpandedToolMode } from './ExpandedToolbar'
import { ExpandedLeadTrace } from './ExpandedLeadTrace'
import { ExpandedYAxis } from './ExpandedYAxis'
import { ExpandedXAxis } from './ExpandedXAxis'
import { NavigateOverlay } from './NavigateOverlay'
import { LARGE_BOX_PX, LEAD_GROUPS } from '@/types/ecg'
import type { ECGData, LeadName, GainSetting, GridMode } from '@/types/ecg'
import type { CaliperMeasurement, RulerMeasurement, Annotation } from '@/types/ecgTools'
import {
  useEcgAnnotationStore, pixelToDomain, domainToPixel,
  type LeadViewport,
} from '@/stores/ecgAnnotationStore'

interface ExpandedLeadViewProps {
  ecgData: ECGData
  leadName: LeadName
  onBack: () => void
  initialTime?: number
  gain?: GainSetting
  onGainChange?: (gain: GainSetting) => void
  gridMode?: GridMode
  onGridModeChange?: (mode: GridMode) => void
  onLeadChange?: (lead: LeadName) => void
  /** Top predicted diagnoses to seed the template overlay (C2.11). */
  topDiagnoses?: string[]
  /**
   * Recording these marks belong to. When set, annotations are kept in the
   * shared store in signal units so the ECG Agent can send them to the model;
   * without it they stay local to this view and the agent cannot see them.
   */
  recordingId?: string
  className?: string
}

/**
 * ExpandedLeadView — Full-width single lead with all measurement tools
 *
 * Pan/zoom navigation, caliper, draw annotation, ruler, HR calculator.
 * Keyboard shortcuts: V=navigate, C=caliper, A=draw, R=ruler, H=HR calc, P=precision, Esc=back.
 */
export function ExpandedLeadView({
  ecgData, leadName, onBack, initialTime = 0,
  gain = 10, onGainChange: _onGainChange, gridMode = 'full', onGridModeChange: _onGridModeChange,
  onLeadChange, topDiagnoses, recordingId, className,
}: ExpandedLeadViewProps) {
  const [activeTool, setActiveTool] = useState<ExpandedToolMode>('navigate')
  const [zoom, setZoom] = useState(1)
  const [panOffset, setPanOffset] = useState(() => {
    const total = ecgData.numSamples / ecgData.samplingRate
    return Math.min(initialTime, Math.max(0, total - 10))
  })

  const [calipers, setCalipers] = useState<CaliperMeasurement[]>([])
  const [localAnnotations, setLocalAnnotations] = useState<Annotation[]>([])
  const [rulers, setRulers] = useState<RulerMeasurement[]>([])
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null)
  const [precisionMode, setPrecisionMode] = useState(false)

  // C2.11: TemplateOverlay toggle + auto-seed top predicted diagnoses on first activation.
  const [showTemplateOverlay, setShowTemplateOverlay] = useState(false)
  const [activeOverlayDx, setActiveOverlayDx] = useState<string[]>([])
  const overlaySvgRef = useRef<SVGSVGElement>(null)

  const handleToggleTemplateOverlay = useCallback(() => {
    setShowTemplateOverlay((prev) => {
      const next = !prev
      if (next && activeOverlayDx.length === 0 && topDiagnoses && topDiagnoses.length > 0) {
        setActiveOverlayDx(topDiagnoses.slice(0, 3))
      }
      return next
    })
  }, [activeOverlayDx.length, topDiagnoses])

  const handleToggleOverlayDx = useCallback((dx: string) => {
    setActiveOverlayDx((prev) => prev.includes(dx) ? prev.filter((d) => d !== dx) : [...prev, dx])
  }, [])

  const gridRef = useRef<HTMLDivElement>(null)
  const [dims, setDims] = useState({ width: 800, height: 300 })

  useEffect(() => {
    const el = gridRef.current
    if (!el) return
    const update = () => setDims({ width: el.clientWidth, height: el.clientHeight })
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Mouse tracking for crosshair
  useEffect(() => {
    const el = gridRef.current
    if (!el) return
    const move = (e: MouseEvent) => { const r = el.getBoundingClientRect(); setMousePos({ x: e.clientX - r.left, y: e.clientY - r.top }) }
    const leave = () => setMousePos(null)
    el.addEventListener('mousemove', move)
    el.addEventListener('mouseleave', leave)
    return () => { el.removeEventListener('mousemove', move); el.removeEventListener('mouseleave', leave) }
  }, [])

  const samples = ecgData.leads[leadName] || []
  const totalDuration = ecgData.numSamples / ecgData.samplingRate

  // Dynamic grid sizing based on max amplitude
  const maxAmp = useMemo(() => {
    let max = 0
    for (const s of Object.values(ecgData.leads)) for (const v of s) { const a = v < 0 ? -v : v; if (a > max) max = a }
    return max
  }, [ecgData])

  const targetAmp = Math.min(Math.max(maxAmp, 0.5), 3.0)
  const rawBox = dims.height > 0 ? (dims.height * 0.75) / (4 * targetAmp) : LARGE_BOX_PX
  const expandedBoxPx = Math.max(LARGE_BOX_PX * 1.5, Math.min(rawBox, LARGE_BOX_PX * 4))

  const basePixelsPerSecond = expandedBoxPx * 25 / 5 // 25mm/s standard
  const pixelsPerSecond = basePixelsPerSecond * zoom
  const pixelsPerMv = expandedBoxPx * gain / 5
  const baseVisibleDuration = dims.width > 0 ? dims.width / basePixelsPerSecond : 0
  const maxZoom = baseVisibleDuration > 0 ? baseVisibleDuration / 0.1 : 10

  const visDuration = baseVisibleDuration / zoom
  const visStart = panOffset
  const visEnd = Math.min(panOffset + visDuration, totalDuration)
  const yRangeMv = dims.height / pixelsPerMv

  const gridOffsetX = -((panOffset * pixelsPerSecond) % (expandedBoxPx * zoom))
  const gridOffsetY = (dims.height / 2) % expandedBoxPx

  // ── annotations: pixels on screen, signal units in the store ──
  // The store holds (lead, seconds, millivolts) so the marks survive zoom, pan
  // and gain, and so the PNG sent to the agent can place them at its own scale.
  // This view still draws in pixels, so convert in both directions here — the
  // viewport is the only place that knows the mapping.
  const viewport: LeadViewport = {
    visibleStartSeconds: visStart,
    pixelsPerSecond,
    pixelsPerMillivolt: pixelsPerMv,
    baselineY: dims.height / 2,
  }

  const storedAnnotations = useEcgAnnotationStore(
    (s) => (recordingId ? s.byRecording[recordingId] : undefined),
  )
  const addStored = useEcgAnnotationStore((s) => s.add)
  const clearStoredLead = useEcgAnnotationStore((s) => s.clearLead)

  const annotations: Annotation[] = useMemo(() => {
    if (!recordingId) return localAnnotations
    return (storedAnnotations ?? [])
      .filter((a) => a.lead === leadName)
      .map((a) => {
        const { x, y } = domainToPixel(a.tSeconds, a.mV, viewport)
        return {
          id: a.id,
          x,
          y,
          type: a.kind,
          label: a.label,
          color: a.color,
          strokePoints: a.points?.map((p) => domainToPixel(p.tSeconds, p.mV, viewport)),
        }
      })
    // viewport is recreated each render; depend on its parts, not the object.
  }, [recordingId, storedAnnotations, localAnnotations, leadName,
      visStart, pixelsPerSecond, pixelsPerMv, dims.height])

  const handleAddAnnotation = useCallback((a: Annotation) => {
    if (!recordingId) {
      setLocalAnnotations((prev) => [...prev, a])
      return
    }
    const { tSeconds, mV } = pixelToDomain(a.x, a.y, viewport)
    addStored(recordingId, {
      id: a.id,
      lead: leadName,
      tSeconds,
      mV,
      label: a.label,
      color: a.color,
      kind: a.type,
      points: a.strokePoints?.map((p) => pixelToDomain(p.x, p.y, viewport)),
    })
  }, [recordingId, leadName, addStored, visStart, pixelsPerSecond, pixelsPerMv, dims.height])

  const clearAnnotations = useCallback(() => {
    setLocalAnnotations([])
    if (recordingId) clearStoredLead(recordingId, leadName)
  }, [recordingId, leadName, clearStoredLead])

  const removeStored = useEcgAnnotationStore((s) => s.remove)
  const handleRemoveAnnotation = useCallback((id: string) => {
    if (recordingId) removeStored(recordingId, id)
    else setLocalAnnotations((prev) => prev.filter((a) => a.id !== id))
  }, [recordingId, removeStored])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      const k = e.key.toLowerCase()
      if (k === 'escape') onBack()
      if (k === 'v') setActiveTool('navigate')
      if (k === 'c') setActiveTool('caliper')
      if (k === 'a') setActiveTool('annotate')
      if (k === 'r') setActiveTool('ruler')
      if (k === 'h') setActiveTool('hr-calc')
      if (k === 'p') setPrecisionMode(prev => !prev)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onBack])

  const handlePrevLead = useCallback(() => {
    if (!onLeadChange) return
    const all = LEAD_GROUPS.all; const idx = all.indexOf(leadName)
    onLeadChange(all[idx <= 0 ? all.length - 1 : idx - 1]!)
  }, [leadName, onLeadChange])

  const handleNextLead = useCallback(() => {
    if (!onLeadChange) return
    const all = LEAD_GROUPS.all; const idx = all.indexOf(leadName)
    onLeadChange(all[idx >= all.length - 1 ? 0 : idx + 1]!)
  }, [leadName, onLeadChange])

  return (
    <div className={`flex flex-col overflow-hidden bg-white ${className ?? ''}`}>
      {/* Subtab: Back + Lead + Toolbar */}
      <div className="flex items-center h-9 shrink-0 border-b border-slate-200 bg-slate-50 px-4 gap-2">
        <button onClick={onBack} className="flex items-center gap-1.5 px-2 py-1 text-xs font-medium text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-md transition-colors"
          title="Back to 12-Lead (Esc)">
          <ArrowLeft className="w-3.5 h-3.5" /><span>Back</span>
        </button>
        <div className="w-px h-5 bg-slate-200" />
        <div className="flex items-center gap-0.5">
          <button onClick={handlePrevLead} className="p-0.5 text-slate-400 hover:text-slate-600 rounded transition-colors"><ChevronLeft className="w-3.5 h-3.5" /></button>
          <span className="text-xs font-bold text-indigo-700 min-w-[52px] text-center">Lead {leadName}</span>
          <button onClick={handleNextLead} className="p-0.5 text-slate-400 hover:text-slate-600 rounded transition-colors"><ChevronRight className="w-3.5 h-3.5" /></button>
        </div>
        <div className="w-px h-5 bg-slate-200" />
        <ExpandedToolbar activeTool={activeTool} onToolChange={setActiveTool}
          onReset={() => { setCalipers([]); clearAnnotations(); setRulers([]) }}
          caliperCount={calipers.length} annotationCount={annotations.length} rulerCount={rulers.length}
          precisionMode={precisionMode} onPrecisionModeChange={setPrecisionMode} />
        <div className="w-px h-5 bg-slate-200" />
        {/* C2.11: TemplateOverlay toggle */}
        <button
          type="button"
          onClick={handleToggleTemplateOverlay}
          className={`flex items-center gap-1 px-2 py-1 text-xs font-medium rounded-md transition-colors ${
            showTemplateOverlay
              ? 'bg-indigo-100 text-indigo-700 hover:bg-indigo-200'
              : 'text-slate-500 hover:text-slate-700 hover:bg-slate-100'
          }`}
          title="Show diagnosis template overlay (sigma bands)"
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Templates</span>
        </button>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-h-0 pt-1">
        <div className="flex-1 flex min-h-0">
          <ExpandedYAxis gain={gain} boxSize={expandedBoxPx} width={48} containerHeight={dims.height}
            className="flex-shrink-0 border-r border-gray-200" />

          <ECGGridPaper gridMode={gridMode} boxSize={expandedBoxPx} zoomX={zoom}
            gridOffsetX={gridOffsetX} gridOffsetY={gridOffsetY} className="flex-1 relative">
            <div ref={gridRef} className="absolute inset-0">
              <ExpandedLeadTrace leadName={leadName} samples={samples} samplingRate={ecgData.samplingRate}
                totalDuration={totalDuration} basePixelsPerSecond={basePixelsPerSecond}
                gain={gain} zoom={zoom} panOffset={panOffset} boxSize={expandedBoxPx} />

              {precisionMode && mousePos && (
                <CrosshairOverlay mouseX={mousePos.x} mouseY={mousePos.y}
                  containerWidth={dims.width} containerHeight={dims.height}
                  visibleStart={visStart} pixelsPerSecond={pixelsPerSecond} pixelsPerMillivolt={pixelsPerMv} />
              )}

              <NavigateOverlay isActive={activeTool === 'navigate'}
                zoom={zoom} panOffset={panOffset} visibleDuration={baseVisibleDuration}
                totalDuration={totalDuration} onZoomChange={setZoom} onPanChange={setPanOffset} maxZoom={maxZoom} />

              <CaliperOverlay isActive={activeTool === 'caliper'}
                pixelsPerSecond={pixelsPerSecond}
                onAddMeasurement={(m) => setCalipers(prev => [...prev, m])}
                containerHeight={dims.height} yRangeMv={yRangeMv} />

              <DrawOverlay isActive={activeTool === 'annotate'}
                onAddAnnotation={handleAddAnnotation} />

              <RulerOverlay isActive={activeTool === 'ruler'}
                onAddMeasurement={(m) => setRulers(prev => [...prev, m])} />

              <HRCalculatorOverlay isActive={activeTool === 'hr-calc'} pixelsPerSecond={pixelsPerSecond} />

              <MeasurementsDisplayLayer
                caliperMeasurements={calipers} rulerMeasurements={rulers} annotations={annotations}
                onRemoveCaliper={(id) => setCalipers(prev => prev.filter(m => m.id !== id))}
                onRemoveRuler={(id) => setRulers(prev => prev.filter(m => m.id !== id))}
                onRemoveAnnotation={handleRemoveAnnotation} />

              {/* C2.11: TemplateOverlay (sigma bands per diagnosis). TODO markers in
                  TemplateOverlay source are P-tov polish (Tier 7), out-of-scope here. */}
              {showTemplateOverlay && (
                <TemplateOverlay
                  ecgData={ecgData}
                  leadName={leadName}
                  activeDiagnoses={activeOverlayDx}
                  onToggleDiagnosis={handleToggleOverlayDx}
                  svgRef={overlaySvgRef as React.RefObject<SVGSVGElement>}
                  width={dims.width}
                  height={dims.height}
                  gain={gain}
                  paperSpeed={25}
                />
              )}
            </div>
          </ECGGridPaper>
        </div>

        <ExpandedXAxis startTime={visStart} endTime={visEnd} pixelsPerSecond={pixelsPerSecond}
          containerWidth={dims.width} height={28} className="flex-shrink-0 border-t border-gray-200 ml-[48px]" />
      </div>
    </div>
  )
}
