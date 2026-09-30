/**
 * ECG Data Types — IEC 60601-2-25 compliant
 *
 * Grid constants, clinical display settings, calibration math.
 * Ported from FEBE frontend/src/types/ecg.ts.
 */

// ── Grid pixel constants (96 CSS DPI — device-independent) ──

export const PIXELS_PER_MM = 96 / 25.4          // ~3.78px per mm
export const SMALL_BOX_PX = PIXELS_PER_MM       // 1mm minor gridline
export const LARGE_BOX_PX = 5 * PIXELS_PER_MM   // 5mm major gridline (~18.9px)
export const FRAME_GAP_PX = LARGE_BOX_PX        // gap between frames
export const Y_AXIS_WIDTH = 40                   // label column width (px)

// ── Lead types ──

export type LeadName =
  | 'I' | 'II' | 'III'
  | 'aVR' | 'aVL' | 'aVF'
  | 'V1' | 'V2' | 'V3' | 'V4' | 'V5' | 'V6'

export const LEAD_GROUPS = {
  limb: ['I', 'II', 'III', 'aVR', 'aVL', 'aVF'] as LeadName[],
  chest: ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'] as LeadName[],
  all: ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] as LeadName[],
}

/** Standard clinical 3-column x 4-row layout */
export const TWELVE_LEAD_LAYOUT: [LeadName, LeadName, LeadName][] = [
  ['I',   'aVR', 'V1'],
  ['II',  'aVL', 'V2'],
  ['III', 'aVF', 'V3'],
  ['II',  'V4',  'V5'],  // Row 4: II rhythm + V4, V5 (some layouts vary)
]

/** 2x6 grid layout (alternative) */
export const LEAD_GRID_2x6: LeadName[][] = [
  ['I', 'V1'],
  ['II', 'V2'],
  ['III', 'V3'],
  ['aVR', 'V4'],
  ['aVL', 'V5'],
  ['aVF', 'V6'],
]

// ── Display settings ──

export type GridMode = 'full' | 'major' | 'off'
export type PaperSpeed = 10 | 25 | 50
export type GainSetting = 5 | 10 | 20 | 50
export type VisibleRange = 10 | 5 | 2.5 | 1

export type XScaleMode =
  | { mode: 'paperSpeed'; value: PaperSpeed }
  | { mode: 'visibleRange'; value: VisibleRange }

export const DEFAULT_X_SCALE: XScaleMode = { mode: 'paperSpeed', value: 25 }

/** Tool modes for interactive overlays */
export type ToolMode = 'none' | 'caliper' | 'crosshair' | 'draw' | 'ruler' | 'hr-calculator'

/** View mode: twelve-lead grid or expanded single lead */
export type ViewMode = 'twelve-lead' | 'expanded'

// ── Grid colors (IEC compliant) ──

/**
 * Canonical ECG grid color palette (single source of truth as of V3.24).
 *
 * Consumed by: `CalibrationPulse`, `ECGGridPaper`, `LeadTrace`,
 * `LeadTraceAnimated`, `ExpandedLeadTrace`.
 *
 * V3.24 (Q-A3-10) collapsed this to one place: pre-V3.24 there were duplicate
 * (and drifted) definitions in `src/design/tokens.ts` `ecg = {...}` and
 * `src/index.css :root` `--ecg-*` CSS vars, but neither had any consumer.
 * Both removed.
 */
export const ECG_GRID_COLORS = {
  major: '#FF9999',
  minor: '#FFCCCC',
  background: '#FFF5F5',
  trace: '#0F172A',
  traceWidth: 1.5,
} as const

// ── ECG Data ──

/**
 * Lead-sample container — accepts both `number[]` (JSON path) and
 * `Float32Array` (binary path).
 *
 * C0.1: chose `number[] | Float32Array` over `ArrayLike<number>` because
 * the union retains `.slice`, `.map`, and `Symbol.iterator` (methods
 * shared by both Array and TypedArray). `ArrayLike<number>` is too
 * narrow — only `.length` and indexed access — and would force every
 * existing `samples.slice/map/for-of` call site to convert via
 * `Array.from(...)` before use.
 */
export type SampleArray = number[] | Float32Array

/**
 * ECGData carries the patient ECG payload.
 *
 * The backend negotiates content type via `Accept` header. The frontend's
 * `ecgService.loadECGData` opts in to binary (`application/octet-stream`)
 * for ~6x smaller wire payload + 4x less client memory; the JSON path is
 * retained for tests/debugging via `loadECGDataJson`.
 *
 * Operations that work on both number[] and Float32Array:
 *   - indexed access (samples[i])
 *   - .length, .slice, .map, for...of
 *   - pass to uPlot (native typed-array support)
 *   - pass to SVG path generators
 *
 * Operations that DON'T work on Float32Array (audit found zero hits in
 * V3; watch for these on new code):
 *   - .push() / .concat() / .splice() — typed arrays are fixed-size
 *   - JSON.stringify(samples) — gives object form, not array. Use
 *     JSON.stringify(Array.from(samples)) if needed.
 *   - Array.isArray(samples) — returns false. Use 'length' in x or
 *     ArrayBuffer.isView(x) to discriminate.
 *
 */
export interface ECGData {
  patientId: string
  samplingRate: number   // Hz (typically 500)
  numSamples: number
  durationSeconds?: number
  leads: Record<LeadName, SampleArray>
}

// ── Display settings ──

export interface ECGDisplaySettings {
  paperSpeed: PaperSpeed
  gain: GainSetting
  gridMode: GridMode
  showCalibrationPulse: boolean
  showMeasurements: boolean
  rhythmStripLead: LeadName
}

export const DEFAULT_ECG_SETTINGS: ECGDisplaySettings = {
  paperSpeed: 25,
  gain: 10,
  gridMode: 'full',
  showCalibrationPulse: true,
  showMeasurements: true,
  rhythmStripLead: 'II',
}

// ── Math utilities ──

export function getPixelsPerMm(dpi: number = 96): number {
  return dpi / 25.4
}

export function getPixelsPerSecond(paperSpeed: PaperSpeed, pixelsPerMm: number = getPixelsPerMm()): number {
  return paperSpeed * pixelsPerMm
}

export function getPixelsPerMv(gain: GainSetting, pixelsPerMm: number = getPixelsPerMm()): number {
  return gain * pixelsPerMm
}

export function getGridSpacing(paperSpeed: PaperSpeed, gain: GainSetting) {
  const pixelsPerMm = getPixelsPerMm()
  const msPerSmallBox = 1000 / paperSpeed
  const msPerLargeBox = 5000 / paperSpeed
  const mvPerSmallBox = 1 / gain
  const mvPerLargeBox = 5 / gain

  return {
    smallBoxPx: pixelsPerMm,
    largeBoxPx: 5 * pixelsPerMm,
    msPerSmallBox,
    msPerLargeBox,
    mvPerSmallBox,
    mvPerLargeBox,
    pixelsPerSecond: getPixelsPerSecond(paperSpeed, pixelsPerMm),
    pixelsPerMv: getPixelsPerMv(gain, pixelsPerMm),
  }
}

// ── Calibration pulse ──

export const CALIBRATION_PULSE = {
  amplitudeMv: 1.0,
  durationMs: 200,
  getPixelDimensions: (paperSpeed: PaperSpeed, gain: GainSetting) => {
    const spacing = getGridSpacing(paperSpeed, gain)
    return {
      height: spacing.pixelsPerMv,
      width: (200 / 1000) * spacing.pixelsPerSecond,
    }
  },
} as const

// ── Measurement types ──

export interface MeasurementPoint {
  leadName: LeadName
  sampleIndex: number
  timeMs: number
  amplitudeMv: number
}

// ── Setting labels ──

export const SETTING_LABELS = {
  paperSpeed: {
    10: '10 mm/s (Overview)',
    25: '25 mm/s (Standard)',
    50: '50 mm/s (Detailed)',
  },
  gain: {
    5: '5 mm/mV (Half)',
    10: '10 mm/mV (Standard)',
    20: '20 mm/mV (Double)',
    50: '50 mm/mV (Quad)',
  },
} as const
