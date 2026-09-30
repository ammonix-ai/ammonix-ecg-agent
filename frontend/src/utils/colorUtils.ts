/**
 * Color utilities for 3D projection viewer.
 *
 * Handles 7 color modes: cohort, diagnosis, diagnosis_predicted, accuracy,
 * accuracy_pipeline, error_highlight, family. Supports gradient interpolation,
 * multi-diagnosis RGB blending, and customizable palettes.
 */

import type {
  EnrichedProjectionPoint,
  ColorByMode,
  ViewerSettings,
} from '@/types/projection'
import {
  DIAGNOSIS_COLOR_PALETTE,
  DEFAULT_VIEWER_SETTINGS,
} from '@/types/projection'

export const FALLBACK_COLOR = '#A0B4C2'

// =============================================================================
// Color conversion helpers
// =============================================================================

export function hexToRgb(hex: string): [number, number, number] {
  const r = parseInt(hex.slice(1, 3), 16) / 255
  const g = parseInt(hex.slice(3, 5), 16) / 255
  const b = parseInt(hex.slice(5, 7), 16) / 255
  return [r, g, b]
}

export function hexToNumber(hex: string): number {
  return parseInt(hex.replace('#', ''), 16)
}

export function lerpColor(color1: string, color2: string, t: number): string {
  t = Math.max(0, Math.min(1, t))

  const r1 = parseInt(color1.slice(1, 3), 16)
  const g1 = parseInt(color1.slice(3, 5), 16)
  const b1 = parseInt(color1.slice(5, 7), 16)

  const r2 = parseInt(color2.slice(1, 3), 16)
  const g2 = parseInt(color2.slice(3, 5), 16)
  const b2 = parseInt(color2.slice(5, 7), 16)

  const r = Math.round(r1 + (r2 - r1) * t)
  const g = Math.round(g1 + (g2 - g1) * t)
  const b = Math.round(b1 + (b2 - b1) * t)

  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
}

export function interpolateGradient(
  t: number,
  lowColor: string,
  midColor: string,
  highColor: string,
): string {
  t = Math.max(0, Math.min(1, t))
  if (t <= 0.5) return lerpColor(lowColor, midColor, t * 2)
  return lerpColor(midColor, highColor, (t - 0.5) * 2)
}

export function hslToHex(hsl: string): string {
  const match = hsl.match(/hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)/)
  if (!match) return FALLBACK_COLOR

  const h = parseInt(match[1]!) / 360
  const s = parseInt(match[2]!) / 100
  const l = parseInt(match[3]!) / 100

  let r: number, g: number, b: number

  if (s === 0) {
    r = g = b = l
  } else {
    const hue2rgb = (p: number, q: number, t: number) => {
      if (t < 0) t += 1
      if (t > 1) t -= 1
      if (t < 1 / 6) return p + (q - p) * 6 * t
      if (t < 1 / 2) return q
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6
      return p
    }

    const q = l < 0.5 ? l * (1 + s) : l + s - l * s
    const p = 2 * l - q
    r = hue2rgb(p, q, h + 1 / 3)
    g = hue2rgb(p, q, h)
    b = hue2rgb(p, q, h - 1 / 3)
  }

  const toHex = (x: number) => {
    const hex = Math.round(x * 255).toString(16)
    return hex.length === 1 ? '0' + hex : hex
  }

  return `#${toHex(r)}${toHex(g)}${toHex(b)}`
}

export function normalizeColor(color: string): string {
  if (color.startsWith('#')) return color
  if (color.startsWith('hsl')) return hslToHex(color)
  return FALLBACK_COLOR
}

export function blendColors(colors: string[]): string {
  if (colors.length === 0) return FALLBACK_COLOR
  if (colors.length === 1) return colors[0]!

  let totalR = 0, totalG = 0, totalB = 0

  for (const color of colors) {
    totalR += parseInt(color.slice(1, 3), 16)
    totalG += parseInt(color.slice(3, 5), 16)
    totalB += parseInt(color.slice(5, 7), 16)
  }

  const n = colors.length
  const r = Math.round(totalR / n)
  const g = Math.round(totalG / n)
  const b = Math.round(totalB / n)

  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
}

// =============================================================================
// Diagnosis color functions
// =============================================================================

export function extractUniqueDiagnoses(points: EnrichedProjectionPoint[]): string[] {
  const diagnosisSet = new Set<string>()
  for (const point of points) {
    for (const dx of point.diagnoses) {
      diagnosisSet.add(dx)
    }
  }
  return Array.from(diagnosisSet).sort((a, b) =>
    a.toLowerCase().localeCompare(b.toLowerCase()),
  )
}

export function buildDiagnosisColorMap(
  diagnoses: string[],
  customColors: Record<string, string> = {},
): Map<string, string> {
  const sorted = [...diagnoses].sort((a, b) =>
    a.toLowerCase().localeCompare(b.toLowerCase()),
  )

  const colorMap = new Map<string, string>()
  sorted.forEach((diagnosis, index) => {
    if (customColors[diagnosis]) {
      colorMap.set(diagnosis, customColors[diagnosis]!)
    } else {
      colorMap.set(
        diagnosis,
        DIAGNOSIS_COLOR_PALETTE[index % DIAGNOSIS_COLOR_PALETTE.length]!,
      )
    }
  })

  return colorMap
}

export function getDiagnosisColorFromMap(
  diagnoses: string[],
  colorMap: Map<string, string>,
): string {
  if (diagnoses.length === 0) return FALLBACK_COLOR

  // Lowercased to match useDiagnosisColorMap's keys — the wire mixes cases
  // ('STEMI' in labels, 'stemi' as a primary), and an exact-case lookup
  // silently greyed out every record whose label casing differed.
  const colors = diagnoses
    .map((dx) => colorMap.get(dx.toLowerCase()))
    .filter((c): c is string => c !== undefined)

  if (colors.length === 0) return FALLBACK_COLOR
  return blendColors(colors)
}

// =============================================================================
// Dataset color palette — deterministic per-dataset accent for tables/badges
// =============================================================================

/** Distinct accent colors for dataset badges; chosen for hue separation. */
const DATASET_COLOR_PALETTE: readonly string[] = [
  '#3B82F6', // blue
  '#8B5CF6', // violet
  '#10B981', // emerald
  '#F59E0B', // amber
  '#EF4444', // red
  '#EC4899', // pink
  '#14B8A6', // teal
  '#F97316', // orange
  '#06B6D4', // cyan
  '#A855F7', // purple
  '#84CC16', // lime
  '#0EA5E9', // sky
  '#D946EF', // fuchsia
];

function hashStringToInt(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h << 5) - h + s.charCodeAt(i);
    h |= 0; // force int32
  }
  return Math.abs(h);
}

/**
 * Deterministic accent color for a dataset id (e.g. 'ptbxl', 'ningbo').
 * Same id always returns same color across renders + sessions.
 */
export function getDatasetColor(datasetId: string): string {
  if (!datasetId) return FALLBACK_COLOR;
  return DATASET_COLOR_PALETTE[
    hashStringToInt(datasetId) % DATASET_COLOR_PALETTE.length
  ]!;
}

// =============================================================================
// Diagnosis family grouping
// =============================================================================

/**
 * Maps each diagnosis to a clinical family for grouped coloring.
 *
 * Keys are lowercase; `getFamilyColor` lowercases before lookup because the
 * wire mixes cases ('STEMI' in `labels`, 'stemi' in `primary`).
 *
 * Two clinical decisions worth stating, because the map is the only place
 * they are recorded:
 *
 * 1. Both reduced-EF classes live in 'Structural'. A depressed ejection
 *    fraction is systolic dysfunction of the myocardium, not an arrhythmia,
 *    and it belongs next to LVH rather than next to VPBs and VT. 'lvef ≤45%'
 *    used to sit in 'Ventricular', which conflated a chamber-function
 *    measurement with ventricular ectopy; moving it leaves 'Ventricular' as a
 *    clean ventricular-arrhythmia family.
 * 2. 'lvef >=55% control' and 'normal ecg' are statements of normality, not
 *    diagnoses. They get their own neutral family so they are legible without
 *    being painted as disease. 'sinus rhythm' deliberately stays in
 *    'Rhythm & Rate' — it is a rhythm finding that coexists with pathology.
 *
 * 'unknown' is intentionally absent: an unlabelled record should read as grey.
 */
export const DIAGNOSIS_FAMILIES: Record<string, string> = {
  // Rhythm & Rate
  'sinus rhythm': 'Rhythm & Rate',
  'sinus bradycardia': 'Rhythm & Rate',
  'sinus tachycardia': 'Rhythm & Rate',
  'sinus irregularity': 'Rhythm & Rate',
  'pacing rhythm': 'Rhythm & Rate',
  'junctional rhythm': 'Rhythm & Rate',
  'tachycardia': 'Rhythm & Rate',
  'bradycardia': 'Rhythm & Rate',
  // Atrial
  'atrial fibrillation': 'Atrial',
  'atrial flutter': 'Atrial',
  'atrial premature beats': 'Atrial',
  'af risk': 'Atrial',
  'supraventricular tachycardia': 'Atrial',
  'wolff-parkinson-white': 'Atrial',
  'atrial tachycardia': 'Atrial',
  // Ventricular — ventricular arrhythmia and its risk markers only.
  'ventricular premature beats': 'Ventricular',
  'ventricular tachycardia': 'Ventricular',
  'vt-risk': 'Ventricular',
  'sudden death': 'Ventricular',
  // Conduction (includes av_block taxonomy category — no dedicated AV Block family)
  '1st av-block': 'Conduction',
  '2nd av-block': 'Conduction',
  '3rd av-block': 'Conduction',
  'left bundle branch block / variations': 'Conduction',
  'right bundle branch block': 'Conduction',
  'left anterior fascicular block': 'Conduction',
  'bifascicular block': 'Conduction',
  // Ischemia & Injury
  'myocardial infarction': 'Ischemia & Injury',
  'old myocardial infarction': 'Ischemia & Injury',
  'stemi': 'Ischemia & Injury',
  'nstemi': 'Ischemia & Injury',
  'st deviation': 'Ischemia & Injury',
  't wave change': 'Ischemia & Injury',
  't wave inversion': 'Ischemia & Injury',
  'poor r wave progression': 'Ischemia & Injury',
  'brugada syndrome': 'Ischemia & Injury',
  // Structural (includes axis taxonomy category and the reduced-EF classes)
  'left ventricular hypertrophy': 'Structural',
  'left atrial enlargement': 'Structural',
  'low qrs voltages': 'Structural',
  'axis left shift': 'Structural',
  'axis right shift': 'Structural',
  'normal axis': 'Structural',
  'lvef ≤45%': 'Structural',
  'lvef <40% expanded mimic ef>=55 controls': 'Structural',
  'chagas disease': 'Structural',
  // Repolarization
  'qt interval extension': 'Repolarization',
  // ── added by the machine-label recovery ──────────────────────────────
  // Recovering the MIMIC diagnoses introduced 23 primaries this taxonomy had
  // never seen, which put 2,100 points back into the grey fallback. These are
  // all legitimate findings already ranked in backend/diagnosis_severity.py;
  // they simply had no family. Assignments follow the same reasoning as the
  // tiers there.
  'rsr pattern': 'Conduction',                                   // an incomplete-RBBB morphology
  'shortened pr interval': 'Conduction',                         // pre-excitation territory
  'non-specific intraventricular conduction delay': 'Conduction',
  'left posterior fascicular block': 'Conduction',
  '3 av-block': 'Conduction',                                    // spelling variant of 3rd av-block
  'av dissociation': 'Conduction',

  'ectopic atrial rhythm': 'Atrial',
  'ectopic atrial tachycardia': 'Atrial',
  'right atrial enlargement': 'Atrial',
  'biatrial enlargement': 'Atrial',

  'idioventricular rhythm': 'Ventricular',                       // a ventricular escape rhythm

  'right ventricular hypertrophy': 'Structural',
  'amyloidosis': 'Structural',                                   // infiltrative cardiomyopathy
  'counterclockwise rotation': 'Structural',                     // electrical position

  'pericarditis': 'Ischemia & Injury',
  'acute myocardial ischemia': 'Ischemia & Injury',
  'abnormal q wave': 'Ischemia & Injury',

  'early repolarization': 'Repolarization',
  'indeterminate axis': 'Repolarization',                        // grouped with the axis findings

  'undetermined rhythm': 'Rhythm & Rate',

  // Controls & Normal — normality, not pathology.
  'lvef >=55% control': 'Controls & Normal',
  'normal ecg': 'Controls & Normal',
  // Summary verdicts: not findings, but they do represent records that have
  // nothing else, so they get the neutral colour rather than the grey fallback.
  'abnormal ecg': 'Controls & Normal',
  'borderline ecg': 'Controls & Normal',
}

/** Colors for each family (8 families). */
export const FAMILY_COLORS: Record<string, string> = {
  'Rhythm & Rate': '#3B82F6',    // blue
  'Atrial': '#EC4899',            // pink
  'Ventricular': '#EF4444',       // red
  'Conduction': '#F59E0B',        // amber
  'Ischemia & Injury': '#DC2626', // dark red
  'Structural': '#10B981',        // emerald
  'Repolarization': '#8B5CF6',    // violet
  'Controls & Normal': '#64748B', // muted slate — neutral, reads as "not a disease"
}

/** Legend label for primaries with no family (they render in FALLBACK_COLOR). */
export const FAMILY_UNMAPPED_LABEL = 'Unmapped'

export function getFamilyColor(diagnosis: string): string {
  // Case-insensitive lookup: the domain notes use uppercase STEMI in display,
  // but the family map normalizes to lowercase keys. Q-A3-5.
  const family = DIAGNOSIS_FAMILIES[diagnosis.toLowerCase()]
  if (!family) return FALLBACK_COLOR
  return FAMILY_COLORS[family] ?? FALLBACK_COLOR
}

export function getFamilyForDiagnosis(diagnosis: string): string {
  return DIAGNOSIS_FAMILIES[diagnosis.toLowerCase()] ?? 'Other'
}

// =============================================================================
// Error highlight (predicted vs gold)
// =============================================================================

export type ErrorHighlightBucket = 'both' | 'over' | 'missed' | 'match' | 'na'

/**
 * Compare a record's predictions against the gold labels the model could
 * actually have predicted.
 *
 * `scoredClasses` must be the set of classes the model scores, lowercased —
 * the keys of `sourceSafe.thresholds`. Intersecting the gold set with it is
 * the whole point: `point.diagnoses` carries labels the model has no class
 * for ('sinus rhythm', 'normal ecg', 'lvef >=55% control' — ~46k of the 63,256
 * published rows carry at least one). Comparing against the raw gold set
 * scored a normal ECG as having "missed sinus rhythm", a diagnosis it was
 * never able to call. `xgb_correct` never did this; it was computed against
 * gold ∩ scored classes, so the two modes disagreed on the same record.
 *
 * Comparison is case-insensitive because the wire mixes cases: 'STEMI' in
 * `labels` and in `thresholds`, 'stemi' in `primary`.
 *
 * Returns 'na' when the comparison cannot be made — no predictions, or no
 * gold label inside the scored classes. With no scored gold there is nothing
 * to be right or wrong about, so the record is not graded rather than being
 * called an error. An empty/absent `scoredClasses` (a projection that ships
 * no threshold metadata) makes every record 'na': without the class list we
 * cannot say what a miss would be, and guessing is what caused this bug.
 */
export function classifyErrorHighlight(
  point: EnrichedProjectionPoint,
  scoredClasses?: ReadonlySet<string>,
): ErrorHighlightBucket {
  // Without the scored-class list we cannot say what counts as a miss, and
  // guessing is what produced the 31,000 phantom false negatives this function
  // was written to fix. Only genuinely-unknowable points are ungraded.
  if (!scoredClasses || scoredClasses.size === 0) return 'na'

  const preds = new Set((point.xgbPredictions ?? []).map((p) => p.toLowerCase()))
  const golds = new Set(
    (point.diagnoses ?? [])
      .map((g) => g.toLowerCase())
      .filter((g) => scoredClasses.has(g)),
  )

  // An empty set on either side is meaningful, not ungradeable, and the
  // comparison below already says the right thing about it:
  //   nothing predicted, nothing to predict -> match  (correct silence on a
  //     normal recording; 17,474 points, and the model at its best)
  //   predicted something, nothing to predict -> over (called a finding on a
  //     record whose only labels are normal; 27,045 points, the single largest
  //     error category, previously hidden)
  //   nothing predicted, something to predict -> missed
  // Greying these out hid 63% of the model's errors behind the same colour as
  // its biggest success, and left this mode disagreeing with Accuracy.
  const hasFP = [...preds].some((p) => !golds.has(p))
  const hasFN = [...golds].some((g) => !preds.has(g))

  if (hasFP && hasFN) return 'both'
  if (hasFP) return 'over'
  if (hasFN) return 'missed'
  return 'match'
}

/** The four-colour error scheme, unchanged, plus the ungraded fallback. */
/**
 * Green shades for what the model found, warm shades for what it did not.
 *
 * Over-calling sits in the green family deliberately: a call the labels do not
 * carry is not necessarily wrong. The 12SL reference standard is incomplete and
 * conservative, so a large share of these are plausibly real detections the
 * labels miss. Missing a finding is the clinically worse failure, so the warm
 * colours are reserved for it.
 */
export const ERROR_HIGHLIGHT_COLORS: Record<ErrorHighlightBucket, string> = {
  both: '#DC2626',   // red — over-called AND missed
  over: '#15803D',   // dark green — over-called only
  missed: '#D97706', // orange — missed only
  match: '#4ADE80',  // light green — predicted set equals scored gold set
  na: FALLBACK_COLOR,
}

// =============================================================================
// Main point color function
// =============================================================================

export function getPointColor(
  point: EnrichedProjectionPoint,
  colorMode: ColorByMode,
  settings: ViewerSettings = DEFAULT_VIEWER_SETTINGS,
  diagnosisColorMap?: Map<string, string>,
  cohortColorMap?: Map<string, string>,
  scoredClasses?: ReadonlySet<string>,
): string {
  // `settings` is still part of the signature for palette overrides, but the
  // only mode that read settings.colors was 'accuracy', now removed.
  void settings

  switch (colorMode) {
    // One colour per source cohort, assigned by name from the data itself
    // (see useCohortColorMap). This is the only split the published universe
    // actually carries — it ships no fold membership, so there is no
    // training/holdout colouring to offer.
    case 'cohort':
      return cohortColorMap?.get(point.cohort) ?? FALLBACK_COLOR

    // 'accuracy' and 'accuracy_pipeline' removed: both reduced a point to
    // exact-match vs differs, which is the partition error_highlight already
    // makes — while also saying whether the difference was an over-call, a
    // miss, or both.

    case 'diagnosis':
      // Colour by the single diagnosis that represents the record, not by a
      // blend of all its labels.
      //
      // Blending averaged every label's colour together, so ~7,100 multi-label
      // points rendered in a mixture that matched NO entry in the legend — a
      // record with atrial fibrillation and LVH came out a third colour that
      // means nothing. The backend now picks primaryDiagnosis by clinical
      // severity (see backend/diagnosis_severity.py), so the most consequential
      // finding is the one shown, and every point matches a legend entry.
      if (diagnosisColorMap) {
        return (
          diagnosisColorMap.get(point.primaryDiagnosis.toLowerCase()) ??
          // Fall back to the label blend only if the primary is unrecognised,
          // so a point is never grey when it has something colourable.
          getDiagnosisColorFromMap(point.diagnoses, diagnosisColorMap)
        )
      }
      return normalizeColor(point.color)

    case 'diagnosis_predicted': {
      const preds = point.xgbPredictions ?? []
      if (preds.length === 0) return FALLBACK_COLOR
      if (!diagnosisColorMap) return normalizeColor(point.color)
      const predColors = preds
        .map((dx) => diagnosisColorMap.get(dx.toLowerCase()))
        .filter((c): c is string => c !== undefined)
      return predColors.length > 0 ? blendColors(predColors) : FALLBACK_COLOR
    }

    case 'error_highlight':
      return ERROR_HIGHLIGHT_COLORS[classifyErrorHighlight(point, scoredClasses)]

    case 'family':
      return getFamilyColor(point.primaryDiagnosis)

    default:
      return FALLBACK_COLOR
  }
}
