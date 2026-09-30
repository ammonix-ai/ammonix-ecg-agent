/**
 * Colour-mode tests for the universe viewer.
 *
 * Two defects are pinned here:
 *
 * 1. error_highlight used to compare predictions against ALL gold labels,
 *    including labels the model has no class for ('sinus rhythm',
 *    'normal ecg', 'lvef >=55% control'). A normal ECG was painted as having
 *    "missed sinus rhythm" — a diagnosis it was never able to predict. The
 *    comparison now runs over gold ∩ scored classes, the same basis
 *    `xgb_correct` was computed on.
 *
 * 2. DIAGNOSIS_FAMILIES predated this universe, so 46.2% of points fell
 *    through to the grey fallback (the two LVEF classes alone are 29,153 of
 *    them) and 'vt risk' was keyed with a space where the data says 'vt-risk'.
 *
 * The last block re-measures both against the real shipped universe when
 * `_staging/` is present. That directory is gitignored (the payload ships
 * separately), so the block skips in a clean checkout — the synthetic cases
 * above carry the contract on their own.
 */
import { describe, it, expect } from 'vitest'
import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import {
  classifyErrorHighlight,
  getFamilyColor,
  getFamilyForDiagnosis,
  getPointColor,
  DIAGNOSIS_FAMILIES,
  ERROR_HIGHLIGHT_COLORS,
  FALLBACK_COLOR,
  FAMILY_COLORS,
  type ErrorHighlightBucket,
} from '../colorUtils'
import type { EnrichedProjectionPoint } from '@/types/projection'

/* ── helpers ── */

/** The 32 scored classes, lowercased, as `useScoredClasses` builds them from
 *  `sourceSafe.thresholds`. Only the members these tests need. */
const SCORED = new Set([
  'atrial fibrillation',
  'st deviation',
  'nstemi',
  'stemi',
  'sinus bradycardia',
  'lvef <40% expanded mimic ef>=55 controls',
])

function mkPoint(
  diagnoses: string[],
  xgbPredictions: string[],
  primaryDiagnosis = diagnoses[0] ?? 'unknown',
): EnrichedProjectionPoint {
  return {
    patientId: 'SS-00001',
    x: 0, y: 0, z: 0,
    primaryDiagnosis,
    diagnoses,
    cohort: 'Apr28 multi-source',
    color: '#000000',
    symbol: 'sphere',
    xgbPredictions,
    pipelinePredictions: xgbPredictions,
    tribes: [],
    multiTribe: false,
    modelClassClean: false,
  }
}

/* ── defect 1: gold labels the model has no class for ── */

describe('classifyErrorHighlight — gold set is intersected with the scored classes', () => {
  it('does not call a normal ECG a missed diagnosis', () => {
    // The regression case. Gold is only unscored labels, so nothing can be
    // missed here. Before the fix this was 'both' — a phantom false negative
    // on 'sinus rhythm', which the model has no class for.
    // It is still an error though: a finding was called on a record whose only
    // labels are normal, so it is a pure false positive, not ungradeable.
    const p = mkPoint(['sinus rhythm', 'normal ecg'], ['sinus bradycardia'])
    expect(classifyErrorHighlight(p, SCORED)).toBe<ErrorHighlightBucket>('over')
  })

  it('calls correct silence on a normal ECG a match', () => {
    // Nothing to predict and nothing predicted: the model's best behaviour on
    // a normal recording, and 17,474 points of the universe. Greying this out
    // gave it the same colour as the biggest error category.
    const p = mkPoint(['sinus rhythm', 'normal ecg'], [])
    expect(classifyErrorHighlight(p, SCORED)).toBe<ErrorHighlightBucket>('match')
  })

  it('ignores unscored gold labels alongside a scored one', () => {
    // 'sinus rhythm' and 'lvef >=55% control' are not classes; the model
    // called the one class it has, so this is an exact match. Before the fix
    // it was 'missed'.
    const p = mkPoint(
      ['sinus rhythm', 'atrial fibrillation', 'lvef >=55% control'],
      ['atrial fibrillation'],
    )
    expect(classifyErrorHighlight(p, SCORED)).toBe<ErrorHighlightBucket>('match')
  })

  it('still reports a genuine miss of a scored class', () => {
    const p = mkPoint(['atrial fibrillation', 'st deviation'], ['atrial fibrillation'])
    expect(classifyErrorHighlight(p, SCORED)).toBe<ErrorHighlightBucket>('missed')
  })

  it('still reports over-calling', () => {
    const p = mkPoint(['atrial fibrillation'], ['atrial fibrillation', 'nstemi'])
    expect(classifyErrorHighlight(p, SCORED)).toBe<ErrorHighlightBucket>('over')
  })

  it('still reports both directions at once', () => {
    const p = mkPoint(['atrial fibrillation', 'st deviation'], ['atrial fibrillation', 'nstemi'])
    expect(classifyErrorHighlight(p, SCORED)).toBe<ErrorHighlightBucket>('both')
  })

  it('matches across the case difference the wire carries', () => {
    // 'STEMI' in labels and thresholds, 'stemi' in primary.
    const p = mkPoint(['STEMI'], ['stemi'])
    expect(classifyErrorHighlight(p, SCORED)).toBe<ErrorHighlightBucket>('match')
  })

  it('grades nothing when the scored class list is unavailable', () => {
    // Without the class list we cannot say what a miss would be. Guessing is
    // what produced the bug, so the honest answer is "not graded".
    const p = mkPoint(['atrial fibrillation'], ['st deviation'])
    expect(classifyErrorHighlight(p, undefined)).toBe<ErrorHighlightBucket>('na')
    expect(classifyErrorHighlight(p, new Set())).toBe<ErrorHighlightBucket>('na')
  })

  it('reports a miss when the model called nothing but had something to find', () => {
    expect(classifyErrorHighlight(mkPoint(['atrial fibrillation'], []), SCORED)).toBe('missed')
  })
})

describe('getPointColor — error_highlight four-colour scheme', () => {
  // Green shades for what the model found, warm for what it missed. Over-call
  // is green because the reference labels are incomplete, so an extra call is
  // not necessarily an error; a miss is the clinically worse failure.
  const cases: [ErrorHighlightBucket, string][] = [
    ['match', '#4ADE80'],  // light green
    ['over', '#15803D'],   // dark green
    ['missed', '#D97706'], // orange
    ['both', '#DC2626'],   // red
    ['na', FALLBACK_COLOR],
  ]
  it.each(cases)('%s renders %s', (bucket, hex) => {
    expect(ERROR_HIGHLIGHT_COLORS[bucket]).toBe(hex)
  })

  it('threads the scored classes through to the point colour', () => {
    // A call on an all-normal record is a false positive, and must render as
    // one rather than disappearing into the ungraded grey.
    const normal = mkPoint(['sinus rhythm', 'normal ecg'], ['sinus bradycardia'])
    expect(getPointColor(normal, 'error_highlight', undefined, undefined, undefined, SCORED))
      .toBe('#15803D')

    // Grey is now reserved for the one genuinely unknowable case.
    expect(getPointColor(normal, 'error_highlight', undefined, undefined, undefined, undefined))
      .toBe(FALLBACK_COLOR)

    const miss = mkPoint(['atrial fibrillation', 'st deviation'], ['atrial fibrillation'])
    expect(getPointColor(miss, 'error_highlight', undefined, undefined, undefined, SCORED))
      .toBe('#D97706')
  })
})

/* ── defect 2: family coverage ── */

describe('DIAGNOSIS_FAMILIES covers this universe', () => {
  it('puts both reduced-EF classes in the same family', () => {
    // Reduced EF is systolic dysfunction, not an arrhythmia: 'lvef ≤45%' moved
    // out of Ventricular so the two never split across families.
    expect(getFamilyForDiagnosis('lvef ≤45%')).toBe('Structural')
    expect(getFamilyForDiagnosis('lvef <40% expanded mimic ef>=55 controls')).toBe('Structural')
    expect(getFamilyColor('lvef ≤45%')).toBe(getFamilyColor('lvef <40% expanded mimic ef>=55 controls'))
  })

  it('gives the EF>=55 control its own neutral family, not a disease one', () => {
    expect(getFamilyForDiagnosis('lvef >=55% control')).toBe('Controls & Normal')
    expect(getFamilyForDiagnosis('normal ecg')).toBe('Controls & Normal')
    expect(getFamilyColor('lvef >=55% control')).toBe(FAMILY_COLORS['Controls & Normal'])
    expect(getFamilyColor('lvef >=55% control')).not.toBe(FALLBACK_COLOR)
  })

  it('keeps sinus rhythm in Rhythm & Rate', () => {
    expect(getFamilyForDiagnosis('sinus rhythm')).toBe('Rhythm & Rate')
  })

  it("keys vt-risk the way the data spells it", () => {
    expect(getFamilyForDiagnosis('vt-risk')).toBe('Ventricular')
    expect(DIAGNOSIS_FAMILIES['vt risk']).toBeUndefined()
  })

  it('covers the remaining stragglers', () => {
    expect(getFamilyForDiagnosis('sudden death')).toBe('Ventricular')
    expect(getFamilyForDiagnosis('chagas disease')).toBe('Structural')
    expect(getFamilyForDiagnosis('tachycardia')).toBe('Rhythm & Rate')
  })

  it("leaves 'unknown' grey — that is honest", () => {
    expect(DIAGNOSIS_FAMILIES['unknown']).toBeUndefined()
    expect(getFamilyColor('unknown')).toBe(FALLBACK_COLOR)
  })

  it('is case-insensitive, because primary and labels disagree on case', () => {
    expect(getFamilyColor('STEMI')).toBe(FAMILY_COLORS['Ischemia & Injury'])
  })
})

/* ── measured against the real shipped universe (skipped without _staging) ── */

const UNIVERSE = path.resolve(__dirname, '../../../../_staging/universe')
const PATIENTS = path.join(UNIVERSE, 'patients.jsonl')
const METADATA = path.join(UNIVERSE, 'metadata.json')
const havePayload = existsSync(PATIENTS) && existsSync(METADATA)

describe.skipIf(!havePayload)('measured on the published universe', () => {
  const thresholds: Record<string, number> =
    havePayload ? JSON.parse(readFileSync(METADATA, 'utf-8')).thresholds : {}
  const scored = new Set(Object.keys(thresholds).map((k) => k.toLowerCase()))
  const rows: { labels: string[]; predictions: string[]; primary: string }[] =
    havePayload
      ? readFileSync(PATIENTS, 'utf-8').split('\n').filter(Boolean).map((l) => JSON.parse(l))
      : []

  it('ships 63,256 rows scored over 32 classes', () => {
    expect(rows.length).toBe(63256)
    expect(scored.size).toBe(32)
  })

  it('grades every point, and reconciles exactly with Accuracy', () => {
    const counts: Record<ErrorHighlightBucket, number> = {
      both: 0, over: 0, missed: 0, match: 0, na: 0,
    }
    for (const r of rows) {
      counts[classifyErrorHighlight(mkPoint(r.labels ?? [], r.predictions ?? []), scored)]++
    }

    // These moved when the recovered MIMIC diagnoses were merged: 46% of the
    // universe went from having no labels to having real ones, so there is now
    // far more to be right or wrong about. 'missed' rising from 158 to 2,659 is
    // the expected direction — the model was never trained on 12SL's broader
    // definitions, so it under-calls them.
    expect(counts.both).toBe(12946)
    expect(counts.missed).toBe(2659)
    expect(counts.over).toBe(27618)
    expect(counts.match).toBe(20033)
    expect(counts.na).toBe(0)

    // The point of the exercise: this mode must agree with Accuracy, which is
    // computed independently upstream and shipped as xgb_correct.
    const green = rows.filter((r) => (r as { xgb_correct?: boolean }).xgb_correct).length
    expect(counts.match).toBe(green)
    expect(counts.both + counts.over + counts.missed).toBe(rows.length - green)
  })

  it('leaves only the "unknown" primaries grey', () => {
    let grey = 0
    const greyPrimaries = new Set<string>()
    for (const r of rows) {
      if (!DIAGNOSIS_FAMILIES[(r.primary ?? 'unknown').toLowerCase()]) {
        grey++
        greyPrimaries.add(r.primary)
      }
    }
    // 'unknown' is the only honest grey: a record with no labels at all.
    // The label recovery briefly reintroduced 2,100 grey points by adding 23
    // primaries the family taxonomy had never seen — this test is what caught
    // it, so it is asserted by SET, not just by count. A new diagnosis reaching
    // the universe without a family fails here rather than quietly going grey.
    expect(greyPrimaries).toEqual(new Set(['unknown']))
    expect(grey / rows.length).toBeLessThan(0.001) // was 46.2%, then 3.3%
  })

  it('has a family for every class the model scores', () => {
    const missing = [...scored].filter((c) => !DIAGNOSIS_FAMILIES[c])
    expect(missing).toEqual([])
  })
})
