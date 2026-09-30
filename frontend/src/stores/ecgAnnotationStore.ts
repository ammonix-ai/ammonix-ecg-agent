/**
 * Annotations a reader draws on a trace, held in DOMAIN units.
 *
 * Why a store, and why not pixels
 * -------------------------------
 * These used to live in `useState` inside `ExpandedLeadView`, as x/y pixels of
 * that one zoomed single-lead viewport. Two problems followed from that:
 *
 *   1. Nothing outside the component could read them — so the 12-lead PNG sent
 *      to the agent never carried the reader's marks, and asking "what is this
 *      wave I marked?" reached a model that could not see the mark.
 *   2. Pixels are meaningless anywhere else. A mark at x=583 in a view zoomed
 *      2x and panned to 3.2 s lands somewhere entirely different on a 2x6
 *      12-lead render at a different scale.
 *
 * So an annotation is stored as (lead, seconds, millivolts): where it sits on
 * the signal, not where it sat on someone's screen. Any renderer can then place
 * it correctly at its own scale, and it survives zoom, pan and gain changes.
 *
 * Keyed by recordingId so switching recordings does not carry marks across.
 */

import { create } from 'zustand';

export type EcgAnnotationKind = 'marker' | 'label' | 'stroke';

export interface EcgAnnotation {
  id: string;
  /** Lead the annotation belongs to, e.g. "II", "V4". */
  lead: string;
  /** Seconds from the start of the recording. */
  tSeconds: number;
  /** Signal amplitude at the marked point, in millivolts. */
  mV: number;
  label: string;
  color: string;
  kind: EcgAnnotationKind;
  /** Freehand strokes, also in (seconds, millivolts). */
  points?: { tSeconds: number; mV: number }[];
}

interface EcgAnnotationState {
  byRecording: Record<string, EcgAnnotation[]>;
  add: (recordingId: string, annotation: EcgAnnotation) => void;
  remove: (recordingId: string, id: string) => void;
  clearLead: (recordingId: string, lead: string) => void;
  clearRecording: (recordingId: string) => void;
  /** All annotations for a recording; stable empty array when there are none. */
  forRecording: (recordingId: string) => EcgAnnotation[];
}

const EMPTY: EcgAnnotation[] = [];

export const useEcgAnnotationStore = create<EcgAnnotationState>((set, get) => ({
  byRecording: {},

  add: (recordingId, annotation) =>
    set((s) => ({
      byRecording: {
        ...s.byRecording,
        [recordingId]: [...(s.byRecording[recordingId] ?? []), annotation],
      },
    })),

  remove: (recordingId, id) =>
    set((s) => ({
      byRecording: {
        ...s.byRecording,
        [recordingId]: (s.byRecording[recordingId] ?? []).filter((a) => a.id !== id),
      },
    })),

  clearLead: (recordingId, lead) =>
    set((s) => ({
      byRecording: {
        ...s.byRecording,
        [recordingId]: (s.byRecording[recordingId] ?? []).filter((a) => a.lead !== lead),
      },
    })),

  clearRecording: (recordingId) =>
    set((s) => {
      const next = { ...s.byRecording };
      delete next[recordingId];
      return { byRecording: next };
    }),

  forRecording: (recordingId) => get().byRecording[recordingId] ?? EMPTY,
}));

// ---------------------------------------------------------------------------
// pixel <-> domain conversion
// ---------------------------------------------------------------------------

/**
 * Viewport geometry needed to convert between a lead view's pixels and the
 * signal's own units. All four come straight out of `ExpandedLeadView`.
 */
export interface LeadViewport {
  /** Seconds at the left edge of the visible window. */
  visibleStartSeconds: number;
  pixelsPerSecond: number;
  pixelsPerMillivolt: number;
  /** Y pixel of the 0 mV baseline within the viewport. */
  baselineY: number;
}

export function pixelToDomain(
  x: number,
  y: number,
  vp: LeadViewport,
): { tSeconds: number; mV: number } {
  return {
    tSeconds: vp.visibleStartSeconds + x / vp.pixelsPerSecond,
    // Screen y grows downward; millivolts grow upward.
    mV: (vp.baselineY - y) / vp.pixelsPerMillivolt,
  };
}

export function domainToPixel(
  tSeconds: number,
  mV: number,
  vp: LeadViewport,
): { x: number; y: number } {
  return {
    x: (tSeconds - vp.visibleStartSeconds) * vp.pixelsPerSecond,
    y: vp.baselineY - mV * vp.pixelsPerMillivolt,
  };
}
