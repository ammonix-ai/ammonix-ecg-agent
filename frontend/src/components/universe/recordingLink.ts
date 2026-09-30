import type { EnrichedProjectionPoint } from '@/types/projection';

/**
 * Only the 10,876 openly redistributable PhysioNet rows carry a recording id.
 * Every other row in the published universe has just its stable public display
 * id (SS-NNNNN) and no waveform to fetch, so there is nothing to open.
 *
 * The backend sets `patientId` to the recording id when there is one and falls
 * back to the display id otherwise — which is exactly the test here.
 */
export const NO_RECORDING_NA =
  'No waveform is published for this point — the universe row carries only its public display id, ' +
  'not an openly redistributable recording.';

export function recordingIdOf(patient: EnrichedProjectionPoint): string | null {
  const id = patient.patientId;
  if (!id || id.startsWith('idx-')) return null;
  if (patient.displayId && id === patient.displayId) return null;
  return id;
}

/** A point's identity, minimally — either id alone, or both when they differ. */
export type PointIds = { recordingId: string | null; displayId: string };

export function pointIdsOf(patient: EnrichedProjectionPoint): PointIds {
  return {
    recordingId: recordingIdOf(patient),
    displayId: patient.displayId ?? patient.patientId,
  };
}

/**
 * How a point is named in the UI.
 *
 * A record can carry two ids for the same trace: the PhysioNet recording id
 * (HR07445 — what a user searches, and what names the shipped trace file) and
 * the universe's public display id (SS-00936). Showing only one made a search
 * hit look like it had jumped to a different patient, so show both when they
 * differ, recording id first. The 50,810 MIMIC-derived and 1,570
 * clinical-partner rows have no recording id and keep their display id alone —
 * nothing is invented for them.
 */
export function pointIdLabel(patient: EnrichedProjectionPoint): string {
  const { recordingId, displayId } = pointIdsOf(patient);
  return recordingId && recordingId !== displayId
    ? `${recordingId} · ${displayId}`
    : displayId;
}

export function analyzeHref(recordingId: string): string {
  return `/analyze?recording=${encodeURIComponent(recordingId)}`;
}
