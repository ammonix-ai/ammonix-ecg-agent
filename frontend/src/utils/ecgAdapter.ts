/**
 * GET /api/records/{id}/signal → the ECGData shape the 12-lead renderer wants.
 *
 * Lead matching is case-insensitive and tolerant of the WFDB spellings
 * (`AVR`, `avr`, `aVR`, `MDC_ECG_LEAD_I`), because header text varies across
 * the five source archives. A lead the payload does not carry is left out
 * rather than zero-filled — a flat line the machine never recorded would be a
 * fabricated waveform.
 */

import type { ECGData, LeadName, SampleArray } from '@/types/ecg';
import { LEAD_GROUPS } from '@/types/ecg';
import type { RecordSignal } from '@/types/analyze';

const CANONICAL: LeadName[] = LEAD_GROUPS.all;

function normalise(raw: string): LeadName | null {
  const cleaned = raw.trim().replace(/^MDC_ECG_LEAD_/i, '').replace(/[\s._-]/g, '');
  const upper = cleaned.toUpperCase();
  return CANONICAL.find((lead) => lead.toUpperCase() === upper) ?? null;
}

export interface AdaptedSignal {
  ecgData: ECGData;
  /** Canonical leads the payload did not carry. Rendered as a warning, not filled in. */
  missingLeads: LeadName[];
  /** Lead names in the payload that matched nothing canonical. */
  unknownLeads: string[];
}

export function adaptRecordSignal(signal: RecordSignal): AdaptedSignal {
  const leads: Partial<Record<LeadName, SampleArray>> = {};
  const unknownLeads: string[] = [];
  let maxSamples = 0;

  for (const lead of signal.leads ?? []) {
    const name = normalise(lead.name ?? '');
    if (!name) {
      if (lead.name) unknownLeads.push(lead.name);
      continue;
    }
    const samples = lead.samples ?? [];
    leads[name] = samples;
    if (samples.length > maxSamples) maxSamples = samples.length;
  }

  const missingLeads = CANONICAL.filter((lead) => leads[lead] === undefined);

  const ecgData: ECGData = {
    patientId: signal.recordingId,
    samplingRate: signal.samplingRate || 500,
    numSamples: maxSamples,
    durationSeconds:
      signal.durationSec || (signal.samplingRate ? maxSamples / signal.samplingRate : undefined),
    // The renderer indexes leads[name]; absent leads read as undefined and the
    // grid draws nothing for them, which is the honest rendering.
    leads: leads as Record<LeadName, SampleArray>,
  };

  return { ecgData, missingLeads, unknownLeads };
}
