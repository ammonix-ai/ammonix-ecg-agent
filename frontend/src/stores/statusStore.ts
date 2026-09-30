/**
 * Backend status — GET /api/status.
 *
 * Two consumers: the header dots, and the ECG Agent composer, which disables
 * itself when the model endpoint is unreachable. The store distinguishes
 * "backend down" from "backend up, model off"; the agent page says which.
 */

import { create } from 'zustand';
import { getApiStatus, type StatusProbe } from '@/services/statusService';

interface StatusState {
  probe: StatusProbe | null;
  loading: boolean;
  /** epoch ms of the last completed probe, or null. */
  checkedAt: number | null;
  refresh: () => Promise<void>;
}

export const useStatusStore = create<StatusState>((set) => ({
  probe: null,
  loading: false,
  checkedAt: null,

  refresh: async () => {
    set({ loading: true });
    const probe = await getApiStatus();
    set({ probe, loading: false, checkedAt: Date.now() });
  },
}));

/** True only when the backend answered AND reported the model reachable. */
export function llmReady(probe: StatusProbe | null): boolean {
  return Boolean(probe?.reachable && probe.status?.llm === true);
}

export interface InstalledData {
  universe: boolean;
  model: boolean;
  traces: boolean;
  traceCount: number | null;
}

/**
 * Which data packages this backend has installed. A fresh clone of the public
 * repository has none of them (they are not part of the code release), and the
 * pages say so instead of showing raw API errors. An unfinished probe or an
 * older backend that does not report a package counts as installed.
 */
export function installedData(probe: StatusProbe | null): InstalledData {
  const s = probe?.status;
  const u = s?.universe;
  return {
    universe: typeof u === 'object' && u !== null ? u.ready !== false : u !== false,
    model: s?.model?.packagePresent !== false,
    traces: s?.traces?.available !== false,
    traceCount: s?.traces?.records ?? null,
  };
}

/** Shown wherever a missing package empties a page. */
export const MISSING_DATA_NOTE =
  'It is not part of this code release; see "Data and model" in the README. The hosted demo at ecg.ammonix.ai runs with it.';

/**
 * False only when the backend explicitly reported uploads disabled. A backend
 * that predates the flag (or an unfinished probe) keeps the upload controls,
 * matching its behaviour; the server rejects with 403 regardless.
 */
export function uploadsAllowed(probe: StatusProbe | null): boolean {
  return probe?.status?.analyze?.uploadsEnabled !== false;
}
