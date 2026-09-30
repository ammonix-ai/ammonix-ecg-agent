/**
 * Analyze state — shared by the Analyze page and the ECG Agent page so the
 * agent talks about the recording you are actually looking at.
 *
 * Endpoints: GET /api/records, GET /api/records/{id}/signal, POST /api/analyze.
 *
 * No mock fallback anywhere in here. If a call fails the error string is kept
 * and rendered; the store never substitutes a plausible-looking result.
 */

import { create } from 'zustand';
import { listRecords, getRecordSignal } from '@/services/recordsService';
import { analyzeRecord, analyzeUpload } from '@/services/analyzeService';
import { adaptRecordSignal, type AdaptedSignal } from '@/utils/ecgAdapter';
import type {
  AnalyzeResult,
  RecordQuery,
  RecordSignal,
  RecordSummary,
} from '@/types/analyze';

export const PAGE_SIZE = 50;

/** Where the waveform on screen came from. Drives the training-overlap caveat. */
export type RecordOrigin = 'shipped' | 'upload';

function errText(err: unknown, fallback: string): string {
  if (err instanceof Error && err.message) return err.message;
  return fallback;
}

interface AnalyzeState {
  // ── browse ──
  filters: Required<Pick<RecordQuery, 'source' | 'diagnosis' | 'q'>>;
  offset: number;
  records: RecordSummary[];
  total: number;
  /** Gold labels available under the current source, corpus-wide (filter suggestions). */
  diagnoses: string[];
  listLoading: boolean;
  listError: string | null;

  // ── selection ──
  selected: RecordSummary | null;
  origin: RecordOrigin;
  /** File names of the current upload, when origin === 'upload'. */
  uploadNames: string[];

  // ── waveform ──
  signal: RecordSignal | null;
  adapted: AdaptedSignal | null;
  signalLoading: boolean;
  signalError: string | null;

  // ── classification ──
  result: AnalyzeResult | null;
  analyzing: boolean;
  analyzeError: string | null;

  setFilter: (patch: Partial<AnalyzeState['filters']>) => void;
  setOffset: (offset: number) => void;
  loadRecords: () => Promise<void>;
  selectRecord: (record: RecordSummary) => Promise<void>;
  /** Deep-link entry: /analyze?recording=<id>, used by the universe panel. */
  selectByRecordingId: (recordingId: string) => Promise<void>;
  analyzeSelected: () => Promise<void>;
  uploadAndAnalyze: (files: File[]) => Promise<void>;
  clearSelection: () => void;
}

let listSeq = 0;
let signalSeq = 0;

export const useAnalyzeStore = create<AnalyzeState>((set, get) => ({
  filters: { source: '', diagnosis: '', q: '' },
  offset: 0,
  records: [],
  total: 0,
  diagnoses: [],
  listLoading: false,
  listError: null,

  selected: null,
  origin: 'shipped',
  uploadNames: [],

  signal: null,
  adapted: null,
  signalLoading: false,
  signalError: null,

  result: null,
  analyzing: false,
  analyzeError: null,

  setFilter: (patch) => {
    set((s) => ({ filters: { ...s.filters, ...patch }, offset: 0 }));
  },

  setOffset: (offset) => set({ offset: Math.max(0, offset) }),

  loadRecords: async () => {
    const seq = ++listSeq;
    const { filters, offset } = get();
    set({ listLoading: true, listError: null });
    try {
      const query: RecordQuery = { limit: PAGE_SIZE, offset };
      if (filters.source) query.source = filters.source;
      if (filters.diagnosis) query.diagnosis = filters.diagnosis;
      if (filters.q) query.q = filters.q;
      const data = await listRecords(query);
      if (seq !== listSeq) return; // a newer query already landed
      set({
        records: data.records ?? [],
        total: data.total ?? 0,
        // Keep the previous vocabulary if the response omits it, so the
        // suggestion list never empties out mid-session.
        diagnoses: data.diagnoses ?? get().diagnoses,
        listLoading: false,
      });
    } catch (err) {
      if (seq !== listSeq) return;
      set({
        listLoading: false,
        listError: errText(err, 'Could not list the shipped recordings.'),
        records: [],
        total: 0,
      });
    }
  },

  selectRecord: async (record) => {
    const seq = ++signalSeq;
    set({
      selected: record,
      origin: 'shipped',
      uploadNames: [],
      signal: null,
      adapted: null,
      signalLoading: true,
      signalError: null,
      result: null,
      analyzeError: null,
    });
    try {
      const signal = await getRecordSignal(record.recordingId);
      if (seq !== signalSeq) return;
      set({ signal, adapted: adaptRecordSignal(signal), signalLoading: false });
    } catch (err) {
      if (seq !== signalSeq) return;
      set({
        signalLoading: false,
        signalError: errText(err, 'Could not read the waveform for this recording.'),
      });
    }
  },

  selectByRecordingId: async (recordingId) => {
    if (get().selected?.recordingId === recordingId) return;
    set({ signalLoading: true, signalError: null, result: null, analyzeError: null });
    try {
      const found = await listRecords({ q: recordingId, limit: 25 });
      const match = (found.records ?? []).find((r) => r.recordingId === recordingId);
      if (!match) {
        set({
          signalLoading: false,
          signalError: `${recordingId} is not among the shipped recordings.`,
        });
        return;
      }
      await get().selectRecord(match);
    } catch (err) {
      set({
        signalLoading: false,
        signalError: errText(err, `Could not look up ${recordingId}.`),
      });
    }
  },

  analyzeSelected: async () => {
    const { selected, analyzing } = get();
    if (!selected || analyzing) return;
    set({ analyzing: true, analyzeError: null });
    try {
      const result = await analyzeRecord(selected.recordingId);
      set({ result, analyzing: false });
    } catch (err) {
      set({
        analyzing: false,
        analyzeError: errText(err, 'The analysis request failed.'),
      });
    }
  },

  uploadAndAnalyze: async (files) => {
    if (get().analyzing) return;
    const seq = ++signalSeq;
    set({
      analyzing: true,
      analyzeError: null,
      result: null,
      selected: null,
      origin: 'upload',
      uploadNames: files.map((f) => f.name),
      // Cleared so the panel never shows the previous record's trace while the
      // upload is in flight. The analyze response carries the new waveform.
      signal: null,
      adapted: null,
      signalError: null,
      signalLoading: true,
    });
    try {
      const result = await analyzeUpload(files);
      if (seq !== signalSeq) return;
      // Uploads are not retained server-side, so the samples come back with the
      // result rather than from GET /api/records/{id}/signal.
      const uploaded = result.signal ?? null;
      set({
        result: { ...result, uploaded: true },
        analyzing: false,
        signal: uploaded,
        adapted: uploaded ? adaptRecordSignal(uploaded) : null,
        signalLoading: false,
        signalError: uploaded
          ? null
          : 'The analysis succeeded but the server returned no waveform for this upload.',
      });
    } catch (err) {
      if (seq !== signalSeq) return;
      set({
        analyzing: false,
        signalLoading: false,
        analyzeError: errText(err, 'The upload could not be analyzed.'),
      });
    }
  },

  clearSelection: () => {
    signalSeq += 1;
    set({
      selected: null,
      origin: 'shipped',
      uploadNames: [],
      signal: null,
      adapted: null,
      signalLoading: false,
      signalError: null,
      result: null,
      analyzing: false,
      analyzeError: null,
    });
  },
}));
