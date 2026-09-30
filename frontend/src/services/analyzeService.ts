/**
 * Analyze service — the live frozen-classifier pipeline.
 *
 *   raw 12-lead WFDB → QPSI tokenizer → feat_order feature vector
 *   → 32 diagnoses x 5 frozen swarms → probabilities → placement
 *
 * Endpoint: POST /api/analyze (docs/api-contract.md §2). Two bodies:
 * `{recordingId}` for a shipped record, multipart for an upload.
 *
 * Nothing here fits anything. The classifier is frozen and the 63,256-point
 * embedding is fixed; a new recording is *placed into* it, never re-embedded.
 */

import { apiPost, ApiError, API_BASE } from '@/api/client';
import type { AnalyzeResult } from '@/types/analyze';

export function analyzeRecord(
  recordingId: string,
  signal?: AbortSignal,
): Promise<AnalyzeResult> {
  return apiPost<AnalyzeResult>('/api/analyze', { recordingId }, signal);
}

/**
 * Upload branch: a WFDB pair (`.hea` + `.mat`/`.dat`) or a single 12-lead CSV.
 *
 * Sent as multipart/form-data under the field name `files`. The browser sets
 * the boundary, so no Content-Type header is supplied — `apiPost` cannot be
 * reused here because it JSON-encodes the body.
 */
export async function analyzeUpload(
  files: File[],
  signal?: AbortSignal,
): Promise<AnalyzeResult> {
  const url = `${API_BASE}/api/analyze`;
  const form = new FormData();
  for (const f of files) form.append('files', f, f.name);

  const init: RequestInit = { method: 'POST', body: form };
  if (signal) init.signal = signal;

  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err;
    throw new ApiError(0, 'Network Error', err, url);
  }

  const text = await response.text();
  let body: unknown = text;
  try {
    body = JSON.parse(text);
  } catch {
    /* keep the raw text — the error path renders it */
  }

  if (!response.ok) {
    throw new ApiError(response.status, response.statusText, body, url);
  }
  return body as AnalyzeResult;
}

/**
 * Which files make a valid upload. WFDB needs the header AND the sample file;
 * a CSV stands alone. Returns null when the selection is acceptable.
 */
export function validateUpload(files: File[]): string | null {
  if (files.length === 0) return 'Pick a .hea + .mat/.dat pair, or a 12-lead .csv.';
  const names = files.map((f) => f.name.toLowerCase());
  const hasCsv = names.some((n) => n.endsWith('.csv'));
  if (hasCsv) {
    return files.length === 1 ? null : 'Upload the CSV on its own.';
  }
  const hasHea = names.some((n) => n.endsWith('.hea'));
  const hasSamples = names.some((n) => n.endsWith('.mat') || n.endsWith('.dat'));
  if (hasHea && hasSamples) return null;
  if (hasHea) return 'The .hea header needs its .mat or .dat sample file alongside it.';
  if (hasSamples) return 'The sample file needs its .hea header alongside it.';
  return 'Unrecognised files. Expected .hea + .mat/.dat, or a 12-lead .csv.';
}
