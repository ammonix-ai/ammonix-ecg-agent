/**
 * Records service — the 10,876 shipped open-licensed test recordings.
 *
 * Endpoints (docs/api-contract.md §2):
 *  - GET /api/records
 *  - GET /api/records/{recording_id}/signal
 *
 * No mock fallback. A failure surfaces as an error the page renders; it never
 * substitutes fabricated waveforms or labels.
 */

import { apiGet } from '@/api/client';
import type { RecordListResponse, RecordQuery, RecordSignal } from '@/types/analyze';

function buildQuery(query: RecordQuery): string {
  const qs = new URLSearchParams();
  if (query.source) qs.set('source', query.source);
  if (query.diagnosis) qs.set('diagnosis', query.diagnosis);
  if (query.q) qs.set('q', query.q);
  if (query.limit !== undefined) qs.set('limit', String(query.limit));
  if (query.offset !== undefined) qs.set('offset', String(query.offset));
  const s = qs.toString();
  return s ? `?${s}` : '';
}

export function listRecords(
  query: RecordQuery = {},
  signal?: AbortSignal,
): Promise<RecordListResponse> {
  return apiGet<RecordListResponse>(`/api/records${buildQuery(query)}`, signal);
}

export function getRecordSignal(
  recordingId: string,
  signal?: AbortSignal,
): Promise<RecordSignal> {
  return apiGet<RecordSignal>(
    `/api/records/${encodeURIComponent(recordingId)}/signal`,
    signal,
  );
}
