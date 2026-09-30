/**
 * Status service — GET /api/status.
 *
 * The one field the contract pins is `llm`. When it is false the ECG Agent
 * composer disables itself and explains how to point the backend at an
 * OpenAI-compatible endpoint; Universe and Analyze stay fully usable.
 *
 * A network failure is reported as `reachable: false` rather than as
 * `llm: false`, because "the backend is down" and "the backend is up and the
 * model is not" are different situations and the page says which one it is.
 */

import { apiGet } from '@/api/client';
import type { ApiStatus } from '@/types/analyze';

export interface StatusProbe {
  reachable: boolean;
  status: ApiStatus | null;
  error: string | null;
}

export async function getApiStatus(signal?: AbortSignal): Promise<StatusProbe> {
  try {
    const status = await apiGet<ApiStatus>('/api/status', signal);
    return { reachable: true, status, error: null };
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err;
    return {
      reachable: false,
      status: null,
      error: err instanceof Error ? err.message : 'Backend unreachable',
    };
  }
}
