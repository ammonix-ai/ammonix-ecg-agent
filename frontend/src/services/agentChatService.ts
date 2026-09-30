/**
 * ECG Agent chat — POST /api/chat/stream (SSE).
 *
 * Body (docs/api-contract.md §3):
 *   { messages: [{role, content}], recordingId?, imageDataUrl? }
 *
 * The server assembles context from the analyze result, the six nearest
 * universe neighbours and any matching skills, then streams deltas.
 *
 * Wire format tolerated by the reader below, in the order it is tried:
 *   data: {"content": "...", "done": false}     ← the private route's shape
 *   data: {"delta": "..."} / {"text": "..."}    ← common alternates
 *   data: [DONE]                                ← terminator
 *   data: <bare text>                           ← last resort, emitted as-is
 *
 * The agent verifies; it does not diagnose. The diagnosis comes from the
 * classifier, the universe and the skills — the prompt sent from this page
 * never asks the model for an independent read.
 */

import { ApiError, API_BASE } from '@/api/client';
import type { ChatMessage } from '@/types/analyze';

export interface ChatStreamRequest {
  messages: ChatMessage[];
  recordingId?: string | null;
  /** Rendered 12-lead PNG as a `data:image/png;base64,...` URL. */
  imageDataUrl?: string | null;
}

export interface ChatStreamHandlers {
  onDelta: (text: string) => void;
  onComplete: () => void;
  onError: (error: Error) => void;
}

/** Pull the text out of one parsed SSE payload. Returns null for control frames. */
function deltaOf(payload: unknown): string | null {
  if (typeof payload === 'string') return payload;
  if (!payload || typeof payload !== 'object') return null;
  const obj = payload as Record<string, unknown>;
  for (const key of ['content', 'delta', 'text', 'token']) {
    const v = obj[key];
    if (typeof v === 'string' && v.length > 0) return v;
  }
  return null;
}

function isDone(payload: unknown): boolean {
  if (!payload || typeof payload !== 'object') return false;
  return (payload as Record<string, unknown>).done === true;
}

export function streamAgentChat(
  request: ChatStreamRequest,
  handlers: ChatStreamHandlers,
): AbortController {
  const url = `${API_BASE}/api/chat/stream`;
  const controller = new AbortController();

  const run = async () => {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({
        messages: request.messages,
        recordingId: request.recordingId ?? null,
        imageDataUrl: request.imageDataUrl ?? null,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail: unknown = null;
      try {
        const text = await response.text();
        try {
          detail = JSON.parse(text);
        } catch {
          detail = text;
        }
      } catch {
        /* body already gone; the status line is enough */
      }
      throw new ApiError(response.status, response.statusText, detail, url);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('Response carried no body to stream.');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const raw = line.slice(5).trim();
        if (!raw) continue;
        if (raw === '[DONE]') {
          handlers.onComplete();
          return;
        }
        let payload: unknown = raw;
        try {
          payload = JSON.parse(raw);
        } catch {
          /* not JSON — treat the line as a bare text delta */
        }
        const text = deltaOf(payload);
        if (text) handlers.onDelta(text);
        if (isDone(payload)) {
          handlers.onComplete();
          return;
        }
      }
    }
    handlers.onComplete();
  };

  run().catch((err: unknown) => {
    if (err instanceof Error && err.name === 'AbortError') return;
    handlers.onError(err instanceof Error ? err : new Error(String(err)));
  });

  return controller;
}
