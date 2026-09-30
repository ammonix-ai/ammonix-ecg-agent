/**
 * API client for the Ammonix ECG Agent backend.
 *
 * NO mock fallbacks. Every call hits the real backend.
 * Failures are logged and propagated as errors.
 */

/**
 * Where the backend lives in dev. Mirrors the proxy target in vite.config.ts —
 * change both together.
 */
export const DEV_API_ORIGIN = 'http://127.0.0.1:8100'

/**
 * Base for every request. Empty by default, which means same-origin: the dev
 * server proxies `/api` and `/health` to DEV_API_ORIGIN, so requests need no
 * CORS grant and POSTs skip the preflight. Set VITE_API_URL to talk to a
 * backend on another origin (that backend must then allow this origin).
 *
 * Import this rather than reading import.meta.env at each call site — the
 * private repo had four different spellings of the base URL and they drifted.
 */
export const API_BASE = import.meta.env.VITE_API_URL || ''

const API_URL = API_BASE

const log = (level: 'info' | 'warn' | 'error', msg: string, ...args: unknown[]) => {
  const timestamp = new Date().toISOString()
  const prefix = `[API ${level.toUpperCase()}] ${timestamp}`
  if (level === 'error') console.error(prefix, msg, ...args)
  else if (level === 'warn') console.warn(prefix, msg, ...args)
  else console.log(prefix, msg, ...args)
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    public body: unknown,
    public url: string,
  ) {
    super(`API ${status} ${statusText}: ${url}`)
    this.name = 'ApiError'
  }
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const url = `${API_URL}${path}`
  log('info', `${method} ${path}`)

  const init: RequestInit = {
    method,
    headers: { 'Content-Type': 'application/json' },
  }
  if (body !== undefined) {
    init.body = JSON.stringify(body)
  }
  if (signal !== undefined) {
    init.signal = signal
  }

  const t0 = performance.now()
  let response: Response

  try {
    response = await fetch(url, init)
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      log('info', `${method} ${path} aborted`)
      throw err
    }
    log('error', `Network error: ${method} ${path}`, err)
    throw new ApiError(0, 'Network Error', err, url)
  }

  const elapsed = Math.round(performance.now() - t0)

  if (!response.ok) {
    // C2.FW followup: read body once as text, then try JSON.parse.
    // Pre-fix used `try { .json() } catch { .text() }` which is broken
    // because .json() consumes the body stream even on parse failure,
    // so the fallback .text() threw 'body stream already read'. Mirror
    // _readErrorBody's pattern (the SSE helper had the correct fix).
    const errorBody = await _readErrorBody(response)
    log('error', `${method} ${path} -> ${response.status} (${elapsed}ms)`, errorBody)
    throw new ApiError(response.status, response.statusText, errorBody, url)
  }

  // 204 No Content has no body — Response.json() rejects on empty body.
  if (response.status === 204 || response.headers.get('content-length') === '0') {
    log('info', `${method} ${path} -> ${response.status} (${elapsed}ms)`)
    return undefined as T
  }

  const data = await response.json() as T
  log('info', `${method} ${path} -> ${response.status} (${elapsed}ms)`)
  return data
}

export function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  return request<T>('GET', path, undefined, signal)
}

export function apiPost<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>('POST', path, body, signal)
}

export function apiPut<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>('PUT', path, body, signal)
}

export function apiDelete<T>(path: string, signal?: AbortSignal): Promise<T> {
  return request<T>('DELETE', path, undefined, signal)
}

export function apiPatch<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>('PATCH', path, body, signal)
}

// ─── Error-body helper ─────────────────────────────────────
// The generic SSE helpers that used to live here (apiStream,
// apiStreamFormData, connectSSEJson) went with the pages that used them. The
// one stream left in the open build — POST /api/chat/stream — reads its own
// response in services/agentChatService.ts, where the wire format it tolerates
// is documented next to the parser.

async function _readErrorBody(response: Response): Promise<unknown> {
  // Read the body once as text, then try to JSON-parse it. We can't
  // call `.json()` then fall through to `.text()` because the body
  // stream is consumed by the first read (even on parse failure), so
  // the fallback throws 'body stream already read'.
  let text: string
  try {
    text = await response.text()
  } catch {
    return null
  }
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

/**
 * Health check — returns true if backend is reachable.
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const resp = await fetch(`${API_URL}/health`, { method: 'GET' })
    return resp.ok
  } catch {
    return false
  }
}
