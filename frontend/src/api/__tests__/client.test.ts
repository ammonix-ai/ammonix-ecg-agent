/**
 * API Client tests — verifies request construction, error handling, and
 * response parsing for frontend's client.ts.
 *
 * Adapted from frontend/src/api/__tests__/client.test.ts (V1) — V3's client
 * exposes a smaller surface: ApiError carries (status, statusText, body, url)
 * but no NetworkError/TimeoutError classes and no timeout option. Network
 * failures surface as ApiError with status=0.
 *
 * V3 also has a 204/content-length=0 short-circuit returning undefined
 * (added in V3.11 — covers DELETE callers across tribe/branch/skill/etc.).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { ApiError, apiGet, apiPost, apiPut, apiDelete, checkApiHealth } from '../client'

/* ── helpers ── */

function jsonResponse(data: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : '',
    headers: new Headers({ 'content-type': 'application/json' }),
    json: () => Promise.resolve(data),
    text: () => Promise.resolve(JSON.stringify(data)),
  } as unknown as Response
}

function noContentResponse(): Response {
  return {
    ok: true,
    status: 204,
    statusText: 'No Content',
    headers: new Headers({ 'content-length': '0' }),
    json: () => Promise.reject(new SyntaxError('no body')),
    text: () => Promise.resolve(''),
  } as unknown as Response
}

function errorResponse(status: number, body: unknown = {}): Response {
  return {
    ok: false,
    status,
    statusText: status === 404 ? 'Not Found' : status === 500 ? 'Server Error' : '',
    headers: new Headers({ 'content-type': 'application/json' }),
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  } as unknown as Response
}

/* ── fetch mock ── */

const fetchSpy = vi.fn<(...args: unknown[]) => Promise<Response>>()

beforeEach(() => {
  fetchSpy.mockReset()
  vi.stubGlobal('fetch', fetchSpy)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

/* ═══════════════════════════════════════════════════════════════════
   apiGet
   ═══════════════════════════════════════════════════════════════════ */

describe('apiGet', () => {
  it('resolves parsed JSON on 200', async () => {
    const payload = { items: [1, 2, 3], total: 3 }
    fetchSpy.mockResolvedValueOnce(jsonResponse(payload))

    const result = await apiGet<typeof payload>('/api/widgets')

    expect(result).toEqual(payload)
    expect(fetchSpy).toHaveBeenCalledOnce()
    expect(fetchSpy.mock.calls[0]?.[0]).toBe('/api/widgets')
  })

  it('returns undefined on 204 No Content', async () => {
    fetchSpy.mockResolvedValueOnce(noContentResponse())

    const result = await apiGet<undefined>('/api/fire-and-forget')

    expect(result).toBeUndefined()
  })

  it('throws ApiError on 404 with correct shape', async () => {
    fetchSpy.mockResolvedValueOnce(errorResponse(404, { detail: 'Not found' }))

    try {
      await apiGet('/api/missing')
      expect.unreachable('should have thrown')
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError)
      const err = e as ApiError
      expect(err.status).toBe(404)
      expect(err.body).toEqual({ detail: 'Not found' })
    }
  })

  it('throws ApiError on 500 with body captured', async () => {
    fetchSpy.mockResolvedValueOnce(errorResponse(500, { detail: 'Internal error' }))

    try {
      await apiGet('/api/broken')
      expect.unreachable('should have thrown')
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError)
      expect((e as ApiError).status).toBe(500)
    }
  })

  it('throws ApiError(status=0) on fetch TypeError (network failure)', async () => {
    fetchSpy.mockRejectedValueOnce(new TypeError('Failed to fetch'))

    try {
      await apiGet('/api/offline')
      expect.unreachable('should have thrown')
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError)
      expect((e as ApiError).status).toBe(0)
    }
  })

  it('uses GET method', async () => {
    fetchSpy.mockResolvedValueOnce(jsonResponse({}))

    await apiGet('/api/check')

    const init = fetchSpy.mock.calls[0]?.[1] as RequestInit
    expect(init.method).toBe('GET')
  })
})

/* ═══════════════════════════════════════════════════════════════════
   apiPost
   ═══════════════════════════════════════════════════════════════════ */

describe('apiPost', () => {
  it('sends body with correct Content-Type', async () => {
    const body = { patient_id: 'A0001', cohort: 'Ar' }
    fetchSpy.mockResolvedValueOnce(jsonResponse({ success: true }))

    await apiPost('/api/inference/run', body)

    const init = fetchSpy.mock.calls[0]?.[1] as RequestInit
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json')
    expect(init.body).toBe(JSON.stringify(body))
  })

  it('uses POST method', async () => {
    fetchSpy.mockResolvedValueOnce(jsonResponse({}))

    await apiPost('/api/create', { name: 'test' })

    const init = fetchSpy.mock.calls[0]?.[1] as RequestInit
    expect(init.method).toBe('POST')
  })

  it('resolves JSON on success', async () => {
    const payload = { id: 42, created: true }
    fetchSpy.mockResolvedValueOnce(jsonResponse(payload, 201))

    const result = await apiPost<typeof payload>('/api/items', {})

    expect(result).toEqual(payload)
  })
})

/* ═══════════════════════════════════════════════════════════════════
   apiPut + apiDelete
   ═══════════════════════════════════════════════════════════════════ */

describe('apiPut', () => {
  it('uses PUT method and sends body', async () => {
    fetchSpy.mockResolvedValueOnce(jsonResponse({ updated: true }))

    await apiPut('/api/items/1', { name: 'new' })

    const init = fetchSpy.mock.calls[0]?.[1] as RequestInit
    expect(init.method).toBe('PUT')
    expect(init.body).toBe(JSON.stringify({ name: 'new' }))
  })
})

describe('apiDelete', () => {
  it('uses DELETE method without body', async () => {
    fetchSpy.mockResolvedValueOnce(noContentResponse())

    await apiDelete('/api/items/1')

    const init = fetchSpy.mock.calls[0]?.[1] as RequestInit
    expect(init.method).toBe('DELETE')
    expect(init.body).toBeUndefined()
  })

  it('returns undefined on 204 (the V3.11 fix path)', async () => {
    fetchSpy.mockResolvedValueOnce(noContentResponse())

    const result = await apiDelete('/api/skills/skill-42')

    expect(result).toBeUndefined()
  })
})

/* ═══════════════════════════════════════════════════════════════════
   checkApiHealth
   ═══════════════════════════════════════════════════════════════════ */

describe('checkApiHealth', () => {
  it('returns true on 200', async () => {
    fetchSpy.mockResolvedValueOnce(jsonResponse({ status: 'healthy' }))

    const result = await checkApiHealth()

    expect(result).toBe(true)
  })

  it('returns false on network error', async () => {
    fetchSpy.mockRejectedValueOnce(new TypeError('Failed to fetch'))

    const result = await checkApiHealth()

    expect(result).toBe(false)
  })

  it('calls /health endpoint', async () => {
    fetchSpy.mockResolvedValueOnce(jsonResponse({ status: 'ok' }))

    await checkApiHealth()

    expect(fetchSpy.mock.calls[0]?.[0]).toBe('/health')
  })
})

/* ═══════════════════════════════════════════════════════════════════
   AbortSignal forwarding (C2.CI — cancellable apiGet/apiPost/…)
   ═══════════════════════════════════════════════════════════════════ */

describe('AbortSignal forwarding', () => {
  it('forwards AbortSignal to fetch on apiGet', async () => {
    fetchSpy.mockResolvedValueOnce(jsonResponse({ ok: true }))
    const controller = new AbortController()
    await apiGet('/api/cancellable', controller.signal)
    const init = fetchSpy.mock.calls[0]?.[1] as RequestInit
    expect(init.signal).toBe(controller.signal)
  })

  it('omits signal when none provided (legacy callers unaffected)', async () => {
    fetchSpy.mockResolvedValueOnce(jsonResponse({ ok: true }))
    await apiGet('/api/no-signal')
    const init = fetchSpy.mock.calls[0]?.[1] as RequestInit
    expect(init.signal).toBeUndefined()
  })

  it('re-throws DOMException AbortError without wrapping in ApiError', async () => {
    const abortErr = new DOMException('aborted', 'AbortError')
    fetchSpy.mockRejectedValueOnce(abortErr)
    const controller = new AbortController()
    controller.abort()
    await expect(apiGet('/api/will-abort', controller.signal)).rejects.toBe(abortErr)
  })

  it('forwards AbortSignal to fetch on apiPost', async () => {
    fetchSpy.mockResolvedValueOnce(jsonResponse({ ok: true }))
    const controller = new AbortController()
    await apiPost('/api/cancellable', { x: 1 }, controller.signal)
    const init = fetchSpy.mock.calls[0]?.[1] as RequestInit
    expect(init.signal).toBe(controller.signal)
  })
})
