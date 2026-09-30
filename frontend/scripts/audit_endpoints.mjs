#!/usr/bin/env node
/**
 * Hit every backend endpoint this frontend depends on and print what came back.
 *
 * Usage:
 *   node scripts/audit_endpoints.mjs
 *   npm run audit-endpoints
 *   AUDIT_BASE=http://127.0.0.1:8102 npm run audit-endpoints
 *   AUDIT_RECORD=HR16370 npm run audit-endpoints    # which record to analyze
 *
 * The list below is the whole contract (docs/api-contract.md). If a page calls
 * something that is not here, one of the two is wrong.
 *
 * POST /api/analyze runs the real pipeline — tokenizer plus 32 diagnoses x 5
 * swarms — so it can take tens of seconds. It is last for that reason.
 */

const BASE = process.env.AUDIT_BASE || 'http://127.0.0.1:8100';
const RECORD = process.env.AUDIT_RECORD || 'HR16370';

async function report(method, path, label, body) {
  const url = `${BASE}${path}`;
  const t0 = Date.now();
  let status = '';
  let size = 0;
  let snippet = '';

  try {
    const init = { method };
    if (body !== undefined) {
      init.headers = { 'Content-Type': 'application/json' };
      init.body = JSON.stringify(body);
    }
    const res = await fetch(url, init);
    status = String(res.status);
    const text = await res.text();
    size = Buffer.byteLength(text, 'utf8');
    snippet = text.slice(0, 200).replace(/\s+/g, ' ').trim();
  } catch (err) {
    status = 'ERR';
    snippet = err instanceof Error ? `(${err.message})` : '(unknown error)';
  }

  console.log(`${method} ${path}`);
  console.log(`  used by: ${label}`);
  console.log(`  status:  ${status}  ${size} bytes  ${Date.now() - t0} ms`);
  console.log(`  body:    ${snippet}`);
  console.log('');
}

async function main() {
  console.log('==========================================');
  console.log('Ammonix ECG Agent — ENDPOINT AUDIT');
  console.log(`Base: ${BASE}`);
  console.log(`Date: ${new Date().toISOString()}`);
  console.log('==========================================\n');

  console.log('### Health & status ###');
  await report('GET', '/health', 'nothing in the UI; sanity check');
  await report('GET', '/api/status', 'header status dots + ECG Agent composer gate');

  console.log('### Universe ###');
  await report('GET', '/api/admin/cv/lattice-status', 'Universe — cohort selector');
  await report(
    'GET',
    '/api/admin/cv/projection/tsne?cohort_id=-60603',
    'Universe — 63,256-point projection (large response)',
  );
  await report(
    'GET',
    '/api/admin/cv/source-safe-roc?diagnosis=sinus%20rhythm',
    'Universe — per-class ROC in the model stats panel',
  );

  console.log('### Analyze ###');
  await report('GET', '/api/records?limit=3', 'Analyze — record browser');
  await report(
    'GET',
    `/api/records/${encodeURIComponent(RECORD)}/signal`,
    'Analyze + ECG Agent — 12-lead waveform',
  );

  console.log('### ECG Agent ###');
  await report(
    'POST',
    '/api/chat/stream',
    'ECG Agent — SSE chat (expect text/event-stream, or a clear error when llm is off)',
    { messages: [{ role: 'user', content: 'audit ping' }], recordingId: RECORD },
  );

  console.log('### Analyze — full pipeline (slow) ###');
  await report('POST', '/api/analyze', 'Analyze — the run button', { recordingId: RECORD });

  console.log('==========================================');
  console.log('AUDIT COMPLETE');
  console.log('==========================================');
}

main().catch((err) => {
  console.error(err instanceof Error ? err.stack : err);
  process.exit(1);
});
