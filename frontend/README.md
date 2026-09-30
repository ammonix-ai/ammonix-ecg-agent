# Ammonix ECG Agent — frontend

Three pages, no more:

| Page | Route | What it does |
|---|---|---|
| Universe | `/universe` | The fixed 63,256-point diagnosis embedding. t-SNE, UMAP, PCA. |
| Analyze | `/analyze` | Browse the 10,876 shipped recordings (or upload one), read the 12 leads, run the frozen classifier. |
| ECG Agent | `/agent` | Chat that *verifies* the classifier's read against the trace. |

## Running it

```bash
npm install
npm run dev          # http://localhost:5174
```

The dev server proxies `/api` and `/health` to **http://127.0.0.1:8100**, so
start the backend there:

```bash
cd ../backend && python -m uvicorn main:app --port 8100
```

Requests leave the browser same-origin and are proxied, so the backend needs no
CORS grant and POSTs skip the preflight. To talk to a backend somewhere else:

```bash
VITE_API_URL=http://127.0.0.1:9000 npm run dev     # moves the proxy target
```

Setting `VITE_API_URL` also makes the client address that origin directly, which
means that backend has to allow this one.

Other scripts: `npm run typecheck`, `npm test`, `npm run build`,
`npm run audit-endpoints` (hits every endpoint the app depends on and prints
what came back).

## Two things the UI is careful about

**Score kinds are never blended.** `ensemble_5fold` (the mean over all five
frozen swarms) and `oof_single_fold` (the one fold that held a record out) are
different quantities. Every probability on screen is rendered next to the kind
that produced it, and the live score and the stored universe score sit on
separate rows. For a recording that is already in the published universe, most
of the five swarms trained on it — Analyze says so in as many words rather than
presenting the number as out-of-sample performance. `utils/scoreKind.ts` holds
that vocabulary.

**Approximate placements are labelled approximate.** PCA is linear, so a new
point's coordinate is recovered exactly. t-SNE has no out-of-sample operator and
UMAP ships no fitted transformer, so those points are placed at the k-NN
centroid in probability space and flagged `knn_approx` in the UI.

## Notes

- The image the agent sends is drawn from the raw samples onto a canvas
  (`utils/renderEcgPng.ts`), not screenshotted from the panel. Same samples,
  same calibration, different renderer — `html-to-image` never resolved against
  this document.
- No mock fallbacks anywhere. A failed call renders as a failed call; nothing
  substitutes plausible-looking clinical numbers.
