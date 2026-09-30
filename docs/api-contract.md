# Ammonix ECG Agent — API contract

Three capabilities, one backend. Ports: backend `8100`, frontend `5174`.

Nav is **Universe · Analyze · ECG Agent**. Cohort, World Model and Skills are not
part of the open build; skills survive as *context the agent reasons with*, not
as a page.

Two invariants hold everywhere:

* **The classifier is frozen.** Inference only. No training, no fine-tuning, no
  threshold re-fitting at runtime.
* **The universe is fixed.** The 63,256-point embedding never refits. New
  recordings are placed into it by out-of-sample transform, so every user sees
  the identical universe.

---

## 1. Universe (already built)

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/admin/cv/lattice-status` | cohort list; source-safe row |
| `GET` | `/api/admin/cv/projection/{method}` | `?cohort_id=-60603`; 63,256 points |
| `GET` | `/api/admin/cv/source-safe-roc` | `?diagnosis=<name>` (query param — names contain `/`) |

Methods: `tsne`, `umap`, `pca`. No `clinical`. No perplexity or fold controls —
`perplexity` is pinned at 100 server-side.

---

## 2. Analyze

### `GET /api/records`

Lists the shipped test recordings (the 10,876 open-licensed records).

```
?source=PTB-XL|CPSC|CPSC-Extra|Georgia|Chapman   optional
?diagnosis=<canonical name>                       optional
?q=<substring of recording_id>                    optional
?limit=50&offset=0
```

```jsonc
{ "total": 10876,
  "records": [
    { "recordingId": "HR16370", "displayId": "SS-00930", "source": "PTB-XL",
      "labels": ["sinus rhythm"], "primary": "sinus rhythm",
      "storedTopPrediction": "sinus rhythm", "storedMaxScore": 0.93 }
  ] }
```

### `GET /api/records/{recording_id}/signal`

Raw waveform for display. Read from the shipped WFDB pair; never from the
private tree.

```jsonc
{ "recordingId": "HR16370", "samplingRate": 500, "durationSec": 10,
  "units": "mV",
  "leads": [ { "name": "I", "samples": [ 0.01, ... ] }, ... ]   // 12 leads
}
```

### `POST /api/analyze`

Runs the **full live pipeline**. Accepts either a shipped record or an upload.

```jsonc
// body A
{ "recordingId": "HR16370" }
// body B: multipart/form-data with .hea + .mat/.dat, or a 12-lead CSV
```

```jsonc
{ "recordingId": "HR16370",
  "scoreKind": "ensemble_5fold",        // see note below
  "featureCount": 26490,
  "probabilities": { "sinus rhythm": 0.94, ... },   // 32 classes
  "predictions": ["sinus rhythm"],                  // thresholded
  "topFeatures": [ { "name": "...", "value": 1.23, "contribution": 0.08 } ],
  "projection": { "pca": [x, y, z], "umap": [x, y, z], "tsne": [x, y, z] },
  "tsnePlacement": "knn_approx",        // t-SNE has no out-of-sample transform
  "neighbors": [ { "displayId": "SS-01234", "distance": 0.11,
                   "primary": "sinus rhythm" } ],   // k=6 in probability space
  "stored": { "topPrediction": "sinus rhythm", "maxScore": 0.93,
              "scoreKind": "ensemble_5fold" }       // present for shipped records only
}
```

**`scoreKind` matters and must be surfaced in the UI.**

The shipped universe was scored by averaging **all five** fold-classifiers over
every row (`source_safe_universe.py:425-428` in the private builder), so
`probabilities.npy` is an `ensemble_5fold` matrix. It is **not** a per-record
held-out score. Verified: the live pipeline reproduces the stored values to
~3e-8 across all five sources.

The consequence is important and must be stated wherever scores are shown:

* For an **uploaded** recording, all five folds are genuinely unseen, so the
  ensemble mean is an honest out-of-sample score.
* For a **shipped** recording, four of the five folds were typically trained on
  it, so the same arithmetic is partly in-sample. Example — `HR16370`,
  *sinus irregularity*, per fold: `0.164, 0.787, 0.807, 0.797, 0.832`. The one
  fold that held it out says 0.164; the mean says 0.677.

`classify()` therefore returns `foldProbabilities` so the UI can show the
spread. Scores on shipped records must **not** be described as out-of-sample
performance.

A true single-fold OOF score cannot be served from what ships: the per-diagnosis
`validation_scores.npy` holds it, but the record ordering that indexes it lives
in the private training corpus.

**Projection.** `pca` is re-derived by least squares from the shipped
probabilities onto the shipped coordinates (residual ~2.4e-8, i.e. exact for
practical purposes) — a re-derivation of a linear map, not a refit.
`umap` and `tsne` have no persisted fitted transformer, so both place the point
at the probability-space k-NN centroid (k=6, distance-weighted) and report
`knn_approx`. If `projections/umap_model.joblib` is ever shipped, `projection.py`
picks it up and uses `transform()` instead. The embedding is never refit.

---

## 3. ECG Agent

### `POST /api/chat/stream` — SSE

```jsonc
{ "messages": [ { "role": "user", "content": "why atrial flutter and not AF?" } ],
  "recordingId": "HR16370",
  "imageDataUrl": "data:image/png;base64,..."   // rendered 12-lead, optional
}
```

Server assembles context from: the analyze result (probabilities, thresholds,
top features), the 6 nearest universe neighbours and their labels, and any
matching skills. Streams `text/event-stream` deltas.

The model is reached over an **OpenAI-compatible** endpoint:

```
OPENAI_BASE_URL   default http://127.0.0.1:8001/v1
OPENAI_API_KEY    default "EMPTY"
MODEL_NAME        default the published finetuned Qwen
```

Works unchanged against vLLM, Ollama or LM Studio. When unreachable,
`GET /api/status` reports `llm: false`, the chat UI disables itself with an
explanation, and Universe + Analyze remain fully usable.

**The agent verifies; it does not diagnose.** The diagnosis comes from the
classifier, the universe and the skills. The model's job is to confirm the
classifier's leading features against the image and explain them. Prompts must
not invite it to produce an independent read.

---

## QPSI tokenizer

Every recording — shipped or uploaded — runs through the QPSI tokenizer before
classification:

```
raw 12-lead WFDB → QPSI tokenizer → 26,490 features → frozen classifier
```

QPSI and the classifier are already aligned; the feature vector is ordered by
the model's `feat_order`, read from the shipped package. Feature counts are
derived from `feat_order` and never hardcoded.
