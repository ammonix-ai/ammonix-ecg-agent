# Ammonix ECG Agent

A mechanistic 12-lead ECG agent: a q_psi feature pipeline, a frozen classifier swarm, a 3D
diagnosis universe, and a chat agent that **verifies the classifier's calls against the trace
rather than diagnosing on its own**.

Three capabilities behind one app:

| | what it does |
|---|---|
| **Universe** | Serve-only. Projections, probabilities and per-record rows are pre-computed; nothing is scored at request time. |
| **Analyze** | The live frozen pipeline: raw 12-lead → q_psi tokenizer → the 5-swarm classifier → placement into the fixed universe. It fits nothing. |
| **ECG Agent** | SSE chat against an OpenAI-compatible endpoint. It checks the classifier's features against the trace and explains them; it does not make the diagnosis. |

## Understanding the numbers

The Universe page reports three AUROCs per diagnosis — `primary`, `final` and `area` — and they
are **three different questions, not three estimates of one**. Before quoting any of them, read:

### → [docs/auroc-metrics.md](docs/auroc-metrics.md)

Short version: `primary` compares cases against controls from the same sources and is the number
to quote; `final` adds every other negative the model holds, including clean normals, so it is
usually higher; `area` describes the displayed universe and moves with that population's
composition rather than with the classifier. The doc gives all 32 classes and the two traps —
a high `area` over a low `primary` means easy negatives, and an `area` far below `primary` usually
means the universe holds positives the model never saw.

## Layout

```
backend/     FastAPI app — universe, analyze, chat
frontend/    React + Vite viewer (react-three-fiber)
qpsi/        the q_psi feature pipeline (Python + Rust accelerator)
packaging/   release/scrubbing scripts for the public data package
analysis/    the analyses behind specific classifier decisions (aggregate results only)
docs/        API contract and metric definitions
models/      frozen classifier package        (not in git — see below)
_staging/    universe + shipped traces        (not in git — see below)
```

## Research demonstration, not a medical device

**Research demonstration — not a medical device, not for clinical use.** Nothing this
software outputs is a diagnosis, and it must not be used to make or support a clinical
decision about a patient. It has not been evaluated prospectively, and no regulatory
authority has reviewed it.

## The Ammonix family

This is one of the companion releases of the Ammonix research program:

| | |
|---|---|
| Foundation paper | https://doi.org/10.5281/zenodo.22859098 — the Ammonix method: retrospective harness optimization with verifiable rewards |
| **ECG agent (this repo)** | https://doi.org/10.5281/zenodo.22871232 — *The Ammonix ECG Agent: A Local Specialized AI Agent to Transparently Read and Discuss 12-Lead Electrocardiograms* |
| RCM agent | https://github.com/ammonix-ai/ammonix-rcm-agent · https://doi.org/10.5281/zenodo.23078509 |
| Control-room agent | https://github.com/ammonix-ai/ammonix-industrial-control-room-agent · https://doi.org/10.5281/zenodo.22871228 |
| Rocket launch agent | https://github.com/ammonix-ai/ammonix-rocket-launch |
| Wild Departures | https://github.com/ammonix-ai/ammonix-wild-departures |
| Ask Ammonix | coming later — a local application that answers questions about the architecture, its evidence and its limits, from authored, source-linked text |
| Ammonix**Code** | coming later — our architecture-native coding agent, purpose-built to create systems based on the Ammonix architecture |

## What this repository is, and what it is not

This is the serving application described in the paper: the q_psi feature pipeline, the
backend that places a recording in the diagnosis universe and scores it with a frozen
classifier swarm, the chat agent that checks those calls against the trace, and the viewer.
The scripts that recompute the per-class AUROCs and operating points from a universe
package are in `packaging/` and documented in `docs/auroc-metrics.md`.

It is not the experiment code behind the paper's 17-diagnosis Skill study (Skill induction,
the dual-gate validation, the untrained-language-model and data-matched controls, and the
comparator models), and the learned Skill store is not included. Those results are reported
in the paper and cannot be re-run from this repository.

## Data and model

`models/` and `_staging/` are excluded by `.gitignore`: the repository holds code and
documentation only. **The universe package and the classifier package are not part of this
release.** They are built from several source datasets with different terms, and they will
be published separately, under terms that respect each source, once that review is
complete. Until then a fresh checkout starts, and the Universe and Analyze pages say which
package is missing.

The hosted demo at https://ecg.ammonix.ai runs this code with both packages installed. It
lets anyone browse the universe: for each of the 63,256 recordings its position, its
diagnosis labels and the classifier's calls, under pseudonymous IDs, with the 12-lead traces
of the open PhysioNet recordings. What is not yet released is the downloadable packages
themselves.

Source datasets and their terms:

| Source | Terms | Used for |
|---|---|---|
| PTB-XL | CC BY 4.0 | universe, training, shipped traces |
| CPSC 2018 and CPSC-Extra, Georgia, Chapman-Shaoxing (PhysioNet/CinC Challenge 2020) | CC BY 4.0 | universe, training, shipped traces |
| MIMIC-IV-ECG (waveforms and machine measurements) | Open Database License 1.0 | universe, training |
| HUCA Brugada cohort (PhysioNet) | CC BY-SA 4.0 | Brugada class |
| Pre-/Post-STEMI ECG Database (University of Michigan, Deep Blue Data) | as stated on the Deep Blue record | universe, training |
| MIMIC-IV-Note, MIMIC-IV (PhysioNet, credentialed access) | PhysioNet Credentialed Health Data License 1.5.0 | labels only: infarction diagnoses from discharge summaries, echocardiographic LVEF |

No record-level data from the credentialed sources is in this repository, only aggregate
results such as the per-class AUROCs in `docs/auroc-metrics.md`. No credentialed data is
needed to run it.

Raw traces will be published for the open PhysioNet sources only. No patient identifiers
are contained in any file of this repository.

## Reproducing the paper

`REPRODUCING.md` goes through every table and figure of the paper and says what can be
recomputed from this repository and what cannot. `docs/paper_alignment_audit.md` lists where
the paper and the repository differ.

## Status and limitations

The classifier is frozen and its calls are reported as they are. The label-editing Skills of
the paper's first experiment are not part of this application; the domain rules that ship
run in observation mode and never change a prediction. The evidence in the paper is
retrospective. Thresholds and neighbourhood sizes are design parameters, not validated
clinical settings.

## License

Released under the Ammonix Research License (`LICENSE.md`): research, educational, and
evaluation use is free — including evaluation by a commercial organization deciding whether
to seek a commercial license. Any commercial use requires a separate license from Ammonix —
contact licensing@ammonix.ai. The license covers the software. Datasets keep their own
terms (table above); fonts keep theirs (`NOTICE.md`).

## Citing

See `CITATION.cff`.

## Quickstart (15 minutes, no data package, no key)

This runs the tokenizer on one public PTB-XL recording and prints its named features. It needs
Python 3.10 or newer and internet access to PhysioNet.

```bash
git clone https://github.com/ammonix-ai/ammonix-ecg-agent && cd ammonix-ecg-agent
python -m venv .venv && .venv/Scripts/activate    # Windows; use bin/activate on Unix
pip install -r backend/requirements.txt -e qpsi
python examples/tokenize_ptbxl_record.py
```

Expected: "Extracted 26,490 named features", then a few of them. The viewer and the backend
below start from the same install; without the data packages the Universe and Analyze pages
say what is missing.

## Running it

Backend:

```bash
python -m uvicorn main:app --port 8100
```

from `backend/`. It reads `UNIVERSE_DATA_DIR`, `TRACES_DIR` and `MODEL_PACKAGE_DIR`
(defaults `../_staging/universe`, `../_staging/traces`, `../models/source_safe_canonical_v1`),
and talks to a model endpoint at `OPENAI_BASE_URL` (default `http://127.0.0.1:8001/v1`).

The classifier package is large and is loaded once at startup, so first boot takes a few seconds.
**A deployment without it still starts** — the Universe endpoints work and Analyze answers 503.
Likewise an unreachable model endpoint only disables chat; `GET /api/status` reports `llm: false`.

Frontend:

```bash
npm install && npm run dev
```

from `frontend/`.
