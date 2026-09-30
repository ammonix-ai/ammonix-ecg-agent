"""Analyze: browse the shipped recordings, read a waveform, run the live pipeline.

    GET  /api/records                        the 10,876 shipped test recordings
    GET  /api/records/{recording_id}/signal  12-lead waveform for display
    POST /api/analyze                        {recordingId} or a multipart upload

`POST /api/analyze` is the whole frozen chain in one call:

    raw 12-lead -> QPSI tokenizer -> feat_order vector -> 32 diagnoses x 5 folds
                -> probabilities -> placement into the fixed 63,256-point universe

Nothing fits. The classifier, its thresholds and the embedding are all read from
disk exactly as published; see services/inference.py and services/projection.py.

Three things this router is deliberate about:

* **Scores are labelled, not blended.** The live score is the mean of the five
  folds (`ensemble_5fold`). For a shipped record the `stored` block reports the
  universe's own number alongside it, with its true `scoreKind` — see
  `_stored_block()` for why that is not the contract's `oof_single_fold`.
* **A shipped record is already a point in the universe.** Placing it therefore
  lands on itself at distance 0. The coordinates keep that (they are the honest
  answer: it *is* that point), but the neighbour list drops the self-hit and
  reports it separately as `selfMatch`, so six informative neighbours come back
  instead of five plus a mirror.
* **Extraction is single-flight.** One recording costs ~1.4 s of CPU with the
  Rust kernels and ~21 s without, and XGBoost boosters are shared mutable state;
  `ANALYZE_CONCURRENCY` (default 1) bounds how many run at once, and the work
  runs in a worker thread so the event loop stays responsive.

Environment:
    ANALYZE_CONCURRENCY   concurrent analyses (default 1)
    UPLOAD_MAX_BYTES      upload cap in bytes (default 32 MB)
    TRACES_DIR            shipped WFDB corpus (services/signals.py)
    MODEL_PACKAGE_DIR     frozen classifier (services/inference.py)
    UNIVERSE_DATA_DIR     universe payload (services/source_safe_universe.py)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from services import catalog as catalog_service
from services.catalog import MAX_PAGE_SIZE, CatalogRecord
from services.inference import (
    CONTRACT_STORED_SCORE_KIND,
    STORED_SCORE_KIND,
    FeatureExtractionError,
    ModelPackageMissing,
    classify,
    extract_features,
    load_runtime,
    top_feature_contributions,
)
from services.projection import (
    DEFAULT_K,
    UniverseDataMissing,
    load_placement_model,
    nearest_neighbors,
    place_point,
)
from services.signals import (
    LeadSignals,
    RecordNotFound,
    TracesMissing,
    load_record,
)
from services.uploads import (
    UploadRejected,
    UploadTooLarge,
    max_upload_bytes,
    parse_sampling_rate,
    save_uploads,
    scrub_paths,
    signals_from_upload,
    uploads_enabled,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analyze"])

#: How many top features the explanation returns.
TOP_FEATURES = 10
#: Millivolt precision on the wire: 1e-4 mV = 0.1 uV, an order of magnitude
#: below the ADC step of every source in the corpus, and it halves the payload.
SIGNAL_DECIMALS = 4
#: A JSON analyze body is a single id; anything larger is a mistake or an attack.
MAX_JSON_BODY_BYTES = 64 * 1024
#: Longest recording id we will even look up.
MAX_RECORDING_ID_LEN = 64

_CONCURRENCY = max(1, int(os.environ.get("ANALYZE_CONCURRENCY", "1") or 1))
_ANALYSIS_SLOT = asyncio.Semaphore(_CONCURRENCY)


# ---------------------------------------------------------------------------
# response models
# ---------------------------------------------------------------------------

class RecordSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    recordingId: str
    displayId: Optional[str] = None
    source: Optional[str] = None
    labels: list[str] = Field(default_factory=list)
    primary: Optional[str] = None
    storedTopPrediction: Optional[str] = None
    storedMaxScore: Optional[float] = None


class RecordListResponse(BaseModel):
    total: int
    records: list[RecordSummary]
    #: Every gold label available under the current `source`, for the filter's
    #: suggestion list. Corpus-wide, not page-derived, and not narrowed by the
    #: caller's own `diagnosis`/`q`.
    diagnoses: list[str] = []


class SignalLead(BaseModel):
    name: str
    samples: list[float]


class SignalResponse(BaseModel):
    recordingId: str
    samplingRate: int
    durationSec: float
    units: str
    leads: list[SignalLead]


class AnalyzeRequest(BaseModel):
    """Body A of POST /api/analyze."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    recording_id: str = Field(alias="recordingId", min_length=1, max_length=MAX_RECORDING_ID_LEN)


# ---------------------------------------------------------------------------
# error mapping — a client always learns what to fix, and never learns a path
# ---------------------------------------------------------------------------

def _http(status: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status, detail=scrub_paths(detail))


def _dependency_error(exc: Exception) -> HTTPException:
    """Missing published artefacts are a deployment state, not a bad request."""
    if isinstance(exc, TracesMissing):
        return _http(503, f"{exc} The shipped recordings are distributed separately from the source.")
    if isinstance(exc, ModelPackageMissing):
        return _http(503, str(exc))
    if isinstance(exc, UniverseDataMissing):
        return _http(503, str(exc))
    raise exc


def _checked_recording_id(recording_id: str) -> str:
    rid = (recording_id or "").strip()
    if not rid or len(rid) > MAX_RECORDING_ID_LEN:
        raise _http(404, f"Unknown recording id '{rid[:MAX_RECORDING_ID_LEN]}'.")
    return rid


# ---------------------------------------------------------------------------
# GET /api/records
# ---------------------------------------------------------------------------

@router.get("/records", response_model=RecordListResponse)
async def list_records(
    source: Optional[str] = Query(None, description="PTB-XL | CPSC | CPSC-Extra | Georgia | Chapman"),
    diagnosis: Optional[str] = Query(None, description="Canonical class name; matches gold labels"),
    q: Optional[str] = Query(None, max_length=64, description="Substring of the recording id"),
    limit: int = Query(50, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """The shipped test corpus, filtered and paged.

    `total` is the size of the filtered set (10,876 unfiltered), `records` is the
    requested page. `diagnosis` matches the record's gold labels, never the
    model's own prediction.
    """
    try:
        total, page = await asyncio.to_thread(
            catalog_service.query,
            source=source,
            diagnosis=diagnosis,
            q=q,
            limit=limit,
            offset=offset,
        )
        vocabulary = await asyncio.to_thread(catalog_service.diagnosis_vocabulary, source)
    except (TracesMissing, UniverseDataMissing) as exc:
        raise _dependency_error(exc)
    return {
        "total": total,
        "records": [r.to_summary() for r in page],
        "diagnoses": vocabulary,
    }


# ---------------------------------------------------------------------------
# GET /api/records/{recording_id}/signal
# ---------------------------------------------------------------------------

@router.get("/records/{recording_id}/signal", response_model=SignalResponse)
async def record_signal(recording_id: str) -> dict[str, Any]:
    """The 12-lead waveform of one shipped record, in millivolts.

    Read from the shipped WFDB pair under TRACES_DIR; there is no other source.
    Lead order is canonical (I, II, III, aVR, aVL, aVF, V1-V6) regardless of how
    the file stored it.
    """
    rid = _checked_recording_id(recording_id)
    try:
        signals = await asyncio.to_thread(load_record, rid)
    except RecordNotFound as exc:
        raise _http(404, str(exc))
    except TracesMissing as exc:
        raise _dependency_error(exc)
    except Exception as exc:  # pragma: no cover - a corrupt shipped file
        logger.exception("Reading shipped record %s failed", rid)
        raise _http(500, f"Could not read recording '{rid}': {type(exc).__name__}")
    return _signal_payload(signals)


def _signal_payload(signals: LeadSignals) -> dict[str, Any]:
    rounded = np.round(signals.signal, SIGNAL_DECIMALS)
    return {
        "recordingId": signals.recording_id,
        "samplingRate": int(signals.sampling_rate),
        "durationSec": round(signals.duration_sec, 3),
        "units": signals.units,
        "leads": [
            {"name": name, "samples": rounded[i].tolist()}
            for i, name in enumerate(signals.lead_names)
        ],
    }


# ---------------------------------------------------------------------------
# POST /api/analyze
# ---------------------------------------------------------------------------

@router.post("/analyze")
async def analyze(request: Request) -> dict[str, Any]:
    """Run the frozen pipeline over a shipped record or an uploaded recording.

    Body A: `application/json` `{"recordingId": "HR16370"}`.
    Body B: `multipart/form-data` with a WFDB `.hea` + `.mat`/`.dat` pair, or a
    12-lead `.csv` (optional `samplingRate` field, default 500 Hz). A
    `recordingId` form field is also accepted, which behaves like body A.
    """
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()

    if content_type == "multipart/form-data":
        signals, stored, upload_info = await _read_multipart(request)
    elif content_type in ("application/json", ""):
        stored = await _read_json_recording(request)
        upload_info = None
        try:
            signals = await asyncio.to_thread(load_record, stored.recording_id)
        except RecordNotFound as exc:
            raise _http(404, str(exc))
        except (TracesMissing, UniverseDataMissing) as exc:
            raise _dependency_error(exc)
        except Exception as exc:  # pragma: no cover
            logger.exception("Reading shipped record %s failed", stored.recording_id)
            raise _http(500, f"Could not read recording '{stored.recording_id}': {type(exc).__name__}")
    else:
        raise _http(
            415,
            f"Unsupported Content-Type '{content_type}'. Send application/json "
            '{"recordingId": "..."} or multipart/form-data with a .hea + .mat/.dat '
            "pair or a 12-lead .csv.",
        )

    async with _ANALYSIS_SLOT:
        try:
            return await asyncio.to_thread(_run_pipeline, signals, stored, upload_info)
        except FeatureExtractionError as exc:
            # The file parsed but carries no measurable beats: the client's
            # problem, not the server's.
            raise _http(422, str(exc))
        except (ModelPackageMissing, UniverseDataMissing) as exc:
            raise _dependency_error(exc)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Analysis failed for %s", signals.recording_id)
            raise _http(500, f"Analysis failed: {type(exc).__name__}")


async def _read_json_recording(request: Request) -> CatalogRecord:
    """Parse body A and resolve it to a shipped record."""
    raw = await request.body()
    if len(raw) > MAX_JSON_BODY_BYTES:
        raise _http(413, "Request body is too large for a JSON analyze request.")
    if not raw.strip():
        raise _http(400, 'Empty body. Send {"recordingId": "..."} or a multipart upload.')
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise _http(400, f"Body is not valid JSON: {exc}")
    if not isinstance(payload, dict):
        raise _http(400, 'Body must be a JSON object, e.g. {"recordingId": "HR16370"}.')
    try:
        parsed = AnalyzeRequest.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(p) for p in first.get("loc", ())) or "body"
        raise _http(400, f"Invalid analyze request: {field}: {first.get('msg')}")

    rid = _checked_recording_id(parsed.recording_id)
    try:
        record = catalog_service.get_record(rid)
    except (TracesMissing, UniverseDataMissing) as exc:
        raise _dependency_error(exc)
    if record is None:
        raise _http(404, f"Unknown recording id '{rid}'. See GET /api/records.")
    return record


async def _read_multipart(
    request: Request,
) -> tuple[LeadSignals, CatalogRecord | None, dict[str, Any] | None]:
    """Parse body B: files into a temp dir, or a recordingId field."""
    declared = request.headers.get("content-length")
    cap = max_upload_bytes()
    if declared and declared.isdigit() and int(declared) > cap:
        raise _http(413, f"Upload exceeds the {cap // (1024 * 1024)} MB limit.")

    try:
        form = await request.form(max_files=8, max_fields=8, max_part_size=cap)
    except Exception as exc:
        raise _http(400, f"Malformed multipart body: {type(exc).__name__}: {scrub_paths(exc)}")

    # multi_items(), not items(): a client sends both parts of a WFDB pair under
    # one field name, and the dict view keeps only the last value per key.
    parts = form.multi_items()
    try:
        files = [v for _, v in parts if getattr(v, "filename", None)]
        fields = {k: v for k, v in parts if isinstance(v, str)}

        if files and not uploads_enabled():
            raise _http(
                403,
                "Uploads are disabled on this deployment. Analyze one of the "
                "shipped recordings instead — see GET /api/records.",
            )

        if not files:
            # A multipart body carrying only a recordingId is body A in disguise.
            rid = (fields.get("recordingId") or fields.get("recording_id") or "").strip()
            if not rid:
                raise _http(
                    400,
                    "Multipart body has no files. Attach a .hea plus its .mat/.dat, "
                    "or a 12-lead .csv, or send a recordingId field.",
                )
            rid = _checked_recording_id(rid)
            record = catalog_service.get_record(rid)
            if record is None:
                raise _http(404, f"Unknown recording id '{rid}'. See GET /api/records.")
            try:
                return await asyncio.to_thread(load_record, rid), record, None
            except RecordNotFound as exc:
                raise _http(404, str(exc))

        sampling_rate = parse_sampling_rate(
            fields.get("samplingRate") or fields.get("sampling_rate") or fields.get("fs")
        )

        # Uploads never leave this directory, and it is removed on the way out.
        with tempfile.TemporaryDirectory(prefix="open-ecg-upload-") as tmp:
            saved = await save_uploads(files, Path(tmp))
            signals, info = await asyncio.to_thread(
                signals_from_upload, saved, sampling_rate=sampling_rate
            )
            # LeadSignals holds the samples in memory, so the files can go now.
            info["bytes"] = sum(f.size for f in saved)
            return signals, None, info
    except UploadTooLarge as exc:
        raise _http(413, str(exc))
    except UploadRejected as exc:
        raise _http(400, str(exc))
    except (TracesMissing, UniverseDataMissing) as exc:
        raise _dependency_error(exc)
    finally:
        for _, part in parts:
            if hasattr(part, "close"):
                try:
                    await part.close()
                except Exception:  # pragma: no cover - best effort
                    pass


# ---------------------------------------------------------------------------
# the pipeline itself (synchronous; always called in a worker thread)
# ---------------------------------------------------------------------------

def _run_pipeline(
    signals: LeadSignals,
    stored: CatalogRecord | None,
    upload_info: dict[str, Any] | None,
) -> dict[str, Any]:
    runtime = load_runtime()

    t0 = time.perf_counter()
    features = extract_features(
        signals.signal, signals.sampling_rate, recording_id=signals.recording_id
    )
    t_extract = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = classify(features, runtime)
    t_classify = time.perf_counter() - t0

    t0 = time.perf_counter()
    placement = place_point(result["probabilities"])
    t_place = time.perf_counter() - t0

    t0 = time.perf_counter()
    top_features = top_feature_contributions(
        features, result["topPrediction"], k=TOP_FEATURES, runtime=runtime
    )
    t_explain = time.perf_counter() - t0

    neighbors = placement["neighbors"]
    self_match: dict[str, Any] | None = None
    if stored is not None and stored.display_id:
        hit = next((nb for nb in neighbors if nb["displayId"] == stored.display_id), None)
        if hit is not None:
            self_match = {
                **hit,
                "note": (
                    "This recording is already one of the 63,256 universe points, so it "
                    "matches itself at distance ~0 and the placement lands on its own "
                    "coordinates. It is excluded from `neighbors` below."
                ),
            }
            neighbors = [
                nb
                for nb in nearest_neighbors(result["probabilities"], k=DEFAULT_K + 1)
                if nb["displayId"] != stored.display_id
            ][:DEFAULT_K]

    # An upload has no GET /api/records/{id}/signal to read back from — the
    # files are not retained after the request — so the waveform travels with
    # the result or the reader never sees the trace they just submitted.
    # Shipped records are omitted deliberately: they already have an endpoint,
    # and this block is ~380 KB per recording.
    upload_signal: dict[str, Any] | None = None
    if stored is None:
        upload_signal = {
            "recordingId": signals.recording_id,
            "samplingRate": int(signals.sampling_rate),
            "durationSec": round(signals.duration_sec, 3),
            "units": signals.units,
            "leads": [
                {"name": name, "samples": np.round(row, 4).tolist()}
                for name, row in zip(signals.lead_names, signals.signal)
            ],
        }

    payload: dict[str, Any] = {
        "recordingId": signals.recording_id,
        "inputKind": "shipped" if stored is not None else "upload",
        "signal": upload_signal,
        "source": signals.source or (stored.source if stored else None),
        "samplingRate": int(signals.sampling_rate),
        "durationSec": round(signals.duration_sec, 3),
        "scoreKind": result["scoreKind"],
        "featureCount": result["featureCount"],
        "modelVersion": result["modelVersion"],
        "probabilities": result["probabilities"],
        "predictions": result["predictions"],
        "thresholds": result["thresholds"],
        "thresholdKind": result["thresholdKind"],
        "topPrediction": result["topPrediction"],
        "maxScore": result["maxScore"],
        "foldProbabilities": result["foldProbabilities"],
        "topFeatures": top_features,
        "projection": {
            "pca": placement.get("pca"),
            "umap": placement.get("umap"),
            "tsne": placement.get("tsne"),
        },
        "pcaPlacement": placement.get("pcaPlacement"),
        "pcaResidualRms": placement.get("pcaResidualRms"),
        "umapPlacement": placement.get("umapPlacement"),
        "tsnePlacement": placement.get("tsnePlacement"),
        "neighbors": neighbors,
        "neighborSpace": placement.get("neighborSpace"),
        "universePoints": placement.get("universePoints"),
        "timings": {
            "extractSeconds": round(t_extract, 3),
            "classifySeconds": round(t_classify, 3),
            "placeSeconds": round(t_place, 3),
            "explainSeconds": round(t_explain, 3),
        },
    }
    if placement.get("umapNote"):
        payload["umapNote"] = placement["umapNote"]
    if self_match is not None:
        payload["selfMatch"] = self_match
    if upload_info is not None:
        payload["upload"] = upload_info
    if stored is not None:
        payload["stored"] = _stored_block(stored, result)
    return payload


def _stored_block(record: CatalogRecord, result: dict[str, Any]) -> dict[str, Any]:
    """What the published universe says about this record, labelled honestly.

    docs/api-contract.md calls the stored score `oof_single_fold` — the fold that
    held the record out. It is not that. The universe builder averaged all five
    folds, the same rule the live score uses, which is why the two agree to ~1e-8
    (`maxAbsDeltaVsLive` below proves it per request). Both labels are returned:
    `scoreKind` is what the number actually is, `contractScoreKind` is what the
    contract currently claims, so the discrepancy is visible in the payload
    rather than silently resolved in the UI's favour.
    """
    block: dict[str, Any] = {
        "displayId": record.display_id,
        "topPrediction": record.stored_top_prediction,
        "maxScore": record.stored_max_score,
        "labels": list(record.labels),
        "primary": record.primary,
        "scoreKind": STORED_SCORE_KIND,
        "contractScoreKind": CONTRACT_STORED_SCORE_KIND,
        "scoreKindNote": (
            "The universe's stored score is the mean of all five folds, the same "
            "quantity as the live score — not the single held-out fold the API "
            "contract names. Most shipped PhysioNet records were in the training "
            "pool for some diagnoses, so a stored score on a shipped record is not "
            "an out-of-sample result; see foldProbabilities for the per-fold spread."
        ),
    }
    if record.universe_index is not None:
        try:
            model = load_placement_model()
            stored_vec = model.probabilities[record.universe_index].astype(np.float64)
            live_vec = np.array(
                [float(result["probabilities"].get(dx, 0.0)) for dx in model.classes],
                dtype=np.float64,
            )
            block["maxAbsDeltaVsLive"] = float(np.abs(stored_vec - live_vec).max())
        except Exception as exc:  # pragma: no cover - reproducibility check only
            logger.warning("Stored/live comparison unavailable: %s", exc)
    return block
