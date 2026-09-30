"""Ammonix ECG Agent — backend.

Three capabilities behind one app:

* **Universe** (`/api/admin/cv/...`) — serve path only. The projections,
  probabilities and per-record rows are pre-computed on disk; nothing is scored.
* **Analyze** (`/api/records`, `/api/analyze`) — the live frozen pipeline. Raw
  12-lead -> QPSI tokenizer -> the frozen 5-swarm classifier -> placement into
  the fixed universe. It fits nothing.
* **ECG Agent** (`/api/chat/stream`) — SSE chat against an OpenAI-compatible
  model endpoint. The agent verifies the classifier's calls against the trace;
  it does not diagnose. Unreachable model endpoint means `GET /api/status`
  reports `llm: false` and the chat route answers 503 — Universe and Analyze
  are unaffected.

The classifier package is ~460 MB of boosters and takes ~8 s to load, so it is
loaded once at startup rather than per request. A deployment without the model
package still starts: the universe endpoints work and Analyze answers 503.

Run:
    python -m uvicorn main:app --port 8100        # from this directory
Environment:
    UNIVERSE_DATA_DIR   directory holding metadata.json / patients.jsonl /
                        probabilities.npy / projections/  (default: ../_staging/universe)
    TRACES_DIR          shipped WFDB corpus (default: ../_staging/traces)
    MODEL_PACKAGE_DIR   frozen classifier (default: ../models/source_safe_canonical_v1)
    CORS_ORIGINS        comma-separated allowed origins (default: *)
    PRELOAD_MODEL       "0" to defer the model load to the first analysis
    OPENAI_BASE_URL     model endpoint (default: http://127.0.0.1:8001/v1)
    OPENAI_API_KEY      default "EMPTY"
    MODEL_NAME          unset -> first model reported by GET /v1/models
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Allow `uvicorn main:app` from this directory and `uvicorn backend.main:app`
# from the repo root to resolve `routers` / `services` / `diagnosis_style`.
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402

from routers.analyze import router as analyze_router  # noqa: E402
from routers.chat import router as chat_router  # noqa: E402
from routers.status import router as status_router  # noqa: E402
from routers.universe import router as universe_router  # noqa: E402
from services.source_safe_universe import (  # noqa: E402
    SOURCE_SAFE_COHORT_ID,
    available_projections,
    universe_ready,
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)


def _preload() -> None:
    """Warm the expensive singletons so the first analysis is not the slow one.

    Each part is optional: a checkout without the 167 MB model package, without
    the traces or without the universe payload still serves what it does have,
    and the endpoints that need the missing piece answer 503.
    """
    from services.catalog import load_catalog
    from services.inference import load_runtime
    from services.projection import load_placement_model

    for what, load in (
        ("record catalog", load_catalog),
        ("universe placement model", load_placement_model),
        ("frozen classifier", load_runtime),
    ):
        try:
            load()
        except Exception as exc:
            logger.warning("%s unavailable at startup: %s", what.capitalize(), exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("PRELOAD_MODEL", "1").strip() not in ("0", "false", "no"):
        await asyncio.to_thread(_preload)
    yield


app = FastAPI(
    title="Ammonix ECG Agent API",
    version="1.0.0",
    description=(
        "Research demonstration - not a medical device, not for clinical use. "
        "The published source-safe ECG universe, and the frozen inference "
        "pipeline that places new recordings into it."
    ),
    lifespan=lifespan,
)

# The projection response is ~47 MB uncompressed for 63,256 points; gzip is not
# optional here. Added before CORS so CORS ends up outermost and error responses
# still carry its headers.
app.add_middleware(GZipMiddleware, minimum_size=1000)

_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,  # no cookies or auth on a public API
    allow_methods=["GET", "POST", "OPTIONS"],  # POST is /api/analyze
    allow_headers=["*"],
)

app.include_router(universe_router)
app.include_router(analyze_router)
app.include_router(status_router)
app.include_router(chat_router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, object]:
    """Liveness plus which published artefacts this deployment actually has.

    Reports state; never loads anything. `analyzeReady` is true once the frozen
    package has been loaded into this process (at startup unless PRELOAD_MODEL=0,
    otherwise on the first analysis).
    """
    from services import inference, signals

    return {
        "status": "ok" if universe_ready() else "degraded",
        "universeReady": universe_ready(),
        "cohortId": SOURCE_SAFE_COHORT_ID,
        "projections": available_projections(),
        "tracesReady": (signals.traces_dir() / signals.MANIFEST_NAME).exists(),
        "modelPackageReady": (inference.model_package_dir() / inference.MANIFEST_NAME).exists(),
        "analyzeReady": inference.runtime_loaded(),
    }
