"""One place to ask what this backend can currently do.

    GET /api/status

Every field is cheap: nothing here loads the 167 MB classifier, parses the 21 MB
universe row file, or waits longer than ``LLM_PROBE_TIMEOUT_SECONDS`` on the
model endpoint. It reports what is *present* and what is *already loaded*, so a
page can decide which tabs to enable before the first heavy request.

``llm`` is a plain boolean because the API contract says so; ``llmInfo`` carries
the detail (endpoint, model, error) that the UI shows when it is false.

No absolute paths are returned. Presence, versions and counts only.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter

from services import case as case_service
from services import inference, llm as llm_service, signals, universe_rows
from services.uploads import uploads_enabled
from services.source_safe_universe import (
    SOURCE_SAFE_COHORT_ID,
    available_projections,
    load_universe_metadata,
    universe_ready,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["meta"])


def _universe_status() -> dict[str, Any]:
    ready = universe_ready()
    payload: dict[str, Any] = {
        "ready": ready,
        "cohortId": SOURCE_SAFE_COHORT_ID,
        "projections": available_projections() if ready else [],
        "rowsIndexed": universe_rows.rows_indexed(),
    }
    if ready:
        try:
            metadata = load_universe_metadata()
            payload.update({
                "points": metadata.get("n_patients"),
                "classes": metadata.get("n_classes"),
                "modelVersion": metadata.get("model_version"),
            })
        except Exception as exc:
            logger.warning("Universe metadata unreadable: %s", exc)
            payload["error"] = "Universe metadata could not be read."
    return payload


def _model_status() -> dict[str, Any]:
    """Classifier state without triggering the 8 s load.

    ``inference._RUNTIME`` is the module's own cache; peeking at it with a
    default is how we answer "is it warm?" without warming it.
    """
    loaded = getattr(inference, "_RUNTIME", None) is not None
    package_dir = inference.model_package_dir()
    manifest_path = package_dir / "manifest.json"
    payload: dict[str, Any] = {
        "packagePresent": manifest_path.exists(),
        "loaded": loaded,
        "frozen": True,
    }
    if loaded:
        try:
            payload.update(inference.runtime_summary())
        except Exception as exc:  # pragma: no cover - only if the cache is odd
            logger.warning("Runtime summary failed: %s", exc)
    elif payload["packagePresent"]:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload.update({
                "modelVersion": manifest.get("version") or manifest.get("model_version"),
                "modelFamily": manifest.get("model_family"),
                "createdAt": manifest.get("created_at"),
                "nClasses": len(manifest.get("diagnoses") or manifest.get("classes") or ()),
            })
        except Exception as exc:
            logger.warning("Model manifest unreadable: %s", exc)
    return payload


def _traces_status() -> dict[str, Any]:
    try:
        rows = signals.manifest_rows()
    except FileNotFoundError:
        return {"available": False, "records": 0}
    sources = sorted({row.get("source", "") for row in rows if row.get("source")})
    return {"available": True, "records": len(rows), "sources": sources}


@router.get("/status")
async def status() -> dict[str, Any]:
    """Universe, classifier, traces and model-endpoint state in one call."""
    llm_state = await llm_service.probe()
    universe = _universe_status()
    model = _model_status()
    traces = _traces_status()

    degraded = not (universe.get("ready") and model.get("packagePresent") and traces["available"])
    return {
        "status": "degraded" if degraded else "ok",
        # Contract: `GET /api/status` reports `llm: false` when the endpoint is
        # unreachable, and the chat UI disables itself on that boolean.
        "llm": llm_state.available,
        "llmInfo": llm_state.to_status(),
        "universe": universe,
        "model": model,
        "traces": traces,
        "analyze": {"uploadsEnabled": uploads_enabled()},
        "chat": {
            "endpoint": "/api/chat/stream",
            "transport": "sse",
            "contextEndpoint": "/api/chat/context",
            "caseCache": case_service.cache_info(),
        },
    }
