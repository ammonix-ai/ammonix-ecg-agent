"""The ECG Agent chat route.

    POST /api/chat/stream    text/event-stream

Request
-------
```jsonc
{ "messages": [ { "role": "user", "content": "why atrial flutter and not AF?" } ],
  "recordingId": "HR16370",
  "imageDataUrl": "data:image/png;base64,...",   // rendered 12-lead, optional
  "analysis": { ... }                            // optional: a POST /api/analyze
}                                                //   body, to skip recomputation
```

Response — one JSON object per SSE ``data:`` line, each with a ``type``:

```
data: {"type":"meta","recordingId":"HR16370","model":"...","predictions":[...],
       "neighbors":[...],"skills":[...],"hasImage":true,"scoreKind":"ensemble_5fold"}
data: {"type":"delta","text":"The classifier's leading call is "}
data: {"type":"reasoning","text":"..."}        // only if LLM_ENABLE_THINKING=1
data: {"type":"warning","message":"..."}       // context degraded but answering
data: {"type":"done","finish":"stop","chars":812}
data: [DONE]
```

Failure modes, all of them clean:

* **model endpoint unreachable** — HTTP 503 with a JSON body naming the endpoint
  and the reason. The stream never opens, so the client is not left hanging.
  ``GET /api/status`` reports ``llm: false`` for the same reason.
* **unknown recording** — HTTP 404.
* **classifier or universe payload missing** — HTTP 200, the stream opens, a
  ``warning`` event says the case brief could not be built, and the agent is told
  it has no classifier output to verify.
* **endpoint dies mid-stream** — an ``error`` event, then ``[DONE]``. Whatever
  arrived before the failure stays on screen.

The agent verifies; it does not diagnose. That framing lives in
``services.case.SYSTEM_PROMPT`` and is not overridable from the request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from services import case as case_service
from services import llm as llm_service
from services.inference import FeatureExtractionError, ModelPackageMissing
from services.signals import RecordNotFound, TracesMissing

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

#: Cap on the rendered-ECG data URL. A 12-lead PNG at display resolution is
#: well under 1 MB; anything past this is a mistake or an attack.
MAX_IMAGE_BYTES = int(os.environ.get("CHAT_IMAGE_MAX_BYTES", 8 * 1024 * 1024))
MAX_MESSAGES = 40
MAX_MESSAGE_CHARS = 8000

#: Building a brief for an uncached recording runs the tokenizer and the 160
#: boosters. One at a time by default, so a burst of chat tabs cannot pile
#: several extractions onto the CPU at once. `routers.analyze` keeps its own slot
#: for the same reason; the two should eventually share one.
_PIPELINE_SLOT = asyncio.Semaphore(
    max(1, int(os.environ.get("CHAT_PIPELINE_CONCURRENCY", "1") or 1))
)

_DATA_URL_RE = re.compile(r"^data:image/(png|jpe?g|webp);base64,[A-Za-z0-9+/=\s]+$")
_BARE_B64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")


# ---------------------------------------------------------------------------
# request model
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatStreamRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    messages: List[ChatMessage] = Field(default_factory=list)
    recording_id: Optional[str] = Field(default=None, alias="recordingId")
    image_data_url: Optional[str] = Field(default=None, alias="imageDataUrl")
    #: Optional POST /api/analyze body. When present the chat uses those numbers
    #: instead of re-running the pipeline, and says so in the brief's provenance.
    analysis: Optional[Dict[str, Any]] = None


def _normalise_image(raw: str | None) -> str | None:
    """Validate the image and return a data URL, or raise 4xx.

    Accepts a full ``data:image/...;base64,`` URL, or bare base64 (assumed PNG,
    which is what the frontend's canvas export produces).
    """
    if not raw:
        return None
    value = raw.strip()
    if len(value) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"imageDataUrl is {len(value)} bytes; the limit is {MAX_IMAGE_BYTES}. "
                "Send the rendered 12-lead PNG, not the raw waveform."
            ),
        )
    if value.startswith("data:"):
        if not _DATA_URL_RE.match(value):
            raise HTTPException(
                status_code=400,
                detail="imageDataUrl must be a base64 data URL of type png, jpeg or webp.",
            )
        return value
    if _BARE_B64_RE.match(value) and len(value) > 64:
        return "data:image/png;base64," + value
    raise HTTPException(status_code=400, detail="imageDataUrl is not base64 image data.")


def _validate_messages(messages: List[ChatMessage]) -> list[dict[str, str]]:
    if len(messages) > MAX_MESSAGES:
        raise HTTPException(
            status_code=400, detail=f"Too many messages (limit {MAX_MESSAGES}).",
        )
    out: list[dict[str, str]] = []
    for message in messages:
        role = message.role.strip().lower()
        if role not in ("user", "assistant"):
            # A caller-supplied system turn would let the page overwrite the
            # verify-not-diagnose framing. Drop it.
            logger.info("Dropping message with role %r from chat request.", message.role)
            continue
        content = message.content.strip()
        if not content:
            continue
        out.append({"role": role, "content": content[:MAX_MESSAGE_CHARS]})
    return out


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _meta_event(
    brief: case_service.CaseBrief | None,
    state: llm_service.LLMState,
    *,
    has_image: bool,
    warning: str | None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "type": "meta",
        "model": state.model,
        "baseUrl": state.base_url,
        "hasImage": has_image,
        "contextReady": brief is not None,
    }
    if warning:
        meta["warning"] = warning
    if brief is not None:
        meta.update({
            "recordingId": brief.recording_id,
            "displayId": brief.display_id,
            "scoreKind": brief.score_kind,
            "predictions": brief.predictions,
            "topPrediction": brief.top_prediction,
            "neighbors": [
                {"displayId": n["displayId"], "distance": n["distance"],
                 "primary": n.get("primary"), "labels": n.get("labels", [])}
                for n in brief.neighbors
            ],
            "skills": [
                {"id": s["id"], "kind": s["kind"], "title": s["title"]}
                for s in brief.skills
            ],
            "provenance": brief.provenance,
        })
    return meta


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------

@router.post("/stream")
async def chat_stream(request: ChatStreamRequest) -> StreamingResponse:
    """Stream the agent's verification of the classifier's calls."""
    image = _normalise_image(request.image_data_url)
    history = _validate_messages(request.messages)

    # Pre-flight the model endpoint *before* spending seconds on feature
    # extraction, so an unreachable LLM fails in milliseconds with a clear
    # message instead of after a 20 s pipeline run.
    state = await llm_service.probe()
    if not state.available:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "llm_unavailable",
                "message": (
                    state.error
                    or f"No model endpoint is reachable at {state.base_url}."
                ),
                "baseUrl": state.base_url,
                "hint": (
                    "Start an OpenAI-compatible server (vLLM, Ollama, LM Studio) and point "
                    "OPENAI_BASE_URL at it, then set MODEL_NAME if it serves more than one "
                    "model. Universe and Analyze do not need it."
                ),
            },
        )

    brief: case_service.CaseBrief | None = None
    warning: str | None = None
    if request.recording_id:
        try:
            async with _PIPELINE_SLOT:
                brief = await asyncio.to_thread(
                    case_service.build_case,
                    request.recording_id,
                    analysis=request.analysis,
                )
        except RecordNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ModelPackageMissing, TracesMissing, FileNotFoundError) as exc:
            warning = (
                f"The case brief could not be assembled: {exc} Answering without a prior "
                "analysis of this recording."
            )
            logger.warning("Chat context unavailable for %s: %s", request.recording_id, exc)
        except FeatureExtractionError as exc:
            warning = (
                f"The waveform for {request.recording_id} could not be processed: {exc} "
                "Answering without a prior analysis of this recording."
            )
            logger.warning("Feature extraction failed for %s: %s", request.recording_id, exc)
        except Exception as exc:  # never 500 the chat over context assembly
            warning = (
                f"The case brief could not be assembled ({type(exc).__name__}). Answering "
                "without a prior analysis of this recording."
            )
            logger.exception("Unexpected chat context failure for %s", request.recording_id)

    messages = case_service.build_messages(
        brief,
        history,
        image_data_url=image,
        unavailable_context=(
            warning if warning
            else None if request.recording_id
            else "No recordingId was sent with this conversation."
        ),
    )

    async def event_stream() -> AsyncIterator[str]:
        yield _sse(_meta_event(brief, state, has_image=bool(image), warning=warning))
        if warning:
            yield _sse({"type": "warning", "message": warning})

        chars = 0
        try:
            async for kind, text in llm_service.stream_completion(messages, state=state):
                if kind == llm_service.KIND_REASONING:
                    yield _sse({"type": "reasoning", "text": text})
                    continue
                chars += len(text)
                yield _sse({"type": "delta", "text": text})
        except llm_service.LLMUnavailable as exc:
            logger.warning("Chat stream failed: %s", exc)
            yield _sse({
                "type": "error",
                "code": "llm_unavailable",
                "message": str(exc),
                "partial": chars > 0,
            })
            yield _sse({"type": "done", "finish": "error", "chars": chars})
            yield "data: [DONE]\n\n"
            return
        except asyncio.CancelledError:
            logger.info("Chat stream cancelled by the client.")
            raise
        except Exception as exc:
            logger.exception("Unexpected chat stream failure")
            yield _sse({
                "type": "error",
                "code": "internal",
                "message": f"The chat stream failed: {type(exc).__name__}.",
                "partial": chars > 0,
            })
            yield _sse({"type": "done", "finish": "error", "chars": chars})
            yield "data: [DONE]\n\n"
            return

        yield _sse({"type": "done", "finish": "stop", "chars": chars})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # nginx must not buffer an SSE body
        },
    )


@router.post("/context")
async def chat_context(request: ChatStreamRequest) -> dict[str, Any]:
    """The assembled case brief and the exact prompt, without calling a model.

    Diagnostic sibling of ``/stream``: it is how you check what the agent is
    being told, and it works with no LLM running.
    """
    if not request.recording_id:
        raise HTTPException(status_code=400, detail="recordingId is required.")
    image = _normalise_image(request.image_data_url)
    history = _validate_messages(request.messages)
    try:
        async with _PIPELINE_SLOT:
            brief = await asyncio.to_thread(
                case_service.build_case, request.recording_id, analysis=request.analysis,
            )
    except RecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ModelPackageMissing, TracesMissing, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    messages = case_service.build_messages(brief, history, image_data_url=image)
    return {
        "brief": brief.to_dict(),
        "prompt": {
            "system": messages[0]["content"],
            "turns": len(messages),
            "final": (
                messages[-1]["content"]
                if isinstance(messages[-1]["content"], str)
                else messages[-1]["content"][-1]["text"]
            ),
            "hasImage": bool(image),
        },
    }
