#!/usr/bin/env python
"""A STUB OpenAI-compatible server. It is not a model and it cannot read an ECG.

It exists so the chat plumbing — SSE framing, multimodal content parts, the
frontend's event handling — can be exercised with no GPU in the room. Every
reply it produces is a canned string that says what it received. Never present
its output as a model's reading of an ECG.

    python scripts/stub_openai_server.py --port 8104
    OPENAI_BASE_URL=http://127.0.0.1:8104/v1 MODEL_NAME=stub-echo \\
        python -m uvicorn main:app --port 8103

Implements exactly what services/llm.py calls:
    GET  /v1/models
    POST /v1/chat/completions   (stream=true and non-streamed)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MODEL_ID = "stub-echo"

app = FastAPI(title="Stub OpenAI-compatible server (not a model)")


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "stub"}],
    }


def _describe(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    system = next((m for m in messages if m.get("role") == "system"), {})
    last = messages[-1] if messages else {}
    content = last.get("content")
    image_parts, text = 0, ""
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "image_url":
                image_parts += 1
            elif part.get("type") == "text":
                text = part.get("text", "")
    else:
        text = str(content or "")

    verifies = "verification layer" in str(system.get("content", ""))
    return (
        f"[STUB REPLY — no model ran] Received {len(messages)} messages, "
        f"{image_parts} image part(s), a {len(text)} character final turn, and a "
        f"{len(str(system.get('content', '')))} character system prompt "
        f"({'verify-not-diagnose framing present' if verifies else 'framing MISSING'}). "
        "This server cannot read an ECG."
    )


async def _stream(text: str, model: str) -> AsyncIterator[str]:
    created = int(time.time())
    for i in range(0, len(text), 24):
        chunk = {
            "id": "stub-1", "object": "chat.completion.chunk", "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text[i:i + 24]}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.01)
    done = {
        "id": "stub-1", "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    model = payload.get("model", MODEL_ID)
    text = _describe(payload)
    if payload.get("stream"):
        return StreamingResponse(_stream(text, model), media_type="text/event-stream")
    return JSONResponse({
        "id": "stub-1", "object": "chat.completion", "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8104)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
