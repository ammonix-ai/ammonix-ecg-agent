"""OpenAI-compatible chat client for the ECG Agent.

Anything that speaks the OpenAI HTTP API works unchanged: vLLM
(``--served-model-name``), Ollama (``/v1``), LM Studio, or a hosted endpoint.
The client only ever calls two paths:

    GET  {OPENAI_BASE_URL}/models              liveness + model discovery
    POST {OPENAI_BASE_URL}/chat/completions    streamed completion

Environment
-----------
``OPENAI_BASE_URL``   default ``http://127.0.0.1:8001/v1``
``OPENAI_API_KEY``    default ``EMPTY`` (what local servers expect)
``MODEL_NAME``        unset -> the first model ``GET /models`` reports. Set it
                      when the server hosts more than one, or when the server
                      does not implement ``/models``.
``LLM_TIMEOUT_SECONDS``        default 120  (whole streamed completion)
``LLM_PROBE_TIMEOUT_SECONDS``  default 2.0  (liveness probe; keeps
                               ``GET /api/status`` and the chat pre-flight fast
                               when nothing is listening)
``LLM_PROBE_TTL_SECONDS``      default 10   (probe result cache)
``LLM_MAX_TOKENS``             default 900
``LLM_TEMPERATURE``            default 0.0
``LLM_ENABLE_THINKING``        default false; when true, sends the vLLM/Qwen
                               ``chat_template_kwargs.enable_thinking`` flag and
                               surfaces reasoning deltas separately.

Degradation is the point: nothing here raises out of a probe. When the endpoint
is unreachable, :func:`probe` returns ``available=False`` with the reason, and
:func:`stream_completion` raises :class:`LLMUnavailable` with a message that can
be shown to a user verbatim.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping, Sequence

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_TIMEOUT = 120.0
DEFAULT_PROBE_TIMEOUT = 2.0
DEFAULT_PROBE_TTL = 10.0
DEFAULT_MAX_TOKENS = 900
DEFAULT_TEMPERATURE = 0.0

#: Yielded by :func:`stream_completion` as the first element of each chunk.
KIND_CONTENT = "content"
KIND_REASONING = "reasoning"


class LLMUnavailable(RuntimeError):
    """The model endpoint could not be reached, or refused the request.

    The message is written to be shown to a user: it names the endpoint and
    what went wrong, and never contains a stack trace.
    """


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, float(default)))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMSettings:
    """Resolved connection settings. Read from the environment on each use."""

    base_url: str
    api_key: str
    model: str | None
    timeout: float
    probe_timeout: float
    probe_ttl: float
    max_tokens: int
    temperature: float
    enable_thinking: bool

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/models"

    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"


def settings() -> LLMSettings:
    """Current settings from the environment (no caching, so restarts are seen)."""
    base_url = (os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model = (os.environ.get("MODEL_NAME") or "").strip() or None
    return LLMSettings(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY") or DEFAULT_API_KEY,
        model=model,
        timeout=_env_float("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT),
        probe_timeout=_env_float("LLM_PROBE_TIMEOUT_SECONDS", DEFAULT_PROBE_TIMEOUT),
        probe_ttl=_env_float("LLM_PROBE_TTL_SECONDS", DEFAULT_PROBE_TTL),
        max_tokens=_env_int("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        temperature=_env_float("LLM_TEMPERATURE", DEFAULT_TEMPERATURE),
        enable_thinking=_env_bool("LLM_ENABLE_THINKING", False),
    )


# ---------------------------------------------------------------------------
# liveness
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMState:
    """What the last probe found. Safe to serialise into ``GET /api/status``."""

    available: bool
    base_url: str
    model: str | None
    models: tuple[str, ...] = ()
    error: str | None = None
    latency_ms: float | None = None
    checked_at: float = field(default_factory=time.time)
    model_configured: str | None = None

    def to_status(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "available": self.available,
            "baseUrl": self.base_url,
            "model": self.model,
            "models": list(self.models),
            "checkedAt": self.checked_at,
        }
        if self.latency_ms is not None:
            payload["latencyMs"] = round(self.latency_ms, 1)
        if self.error:
            payload["error"] = self.error
        if self.model_configured:
            payload["modelConfigured"] = self.model_configured
            payload["modelServed"] = (
                not self.models or self.model_configured in self.models
            )
        return payload


_STATE: LLMState | None = None
_STATE_LOCK = asyncio.Lock()


def _friendly_error(exc: Exception, url: str) -> str:
    if isinstance(exc, httpx.ConnectError):
        return f"No model server is listening at {url}."
    if isinstance(exc, httpx.ConnectTimeout):
        return f"Connecting to {url} timed out."
    if isinstance(exc, httpx.ReadTimeout):
        return f"{url} accepted the connection but sent no response in time."
    if isinstance(exc, httpx.HTTPError):
        return f"{url} could not be reached: {type(exc).__name__}: {exc}"
    return f"{url} could not be reached: {exc}"


async def probe(*, force: bool = False) -> LLMState:
    """Check the endpoint, at most once per ``LLM_PROBE_TTL_SECONDS``.

    Never raises. A failure is a returned ``LLMState`` with ``available=False``
    and a human-readable ``error``.
    """
    global _STATE
    cfg = settings()
    cached = _STATE
    if (
        not force
        and cached is not None
        and cached.base_url == cfg.base_url
        and (time.time() - cached.checked_at) < cfg.probe_ttl
    ):
        return cached

    async with _STATE_LOCK:
        cached = _STATE
        if (
            not force
            and cached is not None
            and cached.base_url == cfg.base_url
            and (time.time() - cached.checked_at) < cfg.probe_ttl
        ):
            return cached
        _STATE = await _probe_uncached(cfg)
        return _STATE


async def _probe_uncached(cfg: LLMSettings) -> LLMState:
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=cfg.probe_timeout) as client:
            response = await client.get(cfg.models_url, headers=_headers(cfg))
    except Exception as exc:  # httpx raises a family; all mean "not usable"
        return LLMState(
            available=False,
            base_url=cfg.base_url,
            model=None,
            error=_friendly_error(exc, cfg.models_url),
            model_configured=cfg.model,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0

    if response.status_code >= 400:
        # A server that 404s /models can still serve completions (some proxies
        # do). Treat a configured MODEL_NAME as sufficient in that case.
        if cfg.model:
            return LLMState(
                available=True,
                base_url=cfg.base_url,
                model=cfg.model,
                models=(),
                error=(
                    f"{cfg.models_url} returned HTTP {response.status_code}; "
                    f"using MODEL_NAME={cfg.model} without discovery."
                ),
                latency_ms=latency_ms,
                model_configured=cfg.model,
            )
        return LLMState(
            available=False,
            base_url=cfg.base_url,
            model=None,
            error=(
                f"{cfg.models_url} returned HTTP {response.status_code}. Set MODEL_NAME "
                "if this server does not implement model discovery."
            ),
            latency_ms=latency_ms,
            model_configured=cfg.model,
        )

    try:
        payload = response.json()
        served = tuple(
            str(item.get("id"))
            for item in payload.get("data", [])
            if isinstance(item, Mapping) and item.get("id")
        )
    except Exception:
        served = ()

    model = cfg.model or (served[0] if served else None)
    if model is None:
        return LLMState(
            available=False,
            base_url=cfg.base_url,
            model=None,
            models=served,
            error=(
                f"{cfg.models_url} is reachable but lists no models. Start a model, "
                "or set MODEL_NAME."
            ),
            latency_ms=latency_ms,
            model_configured=cfg.model,
        )

    return LLMState(
        available=True,
        base_url=cfg.base_url,
        model=model,
        models=served,
        latency_ms=latency_ms,
        model_configured=cfg.model,
    )


def invalidate_probe() -> None:
    """Drop the cached probe (used after a stream fails mid-flight)."""
    global _STATE
    _STATE = None


# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------

def _headers(cfg: LLMSettings) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }


def user_message(text: str, image_data_url: str | None = None) -> dict[str, Any]:
    """A user turn, multimodal when an image data URL is supplied.

    The image goes first: vision models attend to a leading image more
    reliably, and it matches how the private client ordered the parts.
    """
    if not image_data_url:
        return {"role": "user", "content": text}
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": text},
        ],
    }


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _partial_tail(text: str, tag: str) -> int:
    """Length of the suffix of ``text`` that could be the start of ``tag``."""
    for n in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


class _ThinkSplitter:
    """Reclassify inline ``<think>...</think>`` spans as reasoning, not content.

    Belt and braces behind the explicit ``enable_thinking`` flag. If a chat
    template ignores the flag, or the model emits the tags anyway, the entire
    chain of thought would otherwise stream straight into the user's answer —
    which is exactly what it did before this existed.

    Tags routinely straddle chunk boundaries (``"<thi"`` at the end of one
    delta, ``"nk>"`` at the start of the next), so a per-chunk regex is not
    enough: any suffix that could still become a tag is held back until the
    next chunk resolves it.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        self._buf += text
        out: list[tuple[str, str]] = []
        while self._buf:
            tag = _THINK_CLOSE if self._in_think else _THINK_OPEN
            kind = KIND_REASONING if self._in_think else KIND_CONTENT
            idx = self._buf.find(tag)
            if idx == -1:
                hold = _partial_tail(self._buf, tag)
                emit = self._buf[: len(self._buf) - hold]
                if emit:
                    out.append((kind, emit))
                self._buf = self._buf[len(self._buf) - hold :]
                break
            if idx:
                out.append((kind, self._buf[:idx]))
            self._buf = self._buf[idx + len(tag) :]
            self._in_think = not self._in_think
        return out

    def flush(self) -> list[tuple[str, str]]:
        """Emit whatever is still buffered at end of stream."""
        if not self._buf:
            return []
        kind = KIND_REASONING if self._in_think else KIND_CONTENT
        out = [(kind, self._buf)]
        self._buf = ""
        return out


def _extract_delta(chunk: Mapping[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for choice in chunk.get("choices") or ():
        if not isinstance(choice, Mapping):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            # Some servers echo the whole message on the final chunk.
            delta = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            out.append((KIND_REASONING, reasoning))
        content = delta.get("content")
        if isinstance(content, str) and content:
            out.append((KIND_CONTENT, content))
    return out


async def stream_completion(
    messages: Sequence[Mapping[str, Any]],
    *,
    state: LLMState | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> AsyncIterator[tuple[str, str]]:
    """Stream one completion, yielding ``(kind, text)`` where kind is
    ``"content"`` or ``"reasoning"``.

    Raises :class:`LLMUnavailable` if the endpoint cannot be reached or refuses
    the request. Once the first chunk has been yielded, a mid-stream failure
    also raises :class:`LLMUnavailable` — the caller owns the half-written
    response and decides what to tell the user.
    """
    cfg = settings()
    resolved = state or await probe()
    model = resolved.model or cfg.model
    if not resolved.available or not model:
        raise LLMUnavailable(
            resolved.error or f"No model available at {cfg.base_url}."
        )

    body: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "stream": True,
        "max_tokens": max_tokens if max_tokens is not None else cfg.max_tokens,
        "temperature": temperature if temperature is not None else cfg.temperature,
    }
    # State the flag EXPLICITLY in both directions. Omitting it does not mean
    # "off": Qwen3.x chat templates default thinking ON, so leaving the key out
    # makes the model reason out loud, and — with no --reasoning-parser on the
    # vLLM side — that reasoning arrives as ordinary content and lands in the
    # user's answer.
    body["chat_template_kwargs"] = {"enable_thinking": bool(cfg.enable_thinking)}
    # Gateway endpoints (e.g. OpenRouter) ignore chat_template_kwargs and use
    # their own control fields (reasoning toggle, provider pin). Operator-set
    # OPENAI_EXTRA_BODY (a JSON object) merges into the request; off by
    # default, local wiring unchanged.
    extra_raw = os.environ.get("OPENAI_EXTRA_BODY", "").strip()
    if extra_raw:
        extra = json.loads(extra_raw)
        if isinstance(extra, dict):
            body.update(extra)

    timeout = httpx.Timeout(cfg.timeout, connect=min(cfg.probe_timeout * 2, cfg.timeout))
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", cfg.completions_url, headers=_headers(cfg), json=body,
            ) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "replace")[:400]
                    invalidate_probe()
                    raise LLMUnavailable(
                        f"{cfg.completions_url} returned HTTP {response.status_code}: "
                        f"{detail.strip() or 'no detail'}"
                    )
                splitter = _ThinkSplitter()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        if payload == "[DONE]":
                            for item in splitter.flush():
                                yield item
                            return
                        continue
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        logger.debug("Skipping unparseable SSE chunk: %.120s", payload)
                        continue
                    if not isinstance(chunk, Mapping):
                        continue
                    if chunk.get("error"):
                        invalidate_probe()
                        raise LLMUnavailable(f"Model server error: {chunk['error']}")
                    for kind, text in _extract_delta(chunk):
                        # Reasoning the server already separated passes through;
                        # content is scanned for inline <think> spans.
                        if kind == KIND_CONTENT:
                            for item in splitter.feed(text):
                                yield item
                        else:
                            yield (kind, text)
    except LLMUnavailable:
        raise
    except (httpx.HTTPError, OSError) as exc:
        invalidate_probe()
        raise LLMUnavailable(_friendly_error(exc, cfg.completions_url)) from exc
