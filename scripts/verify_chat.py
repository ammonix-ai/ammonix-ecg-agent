#!/usr/bin/env python
"""Proof harness for the ECG Agent chat route.

Shows, without needing a model:

* what the model endpoint probe currently reports (`--probe`);
* the exact case brief and system prompt the agent would receive for a
  recording, assembled by the frozen pipeline (`<recording_id>`);
* which skills matched and why.

    PYTHONIOENCODING=utf-8 python scripts/verify_chat.py --probe
    PYTHONIOENCODING=utf-8 python scripts/verify_chat.py HR16370
    PYTHONIOENCODING=utf-8 python scripts/verify_chat.py HR16370 --json

Several diagnosis names are non-ASCII, hence PYTHONIOENCODING on Windows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "backend")]

from services import case as case_service  # noqa: E402
from services import llm as llm_service  # noqa: E402


def probe() -> int:
    state = asyncio.run(llm_service.probe(force=True))
    cfg = llm_service.settings()
    print("MODEL ENDPOINT PROBE")
    print(f"  base url        {cfg.base_url}")
    print(f"  MODEL_NAME      {cfg.model or '(unset — discover from /models)'}")
    print(f"  probe timeout   {cfg.probe_timeout}s")
    print(f"  available       {state.available}")
    print(f"  model           {state.model}")
    print(f"  served models   {list(state.models) or '(none reported)'}")
    if state.latency_ms is not None:
        print(f"  latency         {state.latency_ms:.0f} ms")
    if state.error:
        print(f"  error           {state.error}")
    print()
    print(f"  /api/status would report  llm: {json.dumps(state.available)}")
    return 0 if state.available else 1


def show_case(recording_id: str, *, as_json: bool, question: str | None, with_image: bool) -> int:
    t0 = time.perf_counter()
    brief = case_service.build_case(recording_id)
    elapsed = time.perf_counter() - t0

    if as_json:
        print(json.dumps(brief.to_dict(), indent=2, ensure_ascii=False))
        return 0

    messages = case_service.build_messages(
        brief,
        [{"role": "user", "content": question}] if question else [],
        image_data_url=("data:image/png;base64,STUB" if with_image else None),
    )
    system = messages[0]["content"]
    final = messages[-1]["content"]
    if not isinstance(final, str):
        final = next(p["text"] for p in final if p.get("type") == "text")

    print(f"# case assembled in {elapsed:.1f} s "
          f"({'cached' if elapsed < 0.05 else 'live pipeline'})")
    print("=" * 78)
    print("SYSTEM PROMPT")
    print("=" * 78)
    print(system)
    print("=" * 78)
    print("USER TURN (case brief + question)")
    print("=" * 78)
    print(final)
    print("=" * 78)
    print(f"system {len(system)} chars, user turn {len(final)} chars, "
          f"{len(messages)} messages, image attached: {with_image}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_id", nargs="?", help="shipped recording id, e.g. HR16370")
    parser.add_argument("--probe", action="store_true", help="probe the model endpoint and exit")
    parser.add_argument("--json", action="store_true", help="dump the brief as JSON")
    parser.add_argument("--question", default=None, help="clinician question to render")
    parser.add_argument("--with-image", action="store_true",
                        help="pretend an ECG image is attached (changes the system prompt)")
    args = parser.parse_args()

    if args.probe:
        return probe()
    if not args.recording_id:
        parser.error("give a recording id, or --probe")
    return show_case(
        args.recording_id, as_json=args.json,
        question=args.question, with_image=args.with_image,
    )


if __name__ == "__main__":
    raise SystemExit(main())
