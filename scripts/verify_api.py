#!/usr/bin/env python3
"""Smoke-test the Analyze endpoints against a running backend.

Checks the happy path of all three routes and the error paths that must never
become a 500:

    GET  /api/records                        filters, paging, totals
    GET  /api/records/{id}/signal            12 leads, canonical order
    POST /api/analyze  {"recordingId": ...}  shipped record, with `stored`
    POST /api/analyze  multipart             WFDB pair and 12-lead CSV upload
    POST /api/analyze  malformed input       4xx with a usable message

The CSV upload is generated from a shipped record, so the script needs no
fixtures. It asserts nothing about clinical content — only about the contract's
shape, the score labelling, and the absence of filesystem paths in responses.

Usage:
    python scripts/verify_api.py                       # http://127.0.0.1:8100
    python scripts/verify_api.py --base-url http://127.0.0.1:8102
    python scripts/verify_api.py --record HR16370

Set PYTHONIOENCODING=utf-8 on Windows: several diagnosis names are non-ASCII.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import urllib.error
import urllib.request
import uuid

DEFAULT_BASE_URL = "http://127.0.0.1:8100"
DEFAULT_RECORD = "HR16370"
CANONICAL_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
PATH_LIKE = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\)")

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if condition else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not condition:
        _failures.append(label)


def get(url: str) -> tuple[int, object]:
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def post_json(url: str, payload: object) -> tuple[int, object]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def post_files(url: str, files: list[tuple[str, str, bytes]], fields: dict[str, str] | None = None):
    """Minimal multipart/form-data POST: (field, filename, content) triples."""
    boundary = f"----open-ecg-{uuid.uuid4().hex}"
    buf = io.BytesIO()
    for name, value in (fields or {}).items():
        buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode())
        buf.write(f"{value}\r\n".encode())
    for field, filename, content in files:
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n".encode()
        )
        buf.write(content)
        buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        url,
        data=buf.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def no_paths(payload: object) -> bool:
    return not PATH_LIKE.search(json.dumps(payload))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--record", default=DEFAULT_RECORD)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print(f"\nGET {base}/api/records")
    status, body = get(f"{base}/api/records?limit=3")
    check("200", status == 200, str(body)[:120] if status != 200 else "")
    if status != 200:
        return 1
    check("total is the whole corpus", body["total"] > 10000, f"total={body['total']}")
    check("page respects limit", len(body["records"]) == 3)
    first = body["records"][0]
    check(
        "record shape",
        set(first) >= {"recordingId", "displayId", "source", "labels", "primary",
                       "storedTopPrediction", "storedMaxScore"},
        ", ".join(sorted(first)),
    )
    status, filtered = get(f"{base}/api/records?source=Chapman&limit=1")
    check("source filter narrows the set", filtered["total"] < body["total"],
          f"Chapman={filtered['total']}")
    status, searched = get(f"{base}/api/records?q={args.record}")
    check("q finds the target record", any(r["recordingId"] == args.record for r in searched["records"]))

    print(f"\nGET {base}/api/records/{args.record}/signal")
    status, signal = get(f"{base}/api/records/{args.record}/signal")
    check("200", status == 200, str(signal)[:120] if status != 200 else "")
    if status == 200:
        check("12 leads in canonical order", tuple(l["name"] for l in signal["leads"]) == CANONICAL_LEADS)
        check("samples present", len(signal["leads"][0]["samples"]) > 1000,
              f"{len(signal['leads'][0]['samples'])} samples @ {signal['samplingRate']} Hz")
        check("no filesystem paths", no_paths(signal))

    print(f"\nPOST {base}/api/analyze  (shipped record)")
    status, analysis = post_json(f"{base}/api/analyze", {"recordingId": args.record})
    check("200", status == 200, str(analysis)[:200] if status != 200 else "")
    if status == 200:
        check("scoreKind is the ensemble", analysis["scoreKind"] == "ensemble_5fold")
        check("32 probabilities", len(analysis["probabilities"]) == 32)
        check("featureCount from the package", analysis["featureCount"] > 0,
              str(analysis["featureCount"]))
        check("predictions are thresholded", all(
            analysis["probabilities"][dx] >= analysis["thresholds"][dx] for dx in analysis["predictions"]
        ))
        check("topFeatures present", len(analysis["topFeatures"]) > 0)
        check("projection has all three methods",
              set(analysis["projection"]) == {"pca", "umap", "tsne"})
        check("tsne is flagged approximate", analysis["tsnePlacement"] == "knn_approx")
        check("six neighbours", len(analysis["neighbors"]) == 6)
        check("stored block present", "stored" in analysis)
        stored = analysis.get("stored", {})
        check("stored score labelled", bool(stored.get("scoreKind")), str(stored.get("scoreKind")))
        if "maxAbsDeltaVsLive" in stored:
            check("live run reproduces the stored row", stored["maxAbsDeltaVsLive"] < 1e-5,
                  f"max|delta|={stored['maxAbsDeltaVsLive']:.2e}")
        check("no filesystem paths", no_paths(analysis))

    print(f"\nPOST {base}/api/analyze  (uploads)")
    status, uploaded = post_files(
        f"{base}/api/analyze",
        [("files", "unseen.csv", _csv_from_signal(signal).encode("utf-8"))],
        {"samplingRate": str(signal["samplingRate"])},
    )
    check("csv upload 200", status == 200, str(uploaded)[:200] if status != 200 else "")
    if status == 200:
        check("upload has no stored block", "stored" not in uploaded)
        check("upload reports how it was read", uploaded.get("upload", {}).get("kind") == "csv")
        check("upload matches the shipped run's leading diagnosis",
              uploaded["topPrediction"] == analysis["topPrediction"],
              f"{uploaded['topPrediction']} vs {analysis['topPrediction']}")

    print(f"\nPOST {base}/api/analyze  (rejections — none may be a 500)")
    for label, status_code, call in (
        ("unknown recordingId -> 404", 404,
         lambda: post_json(f"{base}/api/analyze", {"recordingId": "NO-SUCH-RECORD"})),
        ("unknown field -> 400", 400,
         lambda: post_json(f"{base}/api/analyze", {"record": args.record})),
        ("3-lead csv -> 400", 400,
         lambda: post_files(f"{base}/api/analyze",
                            [("f", "bad.csv", b"I,II,III\n0.1,0.2,0.3\n0.1,0.2,0.3\n")])),
        ("data file without header -> 400", 400,
         lambda: post_files(f"{base}/api/analyze", [("f", "orphan.mat", b"\x00" * 64)])),
        ("unsupported extension -> 400", 400,
         lambda: post_files(f"{base}/api/analyze", [("f", "payload.exe", b"MZ")])),
    ):
        code, detail = call()
        check(label, code == status_code, f"got {code}: {str(detail.get('detail'))[:90]}")

    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        return 1
    print("all checks passed")
    return 0


def _csv_from_signal(signal: dict) -> str:
    """Rebuild a 12-lead CSV from a /signal response, to upload as an 'unseen' file."""
    leads = signal["leads"]
    rows = [",".join(lead["name"] for lead in leads)]
    for i in range(len(leads[0]["samples"])):
        rows.append(",".join(f"{lead['samples'][i]:.4f}" for lead in leads))
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
