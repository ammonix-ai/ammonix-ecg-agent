"""Catalog of the shipped test recordings: the WFDB pairs joined to universe rows.

Two artefacts describe the same 10,876 open-licensed PhysioNet records:

* ``_staging/traces/manifest.csv`` — what is actually on disk (display_id,
  recording_id, source, path, bytes). This is the authoritative list: a record
  that is not in the manifest cannot be read, so it is not offered.
* ``_staging/universe/patients.jsonl`` — the universe row for each record, which
  carries the gold ``labels``, the ``primary`` label and the universe's stored
  score (``top_prediction`` / ``max_score``). Only the 10,876 open rows carry a
  ``recording_id``; the other 52,380 universe points are label-only and are not
  part of this catalog.

The join is 1:1 on ``recording_id`` (verified: 0 manifest ids missing from the
universe, 0 display_id or source mismatches), but the code does not assume it —
a manifest row with no universe row is still listed, with null labels.

Nothing here reads a signal or scores anything; it is a text index over two
files, loaded once (~10 MB of Python objects) and cached process-wide.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from services.signals import manifest_rows, traces_dir
from services.source_safe_universe import UniverseDataMissing, universe_data_dir

logger = logging.getLogger(__name__)

PATIENTS_NAME = "patients.jsonl"

#: Hard ceiling on `?limit=`; the full list is 10,876 rows and a browser does
#: not need them in one response.
MAX_PAGE_SIZE = 500


@dataclass(frozen=True)
class CatalogRecord:
    """One shipped recording: what it is on disk, and what the universe says."""

    recording_id: str
    display_id: str | None
    source: str | None
    labels: tuple[str, ...]
    primary: str | None
    stored_top_prediction: str | None
    stored_max_score: float | None
    universe_index: int | None

    def to_summary(self) -> dict[str, Any]:
        """The `records[]` shape of GET /api/records. No paths, ever."""
        return {
            "recordingId": self.recording_id,
            "displayId": self.display_id,
            "source": self.source,
            "labels": list(self.labels),
            "primary": self.primary,
            "storedTopPrediction": self.stored_top_prediction,
            "storedMaxScore": self.stored_max_score,
        }


@dataclass(frozen=True)
class Catalog:
    records: tuple[CatalogRecord, ...]
    by_id: dict[str, CatalogRecord]
    sources: tuple[str, ...]
    diagnoses: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.records)


_CATALOG: Catalog | None = None
_CATALOG_LOCK = threading.Lock()


def load_catalog(*, force: bool = False) -> Catalog:
    """Build (or return) the cached catalog. Raises TracesMissing without a manifest."""
    global _CATALOG
    if _CATALOG is not None and not force:
        return _CATALOG
    with _CATALOG_LOCK:
        if _CATALOG is not None and not force:
            return _CATALOG
        _CATALOG = _build_catalog()
    return _CATALOG


def _universe_rows_by_recording_id() -> dict[str, dict[str, Any]]:
    """The universe rows that carry a recording_id, keyed by it.

    A missing universe payload is not fatal here: the manifest alone still lists
    the recordings, they just arrive without labels or a stored score.
    """
    path: Path = universe_data_dir() / PATIENTS_NAME
    if not path.exists():
        logger.warning(
            "Universe rows not found (%s missing); records will be listed without "
            "labels or stored scores.", PATIENTS_NAME,
        )
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = row.get("recording_id")
            if rid:
                out[str(rid)] = row
    return out


def _build_catalog() -> Catalog:
    rows = manifest_rows()  # raises TracesMissing if the corpus is not on disk
    try:
        universe = _universe_rows_by_recording_id()
    except UniverseDataMissing:  # pragma: no cover - defensive
        universe = {}

    records: list[CatalogRecord] = []
    sources: set[str] = set()
    diagnoses: set[str] = set()
    for row in rows:
        rid = (row.get("recording_id") or "").strip()
        if not rid:
            continue
        u = universe.get(rid, {})
        labels = tuple(str(x) for x in (u.get("labels") or []))
        source = (u.get("source") or row.get("source") or "").strip() or None
        max_score = u.get("max_score")
        records.append(
            CatalogRecord(
                recording_id=rid,
                display_id=(u.get("display_id") or row.get("display_id") or "").strip() or None,
                source=source,
                labels=labels,
                primary=(str(u["primary"]) if u.get("primary") else None),
                stored_top_prediction=(str(u["top_prediction"]) if u.get("top_prediction") else None),
                stored_max_score=(float(max_score) if isinstance(max_score, (int, float)) else None),
                universe_index=(int(u["idx"]) if isinstance(u.get("idx"), int) else None),
            )
        )
        if source:
            sources.add(source)
        diagnoses.update(labels)

    catalog = Catalog(
        records=tuple(records),
        by_id={r.recording_id: r for r in records},
        sources=tuple(sorted(sources)),
        diagnoses=tuple(sorted(diagnoses, key=str.casefold)),
    )
    logger.info(
        "Record catalog ready: %d shipped recordings, %d with universe labels, sources=%s",
        len(catalog), sum(1 for r in records if r.labels), ", ".join(catalog.sources),
    )
    return catalog


def get_record(recording_id: str) -> CatalogRecord | None:
    return load_catalog().by_id.get(recording_id)


def _matches_diagnosis(record: CatalogRecord, wanted: str) -> bool:
    """Gold-label match, case-insensitive.

    Filters on what the record *is* (its labels, plus the `primary` label, which
    is a lower-cased pick from them), never on what the model predicted — a
    diagnosis filter that silently returned the model's own calls would make the
    list look better than the classifier is.
    """
    target = wanted.casefold()
    if record.primary and record.primary.casefold() == target:
        return True
    return any(label.casefold() == target for label in record.labels)


def query(
    *,
    source: str | None = None,
    diagnosis: str | None = None,
    q: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[int, list[CatalogRecord]]:
    """Filter, then page. `total` is the size of the filtered set, not the page."""
    catalog = load_catalog()
    selected: Iterable[CatalogRecord] = catalog.records

    if source:
        wanted = source.casefold()
        selected = [r for r in selected if (r.source or "").casefold() == wanted]
    if diagnosis:
        selected = [r for r in selected if _matches_diagnosis(r, diagnosis)]
    if q:
        needle = q.casefold()
        selected = [r for r in selected if needle in r.recording_id.casefold()]

    selected = list(selected)
    total = len(selected)
    limit = max(1, min(int(limit), MAX_PAGE_SIZE))
    offset = max(0, int(offset))
    return total, selected[offset : offset + limit]


def diagnosis_vocabulary(source: str | None = None) -> list[str]:
    """Every gold label present in the corpus, optionally within one source.

    Deliberately NOT derived from the caller's current page: the browser lists 50
    records at a time out of 10,876, so a page-derived vocabulary offers whichever
    handful of labels happens to be on screen. It is also not filtered by the
    caller's `diagnosis`/`q`, so the suggestion list does not collapse while the
    user is typing into it.
    """
    catalog = load_catalog()
    if not source:
        return list(catalog.diagnoses)
    wanted = source.casefold()
    seen: set[str] = set()
    for record in catalog.records:
        if (record.source or "").casefold() == wanted:
            # labels only, like Catalog.diagnoses: `primary` is a lower-cased pick
            # from them, so adding it would emit case-variant duplicates.
            seen.update(record.labels)
    return sorted(seen, key=str.casefold)


def catalog_summary() -> dict[str, Any]:
    """Counts for /api/status-style callers. Repo-relative paths only."""
    catalog = load_catalog()
    counts: dict[str, int] = {}
    for record in catalog.records:
        key = record.source or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return {
        "shippedRecordings": len(catalog),
        "sources": counts,
        "labelledFromUniverse": sum(1 for r in catalog.records if r.labels),
        "tracesPresent": (traces_dir() / "manifest.csv").exists(),
    }
