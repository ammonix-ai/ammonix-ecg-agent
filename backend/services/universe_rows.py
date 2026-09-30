"""Row lookup over the published universe (``patients.jsonl``).

``services.projection`` reads the same file for display ids and primaries; this
module keeps the rest of each row — the reference labels, the source, the stored
prediction — so a neighbour can be shown with what it actually is. That is the
difference between "SS-01234, distance 0.11" and "SS-01234, a Chapman recording
labelled left ventricular hypertrophy and sinus bradycardia".

One pass over the 21 MB file, ~63k rows, held as interned strings. Loaded
lazily and cached process-wide.

No identifiers leave this file that are not already public: ``display_id`` is
the published key and ``recording_id`` exists only on the open PhysioNet rows.

Environment:
    UNIVERSE_DATA_DIR   universe payload (default: <repo>/_staging/universe)
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from dataclasses import dataclass
from typing import Any

from services.source_safe_universe import universe_data_dir

logger = logging.getLogger(__name__)

PATIENTS_NAME = "patients.jsonl"


class UniverseRowsMissing(FileNotFoundError):
    """patients.jsonl is not where UNIVERSE_DATA_DIR says it is."""


@dataclass(frozen=True, slots=True)
class UniverseRow:
    """One published universe record."""

    idx: int
    display_id: str
    source: str
    cohort: str
    labels: tuple[str, ...]
    primary: str
    predictions: tuple[str, ...]
    top_prediction: str
    max_score: float
    xgb_correct: bool | None
    recording_id: str | None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "displayId": self.display_id,
            "source": self.source,
            "cohort": self.cohort,
            "labels": list(self.labels),
            "primary": self.primary,
            "storedPredictions": list(self.predictions),
            "storedTopPrediction": self.top_prediction,
            "storedMaxScore": self.max_score,
        }
        if self.xgb_correct is not None:
            # The universe builder's `xgb_correct` is set equality between the
            # thresholded prediction set and the record's labels (restricted to
            # the model's classes) — all-or-nothing, so one extra call makes it
            # False. Named for what it measures, not for what it sounds like.
            payload["storedPredictionSetExact"] = self.xgb_correct
        if self.recording_id:
            payload["recordingId"] = self.recording_id
        return payload


@dataclass(frozen=True)
class UniverseRowIndex:
    by_display_id: dict[str, UniverseRow]
    by_recording_id: dict[str, UniverseRow]

    @property
    def n_rows(self) -> int:
        return len(self.by_display_id)


_INDEX: UniverseRowIndex | None = None
_LOCK = threading.Lock()


def load_row_index(*, force: bool = False) -> UniverseRowIndex:
    """Parse patients.jsonl once. ~1 s, ~60 MB resident."""
    global _INDEX
    if _INDEX is not None and not force:
        return _INDEX
    with _LOCK:
        if _INDEX is not None and not force:
            return _INDEX
        _INDEX = _load_uncached()
    return _INDEX


def _load_uncached() -> UniverseRowIndex:
    path = universe_data_dir() / PATIENTS_NAME
    if not path.exists():
        raise UniverseRowsMissing(
            f"Universe rows not found ({PATIENTS_NAME} missing). Point UNIVERSE_DATA_DIR "
            "at the directory holding the published universe payload."
        )

    by_display: dict[str, UniverseRow] = {}
    by_recording: dict[str, UniverseRow] = {}
    intern = sys.intern

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            display_id = str(raw.get("display_id") or f"idx-{raw.get('idx')}")
            recording_id = raw.get("recording_id")
            row = UniverseRow(
                idx=int(raw.get("idx", -1)),
                display_id=display_id,
                source=intern(str(raw.get("source") or "unknown")),
                cohort=intern(str(raw.get("cohort") or "unknown")),
                labels=tuple(intern(str(x)) for x in (raw.get("labels") or ())),
                primary=intern(str(raw.get("primary") or "unknown")),
                predictions=tuple(intern(str(x)) for x in (raw.get("predictions") or ())),
                top_prediction=intern(str(raw.get("top_prediction") or "unknown")),
                max_score=float(raw.get("max_score") or 0.0),
                xgb_correct=(
                    bool(raw["xgb_correct"]) if raw.get("xgb_correct") is not None else None
                ),
                recording_id=str(recording_id) if recording_id else None,
            )
            by_display[display_id] = row
            if row.recording_id:
                by_recording[row.recording_id] = row

    logger.info(
        "Universe rows indexed: %d rows, %d with an open recording id.",
        len(by_display), len(by_recording),
    )
    return UniverseRowIndex(by_display_id=by_display, by_recording_id=by_recording)


def row_by_display_id(display_id: str) -> UniverseRow | None:
    return load_row_index().by_display_id.get(display_id)


def row_by_recording_id(recording_id: str) -> UniverseRow | None:
    return load_row_index().by_recording_id.get(recording_id)


def rows_indexed() -> int | None:
    """Row count if the index is already built; None if it has not been touched."""
    return _INDEX.n_rows if _INDEX is not None else None
