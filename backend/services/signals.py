"""Read raw 12-lead waveforms for the shipped test recordings (and uploads).

The shipped corpus is the 10,876 open-licensed PhysioNet records under
``TRACES_DIR`` (default ``_staging/traces``), one WFDB ``.hea`` + ``.mat`` pair
per record, indexed by ``manifest.csv``:

    display_id,recording_id,source,path,bytes
    SS-00930,HR16370,PTB-XL,PTB-XL/HR16370,120775

Nothing here touches the private tree, a NAS share or a record index database:
a recording is a path under ``TRACES_DIR`` or it does not exist.

Lead order is normalised to the pipeline's canonical order
(I, II, III, aVR, aVL, aVF, V1-V6) before anything downstream sees the array —
the feature extractor addresses leads positionally, so a file that stores them
in a different order would silently produce mislabelled features.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import wfdb

logger = logging.getLogger(__name__)

# The order run_pipeline() assumes when meta carries no explicit lead_names.
CANONICAL_LEADS: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent
_DEFAULT_TRACES_SUBPATH = ("_staging", "traces")

MANIFEST_NAME = "manifest.csv"

# Lead aliases seen in the wild (case and punctuation vary by source).
_LEAD_ALIASES = {
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    "LEAD I": "I", "LEAD II": "II", "LEAD III": "III",
    "MDC_ECG_LEAD_I": "I", "MDC_ECG_LEAD_II": "II", "MDC_ECG_LEAD_III": "III",
}


class RecordNotFound(FileNotFoundError):
    """No WFDB pair for that recording id under TRACES_DIR."""


class TracesMissing(FileNotFoundError):
    """The shipped trace corpus is not on disk."""


@dataclass(frozen=True)
class LeadSignals:
    """A 12-lead recording in millivolts, canonical lead order."""

    recording_id: str
    signal: np.ndarray          # (12, n_samples) float64, mV
    sampling_rate: int
    lead_names: tuple[str, ...]
    units: str = "mV"
    source: str | None = None
    display_id: str | None = None

    @property
    def duration_sec(self) -> float:
        return float(self.signal.shape[1]) / float(self.sampling_rate)


def traces_dir() -> Path:
    override = os.environ.get("TRACES_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return _REPO_ROOT.joinpath(*_DEFAULT_TRACES_SUBPATH)


def _normalise_lead(name: str) -> str:
    key = str(name).strip()
    if key in CANONICAL_LEADS:
        return key
    return _LEAD_ALIASES.get(key.upper(), key)


@lru_cache(maxsize=1)
def _manifest_index() -> dict[str, dict[str, str]]:
    """recording_id -> manifest row. Cached; the corpus is immutable."""
    path = traces_dir() / MANIFEST_NAME
    if not path.exists():
        raise TracesMissing(
            f"Trace manifest not found ({MANIFEST_NAME} missing). Download the "
            "shipped recordings, or point TRACES_DIR at the directory holding them."
        )
    index: dict[str, dict[str, str]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rid = (row.get("recording_id") or "").strip()
            if rid:
                index[rid] = row
    return index


def manifest_rows() -> list[dict[str, str]]:
    """Every manifest row, manifest order."""
    return list(_manifest_index().values())


def has_record(recording_id: str) -> bool:
    try:
        return recording_id in _manifest_index()
    except TracesMissing:
        return False


def record_base_path(recording_id: str) -> Path:
    """Extension-less WFDB base path for a shipped record."""
    row = _manifest_index().get(recording_id)
    if row is None:
        raise RecordNotFound(f"Unknown recording id '{recording_id}'.")
    rel = (row.get("path") or "").strip().replace("\\", "/")
    base = (traces_dir() / rel).resolve()
    root = traces_dir().resolve()
    # A manifest is data, not a capability: never let a row's path escape the
    # corpus directory.
    if root not in base.parents:
        raise RecordNotFound(f"Recording '{recording_id}' resolves outside TRACES_DIR.")
    if not base.with_suffix(".hea").exists():
        raise RecordNotFound(
            f"Recording '{recording_id}' is in the manifest but its .hea is not on disk."
        )
    return base


def read_wfdb(base_path: str | Path, *, recording_id: str | None = None) -> LeadSignals:
    """Read any WFDB pair (.hea + .mat/.dat) into canonical 12-lead form.

    Used for both shipped records and uploads.
    """
    base = Path(base_path)
    if base.suffix:
        base = base.with_suffix("")
    record = wfdb.rdrecord(str(base))
    signal = np.asarray(record.p_signal, dtype=np.float64).T  # (n_sig, n_samples)
    names = [_normalise_lead(n) for n in (record.sig_name or [])]

    missing = [lead for lead in CANONICAL_LEADS if lead not in names]
    if missing:
        raise ValueError(
            f"Recording is not a standard 12-lead study; missing leads: {', '.join(missing)}."
        )
    order = [names.index(lead) for lead in CANONICAL_LEADS]
    if order != list(range(12)):
        logger.info("Reordering leads %s -> canonical order", names)
    signal = np.ascontiguousarray(signal[order, :])

    if not np.isfinite(signal).all():
        # WFDB marks dropouts as NaN; the extractor needs finite samples.
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    return LeadSignals(
        recording_id=recording_id or record.record_name or base.name,
        signal=signal,
        sampling_rate=int(record.fs),
        lead_names=CANONICAL_LEADS,
    )


def load_record(recording_id: str) -> LeadSignals:
    """Read one shipped record by its PhysioNet recording id."""
    row = _manifest_index().get(recording_id)
    if row is None:
        raise RecordNotFound(f"Unknown recording id '{recording_id}'.")
    signals = read_wfdb(record_base_path(recording_id), recording_id=recording_id)
    return LeadSignals(
        recording_id=signals.recording_id,
        signal=signals.signal,
        sampling_rate=signals.sampling_rate,
        lead_names=signals.lead_names,
        units=signals.units,
        source=(row.get("source") or None),
        display_id=(row.get("display_id") or None),
    )


def read_csv_leads(path: str | Path, *, recording_id: str, sampling_rate: int) -> LeadSignals:
    """Read a 12-lead CSV upload: one column per lead, header naming the leads.

    Values are millivolts. Column order is free — the header decides.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError("CSV is empty.")
        names = [_normalise_lead(h) for h in header]
        missing = [lead for lead in CANONICAL_LEADS if lead not in names]
        if missing:
            raise ValueError(
                f"CSV is missing leads: {', '.join(missing)}. Expected a header "
                f"naming all 12 of {', '.join(CANONICAL_LEADS)}."
            )
        cols = [names.index(lead) for lead in CANONICAL_LEADS]
        rows: list[list[float]] = []
        for raw in reader:
            if not raw or all(not cell.strip() for cell in raw):
                continue
            rows.append([float(raw[c]) for c in cols])
    if not rows:
        raise ValueError("CSV has a header but no samples.")
    signal = np.asarray(rows, dtype=np.float64).T
    return LeadSignals(
        recording_id=recording_id,
        signal=np.ascontiguousarray(signal),
        sampling_rate=int(sampling_rate),
        lead_names=CANONICAL_LEADS,
    )
