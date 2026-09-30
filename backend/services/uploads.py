"""Turn an uploaded file set into a 12-lead recording, or reject it clearly.

Everything a client can send is hostile until proved otherwise, so this module
is written as a gate rather than a reader:

* **Nothing is written outside the caller's temp directory.** Upload filenames
  are reduced to a bare, character-whitelisted basename before they touch the
  filesystem; `..`, absolute paths, drive letters and separators cannot survive
  ``safe_name()``.
* **Size is capped while streaming**, not after. Files are read in 1 MB chunks
  and the running total is checked against ``UPLOAD_MAX_BYTES`` (default 32 MB),
  so an oversized body is abandoned mid-read instead of buffered.
* **Every failure is a 4xx**, expressed as ``UploadRejected`` with a message a
  user can act on. Malformed WFDB headers throw a wide variety of exceptions
  from deep inside `wfdb`; the caller catches those and re-raises them as this
  type, with paths scrubbed.

Two upload shapes are accepted, matching the API contract:

    .hea + .mat|.dat    a WFDB pair. Sampling rate comes from the header.
    .csv                one column per lead, header row naming the 12 leads,
                        values in millivolts. Sampling rate comes from the
                        `samplingRate` form field (default 500 Hz).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from services.signals import CANONICAL_LEADS, LeadSignals, read_csv_leads, read_wfdb

logger = logging.getLogger(__name__)

#: Total bytes accepted across all parts of one upload.
DEFAULT_MAX_UPLOAD_BYTES = 32 * 1024 * 1024
#: A 12-lead study is at most a header + one data file; four allows for slop.
MAX_UPLOAD_FILES = 4
CHUNK_BYTES = 1024 * 1024

HEADER_SUFFIXES = (".hea",)
WFDB_DATA_SUFFIXES = (".mat", ".dat")
CSV_SUFFIXES = (".csv", ".txt")
ALLOWED_SUFFIXES = HEADER_SUFFIXES + WFDB_DATA_SUFFIXES + CSV_SUFFIXES

#: Sampling rates a 12-lead diagnostic ECG can plausibly carry. The universe was
#: extracted at 500 Hz; anything else still runs but is flagged in the response.
MIN_SAMPLING_RATE = 50
MAX_SAMPLING_RATE = 2000
REFERENCE_SAMPLING_RATE = 500
DEFAULT_CSV_SAMPLING_RATE = 500

#: QPSI needs beats to measure; a two-second strip is the floor. The ceiling
#: keeps one request from occupying the extractor for minutes.
MIN_DURATION_SEC = 2.0
MAX_DURATION_SEC = 120.0

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")
_UNSAFE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\)[^\s'\"<>|]*")
#: Names Windows resolves to devices no matter the extension — `nul.hea` opens
#: the null device, so a "file" written under that name reads back empty.
_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)
_MAX_STEM = 100


class UploadRejected(ValueError):
    """The upload is malformed, oversized or not a 12-lead study. Always a 4xx."""


class UploadTooLarge(UploadRejected):
    """Body exceeds the configured cap. 413."""


def max_upload_bytes() -> int:
    raw = os.environ.get("UPLOAD_MAX_BYTES", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_MAX_UPLOAD_BYTES


def uploads_enabled() -> bool:
    """Whether this deployment accepts user-supplied recordings at all.

    ``ANALYZE_UPLOADS_ENABLED=0`` turns the upload lane off: POST /api/analyze
    then serves shipped records only, and GET /api/status reports the flag so
    the UI hides the upload controls. Default on, matching the API contract.
    """
    return os.environ.get("ANALYZE_UPLOADS_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def scrub_paths(text: str) -> str:
    """Strip filesystem paths out of a message before it reaches a client.

    Temp directories and the server's own layout are not the caller's business,
    and `wfdb` puts the full path of whatever it failed to open into its
    exceptions.
    """
    return _UNSAFE_PATH.sub("<path>", str(text)).replace(str(Path.home()), "<home>")


def safe_name(raw: str) -> str:
    """A bare basename with a whitelisted character set, or a rejection.

    ``../../etc/passwd`` becomes ``passwd``; ``C:\\Windows\\x.hea`` becomes
    ``x.hea``. Nothing that reaches the filesystem contains a separator.
    """
    base = str(raw or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    base = _SAFE_NAME.sub("_", base).lstrip(".")
    if not base:
        raise UploadRejected("An uploaded file has no usable filename.")
    stem, dot, suffix = base.rpartition(".")
    if not dot:  # no extension at all; the suffix check downstream rejects it
        return base[:_MAX_STEM]
    if stem.lower() in _RESERVED_STEMS:
        stem = f"_{stem}"
    # Truncate the stem, never the extension: the extension is what decides how
    # the file is read, and a 200-character name must not silently lose it.
    return f"{stem[:_MAX_STEM]}.{suffix[:16]}"


@dataclass(frozen=True)
class SavedUpload:
    name: str
    path: Path
    size: int

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    @property
    def stem(self) -> str:
        return self.path.stem


async def save_uploads(files: Iterable[Any], dest: Path) -> list[SavedUpload]:
    """Stream Starlette ``UploadFile`` parts into ``dest`` under safe names.

    ``dest`` must be a directory the caller owns (a ``TemporaryDirectory``).
    Returns what was written; raises ``UploadTooLarge`` as soon as the running
    total crosses the cap, leaving the partial file behind for the caller's
    temp-dir cleanup to remove.
    """
    dest = Path(dest)
    cap = max_upload_bytes()
    saved: list[SavedUpload] = []
    total = 0

    for upload in files:
        if len(saved) >= MAX_UPLOAD_FILES:
            raise UploadRejected(
                f"Too many files in one upload (max {MAX_UPLOAD_FILES}). Send a "
                ".hea plus its .mat/.dat, or a single .csv."
            )
        name = safe_name(getattr(upload, "filename", "") or "")
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise UploadRejected(
                f"'{name}' has an unsupported extension. Accepted: "
                f"{', '.join(ALLOWED_SUFFIXES)}."
            )
        if any(f.name == name for f in saved):
            # Two parts sanitise to one name; writing both would silently lose one.
            raise UploadRejected(f"Two uploaded files resolve to the same name '{name}'.")
        target = (dest / name).resolve()
        if target.parent != dest.resolve():
            # Unreachable after safe_name(); kept as the last line of defence.
            raise UploadRejected(f"Refusing to write '{name}' outside the upload directory.")

        written = 0
        with open(target, "wb") as out:
            while True:
                chunk = await upload.read(CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > cap:
                    raise UploadTooLarge(
                        f"Upload exceeds the {cap // (1024 * 1024)} MB limit."
                    )
                written += len(chunk)
                out.write(chunk)
        if written == 0:
            raise UploadRejected(f"'{name}' is empty.")
        saved.append(SavedUpload(name=name, path=target, size=written))

    if not saved:
        raise UploadRejected(
            "No files in the upload. Send a WFDB .hea plus its .mat/.dat, or a "
            "12-lead .csv."
        )
    return saved


def _is_number(token: str) -> bool:
    try:
        float(token.split("/")[0])
    except ValueError:
        return False
    return True


def _header_data_files(header_path: Path) -> list[str]:
    """Data-file names a WFDB header refers to, one per signal line.

    Parsed here rather than left to `wfdb` so a missing companion file produces
    "you did not upload X" instead of a stack trace about a path.
    """
    try:
        text = header_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - defensive
        raise UploadRejected(f"Could not read the header file: {scrub_paths(exc)}") from exc

    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise UploadRejected("The .hea file is empty.")

    record_line = lines[0].split()
    if "/" in record_line[0] or "+" in record_line[0]:
        raise UploadRejected(
            "Multi-segment WFDB records are not supported; upload a single-segment "
            "12-lead study."
        )
    # `name nsig fs nsamp ...` — check the shape before trusting the rest, so a
    # text file renamed .hea fails as "not a WFDB header" rather than as a
    # confusing complaint about a missing signal file named after some word in it.
    if len(record_line) < 3 or not _is_number(record_line[1]) or not _is_number(record_line[2]):
        raise UploadRejected(
            "That .hea is not a WFDB header. Its first line must read "
            "'<record> <n_signals> <sampling_rate> [<n_samples>]'."
        )
    names: list[str] = []
    for line in lines[1:]:
        token = line.split()[0]
        if token.startswith("-"):  # a duration/info line, not a signal spec
            continue
        names.append(token.replace("\\", "/").rsplit("/", 1)[-1])
    if not names:
        raise UploadRejected("The .hea file lists no signals.")
    return names


def _validate_signals(signals: LeadSignals, *, sampling_rate_source: str) -> dict[str, Any]:
    """Shape, rate and duration checks common to both upload kinds."""
    n_leads, n_samples = signals.signal.shape
    if n_leads != len(CANONICAL_LEADS):
        raise UploadRejected(
            f"Expected {len(CANONICAL_LEADS)} leads, got {n_leads}."
        )
    fs = int(signals.sampling_rate)
    if not (MIN_SAMPLING_RATE <= fs <= MAX_SAMPLING_RATE):
        raise UploadRejected(
            f"Sampling rate {fs} Hz is outside the accepted range "
            f"{MIN_SAMPLING_RATE}-{MAX_SAMPLING_RATE} Hz."
        )
    duration = n_samples / fs
    if duration < MIN_DURATION_SEC:
        raise UploadRejected(
            f"Recording is {duration:.2f} s long; at least {MIN_DURATION_SEC:.0f} s "
            "is needed to measure beats."
        )
    if duration > MAX_DURATION_SEC:
        raise UploadRejected(
            f"Recording is {duration:.0f} s long; the limit is {MAX_DURATION_SEC:.0f} s."
        )

    info: dict[str, Any] = {
        "samplingRateSource": sampling_rate_source,
        "leadCount": n_leads,
        "sampleCount": int(n_samples),
    }
    if fs != REFERENCE_SAMPLING_RATE:
        info["samplingRateNote"] = (
            f"This analysis was built from {REFERENCE_SAMPLING_RATE} Hz "
            f"recordings; this one is {fs} Hz. It is scored anyway, but rate-sensitive "
            "features may shift."
        )
    return info


def signals_from_upload(
    saved: list[SavedUpload],
    *,
    sampling_rate: int | None = None,
) -> tuple[LeadSignals, dict[str, Any]]:
    """Read a saved upload set into canonical 12-lead form.

    Returns the signals plus an info block describing how they were read (never
    a path). Raises ``UploadRejected`` for anything malformed.
    """
    headers = [f for f in saved if f.suffix in HEADER_SUFFIXES]
    data_files = [f for f in saved if f.suffix in WFDB_DATA_SUFFIXES]
    csv_files = [f for f in saved if f.suffix in CSV_SUFFIXES]

    if headers:
        return _read_wfdb_upload(headers, data_files, saved)
    if data_files:
        raise UploadRejected(
            f"'{data_files[0].name}' arrived without its .hea header. WFDB signal "
            "files cannot be read on their own."
        )
    if csv_files:
        return _read_csv_upload(csv_files, sampling_rate)
    raise UploadRejected(
        "Nothing recognisable in the upload. Send a WFDB .hea plus its .mat/.dat, "
        "or a 12-lead .csv."
    )


def _read_wfdb_upload(
    headers: list[SavedUpload],
    data_files: list[SavedUpload],
    saved: list[SavedUpload],
) -> tuple[LeadSignals, dict[str, Any]]:
    if len(headers) > 1:
        raise UploadRejected(
            f"Two headers in one upload ({headers[0].name}, {headers[1].name}); send one record."
        )
    header = headers[0]
    if not data_files:
        raise UploadRejected(
            f"'{header.name}' arrived without a signal file. Upload the matching "
            ".mat or .dat alongside it."
        )

    uploaded = {f.name for f in saved}
    for wanted in _header_data_files(header.path):
        if safe_name(wanted) not in uploaded:
            raise UploadRejected(
                f"The header references signal file '{wanted}', which was not "
                "uploaded. Send the .hea and its data file together, with their "
                "original names."
            )

    try:
        signals = read_wfdb(header.path.with_suffix(""), recording_id=header.stem)
    except UploadRejected:
        raise
    except ValueError as exc:
        # read_wfdb raises this for a study that is not standard 12-lead.
        raise UploadRejected(scrub_paths(exc)) from exc
    except Exception as exc:  # wfdb throws widely on malformed headers
        raise UploadRejected(
            f"Could not read the WFDB record: {type(exc).__name__}: {scrub_paths(exc)}"
        ) from exc

    info = _validate_signals(signals, sampling_rate_source="wfdb_header")
    info["kind"] = "wfdb"
    info["files"] = sorted(f.name for f in saved)
    return signals, info


def _read_csv_upload(
    csv_files: list[SavedUpload],
    sampling_rate: int | None,
) -> tuple[LeadSignals, dict[str, Any]]:
    if len(csv_files) > 1:
        raise UploadRejected(
            f"Two CSV files in one upload ({csv_files[0].name}, {csv_files[1].name}); send one."
        )
    csv_file = csv_files[0]
    if sampling_rate is None:
        fs = DEFAULT_CSV_SAMPLING_RATE
        source = "default"
    else:
        fs = int(sampling_rate)
        source = "form_field"
    if not (MIN_SAMPLING_RATE <= fs <= MAX_SAMPLING_RATE):
        raise UploadRejected(
            f"samplingRate={fs} is outside the accepted range "
            f"{MIN_SAMPLING_RATE}-{MAX_SAMPLING_RATE} Hz."
        )

    try:
        signals = read_csv_leads(csv_file.path, recording_id=csv_file.stem, sampling_rate=fs)
    except UploadRejected:
        raise
    except ValueError as exc:
        raise UploadRejected(scrub_paths(exc)) from exc
    except Exception as exc:
        raise UploadRejected(
            f"Could not read the CSV: {type(exc).__name__}: {scrub_paths(exc)}"
        ) from exc

    info = _validate_signals(signals, sampling_rate_source=source)
    info["kind"] = "csv"
    info["files"] = [csv_file.name]
    if source == "default":
        info["samplingRateAssumed"] = (
            f"A CSV carries no sampling rate; {DEFAULT_CSV_SAMPLING_RATE} Hz was assumed. "
            "Send a samplingRate form field to override."
        )
    return signals, info


def parse_sampling_rate(raw: str | None) -> int | None:
    """Read the optional `samplingRate` form field."""
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError) as exc:
        raise UploadRejected(f"samplingRate '{raw}' is not a number.") from exc
    if value <= 0:
        raise UploadRejected("samplingRate must be positive.")
    return value
