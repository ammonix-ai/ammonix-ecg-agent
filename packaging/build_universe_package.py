"""Build the publishable Ammonix ECG Agent universe package.

Reads the private source-safe universe and emits a scrubbed copy safe for public
release on Hugging Face: coordinates, probabilities and labels only.

What this removes, and why:

  * ``subject``            -- raw MIMIC-IV subject ids (``M:13283399``) on 51,856
                             of 63,256 rows, plus hospital MRN-shaped ids on the
                             clinical-partner rows. Publishing these would let
                             anyone with MIMIC credentials re-join and
                             re-identify the cohort.
  * ``recording_id``       -- kept ONLY for the five open PhysioNet sources,
                             where it is a public record id and the join key to
                             the raw traces we ship. Nulled everywhere else,
                             because for MIMIC rows it is the study id.
  * ``matrix``             -- internal build-matrix filename, meaningless
                             publicly.
  * ``__marker__`` labels  -- internal pipeline markers such as
                             ``__mi_family_quarantine__``.
  * absolute paths         -- ``data_dir``, ``package_dir`` and the
                             ``matrix_specs`` entries in metadata.json all carry
                             ``C://Users//peter//...``.

``display_id`` (``SS-00001``) is the public record key and is stable across
releases.

The field list is an ALLOWLIST: a new field appearing upstream is dropped rather
than published, and ``--strict`` turns it into an error so the packager fails
closed instead of leaking silently.

Usage::

    python build_universe_package.py --src <private universe dir> --out <staging dir>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

# --- source classification ------------------------------------------------

# Open PhysioNet sources. Redistributable, and the only ones whose raw signal
# traces we ship. Their record ids are public, so they survive scrubbing.
OPEN_SOURCES = frozenset({"PTB-XL", "CPSC", "CPSC-Extra", "Georgia", "Chapman"})

# MIMIC-derived. The processed rows ship; the identifiers do not.
MIMIC_SOURCES = frozenset(
    {"MIMIC", "MIMIC-LVEF", "MIMIC-LVEF40", "MIMIC-SR", "MIMIC-Control"}
)

# Clinical-partner data. Rows ship so the rare-disease clusters survive, but the
# source name is genericised and the ids are dropped. The diagnosis itself stays
# in ``labels``/``primary``, which is what actually colours the plot.
PRIVATE_SOURCES = frozenset(
    {
        "Brugada-HUCA",
        "Brugada",
        "private-STEMI",
        "STEMI-NSTEMI",
        "Amyloidosis",
        "Chagas",
        "VT",
        "pre-AF",
    }
)
PRIVATE_LABEL = "clinical-partner"

# Fields carried through to the public package, in order.
KEEP_FIELDS = (
    "idx",            # row index; aligns with probabilities.npy and projections
    "display_id",     # public record key
    "recording_id",   # open sources only (see _scrub_row)
    "source",
    "cohort",
    "labels",
    "primary",
    "predictions",
    "top_prediction",
    "max_score",
    "xgb_correct",
)

DROP_FIELDS = frozenset({"subject", "matrix"})

INTERNAL_LABEL = re.compile(r"^__.*__$")

# Anything that looks like a bare MIMIC study id (8 digits) or a subject
# reference. Used by the post-write audit.
MIMIC_ID = re.compile(r"^\d{8}$")
SUBJECT_REF = re.compile(r"\bM\W?:\s*\d+")


def classify(source: str) -> str:
    if source in OPEN_SOURCES:
        return "open"
    if source in MIMIC_SOURCES:
        return "mimic"
    if source in PRIVATE_SOURCES:
        return "private"
    return "unknown"


def _scrub_row(row: dict, stats: Counter, strict: bool) -> dict:
    source = row.get("source", "")
    kind = classify(source)

    if kind == "unknown":
        stats[f"unknown_source:{source}"] += 1
        if strict:
            raise SystemExit(
                f"Unclassified source {source!r}. Add it to OPEN_SOURCES, "
                f"MIMIC_SOURCES or PRIVATE_SOURCES before packaging."
            )

    unexpected = set(row) - set(KEEP_FIELDS) - DROP_FIELDS
    if unexpected:
        stats[f"unexpected_field:{','.join(sorted(unexpected))}"] += 1
        if strict:
            raise SystemExit(
                f"Unexpected field(s) {sorted(unexpected)} in the source data. "
                f"Classify them explicitly, then re-run."
            )

    out: dict = {}
    for field in KEEP_FIELDS:
        if field not in row:
            continue
        value = row[field]

        if field == "recording_id":
            # Public PhysioNet id for open sources; study id otherwise.
            if kind != "open":
                continue
            stats["recording_id_kept"] += 1

        elif field == "source" and kind == "private":
            value = PRIVATE_LABEL
            stats["source_genericised"] += 1

        elif field in ("labels", "predictions") and isinstance(value, list):
            cleaned = [x for x in value if not INTERNAL_LABEL.match(str(x))]
            if len(cleaned) != len(value):
                stats["internal_labels_stripped"] += len(value) - len(cleaned)
            value = cleaned

        out[field] = value

    stats[f"rows_{kind}"] += 1
    return out


def scrub_patients(src: Path, dst: Path, strict: bool) -> Counter:
    stats: Counter = Counter()
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(json.dumps(_scrub_row(json.loads(line), stats, strict), ensure_ascii=False) + "\n")
            stats["rows_total"] += 1
    return stats


def scrub_metadata(src: Path, dst: Path) -> dict:
    meta = json.loads(src.read_text(encoding="utf-8"))

    # Absolute developer paths.
    meta.pop("data_dir", None)
    meta.pop("package_dir", None)

    # matrix_specs carry four more absolute paths; keep the shape, drop the path.
    for spec in meta.get("matrix_specs", []) or []:
        if "path" in spec:
            spec["path"] = Path(str(spec["path"])).name

    # Fold the private sources into one public label.
    counts = meta.get("source_counts") or {}
    folded: dict[str, int] = {}
    private_total = 0
    for name, n in counts.items():
        if classify(name) == "private":
            private_total += n
        else:
            folded[name] = n
    if private_total:
        folded[PRIVATE_LABEL] = private_total
    meta["source_counts"] = dict(sorted(folded.items(), key=lambda kv: -kv[1]))

    meta["license_note"] = (
        "Processed universe only: coordinates, probabilities and labels. No raw "
        "signals, no MIMIC identifiers. Raw traces are published separately for "
        "the open PhysioNet sources only."
    )

    dst.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def audit(out_dir: Path) -> list[str]:
    """Re-read what was written and look for anything that should not be there."""
    problems: list[str] = []

    patients = out_dir / "patients.jsonl"
    seen_sources: Counter = Counter()
    for lineno, line in enumerate(patients.open(encoding="utf-8"), 1):
        row = json.loads(line)
        if "subject" in row:
            problems.append(f"patients.jsonl:{lineno} still carries 'subject'")
        if "matrix" in row:
            problems.append(f"patients.jsonl:{lineno} still carries 'matrix'")
        rid = row.get("recording_id")
        if rid is not None:
            if classify(row.get("source", "")) != "open":
                problems.append(f"patients.jsonl:{lineno} recording_id on non-open row")
            elif MIMIC_ID.match(str(rid)):
                problems.append(f"patients.jsonl:{lineno} recording_id {rid!r} looks like a MIMIC study id")
        seen_sources[row.get("source", "")] += 1
        if len(problems) > 20:
            problems.append("... further problems suppressed")
            break

    for name in PRIVATE_SOURCES:
        if name in seen_sources:
            problems.append(f"private source label {name!r} survived scrubbing")

    blob = (out_dir / "metadata.json").read_text(encoding="utf-8")
    for needle in ("C:\\\\Users", "C:/Users", "192.168.", "\\\\\\\\192"):
        if needle in blob:
            problems.append(f"metadata.json contains {needle!r}")
    if SUBJECT_REF.search(blob):
        problems.append("metadata.json contains a subject-id-shaped string")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="private source_safe_universe directory")
    ap.add_argument("--out", required=True, type=Path, help="staging directory to write")
    ap.add_argument("--strict", action="store_true", default=True,
                    help="fail on unknown sources or unexpected fields (default: on)")
    ap.add_argument("--no-strict", dest="strict", action="store_false")
    args = ap.parse_args()

    src, out = args.src, args.out
    if not (src / "patients.jsonl").exists():
        print(f"error: {src} does not look like a universe directory", file=sys.stderr)
        return 2

    out.mkdir(parents=True, exist_ok=True)
    (out / "projections").mkdir(exist_ok=True)

    print(f"scrubbing patients.jsonl ...")
    stats = scrub_patients(src / "patients.jsonl", out / "patients.jsonl", args.strict)

    print("scrubbing metadata.json ...")
    scrub_metadata(src / "metadata.json", out / "metadata.json")

    print("copying probabilities + projections ...")
    shutil.copy2(src / "probabilities.npy", out / "probabilities.npy")
    for f in sorted((src / "projections").glob("*")):
        if f.suffix in (".npy", ".json"):
            shutil.copy2(f, out / "projections" / f.name)

    print("\n--- packaging report ---")
    for key in sorted(stats):
        print(f"  {stats[key]:>8,}  {key}")

    print("\n--- audit ---")
    problems = audit(out)
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        print(f"\n{len(problems)} problem(s). Package NOT safe to publish.")
        return 1

    print("  clean: no subject ids, no MIMIC study ids, no private labels, no absolute paths")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
