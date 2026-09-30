"""Collect the publishable raw ECG traces for the Ammonix ECG Agent.

Ships the WFDB traces for the 10,876 universe records that come from the five
open PhysioNet sources, and nothing else.

The hazard this guards against
------------------------------
MIMIC-derived and clinical-partner cohorts live in the SAME ``WFDBRecords/``
tree as the open data -- ``HERG_block/``, ``mimic_lvef40_ecgs/``,
``control_sinus_*_ecgs/``, ``mimic_control_ecgs/``, ``VT_patients/``,
``verified_stemi_nstemi/``, ``Brugada_HUCA/`` and friends are sibling folders.
Several of their headers carry MIMIC ``subject_id``/``Study`` lines and free-text
clinical history. A recursive copy of that tree, or a path index followed
blindly, would publish patient data.

So every record is resolved and then checked against a per-source ALLOWLIST of
dataset roots. A record whose path lands anywhere else is refused, not skipped
quietly. Note ``ptb/`` (PTB Diagnostic) is a different dataset from ``ptb-xl/``
and is NOT shipped.

Each copied header is also scanned for identifier-shaped lines as a second,
independent line of defence.

Usage::

    python build_trace_package.py --plan          # dry run: resolve + audit only
    python build_trace_package.py --copy          # actually write the package
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_UNIVERSE = Path(__file__).resolve().parent.parent / "_staging" / "universe"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "_staging" / "traces"

# The source WFDB tree is a local, private path -- it is never baked into the
# repo. Set WFDB_ROOT, or pass --wfdb-root.
_ENV_WFDB = os.environ.get("WFDB_ROOT")
DEFAULT_WFDB = Path(_ENV_WFDB) if _ENV_WFDB else None

# Per-source allowlist of permitted path prefixes, relative to the WFDB root.
# A resolved record MUST sit under one of its source's prefixes.
ALLOWED_ROOTS: dict[str, tuple[str, ...]] = {
    "PTB-XL": ("ptb-xl",),
    "CPSC": ("cpsc_2018",),
    "CPSC-Extra": ("cpsc_2018_extra",),
    "Georgia": ("georgia",),
    # Chapman is sharded across two-digit numeric directories 01..46.
    "Chapman": tuple(f"{n:02d}" for n in range(1, 47)),
}
OPEN_SOURCES = frozenset(ALLOWED_ROOTS)

# Folders that must never contribute a single file. Belt and braces: the
# allowlist above already excludes them, but an explicit denylist makes a
# mistake loud instead of silent.
FORBIDDEN = (
    "wfdbrecords_mimic",
    "mimic_lvef40_ecgs",
    "mimic_control_ecgs",
    "herg_block",
    "control_sinus_brady_ecgs",
    "control_sinus_tachy_ecgs",
    "vt_patients",
    "pre_af_ecgs",
    "pre_vt_ecgs",
    "verified_stemi_nstemi",
    "pre_post_stemi_ecgs",
    "brugada_huca",
    "brugada_ecgs",
    "amyloidosis",
    "chagas_ecgs",
    "wpw_ecgs",
    "touch_ecg",
    "pediatric",
    "st_petersburg_incart",
    "edb",
    "ptb/",       # PTB Diagnostic, not PTB-XL
    "ptb\\",
)

# Header lines that would indicate patient identifiers leaked into a .hea file.
# The open CinC2020 headers carry only '# Dx:', '# Age:', '# Sex:' and similar.
SUSPECT_HEADER = re.compile(
    r"(?i)#\s*(<?subject_id>?|patient|study|history|name|mrn|admission)\b"
)


def load_open_records(universe_dir: Path) -> list[tuple[str, str, str]]:
    """Return (recording_id, source, display_id) for the rows from open data sources."""
    out = []
    with open(universe_dir / "patients.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            src = row.get("source")
            rid = row.get("recording_id")
            if src in OPEN_SOURCES and rid:
                out.append((str(rid), src, row.get("display_id", "")))
    return out


def load_path_index(wfdb_root: Path) -> dict[str, str]:
    idx_file = wfdb_root / ".wfdb_path_index.json"
    if not idx_file.exists():
        raise SystemExit(f"path index not found: {idx_file}")
    raw = json.loads(idx_file.read_text(encoding="utf-8"))
    # Tolerate either {id: path} or {id: {path: ...}}.
    out: dict[str, str] = {}
    for k, v in raw.items():
        out[k] = v if isinstance(v, str) else (v or {}).get("path", "")
    return out


def is_allowed(rel: Path, source: str) -> bool:
    parts = [p.lower() for p in rel.parts]
    if not parts:
        return False
    head = parts[0]
    return any(head == pref.lower() for pref in ALLOWED_ROOTS[source])


def header_is_clean(hea: Path) -> list[str]:
    bad = []
    try:
        for i, line in enumerate(hea.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if SUSPECT_HEADER.search(line):
                bad.append(f"{hea.name}:{i}: {line.strip()[:80]}")
    except OSError as exc:
        bad.append(f"{hea.name}: unreadable ({exc})")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    ap.add_argument("--wfdb-root", type=Path, default=DEFAULT_WFDB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--copy", action="store_true", help="actually copy files (default is a dry run)")
    ap.add_argument("--plan", action="store_true", help="dry run (default)")
    args = ap.parse_args()

    if args.wfdb_root is None:
        print(
            "error: source WFDB tree not set.\n"
            "       Pass --wfdb-root <path>, or set the WFDB_ROOT environment variable.\n"
            "       It is a local private path and is deliberately not stored in this repo.",
            file=sys.stderr,
        )
        return 2

    do_copy = args.copy and not args.plan

    records = load_open_records(args.universe)
    print(f"open-data records in universe: {len(records):,}")

    index = load_path_index(args.wfdb_root)
    print(f"path index entries:              {len(index):,}\n")

    stats: Counter = Counter()
    per_source: dict[str, Counter] = defaultdict(Counter)
    refused: list[str] = []
    missing: list[str] = []
    manifest: list[dict] = []
    total_bytes = 0

    for rid, source, display_id in records:
        raw = index.get(rid)
        if not raw:
            missing.append(f"{source}/{rid}: not in path index")
            stats["missing"] += 1
            continue

        p = Path(raw)
        try:
            rel = p.relative_to(args.wfdb_root)
        except ValueError:
            # Index holds absolute paths from another machine layout; fall back
            # to the tail after the WFDBRecords component.
            parts = list(p.parts)
            low = [x.lower() for x in parts]
            if "wfdbrecords" in low:
                rel = Path(*parts[low.index("wfdbrecords") + 1:])
            else:
                refused.append(f"{source}/{rid}: path outside WFDBRecords -> {raw}")
                stats["refused"] += 1
                continue

        rel_low = str(rel).lower().replace("\\", "/")
        if any(f.strip("/\\") in rel_low.split("/") or f in rel_low for f in FORBIDDEN):
            refused.append(f"{source}/{rid}: DENYLISTED path -> {rel}")
            stats["refused_denylist"] += 1
            continue

        if not is_allowed(rel, source):
            refused.append(f"{source}/{rid}: outside allowlist for {source} -> {rel}")
            stats["refused_allowlist"] += 1
            continue

        hea = args.wfdb_root / rel.with_suffix(".hea")
        sig = None
        for ext in (".mat", ".dat"):
            cand = args.wfdb_root / rel.with_suffix(ext)
            if cand.exists():
                sig = cand
                break
        if not hea.exists() or sig is None:
            missing.append(f"{source}/{rid}: header or signal absent at {rel}")
            stats["missing_files"] += 1
            continue

        size = hea.stat().st_size + sig.stat().st_size
        total_bytes += size
        stats["ok"] += 1
        per_source[source]["ok"] += 1
        per_source[source]["bytes"] += size

        manifest.append(
            {
                "display_id": display_id,
                "recording_id": rid,
                "source": source,
                "path": f"{source}/{hea.stem}",
                "bytes": size,
            }
        )

        if do_copy:
            dest_dir = args.out / source
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in (hea, sig):
                dest = dest_dir / f.name
                if not dest.exists() or dest.stat().st_size != f.stat().st_size:
                    shutil.copy2(f, dest)

    # ---- report ----------------------------------------------------------
    print("--- per source ---")
    for source in sorted(per_source):
        c = per_source[source]
        print(f"  {source:<12} {c['ok']:>6,} records   {c['bytes']/1024**2:>8.1f} MB")
    print(f"\n  TOTAL        {stats['ok']:>6,} records   {total_bytes/1024**3:>8.2f} GB")

    print("\n--- safety ---")
    for key in ("refused", "refused_denylist", "refused_allowlist", "missing", "missing_files"):
        if stats[key]:
            print(f"  {stats[key]:>6,}  {key}")
    if refused:
        print("\n  REFUSED (first 10):")
        for r in refused[:10]:
            print(f"    {r}")
    if missing:
        print("\n  MISSING (first 10):")
        for m in missing[:10]:
            print(f"    {m}")
    if not refused and not missing:
        print("  no refusals, no missing files")

    # ---- header scan on what we actually copied --------------------------
    if do_copy:
        print("\n--- header identifier scan ---")
        bad: list[str] = []
        for hea in sorted(args.out.rglob("*.hea")):
            bad.extend(header_is_clean(hea))
            if len(bad) > 20:
                break
        if bad:
            print(f"  FAIL: {len(bad)} suspect header line(s)")
            for b in bad[:20]:
                print(f"    {b}")
            return 1
        print("  clean: no subject/patient/study/history lines in any copied header")

        args.out.mkdir(parents=True, exist_ok=True)
        man = args.out / "manifest.csv"
        with open(man, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["display_id", "recording_id", "source", "path", "bytes"])
            w.writeheader()
            w.writerows(manifest)
        print(f"\nwrote {man} ({len(manifest):,} rows)")
        print(f"wrote {args.out}")
    else:
        print("\n(dry run -- pass --copy to write the package)")

    if refused:
        print("\nREFUSALS PRESENT: resolve them before publishing.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
