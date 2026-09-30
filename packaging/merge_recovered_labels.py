"""Merge the machine-recovered diagnoses into the shipped universe.

Background: three cohorts (LVEF EF>=55 controls, LVEF EF<40 cases, MIMIC SR
normals — 48,958 records, 77% of the universe) reached the universe with no ECG
diagnoses. The QPSI pipeline reads diagnoses from the ``#Dx:`` line of the WFDB
header; MIMIC headers have no such line, so every one of them recorded
``"No diagnosis codes available"``. The findings were sitting unused in MIMIC's
``machine_measurements.csv`` — the GE Marquette 12SL statements — which join at
100% on study_id.

``recover_machine_labels.py`` mapped those statements to the universe's own
canonical vocabulary. This script merges the result.

What it does per record:
  * unions the recovered labels with whatever the record already had;
  * recomputes ``primary`` by clinical severity, since a record that had only a
    cohort marker now has real findings to be represented by;
  * recomputes ``xgb_correct``, which is exact-set-match against the scored
    classes — leaving it stale would keep counting correct predictions as
    errors, which is the whole reason this mattered;
  * stamps ``label_source`` so the provenance is not lost. These are a machine's
    reads, not adjudicated human labels, and anything downstream should be able
    to tell.

Writes a timestamped backup first. Idempotent: re-running merges the same union
and produces the same result.

    python merge_recovered_labels.py --dry-run     # report, change nothing
    python merge_recovered_labels.py --apply
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

from diagnosis_severity import clinical_labels, primary_diagnosis  # noqa: E402

UNIVERSE = REPO / "_staging" / "universe"
PATIENTS = UNIVERSE / "patients.jsonl"
METADATA = UNIVERSE / "metadata.json"

#: Written onto every record the merge touches, so a consumer can tell a
#: machine-read label from one that came with the source dataset.
LABEL_SOURCE = "mimic-12sl-machine-read"


def load_recovered(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            did = row.get("display_id")
            if did:
                out[did] = list(row.get("recovered_labels") or [])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recovered", type=Path, required=True)
    ap.add_argument("--apply", action="store_true", help="write the merge (default is a dry run)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    apply = args.apply and not args.dry_run

    recovered = load_recovered(args.recovered)
    print(f"recovered-label records: {len(recovered):,}")

    scored = {k.lower() for k in json.loads(METADATA.read_text(encoding="utf-8"))["thresholds"]}

    rows = [json.loads(l) for l in PATIENTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"universe records       : {len(rows):,}\n")

    stats = collections.Counter()
    primary_changed = collections.Counter()
    correct_before = correct_after = 0

    for row in rows:
        did = row.get("display_id")
        before_labels = list(row.get("labels") or [])
        before_primary = row.get("primary")
        if row.get("xgb_correct"):
            correct_before += 1

        new = recovered.get(did)
        if new:
            merged = sorted(set(before_labels) | set(new))
            if merged != sorted(set(before_labels)):
                stats["records_gained_labels"] += 1
                stats["labels_added"] += len(set(new) - set(before_labels))
            row["labels"] = merged
            row["label_source"] = LABEL_SOURCE
        else:
            stats["records_untouched"] += 1

        # Primary is recomputed for EVERY record, not just merged ones: the
        # severity ordering is the source of truth now and the stored value was
        # alphabetical.
        row["primary"] = primary_diagnosis(row.get("labels") or [])
        if row["primary"] != before_primary:
            primary_changed[(before_primary, row["primary"])] += 1

        # Exact set match over the scored classes only — the same rule the
        # universe used, so Accuracy and Error Highlight stay consistent.
        gold = {d.lower() for d in clinical_labels(row.get("labels") or []) if d.lower() in scored}
        preds = {p.lower() for p in (row.get("predictions") or [])}
        row["xgb_correct"] = gold == preds
        if row["xgb_correct"]:
            correct_after += 1

    print(f"records gaining labels : {stats['records_gained_labels']:,}")
    print(f"labels added           : {stats['labels_added']:,}")
    print(f"records untouched      : {stats['records_untouched']:,}")
    print()
    print(f"xgb_correct  before    : {correct_before:,} ({100*correct_before/len(rows):.1f}%)")
    print(f"xgb_correct  after     : {correct_after:,} ({100*correct_after/len(rows):.1f}%)")
    print()
    print("largest primary shifts:")
    for (a, b), n in primary_changed.most_common(10):
        print(f"  {n:>7,}  {a!r} -> {b!r}")
    print(f"  total primaries changed: {sum(primary_changed.values()):,}")

    if not apply:
        print("\n(dry run — pass --apply to write)")
        return 0

    backup = PATIENTS.with_suffix(f".jsonl.pre_recovery_backup")
    if not backup.exists():
        shutil.copy2(PATIENTS, backup)
        print(f"\nbackup written: {backup.name}")
    else:
        print(f"\nbackup already exists, left alone: {backup.name}")

    tmp = PATIENTS.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(PATIENTS)
    print(f"wrote {PATIENTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
