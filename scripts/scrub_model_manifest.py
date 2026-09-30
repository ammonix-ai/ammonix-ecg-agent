#!/usr/bin/env python3
"""Strip developer paths and private source names from the classifier manifest.

Two separate problems, both of which ship publicly with the 167 MB package:

1. **Build paths.** `package_dir`, `data_dir` and `record_list` are absolute
   paths on the training box. Nothing reads them at inference time — feature
   order comes from `feature_names.json` and fold artefacts are referenced by
   package-relative paths — so they go.

2. **Private source names.** The per-diagnosis provenance blocks
   (`pos_by_source`, `neg_by_source`, `train_sources`, `excluded_train_sources`)
   name the clinical-partner cohorts directly: `Brugada-HUCA` appears 94 times
   and `private-STEMI` 62 times, across 31 of the 32 diagnoses, each with case
   counts. "HUCA" identifies a specific hospital. `build_universe_package.py`
   already folds these to `clinical-partner` in the universe; the model manifest
   must match that policy or the scrubbing is pointless — anyone can read the
   partner names straight out of the model card instead.

Counts are preserved and summed under the generic label, so provenance
arithmetic still works; only the attribution is removed.

Idempotent: run it again after re-copying the package.

    python scripts/scrub_model_manifest.py [--package-dir models/source_safe_canonical_v1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = REPO_ROOT / "models" / "source_safe_canonical_v1"

# field -> replacement. Keys stay present so any consumer reading them still
# finds a string, just not somebody's home directory.
REPLACEMENTS = {
    "package_dir": "models/source_safe_canonical_v1",
    "data_dir": "(not shipped — training feature matrices are not public)",
    "record_list": "(not shipped)",
}

# Must stay in step with PRIVATE_SOURCES in packaging/build_universe_package.py.
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

# Keys whose values name data sources, anywhere in the manifest tree.
SOURCE_MAPPINGS = ("pos_by_source", "neg_by_source")
SOURCE_LISTS = ("train_sources", "excluded_train_sources", "sources")


def _fold_mapping(mapping: dict) -> tuple[dict, int]:
    """Sum private sources into one generic bucket, preserving totals."""
    out: dict = {}
    private_total = 0
    hits = 0
    for name, value in mapping.items():
        if name in PRIVATE_SOURCES:
            hits += 1
            private_total += value if isinstance(value, (int, float)) else 0
        else:
            out[name] = value
    if hits:
        out[PRIVATE_LABEL] = out.get(PRIVATE_LABEL, 0) + private_total
    return out, hits


def _fold_list(values: list) -> tuple[list, int]:
    out: list = []
    hits = 0
    for name in values:
        if name in PRIVATE_SOURCES:
            hits += 1
            if PRIVATE_LABEL not in out:
                out.append(PRIVATE_LABEL)
        else:
            out.append(name)
    return out, hits


def scrub_sources(node, counter: dict) -> object:
    """Walk the manifest and genericise every private source reference."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key in SOURCE_MAPPINGS and isinstance(value, dict):
                value, hits = _fold_mapping(value)
                counter["mappings"] += hits
            elif key in SOURCE_LISTS and isinstance(value, list):
                value, hits = _fold_list(value)
                counter["lists"] += hits
            else:
                value = scrub_sources(value, counter)
            out[key] = value
        return out
    if isinstance(node, list):
        return [scrub_sources(v, counter) for v in node]
    if isinstance(node, str) and node in PRIVATE_SOURCES:
        counter["scalars"] += 1
        return PRIVATE_LABEL
    return node


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()

    path = args.package_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))

    changed = []
    for key, replacement in REPLACEMENTS.items():
        current = manifest.get(key)
        if isinstance(current, str) and current != replacement:
            manifest[key] = replacement
            changed.append(key)

    counter = {"mappings": 0, "lists": 0, "scalars": 0}
    manifest = scrub_sources(manifest, counter)
    folded = sum(counter.values())
    if folded:
        changed.append(
            f"{folded} private source refs -> {PRIVATE_LABEL} "
            f"({counter['mappings']} in count maps, {counter['lists']} in lists, "
            f"{counter['scalars']} scalar)"
        )

    if not changed:
        print(f"{path.name}: already clean")
        return 0

    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{path.name}: scrubbed {'; '.join(changed)}")

    # Fail loudly if anything survived, rather than reporting success.
    blob = path.read_text(encoding="utf-8")
    survivors = [n for n in PRIVATE_SOURCES if f'"{n}"' in blob]
    if survivors:
        print(f"  FAIL: still present after scrubbing: {', '.join(sorted(survivors))}")
        return 1
    print("  verified: no private source name remains in the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
