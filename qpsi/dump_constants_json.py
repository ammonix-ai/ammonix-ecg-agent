"""Emit qpsi_native/constants.json from qpsi/constants.py for build.rs codegen.

H34 / Q-A6-13: ends Python <-> Rust constant duplication. The EXPORTED
registry below is the contract between the two languages; constants.py is
the single source of truth, and qpsi_native/build.rs reads the JSON this
script produces to emit a generated constants.rs into Cargo's OUT_DIR at
build time.

Type policy:
    "f64"   -> Rust ``f64`` (Python float or int)
    "usize" -> Rust ``usize`` (Python non-negative int — array indexing)
    "i64"   -> Rust ``i64`` (Python int)

Constants that intentionally stay native (Rust-only tuning) live in the
respective .rs file and do not appear here.

Usage::

    # As a script (canonical: build.rs invokes it this way):
    python qpsi/dump_constants_json.py qpsi/qpsi_native/constants.json

    # As a module:
    python -m qpsi.dump_constants_json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``qpsi`` importable regardless of how we are invoked (script vs -m).
_QPSI_DIR = Path(__file__).resolve().parent
_PARENT = str(_QPSI_DIR.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from qpsi import constants as _constants  # noqa: E402

# Registry of constants that cross the Python <-> Rust boundary.
# Tuple: (python_attr_name, rust_const_name, rust_type).
EXPORTED: list[tuple[str, str, str]] = [
    # Wave region boundaries (plane.rs::time_to_region_label)
    ("EARLY_P_MS",    "EARLY_P_MS",          "f64"),
    ("NORMAL_P_MS",   "NORMAL_P_MS",         "f64"),
    ("QRS_FRAC_MS",   "QRS_FRAC_MS",         "f64"),
    ("ST_FRAC",       "ST_FRAC",             "f64"),
    ("T_EARLY_FRAC",  "T_EARLY_FRAC",        "f64"),
    ("T_NORMAL_FRAC", "T_NORMAL_FRAC",       "f64"),
    ("delta_angle",   "DEFAULT_DELTA_ANGLE", "f64"),
    # Beat segmentation
    ("PAD_HEAD_R_MS", "PAD_HEAD_R_MS",       "f64"),
    # Pacemaker detection
    ("PACE_SIGMA_MS", "PACE_SIGMA_MS",       "f64"),
    ("PACE_AMP_MV",   "PACE_AMP_MV",         "f64"),
    # Hybrid fitter parameters — paired between plane_analysis.py and plane.rs.
    ("QRS_HYBRID_N_WAVES",      "QRS_N_WAVES",       "usize"),
    ("QRS_HYBRID_SIGMA_MAX",    "QRS_SIGMA_MAX",     "f64"),
    ("QRS_HYBRID_ANGLE_LAMBDA", "QRS_ANGLE_LAMBDA",  "f64"),
    ("QRS_HYBRID_AMP_LAMBDA",   "QRS_AMP_LAMBDA",    "f64"),
    ("T_HYBRID_N_WAVES",        "T_N_WAVES",         "usize"),
    ("T_HYBRID_SIGMA_MAX",      "T_SIGMA_MAX",       "f64"),
    ("T_HYBRID_ANGLE_LAMBDA",   "T_ANGLE_LAMBDA",    "f64"),
    ("T_HYBRID_AMP_LAMBDA",     "T_AMP_LAMBDA",      "f64"),
    ("P_HYBRID_N_WAVES",        "P_N_WAVES",         "usize"),
    ("P_HYBRID_SIGMA_MAX",      "P_SIGMA_MAX",       "f64"),
    ("P_HYBRID_ANGLE_LAMBDA",   "P_ANGLE_LAMBDA",    "f64"),
    ("P_HYBRID_AMP_LAMBDA",     "P_AMP_LAMBDA",      "f64"),
]


def build_payload() -> dict[str, dict[str, object]]:
    """Build {rust_name: {value, type, source_name}} from constants.py."""
    payload: dict[str, dict[str, object]] = {}
    for py_name, rust_name, rust_type in EXPORTED:
        if not hasattr(_constants, py_name):
            raise AttributeError(
                f"qpsi.constants is missing '{py_name}' "
                f"(referenced by EXPORTED in dump_constants_json.py)"
            )
        raw = getattr(_constants, py_name)
        if rust_type == "f64":
            value: object = float(raw)
        elif rust_type == "i64":
            value = int(raw)
        elif rust_type == "usize":
            cast = int(raw)
            if cast < 0:
                raise ValueError(
                    f"Cannot export {py_name}={cast} as usize (negative)"
                )
            value = cast
        else:
            raise ValueError(
                f"Unknown rust_type {rust_type!r} for {py_name}"
            )
        payload[rust_name] = {
            "value": value,
            "type": rust_type,
            "source_name": py_name,
        }
    return payload


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        sys.stderr.write("usage: dump_constants_json.py [OUTPUT_PATH]\n")
        return 2
    if len(argv) == 2:
        out_path = Path(argv[1])
    else:
        out_path = _QPSI_DIR / "qpsi_native" / "constants.json"
    payload = build_payload()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
