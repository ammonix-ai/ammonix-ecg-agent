"""Quickstart: run the ECG tokenizer on one public 12-lead recording.

Downloads record 00001 of PTB-XL (CC BY 4.0) from PhysioNet, reads it the way the
backend reads uploads, and extracts the 26,490 named features the classifier
consumes. Needs no data package, no classifier and no language model.

    python examples/tokenize_ptbxl_record.py

Without the optional Rust accelerator the extraction takes about 20 seconds.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import wfdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from services.inference import extract_features  # noqa: E402
from services.signals import read_wfdb  # noqa: E402

RECORD = "records500/00000/00001_hr"
CACHE = ROOT / "examples" / "data" / "ptb-xl"


def main() -> None:
    base = CACHE / RECORD
    if not base.with_suffix(".hea").exists():
        print(f"Downloading PTB-XL {RECORD} from PhysioNet ...")
        wfdb.dl_files("ptb-xl", str(CACHE), [RECORD + ".hea", RECORD + ".dat"])
    ecg = read_wfdb(base, recording_id="ptbxl-00001")
    print(f"Read {ecg.recording_id}: 12 leads, {ecg.duration_sec:.0f} s at {ecg.sampling_rate} Hz")

    t0 = time.time()
    features = extract_features(ecg.signal, ecg.sampling_rate, recording_id=ecg.recording_id)
    print(f"Extracted {len(features):,} named features in {time.time() - t0:.1f} s\n")

    print("A few of them, lead II, median over beats:")
    for wave in ("P", "QRS", "T"):
        for stat in ("rms_amplitude_mv", "duration_ms_fwhm"):
            name = f"ml_direct::II::{wave}::{stat}::median"
            print(f"  {name} = {features[name]:.3f}")
    per_lead = sum(n.startswith("ml_direct::V1::") for n in features)
    print(f"\nEach lead carries {per_lead:,} features; the frontal and horizontal planes add the rest.")


if __name__ == "__main__":
    main()
