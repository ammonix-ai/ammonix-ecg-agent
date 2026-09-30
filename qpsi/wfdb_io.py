"""WFDB file discovery, loading, header parsing, and SNOMED diagnosis code mapping.

Source: Cell 1. No internal deps. External: wfdb, numpy, pathlib, re, os,
datetime, tempfile, shutil, math.
"""
from __future__ import annotations

import datetime
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import wfdb

# ---------------------------------------------------------------------------
# SNOMED-CT -> plain-English dictionary  (~135 entries)
# ---------------------------------------------------------------------------
SNOMED_TO_TEXT: dict[str, str] = {
    "10370003": "pacing rhythm",
    "106068003": "normal atrial rhythm (sinus)",
    "11157007": "ventricular bigeminy",
    "111975006": "QT interval extension",
    "13640000": "ventricular fusion wave",
    "164865005": "myocardial infarction (any wall)",
    "164873001": "left ventricle hypertrophy",
    "164889003": "atrial fibrillation",
    "164890007": "atrial flutter",
    "164896001": "ventricular fibrillation",
    "164909002": "left bundle branch block / variations",
    "164912004": "P wave Change",
    "164917005": "abnormal Q wave",
    "164930006": "ST extension",
    "164931005": "ST tilt up",
    "164934002": "T wave change",
    "164937009": "U wave",
    "164942001": "fQRS Wave",
    "164947007": "PR interval extension",
    "17338001": "ventricular premature beat",
    "195042002": "2\u00b0 AV-block",
    "195060002": "ventricular preexcitation",
    "195101003": "wandering in the AV node / sinus atrium to atrial wandering rhythm",
    "233896004": "AV Node Reentrant Tachycardia",
    "233897008": "atrioventricular reentrant tachycardia",
    "233917008": "atrioventricular block",
    "251146004": "lower voltage QRS in all leads",
    "251147008": "lower voltage QRS in limb leads",
    "251148003": "lower voltage QRS in chest leads",
    "251164006": "junctional premature beat",
    "251173003": "atrial bigeminy",
    "251180001": "ventricular escape trigeminy",
    "251198002": "clockwise rotation",
    "251199005": "counterclockwise rotation",
    "251205003": "prolonged P wave",
    "251223006": "tall P wave",
    "270492004": "1\u00b0 AV-block",
    "27885002": "3\u00b0 AV-block",
    "28189009": "2\u00b0 AV-block (Type II)",
    "284470004": "atrial premature beats",
    "29320008": "ectopic rhythm",
    "365413008": "R\u2011wave abnormality",
    "39732003": "axis left shift",
    "425856008": "paroxysmal ventricular tachycardia",
    "426177001": "sinus bradycardia",
    "426761007": "supraventricular tachycardia",
    "426783006": "sinus rhythm",
    "426995002": "junctional escape beat",
    "427084000": "sinus tachycardia",
    "427172004": "premature ventricular contractions",
    "427393009": "sinus irregularity",
    "428417006": "early repolarization of the ventricles",
    "428750005": "ST-T change",
    "429622005": "ST drop down",
    "445118002": "left anterior fascicular block",
    "446358003": "right atrial hypertrophy",
    "47665007": "axis right shift",
    "50799005": "atrioventricular dissociation",
    "54016002": "2\u00b0 AV-block (Type I)",
    "55827005": "left ventricular hypertrophy",
    "55930002": "ST\u2011segment changes (generic)",
    "59118001": "right bundle branch block",
    "59931005": "T wave change",
    "61721007": "counter\u2011clockwise vector loop",
    "6374002": "bundle branch block (generic)",
    "67751000119106": "right atrial enlargement",
    "698252002": "intraventricular block / interior diff conduction",
    "713422000": "atrial tachycardia",
    "713426002": "incomplete right bundle branch block",
    "713427006": "complete right bundle branch block",
    "733534002": "complete left bundle branch block",
    "74390002": "WPW",
    "75532003": "ventricular escape beat",
    "81898007": "ventricular escape rhythm",
    "89792004": "right ventricle hypertrophy",
    "111288001": "ventricular flutter",
    "164861001": "myocardial ischemia (any wall)",
    "164867002": "old myocardial infarction",
    "164884008": "ventricular ectopic beats",
    "164895002": "ventricular tachycardia",
    "164921003": "R wave abnormality",
    "164951009": "abnormal QRS complex",
    "17366009": "atrial arrhythmia",
    "195080001": "Atrial fibrillation and flutter",
    "195126007": "atrial hypertrophy",
    "204384007": "congenital incomplete AV-block",
    "233892002": "ectopic atrial tachycardia",
    "251120003": "incomplete left bundle branch block",
    "251139008": "arm lead reversal",
    "251166008": "AV Node Reentrant Tachycardia",
    "251168009": "supraventricular bigeminy",
    "251170000": "blocked premature atrial contraction",
    "251182009": "paired ventricular premature complexes",
    "251187003": "atrial escape complex",
    "251200008": "inferior ischemia",
    "251259000": "tall T wave",
    "251266004": "ventricular pacing",
    "251268003": "atrial pacing",
    "253339007": "right atrial abnormality",
    "253352002": "left atrial abnormality",
    "266249003": "ventricular hypertrophy",
    "266257000": "Transient ischemic attack (TIA)",
    "282825002": "paroxysmal atrial fibrillation",
    "314208002": "rapid atrial fibrillation",
    "368009": "heart valve disorder",
    "370365005": "left ventricular strain",
    "413444003": "acute myocardial ischemia",
    "413844008": "chronic myocardial ischemia",
    "418818005": "Brugada syndrome",
    "425419005": "inferior ischemia",
    "425623009": "lateral ischemia",
    "426183003": "2\u00b0 AV-block (Type II)",
    "426434006": "anterior ischemia",
    "426627000": "Bradycardia",
    "426648003": "junctional tachycardia",
    "426664006": "accelerated junctional rhythm",
    "426749004": "chronic atrial fibrillation",
    "445211001": "left posterior fascicular block",
    "446813000": "left atrial hypertrophy",
    "49260003": "Idioventricular rhythm",
    "49578007": "shortened PR interval",
    "53741008": "sinoatrial block",
    "54329005": "anterior myocardial infarction",
    "5609005": "sinus arrest",
    "57054005": "acute myocardial infarction",
    "60423000": "sinus node dysfunction",
    "61277005": "multifocal premature beats",
    "63593006": "supraventricular premature beats",
    "65778007": "sinoatrial block",
    "67198005": "paroxysmal supraventricular tachycardia",
    "67741000119109": "left atrial enlargement",
    "698247007": "cardiac arrhythmia",
    "704997005": "junctional escape rhythm",
    "74615001": "tachycardia-bradycardia",
    "77867006": "diffuse intraventricular block",
    "82226007": "diffuse intraventricular block",
    "84114007": "heart failure",
    # ---- Flags ----
    "99900010": "LVEF \u226445%",
    "99900011": "LV wall thickness \u226513 mm",
    "99900020": "Aortic stenosis (moderate or greater)",
    "99900021": "Aortic regurgitation (moderate or greater)",
    "99900022": "Mitral regurgitation (moderate or greater)",
    "99900023": "Tricuspid regurgitation (moderate or greater)",
    "99900024": "Pulmonary regurgitation (moderate or greater)",
    "99900025": "RV systolic dysfunction (moderate or greater)",
    "99900026": "Pericardial effusion (moderate or large)",
    "99900027": "Pulmonary artery systolic pressure \u226545 mmHg",
    "99900028": "Tricuspid velocity \u22653.2 m/s",
    "99900029": "Structural heart disease (moderate or greater)",
    # ---- Aortic stenosis ----
    "99900100": "Aortic stenosis \u2013 none",
    "99900101": "Aortic stenosis \u2013 mild",
    "99900102": "Aortic stenosis \u2013 moderate",
    "99900103": "Aortic stenosis \u2013 severe",
    # ---- Aortic regurgitation ----
    "99900110": "Aortic regurgitation \u2013 none",
    "99900111": "Aortic regurgitation \u2013 mild",
    "99900112": "Aortic regurgitation \u2013 moderate",
    "99900113": "Aortic regurgitation \u2013 severe",
    # ---- Mitral regurgitation ----
    "99900120": "Mitral regurgitation \u2013 none",
    "99900121": "Mitral regurgitation \u2013 mild",
    "99900122": "Mitral regurgitation \u2013 moderate",
    "99900123": "Mitral regurgitation \u2013 severe",
    # ---- Tricuspid regurgitation ----
    "99900130": "Tricuspid regurgitation \u2013 none",
    "99900131": "Tricuspid regurgitation \u2013 mild",
    "99900132": "Tricuspid regurgitation \u2013 moderate",
    "99900133": "Tricuspid regurgitation \u2013 severe",
    # ---- Pulmonary regurgitation ----
    "99900140": "Pulmonary regurgitation \u2013 none",
    "99900141": "Pulmonary regurgitation \u2013 mild",
    "99900142": "Pulmonary regurgitation \u2013 moderate",
    "99900143": "Pulmonary regurgitation \u2013 severe",
    # ---- RV systolic function ----
    "99900150": "RV systolic function \u2013 normal",
    "99900151": "RV systolic function \u2013 mildly reduced",
    "99900152": "RV systolic function \u2013 moderately reduced",
    "99900153": "RV systolic function \u2013 severely reduced",
    # ---- Pericardial effusion ----
    "99900160": "Pericardial effusion \u2013 none",
    "99900161": "Pericardial effusion \u2013 trace",
    "99900162": "Pericardial effusion \u2013 small",
    "99900163": "Pericardial effusion \u2013 moderate",
    "99900164": "Pericardial effusion \u2013 large",
    # ------ ST/T special EU Database features ----
    "99900200": "ST deviation (ischemic episode)",
    "99900201": "ST elevation",
    "99900202": "ST depression",
    "99900210": "T-wave deviation (ischemic episode)",
    "99900211": "T-wave elevation",
    "99900212": "T-wave inversion",
    "99900220": "Axis shift (positional change)",
    "99900230": "Noise / poor signal quality",
}


# ---------------------------------------------------------------------------
# WFDB record discovery
# ---------------------------------------------------------------------------

def find_wfdb_records(root_dir: str | Path) -> list[dict[str, str]]:
    """Walk *root_dir* and return every WFDB record that has both .hea and .mat.

    Args:
        root_dir: Top-level directory to search recursively.

    Returns:
        Sorted list of dicts with keys ``record_id``, ``hea_path``,
        ``mat_path``.
    """
    records: list[dict[str, str]] = []
    for subdir, _, files in os.walk(root_dir):
        hea_files = {
            os.path.splitext(f)[0]: os.path.join(subdir, f)
            for f in files if f.endswith('.hea')
        }
        mat_files = {
            os.path.splitext(f)[0]: os.path.join(subdir, f)
            for f in files if f.endswith('.mat')
        }
        for record_id in hea_files.keys() & mat_files.keys():
            records.append({
                'record_id': record_id,
                'hea_path': hea_files[record_id],
                'mat_path': mat_files[record_id],
            })
    records.sort(key=lambda r: r['record_id'])
    return records


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def parse_header_info(hea_file_path: str | Path) -> dict[str, Any]:
    """Extract age, sex, and SNOMED diagnosis codes from a .hea file.

    Parses ``#Age:``, ``#Sex:``, and ``#Dx:`` comment lines in
    PhysioNet-style WFDB headers.

    Args:
        hea_file_path: Path to the ``.hea`` header file.

    Returns:
        Dict with keys ``age`` (str | None), ``sex`` (str | None),
        ``dx_codes`` (list[str]).
    """
    info: dict[str, Any] = {"age": None, "sex": None, "dx_codes": []}
    if not os.path.exists(hea_file_path):
        return info

    with open(hea_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#Age:"):
                info["age"] = line.split(":", 1)[1].strip()
            elif line.startswith("#Sex:"):
                info["sex"] = line.split(":", 1)[1].strip()
            elif line.startswith("#Dx:"):
                dx_part = line.split(":", 1)[1].strip()
                info["dx_codes"] = [x.strip() for x in dx_part.split(",") if x.strip()]

    return info


def clean_header_base_date(hea_path: str | Path) -> None:
    """Patch malformed ``base_date`` fields in a .hea file in-place.

    Replaces bare ``/`` or empty ``base_date`` values with a safe
    fallback date (``01/01/1980``) so that ``wfdb.rdrecord`` does not
    crash on the header.

    Args:
        hea_path: Path to the ``.hea`` header file to fix.
    """
    with open(hea_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    changed = False

    # Patch standalone base_date fields
    for i, line in enumerate(lines):
        if re.match(r".*\bbase_date\s*/\s*$", line):
            lines[i] = re.sub(r"base_date\s*/", "base_date 01/01/1980", line)
            changed = True

    # Patch record line (first line) if it ends with a lone slash
    first_line = lines[0]
    fields = first_line.strip().split()
    if fields and fields[-1] == "/":
        fields[-1] = "01/01/1980"
        lines[0] = " ".join(fields) + "\n"
        changed = True

    # Patch any 'base_date /' elsewhere (paranoid fix)
    for i, line in enumerate(lines):
        if re.search(r"base_date\s*/", line):
            lines[i] = re.sub(r"base_date\s*/", "base_date 01/01/1980", line)
            changed = True

    if changed:
        with open(hea_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


# ---------------------------------------------------------------------------
# Safe WFDB record reading (with header patching)
# ---------------------------------------------------------------------------

def safe_rdrecord(base: str | Path, **kw: Any) -> wfdb.Record:
    """Read a WFDB record, automatically patching broken date fields.

    Copies the header to a temporary directory after fixing any
    malformed ``base_date`` entries, reads the record via
    ``wfdb.rdrecord``, then cleans up the temp files.

    Args:
        base: Base path to the WFDB record (without extension).
        **kw: Extra keyword arguments forwarded to ``wfdb.rdrecord``.

    Returns:
        A ``wfdb.Record`` instance.
    """
    base = Path(base)
    hea = base.with_suffix(".hea")
    lines: list[str] = []
    # Read all header lines
    with hea.open("r", encoding="ascii", errors="ignore") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) >= 9:
                # Check if last field is a plausible base_date, otherwise patch
                try:
                    datetime.datetime.strptime(fields[-1], "%d/%m/%Y")
                except Exception:
                    # Patch only if it looks like a date field or just a slash
                    if re.fullmatch(r"[/\d]{1,10}", fields[-1]) or fields[-1] == "/":
                        fields[-1] = "01/01/1901"
                        line = " ".join(fields) + "\n"
            lines.append(line)
    # Write patched header to temp dir
    tmp_dir = tempfile.mkdtemp(prefix="wfdbfix_")
    tmp_base = Path(tmp_dir) / base.name
    with (tmp_base.with_suffix(".hea")).open("w", encoding="ascii") as dst:
        dst.writelines(lines)
    for ext in (".dat", ".mat"):
        src = base.with_suffix(ext)
        if src.exists():
            shutil.copy(src, tmp_base.with_suffix(ext))
    rec = wfdb.rdrecord(str(tmp_base), **kw)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return rec


# ---------------------------------------------------------------------------
# Full ECG loading
# ---------------------------------------------------------------------------

def load_physionet_ecg(
    base: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """Load a PhysioNet .mat/.hea pair into (signals, metadata).

    Works for both JS**** and A**** style headers (handles ``# Age:``
    etc.).

    Args:
        base: Base path to the WFDB record (with or without ``.mat``
            extension).

    Returns:
        Tuple of ``(ecg_raw, meta)`` where *ecg_raw* is a 2-D numpy
        array of shape ``(n_leads, n_samples)`` and *meta* is a dict
        with keys ``record_name``, ``sampling_frequency``, ``n_sig``,
        ``sig_names``, ``age``, ``sex``, ``dx_codes``.
    """
    base = str(Path(base).with_suffix(""))  # ensure no .mat extension
    record = wfdb.rdrecord(base)
    sigs = record.p_signal.T
    leads = record.sig_name

    meta: dict[str, Any] = {
        "record_name": record.record_name,
        "sampling_frequency": record.fs,
        "n_sig": record.n_sig,
        "sig_names": leads,
        "age": None,
        "sex": None,
        "dx_codes": [],
    }

    # --- Parse the header manually for PhysioNet style comments ---
    hea_path = Path(base + ".hea")
    if hea_path.exists():
        with open(hea_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Normalize both "#Age:" and "# Age:"
                if re.match(r"^#\s*Age", line, re.I):
                    val = re.sub(r"^#\s*Age\s*:\s*", "", line, flags=re.I)
                    meta["age"] = int(val) if val.isdigit() else val
                elif re.match(r"^#\s*Sex", line, re.I):
                    val = re.sub(r"^#\s*Sex\s*:\s*", "", line, flags=re.I)
                    meta["sex"] = val.strip()
                elif re.match(r"^#\s*Dx", line, re.I):
                    val = re.sub(r"^#\s*Dx\s*:\s*", "", line, flags=re.I)
                    # split by comma or whitespace and keep only digits/letters
                    meta["dx_codes"] = [
                        c.strip() for c in re.split(r"[,\s]+", val) if c.strip()
                    ]
    return sigs, meta


# ---------------------------------------------------------------------------
# Pure math helper
# ---------------------------------------------------------------------------

def wave_3d_vector(angle_xy: float, angle_xz: float) -> tuple[float, float, float]:
    """Convert two plane angles into a unit 3-D direction vector.

    Args:
        angle_xy: Angle in the XY plane (radians).
        angle_xz: Angle in the XZ plane (radians).

    Returns:
        Tuple ``(vx, vy, vz)`` representing the unit vector.
    """
    vx = math.cos(angle_xz) * math.cos(angle_xy)
    vy = math.cos(angle_xz) * math.sin(angle_xy)
    vz = math.sin(angle_xz)
    return (vx, vy, vz)
