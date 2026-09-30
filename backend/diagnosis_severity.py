"""Clinical severity ordering used to pick the diagnosis that represents a record.

WHY THIS EXISTS
---------------
``primary`` in the shipped universe was a display convention, not a clinical
judgement: it sorted the labels alphabetically and took the first that passed a
couple of filters. So 47% of records were represented by whichever label sorted
first, 2,749 records with several competing pathologies were decided by the
alphabet (``lvef <=45%`` beat ``myocardial infarction`` because *l* precedes
*m*), and because the sort ran on raw strings, uppercase ``STEMI`` jumped ahead
of everything lowercase on ASCII order.

This module replaces that with an explicit ranking: when a recording carries
several findings, the most clinically consequential one represents it, and the
same ranking drives the colour it gets in the universe.

STATUS: DRAFT, NOT CLINICALLY APPROVED
--------------------------------------
Drafted by a non-clinician, pending review by a cardiac electrophysiologist. Do
not present it as validated clinical content. Reorder freely — position IS the
rank, and nothing else in the codebase hardcodes a position.

The data carries 96 distinct labels while the model scores 32 classes, so this
ranks well beyond the scored set. Anything left unranked (raw SNOMED codes,
acquisition notes like "lead(s) unsuitable for analysis") sorts last, which is
the honest answer for a string nobody can interpret.

Note: no ventricular tachycardia class exists among the 32 scored diagnoses, so
sustained VT is represented only by the unscored ``vt-risk`` label.
"""

from __future__ import annotations

import re

# Ordered most consequential first. The tier comments are the argument for each
# grouping and are the part most worth disagreeing with.
SEVERITY_ORDER: tuple[str, ...] = (
    # ── 1. Acute ischaemia and infarction — minutes matter ───────────────
    "STEMI",
    "acute myocardial ischemia",
    "nstemi",
    "myocardial infarction",
    "inferior ischemia",

    # ── 2. Failing rhythm or conduction — syncope, asystole, arrest ──────
    # Complete heart block and escape rhythms sit here rather than with the
    # other conduction disease: an escape rhythm means the level above it has
    # already failed.
    "sudden death",
    "3rd av-block",
    "3 av-block",
    "av dissociation",
    "ventricular escape rhythm",
    "idioventricular rhythm",
    "vt-risk",

    # ── 3. Sudden-death substrate — the ECG is the whole diagnosis ───────
    "brugada syndrome",
    "wolff-parkinson-white",
    "qt interval extension",
    "shortened pr interval",

    # ── 4. Tachyarrhythmia with embolic or haemodynamic consequence ──────
    # Above structural disease because AF changes anticoagulation today,
    # whereas a reduced EF changes management over weeks. Reasonable to
    # disagree and lift tier 5 above this one.
    "atrial fibrillation",
    "atrial flutter",
    "supraventricular tachycardia",
    "av node reentrant tachycardia",
    "ectopic atrial tachycardia",
    "af risk",

    # ── 5. Ventricular dysfunction, cardiomyopathy, chamber disease ──────
    "heart failure",
    "lvef <40% expanded MIMIC EF>=55 controls",
    "lvef ≤45%",
    "amyloidosis",
    "chagas disease",
    "left ventricular hypertrophy",
    "right ventricular hypertrophy",
    "right ventricle hypertrophy",
    "old myocardial infarction",
    "abnormal q wave",
    "pericarditis",
    "low qrs voltages",
    "lower voltage qrs in all leads",
    "left atrial enlargement",
    "left atrial hypertrophy",
    "right atrial enlargement",
    "biatrial enlargement",
    "poor r wave progression",

    # ── 6. Conduction disease ────────────────────────────────────────────
    # LBBB first: a new LBBB with symptoms is an ACS equivalent and a single
    # ECG cannot say whether it is new. Bifascicular above RBBB for its risk
    # of progressing to complete block.
    "left bundle branch block / variations",
    "bifascicular block",
    "diffuse intraventricular block",
    "right bundle branch block",
    "left anterior fascicular block",
    "left posterior fascicular block",
    "non-specific intraventricular conduction delay",
    "1st av-block",
    "1 av-block",

    # ── 7. Ectopy, escape and device rhythm ──────────────────────────────
    # QUESTIONABLE: a paced rhythm is a device state, not a severity. It also
    # invalidates any reading of repolarisation and axis, so there is a case
    # for ranking it near the top purely so the reader is not misled by
    # everything below it.
    "ventricular premature beats",
    "ventricular bigeminy",
    "pacing rhythm",
    "junctional rhythm",
    "ectopic atrial rhythm",
    "ectopic rhythm",
    "atrial premature beats",
    "supraventricular bigeminy",
    "blocked premature atrial contraction",
    "undetermined rhythm",

    # ── 8. Nonspecific morphology and repolarisation ─────────────────────
    # 'st deviation' lives here, not in tier 1. It is the signature of acute
    # injury, but genuine ACS already carries STEMI / nstemi / myocardial
    # infarction, which outrank everything here — so demoting it costs no
    # acute case and stops it representing records whose real story is atrial
    # fibrillation. Kept above the rate descriptors because a repolarisation
    # abnormality is a finding, while a rate is often physiological.
    "st deviation",
    "t wave change",
    "st tilt up",
    "early repolarization",
    "early repolarization of the ventricles",
    "p wave change",
    "abnormal qrs",
    "r-wave abnormality",
    "r wave abnormality",
    "rsr pattern",
    "rsr'(v1) - probable normal variant",
    "broad r or r' in v1 or v2",
    "axis left shift",
    "axis right shift",
    "indeterminate axis",
    "qrs axis superior to -30 degrees",
    "counterclockwise rotation",

    # ── 9. Rate and rhythm descriptors ───────────────────────────────────
    "bradycardia",
    "sinus bradycardia",
    "tachycardia",
    "sinus tachycardia",
    "sinus irregularity",

    # ── 10. Summary statements and normality ─────────────────────────────
    # Last on purpose: these describe the absence of a specific finding, so
    # they should only represent a record that has nothing else to say.
    "abnormal ecg",
    "borderline ecg",
    "sinus rhythm",
    "normal ecg",
)

#: Labels that record COHORT MEMBERSHIP, not a finding on the ECG.
#:
#: 'lvef >=55% control' says the record was recruited as a preserved-EF control
#: for the LVEF study. It is not something read off the trace, and it should
#: never represent a recording or appear in a list of diagnoses. All 23,013
#: records carrying it carry nothing else, so before this they were all
#: represented by a non-diagnosis.
#:
#: The cohort itself is not lost: those same records carry
#: cohort = 'LVEF EF>=55 controls', which is what the Cohort colouring uses.
NON_DIAGNOSTIC_MARKERS: frozenset[str] = frozenset({
    # Cohort membership.
    "lvef >=55% control",

    # 12SL SUMMARY VERDICTS. The machine prints one of these as the last line
    # of every report — they are its overall impression, not a finding, and a
    # reader picking diagnoses should never be offered "abnormal ecg" as one.
    # The label recovery pulled them in with the real statements, which is how
    # 'abnormal ecg' went from 221 to 14,478 and 'borderline ecg' from 0 to
    # 6,596. 'normal ecg' predates it but belongs here for the same reason.
    "normal ecg",
    "abnormal ecg",
    "borderline ecg",
    "otherwise normal ecg",
    "within normal limits",

    # Acquisition and interpretation notes, not clinical findings.
    "poor quality data, interpretation may be affected",
    "interpretation made without knowledge of patient's sex and age",
    "repeat if myocardial injury is suspected",
})

#: Raw SNOMED codes that reached the label set unmapped (e.g. '73795002',
#: 170 records). Diagnoses in principle, but nobody can read one, and offering
#: it in a diagnosis picker is worse than omitting it. Worth mapping through
#: DIAG_CANON properly rather than leaving as-is.
_SNOMED_CODE = re.compile(r"^\d{5,}$")

#: What a recording is called when every label was a cohort marker.
#:
#: 'unknown', NOT 'sinus rhythm'. Preserved EF does not mean a normal ECG —
#: these patients were recruited on echo alone. Checked against MIMIC's
#: machine_measurements.csv, which every one of the 23,013 controls joins to:
#:
#:     40.4%  the machine read says ABNORMAL ECG
#:     21.7%  normal ECG
#:            the rest borderline
#:
#: and the reports carry real findings — atrial fibrillation on 940, RBBB on
#: 976, left axis deviation on 1,621. The classifier already calls something on
#: 82% of them, against 38% for the genuinely-normal MIMIC SR cohort.
#:
#: Their ECG diagnoses were never joined into the universe; only cohort
#: membership was recorded. Until that join is done, 'unknown' is the honest
#: answer: we do not know what these ECGs show. Calling them sinus rhythm
#: labelled AF and atrial-flutter patients as normal.
MARKER_ONLY_FALLBACK = "unknown"


def is_diagnosis(label: str) -> bool:
    """False for anything that is not an ECG finding.

    Excludes cohort markers, 12SL summary verdicts, acquisition notes and raw
    SNOMED codes — none of which belong in a list of diagnoses.
    """
    s = str(label).strip().lower()
    if not s or s in NON_DIAGNOSTIC_MARKERS:
        return False
    return not _SNOMED_CODE.match(s)


def clinical_labels(labels) -> list[str]:
    """``labels`` with cohort markers removed."""
    return [str(dx) for dx in (labels or []) if str(dx).strip() and is_diagnosis(dx)]

#: Lower rank = more severe. Lookup is case-insensitive: the wire mixes cases
#: ('STEMI' in labels, 'stemi' as a primary), and an exact-case lookup is what
#: greyed out 6,140 points in the universe viewer.
_RANK: dict[str, int] = {dx.lower(): i for i, dx in enumerate(SEVERITY_ORDER)}

#: Unranked labels — raw SNOMED codes, acquisition notes — sort after every
#: ranked one, then alphabetically, so they are stable rather than arbitrary
#: and can never outrank a real finding.
UNRANKED = len(SEVERITY_ORDER)


def severity_rank(diagnosis: str) -> int:
    """Rank of a diagnosis; lower is more severe. Unknown labels sort last."""
    return _RANK.get(str(diagnosis).strip().lower(), UNRANKED)


def primary_diagnosis(labels) -> str:
    """The label that best represents a recording: the most severe one.

    Cohort markers are dropped first — they describe how a record was recruited,
    not what the ECG shows. A record left with nothing but markers is a
    preserved-EF control, i.e. sinus rhythm.

    Among equally-unranked labels the alphabetically first wins, so the result
    does not depend on the order labels happened to arrive in.
    """
    raw = [str(dx) for dx in (labels or []) if str(dx).strip()]
    if not raw:
        return "unknown"
    usable = clinical_labels(raw)
    if not usable:
        return MARKER_ONLY_FALLBACK
    return min(usable, key=lambda dx: (severity_rank(dx), dx.lower()))


def coverage(labels) -> dict[str, list[str]]:
    """Which of ``labels`` this ordering ranks and which it misses.

    Used by the tests so a class added to the model cannot silently fall to the
    unranked tail.
    """
    ranked, missing = [], []
    for dx in labels:
        (ranked if severity_rank(dx) < UNRANKED else missing).append(dx)
    known = {str(c).lower() for c in labels}
    return {
        "ranked": ranked,
        "missing": missing,
        "not_in_data": [dx for dx in SEVERITY_ORDER if dx.lower() not in known],
    }
