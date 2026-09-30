"""Canonicalise a raw diagnosis string to the project vocabulary.

Vendored from the private ammonix `domain.DIAG_CANON` + `normalize_diagnosis`,
and the `namespace.normalize_to_model_class` idea of returning None for
anything outside the vocabulary.

Why the universe needs it: labels reach the data from three different places —
PhysioNet `#Dx:` SNOMED codes, the Apr28 label file, and (since the recovery)
GE 12SL machine statements — with different spellings and different ideas of
what counts as a diagnosis. Without canonicalising, the diagnosis picker offered
98 entries including "abnormal ecg", "borderline ecg" and the raw SNOMED code
"73795002", none of which are findings.

Canonicalising collapses that to 36 real classes. Note it FOLDS rather than
drops where the project already has a rule: "normal ecg" becomes
"sinus rhythm", matching normalize_diagnosis().

Returns None for anything with no canonical form — cohort markers, 12SL summary
verdicts, acquisition notes, unmapped SNOMED codes.
"""

from __future__ import annotations

import re

#: Raw variant -> canonical, vendored verbatim from ammonix/domain.py.
DIAG_CANON: dict[str, str] = {
    '1\\u00b0 av block': '1st av-block',
    '1\\u00b0 av-block': '1st av-block',
    '1st av-block': '1st av-block',
    '2\\u00b0 av block': '2nd av-block',
    '2\\u00b0 av-block': '2nd av-block',
    '2nd av-block': '2nd av-block',
    '3\\u00b0 av block': '3rd av-block',
    '3\\u00b0 av-block': '3rd av-block',
    '3rd av-block': '3rd av-block',
    'a-fib': 'atrial fibrillation',
    'acute myocardial infarction': 'myocardial infarction',
    'acute stemi': 'STEMI',
    'af': 'atrial fibrillation',
    'af risk': 'af risk',
    'af risk prediction': 'af risk',
    'afib': 'atrial fibrillation',
    'anterior mi': 'myocardial infarction',
    'anterior myocardial infarction': 'myocardial infarction',
    'anterior stemi': 'STEMI',
    'atrial fibrillation': 'atrial fibrillation',
    'atrial fibrillation risk': 'af risk',
    'atrial flutter': 'atrial flutter',
    'atrial premature beats': 'atrial premature beats',
    'atypical atrial flutter': 'atrial flutter',
    'auricular flutter': 'atrial flutter',
    'axis left shift': 'axis left shift',
    'axis right shift': 'axis right shift',
    'brugada': 'brugada syndrome',
    'brugada pattern': 'brugada syndrome',
    'brugada syndrome': 'brugada syndrome',
    'complete heart block': '3rd av-block',
    'complete left bundle branch block': 'left bundle branch block / variations',
    'complete right bundle branch block': 'right bundle branch block',
    'delta wave': 'wolff-parkinson-white',
    'first degree av block': '1st av-block',
    'flutter waves': 'atrial flutter',
    'incomplete right bundle branch block': 'right bundle branch block',
    'inferior myocardial infarction': 'myocardial infarction',
    'inferior stemi': 'STEMI',
    'irbbb': 'right bundle branch block',
    'lateral myocardial infarction': 'myocardial infarction',
    'lateral stemi': 'STEMI',
    'lbbb': 'left bundle branch block / variations',
    'left axis deviation': 'axis left shift',
    'left bundle branch block': 'left bundle branch block / variations',
    'left ventricle hypertrophy': 'left ventricular hypertrophy',
    'left ventricular hypertrophy': 'left ventricular hypertrophy',
    'leftward axis': 'axis left shift',
    'long qt': 'qt interval extension',
    'lvh': 'left ventricular hypertrophy',
    'myocardial infarction': 'myocardial infarction',
    'myocardial ischemia': 'myocardial infarction',
    'non st elevation myocardial infarction': 'nstemi',
    'non stemi': 'nstemi',
    'non-st elevation myocardial infarction': 'nstemi',
    'non-st-elevation myocardial infarction': 'nstemi',
    'non-stemi': 'nstemi',
    'nonspecific st-t abnormality': 'st deviation',
    'normal atrial rhythm': 'sinus rhythm',
    'normal ecg': 'sinus rhythm',
    'normal sinus rhythm': 'sinus rhythm',
    'nstemi': 'nstemi',
    'old mi': 'old myocardial infarction',
    'old myocardial infarction': 'old myocardial infarction',
    'paroxysmal ventricular tachycardia': 'ventricular tachycardia',
    'pre-excitation': 'wolff-parkinson-white',
    'preexcitation': 'wolff-parkinson-white',
    'premature atrial contractions': 'atrial premature beats',
    'premature ventricular contractions': 'ventricular premature beats',
    'prior mi': 'old myocardial infarction',
    'prior myocardial infarction': 'old myocardial infarction',
    'qt interval extension': 'qt interval extension',
    'qt prolongation': 'qt interval extension',
    'rbbb': 'right bundle branch block',
    'right axis deviation': 'axis right shift',
    'right bundle block': 'right bundle branch block',
    'right bundle branch block': 'right bundle branch block',
    'rightward axis': 'axis right shift',
    'second degree av block': '2nd av-block',
    'sinus arrhythmia': 'sinus irregularity',
    'sinus bradycardia': 'sinus bradycardia',
    'sinus irregularity': 'sinus irregularity',
    'sinus rhythm': 'sinus rhythm',
    'sinus tachycardia': 'sinus tachycardia',
    'st depression': 'st deviation',
    'st deviation': 'st deviation',
    'st drop down': 'st deviation',
    'st elevation': 'st deviation',
    'st elevation myocardial infarction': 'STEMI',
    'st extension': 'st deviation',
    'st segment changes': 'st deviation',
    'st segment depression': 'st deviation',
    'st segment elevation': 'st deviation',
    'st-elevation myocardial infarction': 'STEMI',
    'st-elevation myocardial infarction (stemi)': 'STEMI',
    'st-segment changes': 'st deviation',
    'st-t deviation': 'st deviation',
    'std': 'st deviation',
    'ste': 'st deviation',
    'stemi': 'STEMI',
    'supraventricular premature beats': 'atrial premature beats',
    'supraventricular premature contractions': 'atrial premature beats',
    't wave abnormality': 't wave change',
    't wave change': 't wave change',
    't wave inversion': 't wave inversion',
    't-wave change': 't wave change',
    'third degree av block': '3rd av-block',
    'type 1 brugada': 'brugada syndrome',
    'typical atrial flutter': 'atrial flutter',
    'v-fib': 'ventricular tachycardia',
    'v-flutter': 'ventricular tachycardia',
    'v-tach': 'ventricular tachycardia',
    'ventricular ectopic beats': 'ventricular premature beats',
    'ventricular ectopics': 'ventricular premature beats',
    'ventricular fibrillation': 'ventricular tachycardia',
    'ventricular flutter': 'ventricular tachycardia',
    'ventricular premature beat': 'ventricular premature beats',
    'ventricular premature beats': 'ventricular premature beats',
    'ventricular tachycardia': 'ventricular tachycardia',
    'ventricular tachycardia risk': 'vt risk',
    'vf': 'ventricular tachycardia',
    'vfib': 'ventricular tachycardia',
    'vflutter': 'ventricular tachycardia',
    'vt': 'ventricular tachycardia',
    'vt risk': 'vt risk',
    'vt risk prediction': 'vt risk',
    'vtach': 'ventricular tachycardia',
    'wolff parkinson white': 'wolff-parkinson-white',
    'wolff-parkinson-white': 'wolff-parkinson-white',
    'wpw': 'wolff-parkinson-white',
    'wpw syndrome': 'wolff-parkinson-white',
}

#: Merges applied on top of DIAG_CANON, for labels the vocabulary predates.
#:
#: The EF classes: the Jun3 rebuild scores 'lvef <40% expanded MIMIC EF>=55
#: controls' (6,140 records) while the older Apr28 work labelled 'lvef ≤45%'
#: (1,251, MIMIC-only, zero overlap). Both are echo-derived EF thresholds, not
#: ECG findings, and EF<40 is a strict subset of EF<=45 — so the superset label
#: covers both without asserting anything false. A deliberate choice.
#:
#: NOTE this one renames a SCORED class, so it must be applied to predictions
#: and threshold keys as well as labels or the two sides stop matching. See
#: services/source_safe_universe.py.
MERGES: dict[str, str] = {
    "lvef <40% expanded mimic ef>=55 controls": "lvef ≤45%",
    "lvef <40%": "lvef ≤45%",
    "lvef <=45%": "lvef ≤45%",
    "lvef ≤45%": "lvef ≤45%",
}

#: normalize_diagnosis() maps all of these to sinus rhythm rather than
#: discarding them — a normal ECG IS a rhythm statement.
NORMAL_VARIANTS = frozenset({
    "normal", "norm", "nsr", "normal sinus rhythm", "normal ecg",
    "no abnormalities", "within normal limits", "otherwise normal ecg",
})

_LOOKUP = {k.lower(): v for k, v in DIAG_CANON.items()}
_TRAILING_NUMS = re.compile(r"\s+\d+(\s+\d+)?$")
_PARENS = re.compile(r"\s*\([^)]*\)\s*$")
_WS = re.compile(r"\s+")


def canonicalise(raw: str, extra: dict[str, str] | None = None) -> str | None:
    """Canonical form of one diagnosis string, or None if it is not one.

    extra lets the caller accept the model's own scored classes verbatim —
    several (bifascicular block, pacing rhythm, low qrs voltages) are legitimate
    classes that predate DIAG_CANON and have no entry in it.
    """
    if raw is None:
        return None
    text = str(raw).replace("**", "").strip()
    if not text:
        return None
    text = _TRAILING_NUMS.sub("", text)
    low = text.lower().strip().replace("-", " ").replace("_", " ")
    low = _WS.sub(" ", low).strip()
    low = _PARENS.sub("", low).strip()

    if low in NORMAL_VARIANTS:
        return "sinus rhythm"
    # Merges run before the lookup: they deliberately override, and one of them
    # renames a scored class.
    if low in MERGES:
        return MERGES[low]
    if low in _LOOKUP:
        return _LOOKUP[low]
    if extra:
        if low in extra:
            return extra[low]
        exact = str(raw).strip().lower()
        if exact in extra:
            return extra[exact]
    return None


def canonicalise_all(labels, extra: dict[str, str] | None = None) -> list[str]:
    """Canonical, de-duplicated, sorted labels. Non-diagnoses are dropped."""
    out = {c for c in (canonicalise(l, extra) for l in (labels or [])) if c}
    return sorted(out)
