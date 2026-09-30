"""Recover ECG diagnoses for the 48,958 universe records that have none.

WHY
---
Three cohorts (LVEF EF>=55 controls, LVEF EF<40 cases, MIMIC SR normals) were
assembled as bare file lists.  Their manifests carry recording_id / hea_path /
subject_id and nothing else, so the q_psi jun3 run had no diagnosis codes to
attach and every one of them ended up with

    Label.Diagnosis = ["No diagnosis codes available"]

which the universe build turned into cohort membership only.  46% of the
universe therefore looks unlabelled while the classifier calls a finding on 82%
of the "controls".

MIMIC-IV ships the GE/Marquette 12SL machine read for exactly these records in
machine_measurements.csv (report_0 .. report_17).  All 48,958 join at 100% on
study_id.  This script maps those free-text statements onto the universe's
canonical class names and writes a SEPARATE recovered-labels file.  Nothing is
retrained, refit or re-scored, and patients.jsonl is not modified.

WHAT THE MAPPING IS
-------------------
Pattern rules, not one entry per string.  The 12SL vocabulary is regular:
'inferior infarct - age undetermined', 'possible inferior infarct - age
undetermined' and 'inferior infarct, old' are the same finding with different
qualifiers, so one 'infarct' rule covers all three (plus an 'old' rule that adds
the age qualifier).  Rules are ADDITIVE: every rule is tested against every
statement and the labels union, which is what makes compounds like 'sinus rhythm
with pvc(s) with 1st degree a-v block' decompose correctly into three labels.

A statement gets one of four dispositions:

    labelled   at least one rule fired
    technical  acquisition note, not a clinical finding ('lead(s) unsuitable
               for analysis: v2', 'arm lead reversal', 'report made without
               knowing patient's age')
    no-finding recognised, deliberately carries no diagnosis ('no other
               finding', 'probable normal variant', 'cannot rule out
               anteroseptal infarct' -- an explicit non-call)
    unmapped   nothing fired; reported with counts so the gap is visible

Coverage is reported both ways: labelled-only, and labelled + technical +
no-finding (everything the mapping has an opinion about).

DELIBERATE DEVIATIONS FROM THE BRIEF'S MAPPING NOTES
----------------------------------------------------
1. 'cannot rule out <region> infarct' does NOT produce myocardial infarction.
   The brief says any '* infarct *' maps to MI, but 'cannot rule out' is 12SL's
   explicit non-call -- 'poor r wave progression - cannot rule out septal
   infarct' means PRWP, nonspecific.  Those statements still get their other
   findings; the standalone ones are booked as no-finding.
2. 'abnormal r-wave progression, early transition' maps to counterclockwise
   rotation, not poor r wave progression.  Early transition is the opposite of
   poor progression.  Plain 'abnormal r-wave progression' still maps to PRWP.
3. ST elevation described as early repolarisation maps to 'early
   repolarization' and NOT 'st deviation' -- it is a normal variant.
4. 'x:1 a-v block' inside an atrial-flutter statement is flutter conduction,
   not AV nodal disease, so it does not produce 2nd av-block.

Outputs (scratchpad, never the staged universe):
    recovered_labels.jsonl    one object per affected record
    recovery_report.json      the numbers printed at the end, machine-readable
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRATCH = Path(
    r"/path/to/scratch"
    r""
    r"/37adc070-feaf-456c-9694-b1d7ee490af5/scratchpad"
)
UNIVERSE = Path(r"/path/to/ammonix-ecg-agent/_staging/universe")

AFFECTED_IDS = SCRATCH / "affected_ids.json"
REPORTS = SCRATCH / "reports.json"
PATIENTS = UNIVERSE / "patients.jsonl"
METADATA = UNIVERSE / "metadata.json"

OUT_JSONL = SCRATCH / "recovered_labels.jsonl"
OUT_REPORT = SCRATCH / "recovery_report.json"


# =============================================================================
# NORMALISATION
# =============================================================================

def norm(line: str) -> str:
    """Lowercase, collapse whitespace, drop the trailing full stop.

    Matches the normalisation used to build stmts.json ('Sinus rhythm.' and
    'Sinus rhythm' are one statement, 33,536 occurrences).
    """
    s = re.sub(r"\s+", " ", str(line)).strip().lower()
    return s.rstrip(".").strip()


# =============================================================================
# DISPOSITION 1: TECHNICAL -- acquisition notes, not findings
# =============================================================================

TECHNICAL = [
    r"report made without knowing",
    r"interpretation made without know",
    r"data quality",
    r"age not entered",
    r"lead\(s\) unsuitable",
    r"lead reversal",
    r"analyzed ---",
    r"omitted from analysis",
    r"faulty v\d",
    r"sequence error",
    r"pediatric criteria",
    r"recording unsuitable",
    r"analysis error",
    r"pacer detection suspended",
    r"measurement error",
    r"if rhythm correct",
    r"suggests dextrocardia",
    r"complex qrs morphology",
    r"repeat if myocardial injury is suspected$",
    r"^-{2,}",
    r"leads analyzed",
    r"partial missing lead",
]

# =============================================================================
# DISPOSITION 2: NO-FINDING -- recognised, deliberately unlabelled
# =============================================================================

NO_FINDING = [
    r"^no other finding$",
    r"^probable normal variant$",
    r"equivocal significance",
    r"leads are also involved",
    r"changes may also partly be due to rhythm",
    r"appearances provide additional evidence",
    r"offer additional evidence",
    r"^regular rhythm$",
    r"normal for age",
    r"^these changes may be due to",
    r"^changes may (also |be due )?part",
    # 12SL's explicit non-call; see deviation 1 in the module docstring.
    r"^cannot rule out .*infarct",
    r"^consider (old )?(anterior|inferior|septal|lateral|anteroseptal|anterolateral|extensive) infarct$",
]

# =============================================================================
# DISPOSITION 3: CLINICAL RULES  (name, pattern, labels, exclude-pattern)
# =============================================================================

Rule = tuple[str, str, tuple[str, ...], str | None]

RULES: list[Rule] = [
    # ---- sinus / rate ------------------------------------------------------
    ("sinus rhythm", r"\bsinus rhythm\b", ("sinus rhythm",), None),
    ("sinus tachycardia", r"\bsinus tachycardia\b|sinus \(\? atrial\) tachycardia",
     ("sinus tachycardia",), None),
    ("sinus bradycardia", r"\bsinus bradycardia\b", ("sinus bradycardia",), None),
    ("sinus arrhythmia", r"\bsinus arrhythmia\b|\birregular sinus\b",
     ("sinus irregularity",), None),

    # ---- atrial arrhythmia -------------------------------------------------
    ("atrial fibrillation", r"\batrial fibrillation\b|\bafib\b|afib/flut",
     ("atrial fibrillation",), None),
    ("atrial flutter", r"\batrial flutter\b|afib/flut", ("atrial flutter",), None),
    ("svt", r"supraventricular tachycardia", ("supraventricular tachycardia",), None),
    ("atrial tachycardia", r"atrial tachycardia", ("ectopic atrial tachycardia",), None),
    ("ectopic atrial rhythm", r"ectopic atrial (rhythm|bradycardia)",
     ("ectopic atrial rhythm",), None),
    ("ectopic atrial brady", r"ectopic atrial bradycardia", ("bradycardia",), None),

    # ---- junctional / ventricular / device ---------------------------------
    ("junctional", r"junctional (rhythm|tachycardia)", ("junctional rhythm",), None),
    ("idioventricular", r"idioventricular rhythm", ("idioventricular rhythm",), None),
    ("ventricular escape", r"ventricular escape", ("ventricular escape rhythm",), None),
    # (?<!supra) is load-bearing: without it 'supraventricular tachycardia'
    # matched the VT rule and 35 SVT records were labelled ventricular
    # tachycardia. Calling an SVT a VT is the most dangerous error this mapper
    # could make. \b stops 'ventricular tachycardias' style suffixes drifting.
    ("ventricular tachycardia", r"(?<!supra)ventricular tachycardia\b",
     ("ventricular tachycardia",), None),
    ("pacing", r"\bpacemaker\b|\bpacing\b|\bpaced\b|\bv-paced\b",
     ("pacing rhythm",), None),
    ("av dissociation", r"a-?v dissociation", ("av dissociation",), None),
    ("undetermined rhythm", r"undetermined rhythm|unknown rhythm|regular supraventricular rhythm",
     ("undetermined rhythm",), None),

    # ---- axis --------------------------------------------------------------
    ("axis left", r"left axis deviation|leftward axis|\blad\b|axis leftward|"
                  r"qrs axis superior to -30",
     ("axis left shift",), None),
    ("axis right", r"right axis deviation|rightward axis|\brad\b", ("axis right shift",), None),
    ("axis indeterminate", r"indeterminate (frontal qrs )?axis|extreme qrs axis",
     ("indeterminate axis",), None),

    # ---- AV conduction -----------------------------------------------------
    ("1st av block", r"(1st|first) degree a-?v block|prolonged pr interval",
     ("1st av-block",), None),
    ("2nd av block", r"(2nd|second) degree a-?v block|mobitz|wenckebach",
     ("2nd av-block",), None),
    ("2nd av block ratio", r"\d:1 a-?v block", ("2nd av-block",), r"flutter"),
    ("3rd av block", r"(3rd|third) degree a-?v block|complete heart block|"
                     r"complete a-?v block|a-?v block, complete|\(third degree\)",
     ("3rd av-block",), None),
    ("short pr", r"short pr interval", ("shortened pr interval",), None),

    # ---- intraventricular conduction ---------------------------------------
    ("rbbb", r"\brbbb\b|right bundle branch block", ("right bundle branch block",), None),
    ("lbbb", r"\blbbb\b|left bundle branch block",
     ("left bundle branch block / variations",), None),
    ("lafb", r"\blafb\b|(left )?anterior fascicular block", ("left anterior fascicular block",), None),
    ("lpfb", r"\blpfb\b|left posterior fascicular block", ("left posterior fascicular block",), None),
    ("ivcd", r"\bivcd\b|i\.?v\.? conduction defect|intraventricular conduction (defect|delay)|"
             r"prolonged qrs duration|abnormal ventricular conduction pathways",
     ("non-specific intraventricular conduction delay",), None),
    ("rsr", r"\brsr'|rsr' pattern|broad r or r' in v1", ("rsr pattern",), None),

    # ---- ectopy ------------------------------------------------------------
    # Every 'ventricular' alternative needs the (?<!supra) guard, not just the
    # bigeminy one — 'supraventricular ectopy' and 'supraventricular bigeminy'
    # both reached this rule and emitted a ventricular ectopy call.
    ("vpb", r"\bpvcs?\b|pvc\(s\)|premature ventricular contraction|"
            r"(?<!supra)ventricular premature complex|(?<!supra)ventricular ectop|"
            r"(?<!supra)ventricular couplet|(?<!supra)ventricular (bi|tri)geminy|"
            r"multiple premature complexes",
     ("ventricular premature beats",), None),
    # Same (?<!supra) guard: 'supraventricular bigeminy' was emitting
    # ventricular bigeminy AND ventricular premature beats — and VPB is a
    # scored class, so it inflated the agreement statistic too.
    ("ventricular bigeminy", r"(?<!supra)ventricular bigeminy|bigeminal pvc",
     ("ventricular bigeminy",), None),
    ("apb", r"\bpacs?\b|pac\(s\)|premature atrial contraction|atrial premature complex|"
            r"supraventricular extrasystole|aberrantly conducted supraventricular|"
            r"aberrant conduction of sv|multiple premature complexes",
     ("atrial premature beats",), None),
    # 'supraventricular bigeminy' is not one of the universe's diagnoses, so it
    # is recognised only to stop it falling through as unmapped, and emits
    # nothing. Decision: neglect it.
    ("supraventricular bigeminy", r"supraventricular bigeminy|bigeminal pacs",
     (), None),

    # ---- infarction --------------------------------------------------------
    ("infarct", r"infarct", ("myocardial infarction",), r"cannot rule out"),
    ("old infarct", r"\bold\b.*infarct|infarct.*\bold\b", ("old myocardial infarction",),
     r"cannot rule out"),
    ("q waves", r"\bq waves?\b|^q in ", ("abnormal q wave",), None),

    # ---- ST / T ------------------------------------------------------------
    ("st deviation",
     r"st-t change|st-t abnormal|st changes|st change\b|st depression|st depr\b|"
     r"st elevation|st elev\b|st junctional depression|repolarization (change|abnormalit)|"
     r"repol abnrm|repol abnormality|repolarization abnormality|st-t changes|"
     r"changes possibly due to myocardial ischemia|st-t changes in the",
     ("st deviation",), None),
    # (?<!st-) keeps 'lateral st-t changes are nonspecific' out of the T-wave
    # bucket: per the brief ST-T statements are one finding, 'st deviation'.
    ("t wave change",
     r"t wave change|t wave abnormal|t abnormalities|t abnrm|abnormal t,|abnrm t,|"
     r"(?<!st-)\bt changes|tall t waves|t wave inversion",
     ("t wave change",), r"normal for age"),
    ("pericarditis", r"pericarditis", ("pericarditis",), None),

    # ---- QT ----------------------------------------------------------------
    ("qt", r"prolonged qtc?\b|long qtc?\b|prolonged qt interval", ("qt interval extension",), None),

    # ---- voltage / progression --------------------------------------------
    ("low voltage", r"low qrs voltage|low voltage", ("low qrs voltages",), None),
    ("prwp", r"poor r wave progression|abnormal r-?wave progression|low r\(v2-v4\)",
     ("poor r wave progression",), r"early transition"),
    ("ccw rotation", r"early transition", ("counterclockwise rotation",), None),

    # ---- hypertrophy / chambers -------------------------------------------
    ("lvh", r"left ventricular hypertrophy|\blvh\b|\bbvh\b|biventricular hypertrophy",
     ("left ventricular hypertrophy",), None),
    ("rvh", r"right ventricular hypertrophy|\brvh\b|\bbvh\b|biventricular hypertrophy",
     ("right ventricular hypertrophy",), None),
    ("lae", r"left atrial (abnormality|enlargement)|\blae\b|p terminal force",
     ("left atrial enlargement",), None),
    ("rae", r"right atrial (abnormality|enlargement)", ("right atrial enlargement",), None),
    ("biatrial", r"biatrial enlargement", ("biatrial enlargement",), None),

    # ---- pre-excitation ----------------------------------------------------
    ("wpw", r"\bwpw\b|wolff", ("wolff-parkinson-white",), None),

    # ---- summary statements ------------------------------------------------
    ("normal ecg", r"^(summary: )?normal ecg|^within normal limits$", ("normal ecg",), None),
    ("abnormal ecg", r"^(summary: )?abnormal( ecg)?$|^abnormal ecg", ("abnormal ecg",), None),
    ("borderline ecg", r"^(summary: )?borderline( ecg)?$", ("borderline ecg",), None),
]

COMPILED: list[tuple[str, re.Pattern, tuple[str, ...], re.Pattern | None]] = [
    (name, re.compile(pat), labels, re.compile(exc) if exc else None)
    for name, pat, labels, exc in RULES
]
TECHNICAL_RE = [re.compile(p) for p in TECHNICAL]
NO_FINDING_RE = [re.compile(p) for p in NO_FINDING]

# ---- acute-injury post-rules ------------------------------------------------
ACUTE_RE = re.compile(
    r"acute st elevation mi|consider acute infarct|infarct[ ,-]+(possibly |probably )?acute|"
    r"acute (anterior|inferior|lateral|septal|anteroseptal|anterolateral|extensive) infarct|"
    r"infarct with some acute changes|consider (inferior|anterior|septal|lateral|"
    r"anteroseptal|anterolateral) injury"
)
ACUTE_BLOCK_RE = re.compile(r"cannot rule out|repeat if|reciprocal")
EARLY_REPOL_RE = re.compile(r"early repol")


def map_statement(s: str) -> tuple[set[str], str]:
    """Return (labels, disposition) for one normalised 12SL statement."""
    labels: set[str] = set()
    for _name, pat, out, exc in COMPILED:
        if pat.search(s) and not (exc and exc.search(s)):
            labels.update(out)

    # ST elevation called as a normal variant is not a deviation finding.
    if EARLY_REPOL_RE.search(s):
        labels.discard("st deviation")
        labels.add("early repolarization")

    # Acute injury: resolve the MI family to ONE label, and only on evidence.
    #
    # The project already has a rule for this — mi_family_label_curation, which
    # strips every MI-family label and re-adds exactly one, decided from the
    # DISCHARGE LETTER (letter_stemi drove 356 of its 370 STEMIs) with the
    # machine ST-elevation flag as a secondary signal, quarantining the 439 it
    # could not resolve rather than calling them controls.
    #
    # No discharge letter is available here, so 12SL text is much weaker
    # evidence. The first pass called STEMI on ANY acute-infarct statement
    # without 'st depression', which gave 150 STEMIs whose report never
    # mentions ST elevation at all — and STEMI is rank 1 in the severity
    # ordering, so each of those would headline its record in the viewer.
    #
    # So: STEMI only when the text actually says ST elevation; explicit
    # depression makes it nstemi; and an acute infarct with neither stays
    # 'myocardial infarction' rather than being guessed into a subtype.
    if ACUTE_RE.search(s) and not ACUTE_BLOCK_RE.search(s):
        labels.add("myocardial infarction")
        has_elevation = "st elevation" in s or "st elev" in s
        has_depression = "st depression" in s
        if has_elevation:
            labels.add("STEMI")
        elif has_depression:
            labels.add("nstemi")

    # Bifascicular block is RBBB + a fascicular block, never stated as such.
    if "right bundle branch block" in labels and (
        "left anterior fascicular block" in labels
        or "left posterior fascicular block" in labels
    ):
        labels.add("bifascicular block")

    if labels:
        return labels, "labelled"
    for pat in NO_FINDING_RE:
        if pat.search(s):
            return labels, "no-finding"
    for pat in TECHNICAL_RE:
        if pat.search(s):
            return labels, "technical"
    return labels, "unmapped"


# =============================================================================
# DRIVER
# =============================================================================

MARKERS = {
    "lvef >=55% control",
    "lvef <40% expanded MIMIC EF>=55 controls",
}
SUMMARY_LABELS = {"normal ecg", "abnormal ecg", "borderline ecg", "sinus rhythm"}


def main() -> int:
    affected = json.loads(AFFECTED_IDS.read_text(encoding="utf-8"))
    reports = json.loads(REPORTS.read_text(encoding="utf-8"))
    scored = set(json.loads(METADATA.read_text(encoding="utf-8"))["classes"])

    # display_id -> universe row (labels + predictions)
    rows: dict[str, dict] = {}
    with PATIENTS.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows[r["display_id"]] = r

    # study_id -> cohort, display_id
    order: list[tuple[str, str, str]] = []
    for cohort, mapping in affected.items():
        for study_id, display_id in mapping.items():
            order.append((cohort, study_id, display_id))

    stmt_disp: Counter[str] = Counter()          # disposition -> statement volume
    unmapped: Counter[str] = Counter()
    per_cohort = defaultdict(lambda: {
        "records": 0,
        "stmt_total": 0,
        "stmt_labelled": 0,
        "stmt_technical": 0,
        "stmt_nofinding": 0,
        "stmt_unmapped": 0,
        "records_with_new": 0,
        "records_with_new_nonsummary": 0,
        "records_no_labels": 0,
        "dx": Counter(),
        "pred_total": 0,
        "tp_before": 0,
        "tp_after": 0,
        "labels_before": 0,
        "labels_after": 0,
        "recall_hit_before": 0,
        "recall_hit_after": 0,
    })

    dx_all: Counter[str] = Counter()
    per_class_gain: Counter[str] = Counter()
    tp_before = tp_after = pred_total = 0
    lab_before_tot = lab_after_tot = 0
    hit_before = hit_after = 0
    n_new = n_new_nonsummary = 0

    with OUT_JSONL.open("w", encoding="utf-8") as out:
        for cohort, study_id, display_id in order:
            row = rows.get(display_id, {})
            existing = list(row.get("labels") or [])
            preds = list(row.get("predictions") or [])
            raw = reports.get(study_id, [])

            pc = per_cohort[cohort]
            pc["records"] += 1

            rec_labels: set[str] = set()
            rec_unmapped: list[str] = []
            for line in raw:
                s = norm(line)
                if not s:
                    continue
                pc["stmt_total"] += 1
                labels, disp = map_statement(s)
                stmt_disp[disp] += 1
                if disp == "labelled":
                    pc["stmt_labelled"] += 1
                    rec_labels |= labels
                elif disp == "technical":
                    pc["stmt_technical"] += 1
                elif disp == "no-finding":
                    pc["stmt_nofinding"] += 1
                else:
                    pc["stmt_unmapped"] += 1
                    unmapped[s] += 1
                    rec_unmapped.append(s)

            recovered = sorted(rec_labels)
            for d in recovered:
                dx_all[d] += 1
                pc["dx"][d] += 1

            existing_l = {e.lower() for e in existing}
            new = [d for d in recovered if d.lower() not in existing_l]
            new_ns = [d for d in new if d.lower() not in SUMMARY_LABELS]
            if new:
                n_new += 1
                pc["records_with_new"] += 1
            if new_ns:
                n_new_nonsummary += 1
                pc["records_with_new_nonsummary"] += 1
                for d in new_ns:
                    per_class_gain[d] += 1
            if not recovered:
                pc["records_no_labels"] += 1

            # ---- classifier agreement -------------------------------------
            before = {e for e in existing if e in scored}
            after = before | {d for d in recovered if d in scored}
            pset = set(preds)
            b_tp = len(pset & before)
            a_tp = len(pset & after)
            pred_total += len(pset)
            tp_before += b_tp
            tp_after += a_tp
            lab_before_tot += len(before)
            lab_after_tot += len(after)
            hit_before += len(before & pset)
            hit_after += len(after & pset)
            pc["pred_total"] += len(pset)
            pc["tp_before"] += b_tp
            pc["tp_after"] += a_tp
            pc["labels_before"] += len(before)
            pc["labels_after"] += len(after)
            pc["recall_hit_before"] += len(before & pset)
            pc["recall_hit_after"] += len(after & pset)

            out.write(json.dumps({
                "display_id": display_id,
                "study_id": study_id,
                "cohort": cohort,
                "existing_labels": existing,
                "recovered_labels": recovered,
                "unmapped_statements": rec_unmapped,
                "raw_reports": raw,
            }, ensure_ascii=False) + "\n")

    # ---- report --------------------------------------------------------------
    total_stmt = sum(stmt_disp.values())
    lab = stmt_disp["labelled"]
    tech = stmt_disp["technical"]
    nof = stmt_disp["no-finding"]
    unm = stmt_disp["unmapped"]

    def pct(a, b):
        return 0.0 if not b else round(100.0 * a / b, 2)

    print("=" * 78)
    print("STATEMENT COVERAGE  (statement instances, not distinct strings)")
    print("=" * 78)
    print(f"  total statements       {total_stmt:>8,}")
    print(f"  labelled               {lab:>8,}   {pct(lab, total_stmt):>6}%")
    print(f"  technical (ignored)    {tech:>8,}   {pct(tech, total_stmt):>6}%")
    print(f"  no-finding (ignored)   {nof:>8,}   {pct(nof, total_stmt):>6}%")
    print(f"  UNMAPPED               {unm:>8,}   {pct(unm, total_stmt):>6}%")
    print(f"  COVERED (all handled)  {total_stmt - unm:>8,}   "
          f"{pct(total_stmt - unm, total_stmt):>6}%")

    print()
    print("=" * 78)
    print("PER COHORT")
    print("=" * 78)
    for cohort, pc in per_cohort.items():
        cov = pc["stmt_total"] - pc["stmt_unmapped"]
        print(f"\n{cohort}   ({pc['records']:,} records)")
        print(f"  statements            {pc['stmt_total']:,}"
              f"  covered {pct(cov, pc['stmt_total'])}%"
              f"  labelled {pct(pc['stmt_labelled'], pc['stmt_total'])}%")
        print(f"  gained >=1 new label  {pc['records_with_new']:,}"
              f"  ({pct(pc['records_with_new'], pc['records'])}%)")
        print(f"  gained >=1 new NON-SUMMARY label {pc['records_with_new_nonsummary']:,}"
              f"  ({pct(pc['records_with_new_nonsummary'], pc['records'])}%)")
        print(f"  still no label at all {pc['records_no_labels']:,}")
        print("  top recovered diagnoses:")
        for d, c in pc["dx"].most_common(15):
            print(f"      {c:>7,}  {pct(c, pc['records']):>5}%  {d}")

    print()
    print("=" * 78)
    print("RECOVERED DIAGNOSIS DISTRIBUTION (all 3 cohorts)")
    print("=" * 78)
    for d, c in dx_all.most_common():
        print(f"  {c:>7,}  {pct(c, len(order)):>5}%  {d}")

    print()
    print("=" * 78)
    print(f"TOP 20 UNMAPPED STATEMENTS  ({len(unmapped):,} distinct, {unm:,} instances)")
    print("=" * 78)
    for s, c in unmapped.most_common(20):
        print(f"  {c:>6,}  {s}")

    print()
    print("=" * 78)
    print("CLASSIFIER AGREEMENT  (32 scored classes only)")
    print("=" * 78)
    print(f"  records                       {len(order):,}")
    print(f"  predictions emitted           {pred_total:,}")
    print(f"  TRUE positives  before        {tp_before:,}   "
          f"precision {pct(tp_before, pred_total)}%")
    print(f"  TRUE positives  after         {tp_after:,}   "
          f"precision {pct(tp_after, pred_total)}%")
    print(f"  FALSE positives before        {pred_total - tp_before:,}")
    print(f"  FALSE positives after         {pred_total - tp_after:,}")
    print(f"  scored labels  before         {lab_before_tot:,}   "
          f"recall {pct(hit_before, lab_before_tot)}%")
    print(f"  scored labels  after          {lab_after_tot:,}   "
          f"recall {pct(hit_after, lab_after_tot)}%")
    print("\n  per cohort:")
    for cohort, pc in per_cohort.items():
        print(f"    {cohort:<24} preds {pc['pred_total']:>7,}"
              f"  TP {pc['tp_before']:>7,} -> {pc['tp_after']:>7,}"
              f"   precision {pct(pc['tp_before'], pc['pred_total']):>6}%"
              f" -> {pct(pc['tp_after'], pc['pred_total']):>6}%")

    report = {
        "statements": {
            "total": total_stmt, "labelled": lab, "technical": tech,
            "no_finding": nof, "unmapped": unm,
            "covered_pct": pct(total_stmt - unm, total_stmt),
            "labelled_pct": pct(lab, total_stmt),
        },
        "records": len(order),
        "records_with_new_label": n_new,
        "records_with_new_nonsummary_label": n_new_nonsummary,
        "diagnosis_distribution": dx_all.most_common(),
        "new_label_distribution": per_class_gain.most_common(),
        "top_unmapped": unmapped.most_common(40),
        "classifier": {
            "predictions": pred_total,
            "tp_before": tp_before, "tp_after": tp_after,
            "precision_before": pct(tp_before, pred_total),
            "precision_after": pct(tp_after, pred_total),
            "scored_labels_before": lab_before_tot,
            "scored_labels_after": lab_after_tot,
            "recall_before": pct(hit_before, lab_before_tot),
            "recall_after": pct(hit_after, lab_after_tot),
        },
        "per_cohort": {
            k: {kk: (dict(vv.most_common()) if isinstance(vv, Counter) else vv)
                for kk, vv in v.items()}
            for k, v in per_cohort.items()
        },
    }
    OUT_REPORT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT_JSONL}")
    print(f"wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
