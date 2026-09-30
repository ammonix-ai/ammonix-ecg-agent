"""Domain skills: the rules that say *what to look for* in this case.

Vendored from the private `ammonix.skills` / `ammonix.tribes` pair, reduced to
what the Ammonix ECG Agent needs. Two things came across:

* the **selection rule** from ``ammonix.skills.collect_applicable_skills`` — a
  skill is applicable when the classifier's probability for its diagnosis
  exceeds ``inclusion_ratio x threshold`` (default 0.2), so the agent is told
  about everything in the differential, not only the calls above threshold, and
  the list is ranked by probability/threshold ratio;
* the **domain groups** from ``ammonix.tribes`` — the mutually exclusive
  families (rhythm, ST, axis, bundle branch), the AF/flutter pair, the sinus
  family, and the MI/STEMI equivalence with its expected ST/T companions.

What deliberately did *not* come across:

* the prediction-editing half of the private skills (``skill_sinus_gate_predict``,
  ``skill_mutex_predict``, ``skill_af_flutter_svm_predict``) and the scoring
  adjustments. Those rewrite the classifier's output. In this build the
  classifier is frozen and its calls are reported as they are, so the same rules
  run in *observation* mode: they flag the conflict and name the discriminator,
  and nothing here changes a probability or a prediction;
* the episodic skill store. Those are learned, patient-indexed rows in a private
  database. The per-diagnosis text below is written for this repository.

The output feeds the agent's case brief. It is verification guidance — the
diagnosis has already been made by the classifier and the universe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

#: A skill is offered to the agent when p > INCLUSION_RATIO * threshold.
#: Ported from ammonix.skills.collect_applicable_skills (default 0.2).
INCLUSION_RATIO = 0.2

#: Sinus variants. Plain "sinus rhythm" is not one of the 32 frozen classes —
#: the universe scores deviations from it, so its absence is the baseline.
SINUS_FAMILY: frozenset[str] = frozenset({
    "sinus rhythm", "sinus bradycardia", "sinus tachycardia", "sinus irregularity",
})

NON_SINUS_RHYTHMS: frozenset[str] = frozenset({
    "atrial fibrillation", "atrial flutter", "pacing rhythm", "junctional rhythm",
    "idioventricular rhythm", "ventricular tachycardia", "ventricular fibrillation",
    "supraventricular tachycardia", "asystole", "2nd av-block", "3rd av-block",
})

RHYTHM_DIAGNOSES: frozenset[str] = SINUS_FAMILY | NON_SINUS_RHYTHMS

AF_FLUTTER_PAIR: frozenset[str] = frozenset({"atrial fibrillation", "atrial flutter"})

#: Mutually exclusive families. Intersected with the classifier's actual class
#: list at match time, so members this package does not score are ignored.
MUTEX_GROUPS: dict[str, frozenset[str]] = {
    "rhythm": RHYTHM_DIAGNOSES,
    "st_segment": frozenset({"st depression", "st elevation", "st deviation"}),
    "axis": frozenset({"axis left shift", "axis right shift"}),
    "block": frozenset({
        "left bundle branch block", "right bundle branch block",
        "left bundle branch block / variations",
        "incomplete left bundle branch block",
        "incomplete right bundle branch block",
    }),
}

MUTEX_DISCRIMINATORS: dict[str, str] = {
    "rhythm": (
        "Only one underlying rhythm can be correct. Settle it on P-wave morphology "
        "and regularity in II, III, aVF and V1: discrete upright P before every QRS, "
        "absent/irregular baseline, sawtooth flutter waves, or pacing spikes."
    ),
    "st_segment": (
        "ST deviation direction is one call. Read the J point against the TP "
        "baseline lead by lead and say which leads are elevated and which depressed."
    ),
    "axis": (
        "The frontal axis cannot be both left and right. Read net QRS polarity in "
        "I and aVF, then refine with II."
    ),
    "block": (
        "Only one bundle-branch pattern fits. Read QRS width, then V1 (rSR' vs QS/rS) "
        "and I/V6 (slurred S vs broad notched R)."
    ),
}

#: Diagnoses the MI family treats as interchangeable at scoring time, plus the
#: ST/T findings that are expected companions rather than separate errors.
MI_FAMILY: frozenset[str] = frozenset({
    "STEMI", "nstemi", "myocardial infarction", "old myocardial infarction",
})
MI_EXPECTED_COMPANIONS: frozenset[str] = frozenset({
    "st deviation", "st elevation", "st depression", "t wave change",
    "t wave inversion", "poor r wave progression",
})

#: Rate classes that are consequences of the rhythm rather than rhythms.
RATE_CLASSES: frozenset[str] = frozenset({"tachycardia", "bradycardia"})

#: Per-diagnosis verification guidance, keyed by lower-cased class name.
#: Written for this repository: each entry says what would confirm the
#: classifier's call on the trace, and what would argue against it.
DIAGNOSIS_CHECKS: dict[str, str] = {
    "sinus tachycardia": (
        "Rate above 100/min with a normal upright P before every QRS (positive in "
        "II, negative in aVR) and a constant PR. Against: absent or non-sinus P "
        "waves, or an abrupt rather than gradual rate."
    ),
    "sinus bradycardia": (
        "Rate below 60/min, one upright P per QRS in II, constant PR. Against: "
        "dropped or non-conducted P waves (that is block, not bradycardia), or a "
        "junctional escape with no P."
    ),
    "sinus irregularity": (
        "Beat-to-beat R-R variation with unchanged sinus P morphology and PR. "
        "Against: irregularly irregular R-R with no discernible P (favours AF), or "
        "early P waves of different shape (atrial ectopy)."
    ),
    "atrial fibrillation": (
        "Irregularly irregular R-R with no organised P waves and a fibrillatory "
        "baseline, best seen in V1 and II. Against: any consistent P-QRS relation, "
        "or regular sawtooth undulation at ~300/min."
    ),
    "atrial flutter": (
        "Sawtooth F waves at roughly 250-350/min, classically negative in II, III, "
        "aVF and positive in V1, with regular conduction (often 2:1, ventricular "
        "rate near 150). Against: a chaotic irregular baseline (favours AF)."
    ),
    "supraventricular tachycardia": (
        "Regular narrow-complex tachycardia usually above 150/min with absent or "
        "retrograde P waves. Against: visible sinus P waves at the same rate, or a "
        "wide QRS without prior bundle-branch pattern."
    ),
    "pacing rhythm": (
        "Sharp narrow pacing spikes immediately preceding P (atrial), QRS "
        "(ventricular) or both, with a broad paced QRS, typically LBBB-like from an "
        "RV lead. Against: no spikes anywhere across the twelve leads."
    ),
    "ventricular premature beats": (
        "Early wide QRS with no preceding P, discordant T, usually followed by a "
        "compensatory pause. Against: early *narrow* beats with a preceding "
        "abnormal P (those are atrial)."
    ),
    "atrial premature beats": (
        "Early narrow QRS preceded by a P of different morphology, with a "
        "non-compensatory pause. Against: early wide complexes without a P."
    ),
    "tachycardia": (
        "A rate statement only: ventricular rate above 100/min. Say which rhythm "
        "is driving it rather than treating this as a diagnosis on its own."
    ),
    "bradycardia": (
        "A rate statement only: ventricular rate below 60/min. Name the mechanism "
        "(sinus, junctional escape, or block) from the P-QRS relation."
    ),
    "1st av-block": (
        "PR interval longer than 200 ms, constant, with every P conducted. Against: "
        "progressive PR lengthening with a dropped beat, or dropped P waves."
    ),
    "left bundle branch block / variations": (
        "QRS at or above 120 ms with a broad notched or monophasic R in I, aVL, V5, "
        "V6 and a QS or rS in V1, with discordant ST-T. Against: an rSR' in V1 "
        "(that is right-sided)."
    ),
    "right bundle branch block": (
        "QRS at or above 120 ms with rSR' in V1-V2 and a wide slurred S in I and "
        "V6. Against: a broad monophasic R in V6."
    ),
    "left anterior fascicular block": (
        "Left axis beyond -45 degrees, qR in I and aVL, rS in II, III, aVF, with "
        "QRS still under 120 ms. Against: a wide QRS (then read the bundle first) "
        "or an inferior infarct pattern explaining the axis."
    ),
    "bifascicular block": (
        "RBBB pattern plus a fascicular block axis (usually RBBB with left anterior "
        "fascicular block). Against: only one of the two components present."
    ),
    "wolff-parkinson-white": (
        "Short PR under 120 ms with a delta wave slurring the QRS upstroke and a "
        "widened QRS. Against: a normal crisp QRS onset."
    ),
    "left ventricular hypertrophy": (
        "Voltage criteria (for example S in V1 plus R in V5/V6 above 35 mm, or R in "
        "aVL above 11 mm) with a strain ST-T pattern in the lateral leads. Against: "
        "tall voltages in a thin chest with normal repolarisation."
    ),
    "left atrial enlargement": (
        "P wave in II at or above 120 ms and often notched, with a deep negative "
        "terminal component in V1. Against: a short crisp P."
    ),
    "low qrs voltages": (
        "QRS amplitude under 5 mm in every limb lead or under 10 mm in every chest "
        "lead. Against: any single lead clearly exceeding those limits, and check "
        "the calibration pulse before accepting the call."
    ),
    "poor r wave progression": (
        "R wave failing to grow across V1-V4 (R in V3 under 3 mm), without a full "
        "infarct pattern. Against: normal R growth, or evidence of lead "
        "misplacement."
    ),
    "st deviation": (
        "J point displaced from the TP baseline. Name the leads and the direction, "
        "and whether the shape is concave, horizontal or downsloping."
    ),
    "t wave change": (
        "T wave flattening, inversion or asymmetry, with the leads named and the "
        "regional pattern stated. Against: normal juvenile or lead-specific "
        "inversion (III, aVR, V1)."
    ),
    "qt interval extension": (
        "Prolonged QT once rate-corrected; measure in II or V5 and use the longest "
        "clearly delineated T offset. Against: a T-U fusion mistaken for a long T."
    ),
    "STEMI": (
        "ST elevation in two or more contiguous leads with a regional pattern and "
        "reciprocal depression. Against: diffuse concave elevation with PR "
        "depression (pericarditis), or the discordant elevation expected in LBBB or "
        "a paced rhythm."
    ),
    "nstemi": (
        "Ischaemic ST depression or T inversion without diagnostic elevation. This "
        "label needs troponin to be real; on the trace, confirm only the ischaemic "
        "repolarisation pattern and say the diagnosis is not ECG-only."
    ),
    "myocardial infarction": (
        "Regional pattern of pathological Q waves, ST deviation or T inversion. "
        "Name the territory (anterior V1-V4, inferior II/III/aVF, lateral I/aVL/V5-V6)."
    ),
    "old myocardial infarction": (
        "Pathological Q waves or loss of R with a settled ST segment and no acute "
        "elevation. Against: active ST elevation, which makes it not old."
    ),
    "brugada syndrome": (
        "Coved (type 1) ST elevation at or above 2 mm in V1-V2 with a descending ST "
        "and negative T. Against: a saddleback shape alone, or a right "
        "precordial pattern explained by RBBB or high lead placement."
    ),
    "axis left shift": (
        "Frontal axis left of -30 degrees: positive QRS in I, negative in II and "
        "aVF. Against: a positive net QRS in II."
    ),
    "axis right shift": (
        "Frontal axis right of +90 degrees: negative QRS in I, positive in aVF. "
        "Against: a positive net QRS in I."
    ),
    "lvef <40% expanded mimic ef>=55 controls": (
        "Ejection fraction is NOT readable from a surface ECG. This is a "
        "statistical inference drawn from depolarisation and "
        "repolarisation morphology, trained against echo-confirmed EF<40% cases and "
        "EF>=55% controls. Do not claim to see reduced EF. You may only report "
        "supporting or opposing surface markers (wide QRS or LBBB, low voltages, "
        "poor R progression, LVH, diffuse T change) and state clearly that "
        "confirmation needs an echo."
    ),
}

GENERIC_CHECK = (
    "Name the waveform evidence you can see for this call — lead, wave, interval, "
    "amplitude — and say if the trace does not support it."
)


@dataclass(frozen=True)
class MatchedSkill:
    """One piece of verification guidance that applies to this case."""

    skill_id: str
    kind: str                    # "diagnosis" | "rule"
    title: str
    trigger: str                 # why it fired, with the numbers
    check: str                   # what to verify on the trace
    diagnoses: tuple[str, ...]
    priority: float              # higher first

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.skill_id,
            "kind": self.kind,
            "title": self.title,
            "trigger": self.trigger,
            "check": self.check,
            "diagnoses": list(self.diagnoses),
            "priority": round(self.priority, 4),
        }


def _ratio(prob: float, threshold: float | None) -> float:
    if threshold is None or threshold <= 0:
        return prob / 0.5
    return prob / threshold


def _fmt(dx: str, prob: float, threshold: float | None) -> str:
    if threshold is None or threshold <= 0:
        return f"{dx} p={prob:.3f}"
    return f"{dx} p={prob:.3f} vs threshold {threshold:.3f} ({_ratio(prob, threshold):.2f}x)"


def diagnosis_check(diagnosis: str) -> str:
    """Verification text for one diagnosis (generic fallback if unknown)."""
    return DIAGNOSIS_CHECKS.get(diagnosis, DIAGNOSIS_CHECKS.get(diagnosis.lower(), GENERIC_CHECK))


def above_threshold(
    probabilities: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> list[str]:
    """Classes at or above their own threshold, highest ratio first."""
    calls = [
        (dx, _ratio(p, thresholds.get(dx)))
        for dx, p in probabilities.items()
        if p >= (thresholds.get(dx) or 0.5)
    ]
    calls.sort(key=lambda item: -item[1])
    return [dx for dx, _ in calls]


def applicable_diagnosis_skills(
    probabilities: Mapping[str, float],
    thresholds: Mapping[str, float],
    *,
    inclusion_ratio: float = INCLUSION_RATIO,
    limit: int = 8,
) -> list[MatchedSkill]:
    """Skills for every diagnosis in the differential.

    Selection rule ported from ``ammonix.skills.collect_applicable_skills``:
    include when ``p > inclusion_ratio * threshold``, rank by ``p / threshold``.
    """
    hits: list[MatchedSkill] = []
    for dx, prob in probabilities.items():
        threshold = thresholds.get(dx)
        cut = (threshold if threshold and threshold > 0 else 0.5) * inclusion_ratio
        if prob <= cut:
            continue
        ratio = _ratio(prob, threshold)
        hits.append(MatchedSkill(
            skill_id=f"dx::{dx}",
            kind="diagnosis",
            title=dx,
            trigger=(
                f"{_fmt(dx, prob, threshold)} — "
                + ("above threshold" if ratio >= 1.0 else "in the differential, below threshold")
            ),
            check=diagnosis_check(dx),
            diagnoses=(dx,),
            priority=ratio,
        ))
    hits.sort(key=lambda s: -s.priority)
    return hits[:limit]


def rule_skills(
    probabilities: Mapping[str, float],
    thresholds: Mapping[str, float],
    *,
    rate_bpm: float | None = None,
) -> list[MatchedSkill]:
    """Group-level rules: mutex conflicts, the AF/flutter pair, MI family,
    rhythm coverage, and rate-versus-rhythm consistency.

    Observation only — none of these edits a prediction.
    """
    classes = set(probabilities)
    calls = set(above_threshold(probabilities, thresholds))
    out: list[MatchedSkill] = []

    # --- mutex groups -----------------------------------------------------
    for group, members in MUTEX_GROUPS.items():
        present = sorted(
            (dx for dx in calls if dx in members),
            key=lambda dx: -_ratio(probabilities[dx], thresholds.get(dx)),
        )
        if len(present) < 2:
            continue
        if group == "rhythm" and set(present) <= AF_FLUTTER_PAIR:
            continue  # handled by the dedicated AF/flutter rule below
        detail = "; ".join(_fmt(dx, probabilities[dx], thresholds.get(dx)) for dx in present)
        out.append(MatchedSkill(
            skill_id=f"mutex::{group}",
            kind="rule",
            title=f"Mutually exclusive {group.replace('_', ' ')} calls",
            trigger=(
                f"{len(present)} members of the {group.replace('_', ' ')} group are above "
                f"threshold at once: {detail}. Each diagnosis is scored "
                "independently, so more than one of them can be raised."
            ),
            check=MUTEX_DISCRIMINATORS.get(group, GENERIC_CHECK),
            diagnoses=tuple(present),
            priority=100.0 + max(_ratio(probabilities[dx], thresholds.get(dx)) for dx in present),
        ))

    # --- AF vs flutter ----------------------------------------------------
    if AF_FLUTTER_PAIR <= calls:
        af, fl = "atrial fibrillation", "atrial flutter"
        out.append(MatchedSkill(
            skill_id="rule::af-flutter",
            kind="rule",
            title="Atrial fibrillation and atrial flutter both called",
            trigger=(
                f"{_fmt(af, probabilities[af], thresholds.get(af))} and "
                f"{_fmt(fl, probabilities[fl], thresholds.get(fl))}. A dedicated arbiter "
                "normally separates this pair; it is not available here, so the two calls "
                "stand side by side and the trace decides."
            ),
            check=(
                "Look at the atrial baseline in II, III, aVF and V1. Organised sawtooth F "
                "waves at a constant ~250-350/min with regular conduction favour flutter; a "
                "chaotic undulating baseline with irregularly irregular R-R favours "
                "fibrillation. Say which one the trace shows, or that it is coarse AF and "
                "genuinely ambiguous."
            ),
            diagnoses=(af, fl),
            priority=120.0,
        ))

    # --- MI family --------------------------------------------------------
    mi_calls = sorted(calls & MI_FAMILY)
    if mi_calls:
        companions = sorted(calls & MI_EXPECTED_COMPANIONS)
        out.append(MatchedSkill(
            skill_id="rule::mi-family",
            kind="rule",
            title="Infarct-family calls",
            trigger=(
                "Infarct family above threshold: "
                + "; ".join(_fmt(dx, probabilities[dx], thresholds.get(dx)) for dx in mi_calls)
                + (
                    f". Expected companions also called: {', '.join(companions)}."
                    if companions else
                    ". No ST/T companion call accompanies it."
                )
            ),
            check=(
                "STEMI, NSTEMI, myocardial infarction and old myocardial infarction overlap by "
                "construction — the reference labels treat them as one family, so several can "
                "fire on one trace. Resolve them on timing and territory: acute elevation with "
                "reciprocal change (STEMI), ischaemic depression or T inversion without "
                "elevation (NSTEMI), established Q waves with a settled ST segment (old). "
                "Accompanying ST-deviation and T-wave calls are expected parts of the same "
                "finding, not separate diagnoses."
            ),
            diagnoses=tuple(mi_calls),
            priority=110.0,
        ))

    # --- rhythm coverage --------------------------------------------------
    rhythm_calls = sorted(calls & RHYTHM_DIAGNOSES)
    if not rhythm_calls:
        out.append(MatchedSkill(
            skill_id="rule::rhythm-coverage",
            kind="rule",
            title="No rhythm call",
            trigger=(
                "No rhythm class is above threshold. Plain sinus rhythm is not one of the 32 "
                "scored classes — only departures from it are scored — so no rhythm "
                "abnormality was identified here, not that the rhythm went unexamined."
            ),
            check=(
                "Confirm the baseline rhythm from the trace: one upright P per QRS in "
                "II with a constant PR, regular R-R, rate 60-100. If the trace is not sinus, "
                "say so plainly — that is a discrepancy worth reporting."
            ),
            diagnoses=(),
            priority=90.0,
        ))

    # --- sinus variant vs measured rate ----------------------------------
    sinus_calls = sorted(calls & SINUS_FAMILY)
    rate_calls = sorted(calls & RATE_CLASSES)
    if rate_bpm is not None and (sinus_calls or rate_calls):
        expected = (
            "sinus bradycardia" if rate_bpm < 60
            else "sinus tachycardia" if rate_bpm > 100
            else "a normal rate"
        )
        agree = (
            (rate_bpm < 60 and ("sinus bradycardia" in sinus_calls or "bradycardia" in rate_calls))
            or (rate_bpm > 100 and ("sinus tachycardia" in sinus_calls or "tachycardia" in rate_calls))
            or (60 <= rate_bpm <= 100 and not (set(sinus_calls) & {"sinus bradycardia", "sinus tachycardia"}) and not rate_calls)
        )
        out.append(MatchedSkill(
            skill_id="rule::rate-consistency",
            kind="rule",
            title="Rate versus rate-dependent call",
            trigger=(
                f"Estimated ventricular rate {rate_bpm:.0f}/min (QRS complexes detected in lead "
                f"II over the recording window) implies {expected}. Calls in play: "
                f"{', '.join(sinus_calls + rate_calls) or 'none'}. "
                + ("Consistent." if agree else "These do not line up.")
            ),
            check=(
                "The 60 and 100/min cut-offs assume a measured R-R interval, which is not "
                "available here; the rate above is "
                "a beat-count estimate over the visible window, so treat it as approximate. "
                "Count the R-R spacing on the image and state the rate you actually measure."
            ),
            diagnoses=tuple(sinus_calls + rate_calls),
            priority=95.0 if not agree else 60.0,
        ))

    unknown = [dx for dx in calls if dx not in classes]
    if unknown:  # defensive: probabilities and calls come from the same dict
        out.append(MatchedSkill(
            skill_id="rule::unknown-class",
            kind="rule",
            title="Unrecognised class",
            trigger=f"Calls not in the scored class list: {', '.join(sorted(unknown))}.",
            check=GENERIC_CHECK,
            diagnoses=tuple(sorted(unknown)),
            priority=10.0,
        ))

    out.sort(key=lambda s: -s.priority)
    return out


def match_skills(
    probabilities: Mapping[str, float],
    thresholds: Mapping[str, float],
    *,
    rate_bpm: float | None = None,
    inclusion_ratio: float = INCLUSION_RATIO,
    limit_diagnosis: int = 8,
) -> list[MatchedSkill]:
    """Every skill that applies to this case, rules first then diagnoses."""
    return [
        *rule_skills(probabilities, thresholds, rate_bpm=rate_bpm),
        *applicable_diagnosis_skills(
            probabilities, thresholds,
            inclusion_ratio=inclusion_ratio, limit=limit_diagnosis,
        ),
    ]


def render_skills(skills: Iterable[MatchedSkill | Mapping[str, Any]]) -> str:
    """Skills as prompt text. Accepts the dataclass or its ``to_dict()`` form."""
    lines: list[str] = []
    for skill in skills:
        if isinstance(skill, Mapping):
            skill_id = skill.get("id", "skill")
            title, trigger, check = skill.get("title", ""), skill.get("trigger", ""), skill.get("check", "")
        else:
            skill_id, title, trigger, check = skill.skill_id, skill.title, skill.trigger, skill.check
        lines.append(f"  [{skill_id}] {title}")
        lines.append(f"      why: {trigger}")
        lines.append(f"      check: {check}")
    return "\n".join(lines) if lines else "  (none matched)"
