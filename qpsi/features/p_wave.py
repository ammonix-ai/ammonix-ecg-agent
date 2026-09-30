"""
P-wave feature extractors for ECG analysis (threshold-calibrated).

Source: Cell 9A2 (Cell 10) of q_psi_ai_for_ecg notebook — conservative revision
with thresholds tightened to suppress false positives in normal sinus rhythm
while preserving diagnostic sensitivity.

20 extractors covering: PR depression, atrial bigeminy, second-degree AV block,
atrial fibrillation, atrial F-waves, left/right atrial enlargement, inter-atrial
block, first-degree AV block, short PR syndrome, P-wave notching, low voltage,
wandering pacemaker, alternans, multifocal atrial tachycardia, premature atrial
contractions, ectopic atrial rhythm, axis deviation, retrograde P-waves, and
sinus arrest.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from qpsi.features.context import FeatureContext
from qpsi.features.helpers import format_lead_list


# ============================================================================
# P-WAVE HELPER FUNCTIONS
# ============================================================================

def _p_duration_ms(p_gauss: list[dict]) -> float:
    if not p_gauss:
        return 0.0
    p_start = min(g["center_ms"] - 2 * g["sigma_ms"] for g in p_gauss)
    p_end = max(g["center_ms"] + 2 * g["sigma_ms"] for g in p_gauss)
    return max(0.0, p_end - p_start)


def _p_dominant_amp(p_gauss: list[dict]) -> float:
    return max((abs(g["amp_mv"]) for g in p_gauss), default=0.0)


# H30 / Q-A6-12: previously this file defined a local `get_qrs_gaussians` with
# a Cell-10 narrower filter (`wave_type == "QRS"` only) — dead code never
# called by any P-wave extractor. Canonical Cell-12 broader filter now lives
# at qpsi.features.helpers.get_qrs_gaussians.


def get_p_wave_gaussians(lead_fits: Dict, lead: str, beat_idx: int = None) -> List[Dict]:
    if lead not in lead_fits:
        return []
    if beat_idx is None:
        gaussians = lead_fits[lead].get("avg", [])
    else:
        by_beat = lead_fits[lead].get("by_beat", [])
        if beat_idx < len(by_beat):
            gaussians = by_beat[beat_idx]
        else:
            return []
    return [g for g in gaussians if g.get("wave_type") == "P"]


def calculate_p_wave_axis(lead_fits: Dict) -> float:
    p_I = get_p_wave_gaussians(lead_fits, "I")
    p_aVF = get_p_wave_gaussians(lead_fits, "aVF")
    if not p_I or not p_aVF:
        return None
    amp_I = sum(g["amp_mv"] for g in p_I)
    amp_aVF = sum(g["amp_mv"] for g in p_aVF)
    if amp_I == 0 and amp_aVF == 0:
        return None
    return np.degrees(np.arctan2(amp_aVF, amp_I))


def calculate_p_wave_axis_for_beat(lead_fits: Dict, beat_idx: int) -> float:
    p_I = get_p_wave_gaussians(lead_fits, "I", beat_idx)
    p_aVF = get_p_wave_gaussians(lead_fits, "aVF", beat_idx)
    if not p_I or not p_aVF:
        return None
    amp_I = sum(g["amp_mv"] for g in p_I)
    amp_aVF = sum(g["amp_mv"] for g in p_aVF)
    return np.degrees(np.arctan2(amp_aVF, amp_I)) if (amp_I != 0 or amp_aVF != 0) else None


def calculate_pr_interval(lead_fits: Dict, lead: str = "II") -> float:
    if lead not in lead_fits:
        return None
    all_gaussians = lead_fits[lead].get("avg", [])
    p_gaussians = [g for g in all_gaussians if g.get("wave_type") == "P"]
    qrs_gaussians = [g for g in all_gaussians if g.get("wave_type") in ["Q", "R", "S", "QRS"]]
    if not p_gaussians or not qrs_gaussians:
        return None
    p_start = min(g["center_ms"] - 2 * g["sigma_ms"] for g in p_gaussians)
    qrs_start = min(g["center_ms"] - 2 * g["sigma_ms"] for g in qrs_gaussians)
    return qrs_start - p_start


# ============================================================================
# P-WAVE FEATURE FUNCTIONS
# ============================================================================


# ============================================================================
# 18. PR DEPRESSION
# ============================================================================
def p_wave_pr_depression(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Detect PR segment depression (pericarditis marker).
    ENHANCED sensitivity --- lowered threshold.
    """
    # This requires ST segment analysis at the PR segment level
    # Simplified implementation checking for negative deflection after P-wave

    pr_depressions = []

    for lead in ["II", "V5", "V6", "I", "aVL"]:
        if lead not in ctx.lead_fits:
            continue

        all_gaussians = ctx.lead_fits[lead].get("avg", [])
        p_gaussians = [g for g in all_gaussians if g.get("wave_type") == "P"]
        qrs_gaussians = [g for g in all_gaussians if g.get("wave_type") in ["Q", "R", "S", "QRS"]]

        if not p_gaussians or not qrs_gaussians:
            continue

        p_end = max(g["center_ms"] + 2 * g["sigma_ms"] for g in p_gaussians)
        qrs_start = min(g["center_ms"] - 2 * g["sigma_ms"] for g in qrs_gaussians)

        # Check for depression in PR segment (simplified --- normally would need baseline)
        # Looking for negative components between P and QRS
        pr_segment_gaussians = [g for g in all_gaussians
                                if p_end <= g["center_ms"] <= qrs_start]

        if pr_segment_gaussians:
            min_amp = min(g["amp_mv"] for g in pr_segment_gaussians)
            # LOWERED THRESHOLD: -0.04mV (was -0.08mV)
            if min_amp < -0.04:
                pr_depressions.append({"lead": lead, "depression_mv": abs(min_amp)})

    if len(pr_depressions) >= 2:  # LOWERED from 3 leads
        max_depression = max(p["depression_mv"] for p in pr_depressions)
        affected_leads = [p["lead"] for p in pr_depressions]

        return {
            "p_wave_pr_depression": {
                "present": True,
                "description": f"PR segment depression in {format_lead_list(affected_leads)} - "
                               f"suggests acute pericarditis",
                # QUANTITATIVE PARAMETERS:
                "affected_leads": affected_leads,
                "max_depression_mv": round(max_depression, 3),
                "lead_depressions": {p["lead"]: round(p["depression_mv"], 3) for p in pr_depressions}
            }
        }

    return {}


# ============================================================================
# 19. ATRIAL BIGEMINY
# ============================================================================
def p_wave_atrial_bigeminy(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Detect atrial bigeminy: alternating normal and premature P-waves.
    ENHANCED sensitivity.
    """
    if not ctx.rr_intervals or len(ctx.rr_intervals) < 6:
        return {}

    by_beat = ctx.lead_fits.get("II", {}).get("by_beat", [])

    if len(by_beat) < 6:
        return {}

    # Check for alternating long-short RR pattern
    alternating_count = 0
    mean_rr = np.mean(ctx.rr_intervals)

    for i in range(0, len(ctx.rr_intervals) - 1, 2):
        if i + 1 < len(ctx.rr_intervals):
            # LOWERED THRESHOLD: long > 0.9*mean, short < 0.85*mean (was 1.0 and 0.8)
            long_rr = ctx.rr_intervals[i]
            short_rr = ctx.rr_intervals[i + 1]

            if long_rr > 0.9 * mean_rr and short_rr < 0.85 * mean_rr:
                alternating_count += 1

    bigeminy_fraction = alternating_count / (len(ctx.rr_intervals) / 2)

    # LOWERED THRESHOLD: 50% (was 70%)
    if bigeminy_fraction >= 0.50:
        return {
            "p_wave_atrial_bigeminy": {
                "present": True,
                "description": f"Atrial bigeminy pattern - alternating normal and premature beats "
                               f"({bigeminy_fraction*100:.0f}% of beats)",
                # QUANTITATIVE PARAMETERS:
                "bigeminy_fraction": round(bigeminy_fraction, 2),
                "alternating_pairs": alternating_count,
                "total_intervals": len(ctx.rr_intervals),
                "mean_rr_ms": round(mean_rr, 1)
            }
        }

    return {}


# ============================================================================
# 20. SECOND DEGREE atrioventricular conduction delay (Dropped P-waves)
# ============================================================================
def p_wave_second_degree_av_block(ctx: FeatureContext) -> Dict[str, Any]:
    """P-waves without subsequent QRS."""
    dropped_beats = 0

    # Check each lead for P-waves without QRS
    for lead in ["II", "V1"]:
        if lead not in ctx.lead_fits:
            continue

        by_beat = ctx.lead_fits[lead].get("by_beat", [])

        for beat_idx in range(len(by_beat)):
            p_gaussians = [g for g in by_beat[beat_idx] if g.get("wave_type") == "P"]
            qrs_gaussians = [g for g in by_beat[beat_idx] if g.get("wave_type") == "QRS"]

            # P without QRS
            if p_gaussians and not qrs_gaussians:
                dropped_beats += 1

    if dropped_beats > 0:
        return {
            "second_degree_av_block": {
                "present": True,
                "description": f"Second-degree atrioventricular conduction delay - {dropped_beats} dropped beat(s) detected "
            }
        }
    return {}


# ============================================================================
# 1. Irregular atrial activity (Absence of P-waves)
# ============================================================================
def p_wave_atrial_fibrillation(ctx: FeatureContext) -> Dict[str, Any]:
    """Most important arrhythmia --- complete absence of P-waves."""
    p_wave_detected = False

    # Check multiple leads for any P-wave activity
    for lead in ["II", "V1", "aVF"]:
        if lead in ctx.lead_fits:
            p_gaussians = get_p_wave_gaussians(ctx.lead_fits, lead)
            if p_gaussians and any(abs(g["amp_mv"]) > 0.05 for g in p_gaussians):
                p_wave_detected = True
                break

    if not p_wave_detected and ctx.rr_intervals:
        rr_cv = np.std(ctx.rr_intervals) / np.mean(ctx.rr_intervals)
        if rr_cv > 0.15:  # Irregularly irregular
            return {
                "atrial_fibrillation": {
                    "present": True,
                    "description": "irregular atrial activity with absent P-waves and irregularly "
                                   "irregular ventricular response - HIGH STROKE RISK, "
                                   "consider anticoagulation"
                }
            }
    return {}


# ============================================================================
# 2. Regular atrial activity with sawtooth pattern (Saw-tooth pattern)
# ============================================================================
def p_wave_atrial_f_wave(ctx: FeatureContext) -> Dict[str, Any]:
    """Regular atrial activity at ~300 bpm."""
    f_wave_evidence = 0

    # Look for multiple regular atrial deflections per cycle
    for lead in ["II", "III", "aVF", "V1"]:
        if lead not in ctx.lead_fits:
            continue

        # Check for 2-4 regular P-like deflections per beat
        avg_deflections = ctx.lead_fits[lead].get("avg", [])
        p_like_count = len([g for g in avg_deflections
                           if g.get("wave_type") in ["P"] and abs(g["amp_mv"]) > 0.05])

        if p_like_count >= 2:  # Multiple atrial deflections
            f_wave_evidence += 1

    if f_wave_evidence >= 2:
        return {"atrial_f_wave": {
                "present": True,
                "description": "Atrial f_wave with saw-tooth F_waves, atrial rate ~300 bpm, "
                               "requires rate control and anticoagulation consideration"
            }
        }
    return {}


# ============================================================================
# 4. LEFT ATRIAL ENLARGEMENT
# ============================================================================
def p_wave_left_atrial_enlargement(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Detect LAE: P-wave >=110ms (lowered from 120ms) OR notched in II/V1.
    ENHANCED sensitivity.
    """
    lae_leads = []
    durations = []
    notch_count = 0
    max_duration = 0

    for lead in ["II", "V1", "I"]:
        if lead not in ctx.lead_fits:
            continue

        p_gaussians = get_p_wave_gaussians(ctx.lead_fits, lead)

        if not p_gaussians:
            continue

        # Calculate P-wave duration
        p_start = min(g["center_ms"] - 2 * g["sigma_ms"] for g in p_gaussians)
        p_end = max(g["center_ms"] + 2 * g["sigma_ms"] for g in p_gaussians)
        duration = p_end - p_start
        durations.append(duration)
        max_duration = max(max_duration, duration)

        # Check for notching (multiple P components)
        if len(p_gaussians) >= 2:
            notch_count += 1

        # LOWERED THRESHOLD: 110ms (was 120ms)
        if duration >= 110 or len(p_gaussians) >= 2:
            lae_leads.append(lead)

    if lae_leads:
        avg_duration = np.mean(durations) if durations else 0
        return {
            "p_wave_left_atrial_enlargement": {
                "present": True,
                "description": f"Left atrial enlargement pattern in {format_lead_list(lae_leads)} "
                               f"(P duration {max_duration:.0f}ms)",
                # QUANTITATIVE PARAMETERS:
                "affected_leads": lae_leads,
                "max_duration_ms": round(max_duration, 1),
                "avg_duration_ms": round(avg_duration, 1),
                "notch_count": notch_count,
                "severity_score": min(1.0, max_duration / 150.0)  # Normalized 0-1
            }
        }
    return {}


# ============================================================================
# 5. RIGHT ATRIAL ENLARGEMENT
# ============================================================================
def p_wave_right_atrial_enlargement(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Detect RAE: P-wave amplitude >=0.20mV (lowered from 0.25mV) in II, III, or aVF.
    ENHANCED sensitivity.
    """
    rae_leads = []
    amplitudes = []
    max_amplitude = 0

    for lead in ["II", "III", "aVF", "V1", "V2"]:
        if lead not in ctx.lead_fits:
            continue

        p_gaussians = get_p_wave_gaussians(ctx.lead_fits, lead)

        if p_gaussians:
            total_amp = sum(abs(g["amp_mv"]) for g in p_gaussians)
            amplitudes.append(total_amp)
            max_amplitude = max(max_amplitude, total_amp)

            # LOWERED THRESHOLD: 0.20mV (was 0.25mV)
            if total_amp >= 0.20:
                rae_leads.append(lead)

    if rae_leads:
        avg_amplitude = np.mean(amplitudes) if amplitudes else 0
        return {
            "p_wave_right_atrial_enlargement": {
                "present": True,
                "description": f"Right atrial enlargement pattern in {format_lead_list(rae_leads)} "
                               f"(peak amplitude {max_amplitude:.2f}mV)",
                # QUANTITATIVE PARAMETERS:
                "affected_leads": rae_leads,
                "max_amplitude_mv": round(max_amplitude, 3),
                "avg_amplitude_mv": round(avg_amplitude, 3),
                "severity_score": min(1.0, max_amplitude / 0.35)  # Normalized 0-1
            }
        }
    return {}


# ============================================================================
# 6. INTER-ATRIAL BLOCK
# ============================================================================
def p_wave_inter_atrial_block(ctx: FeatureContext) -> Dict[str, Any]:
    p_gaussians_ii = get_p_wave_gaussians(ctx.lead_fits, "II")
    if not p_gaussians_ii:
        return {}
    p_start = min(g["center_ms"] - 2 * g["sigma_ms"] for g in p_gaussians_ii)
    p_end = max(g["center_ms"] + 2 * g["sigma_ms"] for g in p_gaussians_ii)
    p_dur = p_end - p_start
    if p_dur < 120:  # raised from 110
        return {}
    p_gaussians_v1 = get_p_wave_gaussians(ctx.lead_fits, "V1")
    if len(p_gaussians_v1) >= 2:
        amps = [g["amp_mv"] for g in p_gaussians_v1]
        if any(a > 0 for a in amps) and any(a < 0 for a in amps):
            neg_time = sum(4 * g["sigma_ms"] for g in p_gaussians_v1 if g["amp_mv"] < 0)
            min_neg = min((g["amp_mv"] for g in p_gaussians_v1 if g["amp_mv"] < 0), default=0)
            if neg_time >= 40 and min_neg < -0.02:
                typ = "advanced"
            else:
                typ = "partial"
            return {"p_wave_inter_atrial_block": {
                "present": True,
                "description": f"{typ.capitalize()} inter-atrial block (P {p_dur:.0f} ms, biphasic V1)",
                "type": typ,
                "p_duration_ms": round(p_dur, 1),
                "terminal_negative_duration_ms": round(neg_time, 1),
                "affected_leads": ["II", "V1"]
            }}
    return {}


# ============================================================================
# 7. FIRST-DEGREE AV BLOCK
# ============================================================================
def p_wave_first_degree_av_block(ctx: FeatureContext) -> Dict[str, Any]:
    prs = [calculate_pr_interval(ctx.lead_fits, L) for L in ["II", "V1", "I"] if calculate_pr_interval(ctx.lead_fits, L)]
    if not prs:
        return {}
    max_pr, avg_pr, std_pr = max(prs), np.mean(prs), (np.std(prs) if len(prs) > 1 else 0)
    if max_pr >= 230:  # raised
        sev = "mild" if max_pr < 250 else ("moderate" if max_pr < 300 else "severe")
        return {"p_wave_first_degree_av_block": {
            "present": True,
            "description": f"First-degree AV block (PR {max_pr:.0f} ms, {sev})",
            "max_pr_ms": round(max_pr, 1),
            "avg_pr_ms": round(avg_pr, 1),
            "std_pr_ms": round(std_pr, 1),
            "severity": sev,
            "severity_score": min(1.0, (max_pr - 230) / 230.0)
        }}
    return {}


# ============================================================================
# 8. SHORT PR SYNDROME
# ============================================================================
def p_wave_short_pr_syndrome(ctx: FeatureContext) -> Dict[str, Any]:
    prs = [calculate_pr_interval(ctx.lead_fits, L) for L in ["II", "V1", "I"] if calculate_pr_interval(ctx.lead_fits, L)]
    if not prs:
        return {}
    min_pr = min(prs)
    avg_pr = np.mean(prs)
    if min_pr < 100:  # tightened
        return {"p_wave_short_pr_syndrome": {
            "present": True,
            "description": f"Short PR interval ({min_pr:.0f} ms) \u2013 possible pre-excitation",
            "min_pr_ms": round(min_pr, 1),
            "avg_pr_ms": round(avg_pr, 1),
            "deviation_from_normal": round(120 - min_pr, 1)
        }}
    return {}


# ============================================================================
# 9. P-WAVE NOTCHING
# ============================================================================
def p_wave_notching(ctx: FeatureContext, *, debug: bool = False) -> Dict[str, Any]:
    lead_fits = getattr(ctx, "lead_fits", {})
    if not isinstance(lead_fits, dict) or not lead_fits:
        return {}
    leads_to_check = ["II", "V1", "I"]
    MIN_SEP_MS = 40.0      # raised
    MAX_SEP_MS = 120.0
    MIN_VALLEY_DROP = 0.30  # relaxed back to physiological
    MIN_BEATS_FOR_LEAD = 3  # raised
    SUPPORT_FRAC = 0.40
    hits_by_lead = {}
    nbeats_by_lead = {}
    for L in leads_to_check:
        if L not in lead_fits:
            continue
        by_beat = lead_fits[L].get("by_beat", [])
        if len(by_beat) < MIN_BEATS_FOR_LEAD:
            continue
        nbeats_by_lead[L] = len(by_beat)
        count = 0
        for beat in by_beat:
            p_gauss = [g for g in beat if g.get("wave_type") == "P"]
            if len(p_gauss) < 2:
                continue
            s = sorted(p_gauss, key=lambda x: x["center_ms"])
            for i in range(len(s) - 1):
                g1, g2 = s[i], s[i + 1]
                sep = abs(g2["center_ms"] - g1["center_ms"])
                if MIN_SEP_MS <= sep <= MAX_SEP_MS and g1["amp_mv"] * g2["amp_mv"] > 0:
                    count += 1
                    break
        if count >= 3:
            hits_by_lead[L] = count
    if hits_by_lead:
        tot = sum(hits_by_lead.values())
        leads = list(hits_by_lead.keys())
        return {"p_wave_notching": {
            "present": True,
            "description": f"Notched P-waves in {format_lead_list(leads)} \u2013 suggests left atrial enlargement",
            "affected_leads": leads,
            "notch_counts_by_lead": hits_by_lead,
            "total_notches": tot,
            "beats_analyzed_by_lead": nbeats_by_lead}}
    return {}


# ============================================================================
# 10. LOW VOLTAGE P-WAVES
# ============================================================================
def p_wave_low_voltage(ctx: FeatureContext) -> Dict[str, Any]:
    low, tot, amps = 0, 0, {}
    for L in ctx.leads:
        if L not in ctx.lead_fits:
            continue
        pgs = get_p_wave_gaussians(ctx.lead_fits, L)
        if not pgs:
            continue
        tot += 1
        ma = max(abs(g["amp_mv"]) for g in pgs)
        amps[L] = ma
        if ma < 0.05:  # lowered
            low += 1
    if tot >= 8 and low >= 8:  # stricter
        avg = np.mean(list(amps.values())) if amps else 0
        return {"p_wave_low_voltage": {
            "present": True,
            "description": f"Low-voltage P-waves (<0.05 mV) in {low}/{tot} leads \u2013 possible obesity/emphysema",
            "low_voltage_lead_count": low,
            "total_leads_analyzed": tot,
            "avg_amplitude_mv": round(avg, 3),
            "lead_amplitudes": {k: round(v, 3) for k, v in amps.items()},
            "severity_score": 1.0 - (avg / 0.05)}}
    return {}


# ============================================================================
# 11. WANDERING ATRIAL PACEMAKER
# ============================================================================
def p_wave_wandering_pacemaker(ctx: FeatureContext) -> Dict[str, Any]:
    by_beat = ctx.lead_fits.get("II", {}).get("by_beat", [])
    if len(by_beat) < 8:
        return {}
    axes = []
    for i in range(min(15, len(by_beat))):
        ax = calculate_p_wave_axis_for_beat(ctx.lead_fits, i)
        if ax is not None:
            axes.append(float(np.asarray(ax).item()))
    if len(axes) < 8:
        return {}
    rng = max(axes) - min(axes)
    std = np.std(axes)
    changes = sum(1 for i in range(1, len(axes)) if abs(axes[i] - axes[i - 1]) > 20)  # raised
    if rng > 90 and changes >= 3:  # raised
        return {"p_wave_wandering_pacemaker": {
            "present": True,
            "description": f"Wandering atrial pacemaker \u2013 axis varies {rng:.0f}\u00b0",
            "axis_range_degrees": round(rng, 1),
            "axis_std_degrees": round(std, 1),
            "min_axis": round(min(axes), 1),
            "max_axis": round(max(axes), 1),
            "gradual_change_count": changes,
            "beat_count": len(axes)}}
    return {}


# ============================================================================
# 12. P-WAVE ALTERNANS
# ============================================================================
def p_wave_alternans(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Detect P-wave alternans --- ENHANCED for high sensitivity.
    OLD threshold: 20% even/odd difference.
    NEW threshold: 10% even/odd difference (detect subtle alternans).
    """
    for lead in ["II", "V1", "I"]:  # Added lead I for more coverage
        if lead not in ctx.lead_fits:
            continue

        by_beat = ctx.lead_fits[lead].get("by_beat", [])
        if len(by_beat) < 6:  # Lowered from 8 to 6 beats
            continue

        # Get P-wave amplitudes for consecutive beats
        amplitudes = []
        for beat_idx in range(min(20, len(by_beat))):  # Increased from 16 to 20
            p_gaussians = [g for g in by_beat[beat_idx] if g.get("wave_type") == "P"]
            if p_gaussians:
                total_amp = sum(abs(g["amp_mv"]) for g in p_gaussians)
                amplitudes.append(total_amp)

        if len(amplitudes) >= 6:  # Lowered from 8 to 6
            # Check for alternating pattern (even vs odd beats)
            even_amps = [amplitudes[i] for i in range(0, len(amplitudes), 2)]
            odd_amps = [amplitudes[i] for i in range(1, len(amplitudes), 2)]

            if len(even_amps) >= 3 and len(odd_amps) >= 3:
                even_mean = np.mean(even_amps)
                odd_mean = np.mean(odd_amps)
                even_std = np.std(even_amps)
                odd_std = np.std(odd_amps)

                difference_pct = abs(even_mean - odd_mean) / max(even_mean, odd_mean) * 100

                # LOWERED THRESHOLD: 10% (was 20%)
                if difference_pct > 10.0:
                    return {
                        "p_wave_alternans": {
                            "present": True,
                            "description": f"P-wave alternans in lead {lead} - electrical instability, "
                                           f"increased atrial arrhythmia risk",
                            # QUANTITATIVE PARAMETERS:
                            "affected_leads": [lead],
                            "even_odd_difference_pct": round(difference_pct, 1),
                            "even_mean_mv": round(even_mean, 3),
                            "odd_mean_mv": round(odd_mean, 3),
                            "even_std_mv": round(even_std, 3),
                            "odd_std_mv": round(odd_std, 3),
                            "beat_count": len(amplitudes),
                            "confidence": min(1.0, difference_pct / 20.0)  # Normalized confidence
                        }
                    }
    return {}


# ============================================================================
# 13. MULTIFOCAL ATRIAL TACHYCARDIA
# ============================================================================
def p_wave_multifocal_atrial_tachycardia(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Detect MAT: >=3 distinct P-wave morphologies + variable PR intervals + HR>100 (lowered from HR>110).
    ENHANCED sensitivity.
    """
    # Check heart rate
    if not ctx.rr_intervals or len(ctx.rr_intervals) < 3:
        return {}

    hr = 60000 / np.mean(ctx.rr_intervals)

    # LOWERED THRESHOLD: HR > 100 (was 110)
    if hr < 100:
        return {}

    # Check for P-wave morphology variability across beats
    morphology_changes = 0
    by_beat = ctx.lead_fits.get("II", {}).get("by_beat", [])

    if len(by_beat) < 5:
        return {}

    p_amplitudes = []
    p_durations = []

    for beat_idx in range(min(15, len(by_beat))):
        p_gaussians = [g for g in by_beat[beat_idx] if g.get("wave_type") == "P"]
        if p_gaussians:
            total_amp = sum(abs(g["amp_mv"]) for g in p_gaussians)
            duration = sum(4 * g["sigma_ms"] for g in p_gaussians)
            p_amplitudes.append(total_amp)
            p_durations.append(duration)

    if len(p_amplitudes) >= 5:
        amp_cv = np.std(p_amplitudes) / np.mean(p_amplitudes) if np.mean(p_amplitudes) > 0 else 0
        dur_cv = np.std(p_durations) / np.mean(p_durations) if np.mean(p_durations) > 0 else 0

        # LOWERED THRESHOLD: CV > 0.15 (was 0.25)
        if amp_cv > 0.15 or dur_cv > 0.15:
            return {
                "p_wave_multifocal_atrial_tachycardia": {
                    "present": True,
                    "description": f"Multifocal atrial tachycardia pattern - variable P-wave morphology, HR {hr:.0f} bpm",
                    # QUANTITATIVE PARAMETERS:
                    "heart_rate_bpm": round(hr, 1),
                    "p_amplitude_cv": round(amp_cv, 3),
                    "p_duration_cv": round(dur_cv, 3),
                    "distinct_morphologies_detected": len(set(np.round(p_amplitudes, 2))),
                    "beat_count": len(p_amplitudes)
                }
            }

    return {}


# ============================================================================
# 14. PREMATURE ATRIAL CONTRACTIONS
# ============================================================================
def p_wave_premature_atrial_contractions(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Detect PACs: early P-waves with different morphology.
    ENHANCED sensitivity --- detect subtle PACs.
    """
    if not ctx.rr_intervals or len(ctx.rr_intervals) < 5:
        return {}

    mean_rr = np.mean(ctx.rr_intervals)
    pac_count = 0
    pac_indices = []

    by_beat = ctx.lead_fits.get("II", {}).get("by_beat", [])

    if len(by_beat) < 5:
        return {}

    for i in range(1, min(len(ctx.rr_intervals), len(by_beat) - 1)):
        # LOWERED THRESHOLD: RR < 85% mean (was 80%)
        if ctx.rr_intervals[i] < 0.85 * mean_rr:
            # Check if P-wave morphology differs
            current_p = [g for g in by_beat[i] if g.get("wave_type") == "P"]
            prev_p = [g for g in by_beat[i - 1] if g.get("wave_type") == "P"]

            if current_p and prev_p:
                current_amp = sum(abs(g["amp_mv"]) for g in current_p)
                prev_amp = sum(abs(g["amp_mv"]) for g in prev_p)

                # LOWERED THRESHOLD: 10% difference (was 20%)
                if abs(current_amp - prev_amp) / max(prev_amp, 0.01) > 0.10:
                    pac_count += 1
                    pac_indices.append(i)

    if pac_count > 0:
        pac_rate = pac_count / len(by_beat) * 100

        return {
            "p_wave_premature_atrial_contractions": {
                "present": True,
                "description": f"Premature atrial contractions detected ({pac_count} PACs, {pac_rate:.1f}% of beats)",
                # QUANTITATIVE PARAMETERS:
                "pac_count": pac_count,
                "total_beats": len(by_beat),
                "pac_rate_percent": round(pac_rate, 1),
                "pac_beat_indices": pac_indices[:10],  # First 10 for reference
                "mean_coupling_interval_ms": round(mean_rr * 0.85, 1)
            }
        }

    return {}


# ============================================================================
# 15. ECTOPIC ATRIAL RHYTHM
# ============================================================================
def p_wave_ectopic_atrial_rhythm(ctx: FeatureContext) -> Dict[str, Any]:
    axis = calculate_p_wave_axis(ctx.lead_fits)
    if axis is None:
        return {}
    axis = float(np.asarray(axis).item())
    if axis < -15 or axis > 90:  # tightened
        by_beat_axes = []
        beats = ctx.lead_fits.get("I", {}).get("by_beat", [])
        for i in range(min(10, len(beats))):
            a = calculate_p_wave_axis_for_beat(ctx.lead_fits, i)
            if a is not None:
                by_beat_axes.append(float(np.asarray(a).item()))
        axis_std = np.std(by_beat_axes) if len(by_beat_axes) > 1 else 0
        return {"p_wave_ectopic_atrial_rhythm": {
            "present": True,
            "description": f"Ectopic atrial rhythm \u2013 abnormal P axis ({axis:.0f}\u00b0)",
            "p_axis_degrees": round(axis, 1),
            "axis_variability_deg": round(axis_std, 1),
            "beat_count": len(by_beat_axes)}}
    return {}


# ============================================================================
# 16. P-WAVE AXIS DEVIATION
# ============================================================================
def p_wave_axis_deviation(ctx: FeatureContext) -> Dict[str, Any]:
    axis = calculate_p_wave_axis(ctx.lead_fits)
    if axis is None:
        return {}
    axis = float(np.asarray(axis).item())
    if axis < -15:
        return {"p_wave_left_axis_deviation": {
            "present": True,
            "description": f"Left P-axis deviation ({axis:.0f}\u00b0)",
            "axis_degrees": round(axis, 1),
            "severity": "mild" if axis > -30 else "moderate"}}
    elif axis > 90:
        return {"p_wave_right_axis_deviation": {
            "present": True,
            "description": f"Right P-axis deviation ({axis:.0f}\u00b0)",
            "axis_degrees": round(axis, 1),
            "severity": "mild" if axis < 110 else "moderate"}}
    return {}


# ============================================================================
# 17. RETROGRADE P-WAVES
# ============================================================================
def p_wave_retrograde(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Detect retrograde P-waves: inverted in II, III, aVF.
    ENHANCED sensitivity --- lowered inversion threshold.
    """
    inverted_inferior = []
    amplitudes = {}

    for lead in ["II", "III", "aVF"]:
        if lead not in ctx.lead_fits:
            continue

        p_gaussians = get_p_wave_gaussians(ctx.lead_fits, lead)

        if p_gaussians:
            total_amp = sum(g["amp_mv"] for g in p_gaussians)
            amplitudes[lead] = total_amp

            # LOWERED THRESHOLD: < -0.03mV (was -0.05mV)
            if total_amp < -0.03:
                inverted_inferior.append(lead)

    if len(inverted_inferior) >= 2:
        pr_interval = calculate_pr_interval(ctx.lead_fits, "II")

        return {
            "p_wave_retrograde": {
                "present": True,
                "description": f"Retrograde P-waves (inverted in {format_lead_list(inverted_inferior)}), "
                               f"suggesting junctional rhythm",
                # QUANTITATIVE PARAMETERS:
                "affected_leads": inverted_inferior,
                "lead_amplitudes_mv": {k: round(v, 3) for k, v in amplitudes.items()},
                "pr_interval_ms": round(pr_interval, 1) if pr_interval else None,
                "inversion_count": len(inverted_inferior)
            }
        }

    return {}


# ============================================================================
# 14. SINUS ARREST
# ============================================================================
def p_wave_sinus_arrest(ctx: FeatureContext) -> Dict[str, Any]:
    """
    Detect sinus arrest: missing P-waves with pause >2x mean RR (lowered from 2.5x).
    ENHANCED sensitivity.
    """
    if not ctx.rr_intervals or len(ctx.rr_intervals) < 3:
        return {}

    mean_rr = np.mean(ctx.rr_intervals)
    pauses = []

    for i, rr in enumerate(ctx.rr_intervals):
        # LOWERED THRESHOLD: 2.0x mean RR (was 2.5x)
        if rr > 2.0 * mean_rr:
            pauses.append({"index": i, "duration_ms": rr})

    if pauses:
        max_pause = max(p["duration_ms"] for p in pauses)

        return {
            "p_wave_sinus_arrest": {
                "present": True,
                "description": f"Sinus arrest detected - {len(pauses)} pause(s), longest {max_pause:.0f}ms",
                # QUANTITATIVE PARAMETERS:
                "pause_count": len(pauses),
                "max_pause_ms": round(max_pause, 1),
                "mean_rr_ms": round(mean_rr, 1),
                "pause_ratio": round(max_pause / mean_rr, 2),
                "pause_indices": [p["index"] for p in pauses[:5]]  # First 5
            }
        }

    return {}


# ============================================================================
# REGISTRY OF ALL P-WAVE FEATURES
# ============================================================================

P_WAVE_FEATURES = [
    p_wave_pr_depression,
    p_wave_atrial_bigeminy,
    p_wave_second_degree_av_block,
    p_wave_atrial_fibrillation,
    p_wave_atrial_f_wave,
    p_wave_left_atrial_enlargement,
    p_wave_right_atrial_enlargement,
    p_wave_inter_atrial_block,
    p_wave_first_degree_av_block,
    p_wave_short_pr_syndrome,
    p_wave_notching,
    p_wave_low_voltage,
    p_wave_wandering_pacemaker,
    p_wave_alternans,
    p_wave_multifocal_atrial_tachycardia,
    p_wave_premature_atrial_contractions,
    p_wave_ectopic_atrial_rhythm,
    p_wave_axis_deviation,
    p_wave_retrograde,
    p_wave_sinus_arrest,
]
