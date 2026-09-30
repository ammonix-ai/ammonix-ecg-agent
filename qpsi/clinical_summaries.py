"""
Clinical summary generators for ECG features.

Five domain-specific generators (QRS, Rhythm, Global, P-wave, T-wave) plus
a compact Tier-A summary builder used by the final-record assembler.

Source: Cell 15 lines 118-445 (5 generators)
        Cell 17 lines 1-55 (lead helpers), 56-422 (_short_* functions),
        3376-3394 (build_clinical_summaries_tierA)
No internal dependencies. External: re (for P-axis parsing), typing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Internal helpers for compact summaries (Cell 17)
# ---------------------------------------------------------------------------

_LIMB = ["I", "II", "III", "aVR", "aVL", "aVF"]
_VS = ["V1", "V2", "V3", "V4", "V5", "V6"]


def _lead_compact(leads: List[str]) -> str:
    """Compact lead lists like ['V3','V4','V5','V6','aVL'] -> 'V3\u2013V6, aVL'."""
    if not leads:
        return ""
    limbs = [l for l in leads if l in _LIMB]
    vnums = sorted(int(l[1:]) for l in leads if l in _VS)
    # build V-ranges
    ranges: List[tuple] = []
    if vnums:
        start = prev = vnums[0]
        for x in vnums[1:]:
            if x == prev + 1:
                prev = x
                continue
            ranges.append((start, prev))
            start = prev = x
        ranges.append((start, prev))
    vparts = []
    for a, b in ranges:
        vparts.append(f"V{a}" if a == b else f"V{a}\u2013V{b}")
    parts: List[str] = []
    if vparts:
        parts.append(", ".join(vparts))
    if limbs:
        parts.append(", ".join(limbs))
    return ", ".join(parts)


def _pluck_leads(d: Dict[str, Any]) -> List[str]:
    """Extract lead list from a feature dict, trying common key names."""
    if not isinstance(d, dict):
        return []
    for k in ("affected_leads", "leads", "limb_leads_affected",
              "precordial_leads_affected"):
        v = d.get(k)
        if isinstance(v, list) and v:
            return v
    patt = d.get("pattern")
    if isinstance(patt, dict):
        return list(patt.keys())
    return []


# ---------------------------------------------------------------------------
# Compact _short_* helpers (Cell 17)
# ---------------------------------------------------------------------------

def _short_T(feature_T: Dict[str, Any]) -> str:
    """Compact T-wave summary line."""
    if not isinstance(feature_T, dict):
        return "T: \u2014"
    tokens: List[str] = []
    if feature_T.get("t_wave_biphasic", {}).get("present"):
        L = _pluck_leads(feature_T["t_wave_biphasic"])
        tokens.append(f"biphasic {_lead_compact(L)}" if L else "biphasic")
    if feature_T.get("t_wave_split", {}).get("present"):
        L = _pluck_leads(feature_T["t_wave_split"])
        tokens.append(f"split {_lead_compact(L)}" if L else "split")
    if feature_T.get("t_wave_notched", {}).get("present"):
        L = _pluck_leads(feature_T["t_wave_notched"])
        tokens.append(f"notched {_lead_compact(L)}" if L else "notched")
    if feature_T.get("t_wave_peaked", {}).get("present"):
        L = _pluck_leads(feature_T["t_wave_peaked"])
        tokens.append(f"peaked {_lead_compact(L)}" if L else "peaked")
    if feature_T.get("t_wave_inverted", {}).get("present"):
        L = _pluck_leads(feature_T["t_wave_inverted"])
        tokens.append(f"inverted {_lead_compact(L)}" if L else "inverted")
    if feature_T.get("t_wave_lead_discordance", {}).get("present"):
        L = _pluck_leads(feature_T["t_wave_lead_discordance"])
        tokens.append(f"discordant {_lead_compact(L)}" if L else "discordant")
    if feature_T.get("t_wave_u_on_t", {}).get("present"):
        L = _pluck_leads(feature_T["t_wave_u_on_t"])
        tokens.append(f"U-on-T {_lead_compact(L)}" if L else "U-on-T")
    if feature_T.get("t_wave_fusion", {}).get("present"):
        tokens.append("T\u2013P fusion")
    if not tokens:
        return "T: no specific abnormality"
    return "T: " + "; ".join(tokens)


def _short_P(feature_P: Dict[str, Any]) -> str:
    """Compact P-wave summary line."""
    if not isinstance(feature_P, dict):
        return "P: \u2014"
    tokens: List[str] = []

    # 1) Conduction / PR info
    fd = feature_P.get("first_degree_av_block", {})
    if isinstance(fd, dict) and fd.get("present"):
        pr = fd.get("pr_ms")
        sev = fd.get("severity")
        if isinstance(pr, (int, float)) and pr > 0:
            tokens.append(
                f"1\u00b0 AVB (PR ~{int(round(pr / 10) * 10)} ms"
                f"{f', {sev}' if sev else ''})"
            )
        else:
            tokens.append("1\u00b0 AVB")

    shortpr = feature_P.get("short_pr_syndrome", {})
    if isinstance(shortpr, dict) and shortpr.get("present"):
        tokens.append("short PR")

    # 2) Inter-atrial block (IAB)
    iab = feature_P.get("inter_atrial_block", {})
    if isinstance(iab, dict) and iab.get("present"):
        desc = iab.get("description", "IAB")
        degree = None
        if "advanced" in desc.lower():
            degree = "advanced"
        elif "partial" in desc.lower():
            degree = "partial"
        tokens.append(f"IAB{(' ' + degree) if degree else ''}".strip())

    # 3) Atrial enlargement / tall P / low-voltage / axis
    if feature_P.get("p_wave_left_atrial_enlargement", {}).get("present"):
        L = _pluck_leads(feature_P["p_wave_left_atrial_enlargement"])
        tokens.append(f"LAE {_lead_compact(L)}" if L else "LAE")
    if feature_P.get("p_wave_right_atrial_enlargement", {}).get("present"):
        L = _pluck_leads(feature_P["p_wave_right_atrial_enlargement"])
        tokens.append(f"RAE {_lead_compact(L)}" if L else "RAE")
    if feature_P.get("p_wave_low_voltage", {}).get("present"):
        tokens.append("P low voltage")

    # 4) Morphology: notching, biphasic, alternans
    if feature_P.get("p_wave_notching", {}).get("present"):
        L = _pluck_leads(feature_P["p_wave_notching"])
        tokens.append(f"notched {_lead_compact(L)}" if L else "notched")
    if feature_P.get("p_wave_biphasic", {}).get("present"):
        L = _pluck_leads(feature_P["p_wave_biphasic"])
        tokens.append(f"biphasic {_lead_compact(L)}" if L else "biphasic")
    if feature_P.get("p_wave_alternans", {}).get("present"):
        tokens.append("P alternans")

    # 5) Timing / size numeric hints
    p_dur = feature_P.get("p_duration_ms")
    if isinstance(p_dur, (int, float)) and p_dur > 0:
        if p_dur >= 120:
            tokens.append(f"P prolonged ~{int(round(p_dur))} ms")
        elif p_dur <= 60:
            tokens.append(f"P short ~{int(round(p_dur))} ms")

    # 6) PR depression (pericarditis hint)
    if feature_P.get("p_wave_pr_depression", {}).get("present"):
        tokens.append("PR depression")

    # 7) Ectopy / multifocal / wandering / PACs / bigeminy
    if feature_P.get("p_wave_ectopic_atrial_rhythm", {}).get("present"):
        tokens.append("ectopic atrial rhythm")
    if feature_P.get("p_wave_multifocal_atrial_tachycardia", {}).get("present"):
        tokens.append("multifocal atrial tachycardia")
    if feature_P.get("p_wave_premature_atrial_contractions", {}).get("present"):
        pac = feature_P["p_wave_premature_atrial_contractions"]
        cnt = pac.get("count")
        tokens.append(f"PACs {int(cnt)}" if cnt else "PACs")
    if feature_P.get("p_wave_wandering_pacemaker", {}).get("present"):
        tokens.append("wandering pacemaker")
    if feature_P.get("p_wave_atrial_bigeminy", {}).get("present"):
        tokens.append("atrial bigeminy")
    if feature_P.get("p_wave_retrograde", {}).get("present"):
        tokens.append("retrograde P")

    # 8) Block progression (2nd degree / sinus arrest)
    if feature_P.get("p_wave_second_degree_av_block", {}).get("present"):
        val = feature_P["p_wave_second_degree_av_block"].get("type")
        tokens.append(f"2\u00b0 AV-block{(' ' + val) if val else ''}")
    if feature_P.get("p_wave_sinus_arrest", {}).get("present"):
        tokens.append("sinus arrest")

    # 9) Axis / deviation
    if feature_P.get("p_wave_axis_deviation", {}).get("present"):
        d = feature_P["p_wave_axis_deviation"].get("description", "")
        m = re.search(r"([+-]?\d+)\s*\u00b0", d)
        if m:
            tokens.append(f"P-axis {m.group(1)}\u00b0")
        else:
            tokens.append("P axis dev")

    if not tokens:
        return "P: no specific abnormality"
    return "P: " + "; ".join(tokens)


def _short_QRS(feature_QRS: Dict[str, Any]) -> str:
    """Compact QRS summary line."""
    if not isinstance(feature_QRS, dict):
        return "QRS: \u2014"
    tokens: List[str] = []
    if feature_QRS.get("qrs_low_voltage", {}).get("present"):
        lv = feature_QRS["qrs_low_voltage"]
        limb = _lead_compact(lv.get("limb_leads_affected", []))
        prec = _lead_compact(lv.get("precordial_leads_affected", []))
        if limb and prec:
            tokens.append("low voltage (limb, precordial)")
        elif limb:
            tokens.append("low voltage (limb)")
        elif prec:
            tokens.append("low voltage (precordial)")
        else:
            tokens.append("low voltage")
    if feature_QRS.get("qrs_fragmentation", {}).get("present"):
        L = _pluck_leads(feature_QRS["qrs_fragmentation"])
        tokens.append(f"fragmented {_lead_compact(L)}" if L else "fragmented")
    if feature_QRS.get("rsr_pattern", {}).get("present"):
        L = _pluck_leads(feature_QRS["rsr_pattern"])
        tokens.append(f"RSR' {_lead_compact(L)}" if L else "RSR'")
    return "QRS: " + ("; ".join(tokens) if tokens else "no specific abnormality")


def _short_ST(feature_ST: dict, *, poor_quality: bool) -> str:
    """Compact ST-segment summary line."""
    if not isinstance(feature_ST, dict):
        return "ST: \u2014"
    has_ste = feature_ST.get("st_elevation", {}).get("present")
    has_std = feature_ST.get("st_depression", {}).get("present")
    if poor_quality and not (has_ste or has_std):
        return "ST: assessment limited by poor quality"
    if has_ste:
        ste = feature_ST["st_elevation"]
        L = _lead_compact(ste.get("affected_leads", []))
        mv = ste.get("max_deviation_mv")
        mag = f"; max {mv:.2f} mV" if isinstance(mv, (int, float)) else ""
        return f"ST-elevation {L}{mag}".strip()
    if has_std:
        std = feature_ST["st_depression"]
        L = _lead_compact(std.get("affected_leads", []))
        mv = std.get("max_deviation_mv")
        mag = f"; max {mv:.2f} mV" if isinstance(mv, (int, float)) else ""
        return f"ST-depression {L}{mag}".strip()
    return "ST: no acute ST changes"


def _short_rhythm(feature_rhythm: Dict[str, Any]) -> str:
    """Concise rhythm line for clinical summaries.

    Includes ALL rhythm features detected, with critical findings flagged.
    """
    if not isinstance(feature_rhythm, dict):
        return "Rhythm: \u2014"

    pieces: List[str] = []
    critical: List[str] = []

    # 1) Priority/Critical patterns
    aa = feature_rhythm.get("absolute_arrhythmia", {})
    if isinstance(aa, dict) and aa.get("present"):
        bits: List[str] = []
        cv = aa.get("rr_cv")
        pnn = aa.get("pnn50")
        if isinstance(cv, (int, float)):
            bits.append(f"CV {cv:.2f}")
        if isinstance(pnn, (int, float)):
            bits.append(f"pNN50 {pnn:.0f}%")
        extra = f" ({', '.join(bits)})" if bits else ""
        critical.append("irregular; no stable P sequence" + extra)

    fl = feature_rhythm.get("flutter_like_rr_clustering", {})
    if isinstance(fl, dict) and fl.get("present"):
        centers = fl.get("centers_ms") or []
        if isinstance(centers, list) and centers:
            centers_txt = " / ".join(str(int(round(c))) for c in centers[:3])
            critical.append(
                f"clustered RR ({centers_txt} ms); variable A\u2013V conduction"
            )
        else:
            critical.append("clustered RR; variable A\u2013V conduction")

    fw = feature_rhythm.get("atrial_f_wave", {})
    if isinstance(fw, dict) and fw.get("present"):
        hz = fw.get("f_wave_freq_hz")
        if isinstance(hz, (int, float)):
            critical.append(f"organized atrial activity ~{hz:.1f} Hz")
        else:
            critical.append("organized atrial activity")

    # 2) Regular rhythm features
    if feature_rhythm.get("normal_sinus_rhythm", {}).get("present"):
        nsr = feature_rhythm["normal_sinus_rhythm"]
        hr = nsr.get("heart_rate")
        pieces.append(
            f"sinus ~{int(round(hr))} bpm" if isinstance(hr, (int, float))
            else "sinus"
        )

    if feature_rhythm.get("sinus_bradycardia", {}).get("present"):
        sb = feature_rhythm["sinus_bradycardia"]
        hr = sb.get("heart_rate")
        sev = sb.get("severity", "")
        pieces.append(
            f"sinus brady ~{int(round(hr))} bpm ({sev})" if hr
            else "sinus bradycardia"
        )

    if feature_rhythm.get("sinus_tachycardia", {}).get("present"):
        st = feature_rhythm["sinus_tachycardia"]
        hr = st.get("heart_rate")
        pieces.append(
            f"sinus tachy ~{int(round(hr))} bpm" if isinstance(hr, (int, float))
            else "sinus tachycardia"
        )

    if feature_rhythm.get("irregular_rhythm", {}).get("present"):
        irr = feature_rhythm["irregular_rhythm"]
        pat = irr.get("pattern", "irregular")
        pieces.append(pat)

    # 3) Ventricular ectopy
    if feature_rhythm.get("ventricular_premature_beats", {}).get("present"):
        pvc = feature_rhythm["ventricular_premature_beats"]
        cnt = pvc.get("count")
        burden = pvc.get("burden_percent")
        if cnt and burden is not None:
            pieces.append(f"{cnt} PVCs (~{burden:.1f}%)")
        elif cnt:
            pieces.append(f"{cnt} PVCs")
        else:
            pieces.append("PVCs")

    if feature_rhythm.get("ventricular_bigeminy", {}).get("present"):
        pieces.append("V-bigeminy")
    if feature_rhythm.get("ventricular_trigeminy", {}).get("present"):
        pieces.append("V-trigeminy")

    if feature_rhythm.get("non_sustained_vt", {}).get("present"):
        nsvt = feature_rhythm["non_sustained_vt"]
        longest = nsvt.get("longest_run_beats")
        pieces.append(f"NSVT ({longest} beats)" if longest else "NSVT")

    # 4) Tachycardias
    if feature_rhythm.get("supraventricular_tachycardia", {}).get("present"):
        svt = feature_rhythm["supraventricular_tachycardia"]
        hr = svt.get("heart_rate")
        pieces.append(f"SVT ~{int(round(hr))} bpm" if hr else "SVT")

    if feature_rhythm.get("ventricular_tachycardia", {}).get("present"):
        vt = feature_rhythm["ventricular_tachycardia"]
        hr = vt.get("heart_rate")
        critical.append(f"VT ~{int(round(hr))} bpm" if hr else "VT")

    # 5) Other rhythms
    if feature_rhythm.get("accelerated_idioventricular_rhythm", {}).get("present"):
        aivr = feature_rhythm["accelerated_idioventricular_rhythm"]
        hr = aivr.get("heart_rate")
        pieces.append(f"AIVR ~{int(round(hr))} bpm" if hr else "AIVR")

    if feature_rhythm.get("junctional_rhythm", {}).get("present"):
        jr = feature_rhythm["junctional_rhythm"]
        hr = jr.get("heart_rate")
        pieces.append(
            f"junctional ~{int(round(hr))} bpm" if hr else "junctional"
        )

    if feature_rhythm.get("escape_rhythm", {}).get("present"):
        er = feature_rhythm["escape_rhythm"]
        hr = er.get("heart_rate")
        critical.append(
            f"escape ~{int(round(hr))} bpm" if hr else "escape rhythm"
        )

    # 6) Conduction/pause issues
    if feature_rhythm.get("long_pause", {}).get("present"):
        lp = feature_rhythm["long_pause"]
        pause_ms = lp.get("pause_duration")
        sev = lp.get("severity", "")
        pieces.append(
            f"pause {pause_ms:.0f}ms ({sev})" if pause_ms else "long pause"
        )

    if feature_rhythm.get("r_on_t_phenomenon", {}).get("present"):
        critical.append("R-on-T (HIGH RISK)")

    if feature_rhythm.get("paced_rhythm", {}).get("present"):
        pieces.append("paced")

    # 7) Extrasystoles
    if feature_rhythm.get("extrasystoles", {}).get("present"):
        ext = feature_rhythm["extrasystoles"]
        cnt = ext.get("count")
        burden = ext.get("burden_percent")
        if cnt and burden is not None:
            pieces.append(f"extrasystoles {cnt} ({burden:.1f}%)")
        else:
            pieces.append("extrasystoles")

    # 8) HRV
    if feature_rhythm.get("hrv_metrics", {}).get("present"):
        hrv = feature_rhythm["hrv_metrics"]
        rmssd = hrv.get("rmssd")
        sdnn = hrv.get("sdnn")
        if rmssd and sdnn:
            pieces.append(f"HRV: RMSSD={rmssd:.1f}, SDNN={sdnn:.1f}")

    if feature_rhythm.get("reduced_hrv", {}).get("present"):
        pieces.append("reduced HRV")

    # 9) Panorama
    pano = feature_rhythm.get("rhythm_panorama", {})
    if pano.get("present"):
        reg = pano.get("regular_fraction")
        if reg is not None:
            pieces.append(f"{int(round(reg * 100))}% regular")

    # Build final output
    if critical:
        result = "Rhythm: \u26a0\ufe0f " + "; ".join(critical)
        if pieces:
            result += " + " + "; ".join(pieces)
        return result
    elif pieces:
        return "Rhythm: " + "; ".join(pieces)
    else:
        return "Rhythm: \u2014"


def _short_global(
    feature_global: Dict[str, Any],
    computed_params: Dict[str, Any],
    *,
    poor_quality: bool,
) -> str:
    """Compact global summary line."""
    tokens: List[str] = []
    # QTc
    if (isinstance(feature_global, dict)
            and feature_global.get("qtc_prolongation", {}).get("present")):
        qtc = feature_global["qtc_prolongation"]
        qtc_ms = qtc.get("qtc_ms", None)
        sev = qtc.get("severity", None)
        if isinstance(qtc_ms, (int, float)):
            t = f"QTc {round(qtc_ms)} ms"
            if isinstance(sev, str):
                t += f" ({sev})"
            tokens.append(t)
    # Rate from computed params if available
    hr = None
    rr = (computed_params.get("Main RR Interval mean")
          if isinstance(computed_params, dict) else None)
    if isinstance(rr, (int, float)) and rr > 0:
        hr = round(60000.0 / rr)
    if hr:
        tokens.append(f"rate ~{hr} bpm")
    # Quality tag
    tokens.append("quality: poor" if poor_quality else "quality: ok")
    return "Global: " + "; ".join(tokens)


# ---------------------------------------------------------------------------
# Full-form generators (Cell 15)
# ---------------------------------------------------------------------------

def generate_qrs_clinical_summary(features: Dict[str, Any]) -> str:
    """Generate clinical summary from QRS features.

    Args:
        features: Dict of QRS feature dicts, each with 'present' and
                  'description' keys.

    Returns:
        Clinical summary string.
    """
    if not features:
        return "QRS morphology within normal limits."

    descriptions: List[str] = []
    urgent: List[str] = []

    priority_features = [
        "wide_qrs_complex",
        "pathological_q_waves",
        "delta_wave",
        "epsilon_wave",
        "qrs_alternans",
        "ventricular_tachycardia",
    ]

    for feature_name, feature_data in features.items():
        if isinstance(feature_data, dict) and feature_data.get("present"):
            desc = feature_data.get("description", "")
            if (feature_name in priority_features
                    or any(word in desc.upper()
                           for word in ["EMERGENCY", "HIGH RISK",
                                        "WARNING", "CRITICAL"])):
                urgent.append(desc)
            else:
                descriptions.append(desc)

    summary = "QRS analysis reveals: "

    if urgent:
        summary += "\u26a0\ufe0f IMPORTANT: " + " ".join(urgent)
        if descriptions:
            summary += " Additional findings: "

    if descriptions:
        summary += " ".join(descriptions)

    return summary


def generate_rhythm_clinical_summary(rhythm_flags: Dict[str, Any]) -> str:
    """Generate comprehensive rhythm clinical summary.

    Handles ALL rhythm features without early returns.

    Args:
        rhythm_flags: Dict of rhythm feature dicts.

    Returns:
        Clinical summary string.
    """
    if not isinstance(rhythm_flags, dict) or not rhythm_flags:
        return "Rhythm analysis: no abnormalities detected"

    findings: List[str] = []
    critical: List[str] = []

    features_to_check = [
        # Critical arrhythmias
        ("ventricular_tachycardia", "VT", True),
        ("escape_rhythm", "escape rhythm", True),
        ("r_on_t_phenomenon", "R-on-T phenomenon (HIGH RISK)", True),
        ("absolute_arrhythmia",
         "irregular rhythm without stable P sequence", True),
        # Regular rhythms
        ("normal_sinus_rhythm", None, False),
        ("sinus_bradycardia", None, False),
        ("sinus_tachycardia", None, False),
        # Irregular rhythms
        ("irregular_rhythm", None, False),
        ("atrial_fibrillation", "atrial fibrillation", False),
        ("atrial_f_wave", None, False),
        ("flutter_like_rr_clustering", None, False),
        # Ectopy
        ("ventricular_premature_beats", None, False),
        ("ventricular_bigeminy", "ventricular bigeminy", False),
        ("ventricular_trigeminy", "ventricular trigeminy", False),
        ("non_sustained_vt", "NSVT", False),
        ("extrasystoles", None, False),
        # Other tachycardias
        ("supraventricular_tachycardia", "SVT", False),
        ("accelerated_idioventricular_rhythm", "AIVR", False),
        # Other rhythms
        ("junctional_rhythm", "junctional rhythm", False),
        ("paced_rhythm", "paced rhythm", False),
        # Pauses and blocks
        ("long_pause", None, False),
        # HRV
        ("hrv_metrics", None, False),
        ("reduced_hrv", "reduced heart rate variability", False),
        # Panorama
        ("rhythm_panorama", None, False),
    ]

    for feature_key, default_desc, is_critical in features_to_check:
        if not rhythm_flags.get(feature_key, {}).get("present"):
            continue

        feature_data = rhythm_flags[feature_key]

        # Special handling for features with custom descriptions
        if feature_key == "normal_sinus_rhythm":
            hr = feature_data.get("heart_rate")
            desc = (f"Normal sinus rhythm at {hr:.0f} bpm"
                    if hr else "Normal sinus rhythm")
            findings.append(desc)

        elif feature_key == "sinus_bradycardia":
            hr = feature_data.get("heart_rate")
            severity = feature_data.get("severity", "")
            desc = (f"Sinus bradycardia at {hr:.0f} bpm"
                    if hr else "Sinus bradycardia")
            if severity:
                desc += f" ({severity})"
            findings.append(desc)

        elif feature_key == "sinus_tachycardia":
            hr = feature_data.get("heart_rate")
            desc = (f"Sinus tachycardia at {hr:.0f} bpm"
                    if hr else "Sinus tachycardia")
            findings.append(desc)

        elif feature_key == "irregular_rhythm":
            pattern = feature_data.get("pattern", "irregular")
            cv = feature_data.get("rr_variation")
            desc = f"Irregular rhythm ({pattern})"
            if cv:
                desc += f", CV={cv:.3f}"
            findings.append(desc)

        elif feature_key == "ventricular_premature_beats":
            count = feature_data.get("count")
            burden = feature_data.get("burden_percent")
            if count and burden is not None:
                desc = (f"Ventricular premature beats: "
                        f"{count} PVCs ({burden:.1f}% burden)")
            elif count:
                desc = f"Ventricular premature beats: {count} PVCs"
            else:
                desc = "Ventricular premature beats detected"
            findings.append(desc)

        elif feature_key == "extrasystoles":
            desc = feature_data.get("description", "Extrasystoles present")
            findings.append(desc)

        elif feature_key == "long_pause":
            pause_ms = feature_data.get("pause_duration")
            severity = feature_data.get("severity", "")
            desc = (f"Long pause ({pause_ms:.0f}ms)"
                    if pause_ms else "Long pause detected")
            if severity:
                desc += f" - {severity}"
            findings.append(desc)

        elif feature_key == "atrial_f_wave":
            freq = feature_data.get("f_wave_freq_hz")
            desc = (f"Regular atrial activity at ~{freq:.1f} Hz"
                    if freq else "Regular atrial activity detected")
            findings.append(desc)

        elif feature_key == "flutter_like_rr_clustering":
            centers = feature_data.get("centers_ms", [])
            if centers:
                desc = (
                    "RR clustering suggestive of variable AV conduction "
                    f"(centers: {', '.join(str(int(c)) for c in centers[:3])} ms)"
                )
            else:
                desc = "RR clustering suggestive of variable AV conduction"
            findings.append(desc)

        elif feature_key == "hrv_metrics":
            rmssd = feature_data.get("rmssd")
            sdnn = feature_data.get("sdnn")
            if rmssd and sdnn:
                desc = f"HRV metrics: RMSSD={rmssd:.1f}ms, SDNN={sdnn:.1f}ms"
                findings.append(desc)

        elif feature_key == "rhythm_panorama":
            reg_frac = feature_data.get("regular_fraction")
            if reg_frac is not None:
                desc = f"Rhythm regularity: {int(reg_frac * 100)}% regular beats"
                findings.append(desc)

        else:
            desc = feature_data.get("description", default_desc)
            if desc:
                if is_critical:
                    critical.append(desc)
                else:
                    findings.append(desc)

    # Build final summary
    summary_parts: List[str] = []

    if critical:
        summary_parts.append("\u26a0\ufe0f CRITICAL: " + "; ".join(critical))

    if findings:
        summary_parts.append(
            "Rhythm analysis reveals: " + ". ".join(findings)
        )

    if not summary_parts:
        return "Rhythm analysis: no significant abnormalities detected"

    return " ".join(summary_parts)


def generate_global_clinical_summary(features: Dict[str, Any]) -> str:
    """Generate clinical summary from global features.

    Args:
        features: Dict of global feature dicts.

    Returns:
        Clinical summary string (may be empty).
    """
    if not features:
        return ""

    summaries: List[str] = []
    critical: List[str] = []
    overall_impression: Optional[str] = None

    for feature_name, feature_data in features.items():
        if isinstance(feature_data, dict) and feature_data.get("present"):
            if feature_name == "critical_findings":
                critical.append(feature_data.get("description", ""))
            elif feature_name == "overall_impression":
                overall_impression = feature_data.get("description", "")
            else:
                summaries.append(feature_data.get("description", ""))

    result = ""

    if critical:
        result = "\U0001f6a8 " + " ".join(critical) + "\n\n"

    if summaries:
        result += "Global findings: " + " ".join(summaries)

    if overall_impression:
        if result:
            result += "\n\n"
        result += overall_impression

    return result


def generate_p_wave_clinical_summary(features: Dict[str, Any]) -> str:
    """Generate clinical summary from P-wave features.

    Args:
        features: Dict of P-wave feature dicts.

    Returns:
        Clinical summary string.
    """
    if not features:
        return "P-wave morphology within normal limits."

    descriptions: List[str] = []
    urgent: List[str] = []

    priority_features = [
        "atrial_fibrillation",
        "atrial_f_wave",
        "second_degree_av_block",
        "multifocal_atrial_tachycardia",
        "sinus_arrest",
        "complete_heart_block",
    ]

    for feature_name, feature_data in features.items():
        if isinstance(feature_data, dict) and feature_data.get("present"):
            desc = feature_data.get("description", "")
            if (feature_name in priority_features
                    or any(word in desc.upper()
                           for word in ["WARNING", "HIGH", "URGENT",
                                        "CRITICAL"])):
                urgent.append(desc)
            else:
                descriptions.append(desc)

    summary = "P-wave analysis reveals: "

    if urgent:
        summary += "URGENT FINDINGS: " + " ".join(urgent)
        if descriptions:
            summary += " Additional findings: "

    if descriptions:
        summary += " ".join(descriptions)

    return summary


def generate_t_wave_clinical_summary(features: Dict[str, Any]) -> str:
    """Generate clinical summary from T-wave features.

    Args:
        features: Dict of T-wave feature dicts.

    Returns:
        Clinical summary string.
    """
    if not features:
        return "T-wave morphology within normal limits."

    descriptions: List[str] = []
    warnings: List[str] = []

    for feature_name, feature_data in features.items():
        if isinstance(feature_data, dict) and feature_data.get("present"):
            desc = feature_data.get("description", "")
            if "WARNING" in desc or "HIGH RISK" in desc or "CRITICAL" in desc:
                warnings.append(desc)
            else:
                descriptions.append(desc)

    summary = "T-wave analysis reveals: "

    if warnings:
        summary += " ".join(warnings)
        if descriptions:
            summary += " Additionally: "

    if descriptions:
        summary += " ".join(descriptions)

    return summary


# ---------------------------------------------------------------------------
# Tier-A compact summary builder (Cell 17)
# ---------------------------------------------------------------------------

def build_clinical_summaries_tierA(
    semantic_flags: Dict[str, Any],
    computed_params: Dict[str, Any],
    poor_quality_obj: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Return compact clinical summaries (Tier A).

    Combines all _short_* helpers to produce a 6-key dict of one-liner
    clinical summaries suitable for embedding in the final JSONL record.

    Args:
        semantic_flags: Full structured features dict with domain keys
                        (T, P, QRS, ST, rhythm, global).
        computed_params: Computed parameters dict (heart rate, RR, etc.).
        poor_quality_obj: Optional poor-quality feature dict.

    Returns:
        Dict with keys: T_analysis, P_analysis, QRS_analysis,
        ST_analysis, rhythm_analysis, global_analysis.
    """
    poor = bool(poor_quality_obj and poor_quality_obj.get("present"))
    T = _short_T(semantic_flags.get("T", {}))
    P = _short_P(semantic_flags.get("P", {}))
    QRS = _short_QRS(semantic_flags.get("QRS", {}))
    ST = _short_ST(semantic_flags.get("ST", {}), poor_quality=poor)
    RHY = _short_rhythm(semantic_flags.get("rhythm", {}))
    GLO = _short_global(
        semantic_flags.get("global", {}), computed_params, poor_quality=poor
    )
    return {
        "T_analysis": T,
        "P_analysis": P,
        "QRS_analysis": QRS,
        "ST_analysis": ST,
        "rhythm_analysis": RHY,
        "global_analysis": GLO,
    }
