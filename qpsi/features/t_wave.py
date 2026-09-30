"""
T-wave feature extractors — 21 functions detecting morphological T-wave abnormalities.

Source: Cell 9 of q_psi_ai_for_ecg_Feb14_Adele.ipynb (lines 154-599).
Each extractor receives a FeatureContext and returns a dict (empty if feature absent).

Functions:
    t_wave_alternans            — beat-to-beat amplitude variation >15%
    t_wave_inverted             — dominant T peak <= -0.20 mV
    t_wave_flattened            — T peak < max(0.08, 0.12*QRS)
    t_wave_notched              — two same-polarity peaks with valley >=30%
    t_wave_biphasic             — positive+negative lobes, ratio >=30%, sep >=35 ms
    t_wave_prolonged_duration   — T duration >360 ms
    t_wave_axis_wandering       — T-axis SD >45 degrees across beats
    t_wave_hyperacute           — T >1.2 mV with sigma <50 ms in precordial leads
    t_wave_hidden               — T onset encroaches QRS (RR <=800 ms)
    t_wave_lead_discordance     — T/QRS opposite polarity (|T|>=0.20, |QRS|>=0.7)
    t_wave_fusion               — T extends >65% of RR
    t_wave_u_on_t               — late low-amplitude bump (sep >100 ms, ratio <0.4)
    t_wave_abrupt_offset        — sigma <20 ms (sharp termination)
    t_wave_prolonged_isoelectric_st — isoelectric ST >200 ms in V2-V5
    t_wave_peaked               — T >0.5 mV with sigma <40 ms
    t_wave_asymmetric           — sigma ratio >3:1 between Gaussian components
    t_wave_repol_dispersion     — T-peak timing SD >50 ms across leads
    t_wave_early_termination    — T duration <12% of RR
    t_wave_split                — two lobes >=0.12 mV, sep >=50 ms
    t_wave_plateau              — two similarly-sized broad components
    t_wave_pseudo_normalization — previously inverted T now upright in V1-V3
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from qpsi.features.context import FeatureContext, _f
from qpsi.features.helpers import (
    format_lead_list,
    _contiguous_groups,
    _keep_significant_groups,
    get_t_wave_gaussians,
    get_qrs_gaussians,
    _t_amp_and_duration,
    _sum_wave,
)


# ============================================================================
# T-WAVE FEATURE FUNCTIONS (original set, thresholds adjusted where agreed)
# ============================================================================

# 1. Alternans (variation>15%, beats>=8)
# F3 fix: lowered from 15 to 8 — standard 10s ECG at 75bpm yields ~12 beats,
# so 15 was excluding most recordings.
def t_wave_alternans(ctx: FeatureContext) -> Dict[str, Any]:
    alt = {}
    for lead in ["II", "V1", "V2", "V3", "V4", "V5"]:
        beats = ctx.lead_fits.get(lead, {}).get("by_beat", [])
        if len(beats) < 8:
            continue
        amps = []
        for i in range(len(beats)):
            ts = get_t_wave_gaussians(ctx.lead_fits, lead, i)
            if ts:
                amps.append(sum(abs(_f(g["amp_mv"])) for g in ts))
        if len(amps) < 8:
            continue
        diffs = [amps[i]-amps[i-1] for i in range(1,len(amps))]
        sign_changes = sum(1 for i in range(1,len(diffs)) if np.sign(diffs[i]) != np.sign(diffs[i-1]))
        if sign_changes >= len(diffs)*0.7:
            var = float(np.std(amps)/max(1e-9, np.mean(amps))*100.0)
            if var > 15.0:
                alt[lead] = var
    if not alt: return {}
    leads = list(alt.keys())
    mv = max(alt.values())
    return {"t_wave_alternans": {"present": True,
            "description": f"T-wave alternans in {format_lead_list(leads)} (max variation {mv:.1f}%)"}}

# 2. Inverted (unchanged)
def t_wave_inverted(ctx: FeatureContext) -> Dict[str, Any]:
    inverted = []
    for lead in ctx.leads:
        if lead == "aVR" or lead not in ctx.lead_fits: continue
        ts = get_t_wave_gaussians(ctx.lead_fits, lead)
        if not ts: continue
        dom = max(ts, key=lambda g: abs(_f(g.get("amp_mv", 0.0))))
        if _f(dom.get("amp_mv", 0.0)) <= -0.20:
            inverted.append(lead)
    groups = _keep_significant_groups(inverted, 2)
    if not groups: return {}
    best = max(groups, key=len)
    return {"t_wave_inverted": {"present": True,
            "description": f"T-wave inversion in {format_lead_list(best)} (dominant peak \u2264 \u22120.20 mV)",
            "affected_leads": best}}

# 3. Flattened (unchanged)
def t_wave_flattened(ctx: FeatureContext) -> Dict[str, Any]:
    flats, lf = [], ctx.lead_fits
    for lead in ctx.leads:
        if lead not in lf: continue
        ts = get_t_wave_gaussians(lf, lead)
        if not ts: continue
        t_peak = max(abs(_f(g.get("amp_mv",0.0))) for g in ts)
        qrs = get_qrs_gaussians(lf, lead)
        if qrs:
            qrs_peak = max(abs(_f(g.get("amp_mv",0.0))) for g in qrs)
            thresh = max(0.08, 0.12*qrs_peak)
        else:
            thresh = 0.08
        if t_peak < thresh:
            flats.append(lead)
    groups = _keep_significant_groups(flats, 3)
    if not groups: return {}
    best = max(groups, key=len)
    return {"t_wave_flattened": {"present": True,
            "description": f"Flattened T-waves in {format_lead_list(best)} (T < max(0.08 mV, 0.12\u00d7QRS))",
            "affected_leads": best}}

# 4. Notched (unchanged)
def t_wave_notched(ctx: FeatureContext) -> Dict[str, Any]:
    ok = []
    for lead in ctx.leads:
        ts = get_t_wave_gaussians(ctx.lead_fits, lead)
        if len(ts) < 2: continue
        top = sorted(ts, key=lambda g: abs(_f(g.get("amp_mv",0.0))), reverse=True)[:3]
        hit = False
        for i in range(len(top)):
            for j in range(i+1,len(top)):
                g1, g2 = top[i], top[j]
                a1, a2 = _f(g1["amp_mv"]), _f(g2["amp_mv"])
                if a1*a2 <= 0 or abs(a1) < 0.10 or abs(a2) < 0.10: continue
                sep = abs(_f(g1["center_ms"]) - _f(g2["center_ms"]))
                if sep < 40.0: continue
                t, y = _sum_wave(ts)
                if t is None: continue
                lo, hi = sorted([_f(g1["center_ms"]), _f(g2["center_ms"])])
                msk = (t >= lo) & (t <= hi)
                if not np.any(msk): continue
                valley = float(np.min(y[msk]))
                crest = max(abs(a1), abs(a2))
                drop = 1.0 - (abs(valley)/max(1e-6, crest))
                if drop >= 0.30:
                    hit = True; break
            if hit: break
        if hit: ok.append(lead)
    groups = _keep_significant_groups(ok, 2)
    if not groups: return {}
    best = max(groups, key=len)
    return {"t_wave_notched": {"present": True,
            "description": f"Notched T-waves in {format_lead_list(best)} (peaks \u22650.10 mV, sep \u226540 ms, valley \u226530%)",
            "affected_leads": best}}

# 5. Biphasic (unchanged)
def t_wave_biphasic(ctx: FeatureContext) -> Dict[str, Any]:
    patterns, cand = {}, []
    for lead in ctx.leads:
        if lead == "aVR": continue
        ts = get_t_wave_gaussians(ctx.lead_fits, lead)
        if len(ts) < 2: continue
        pos = [g for g in ts if _f(g.get("amp_mv",0.0)) > 0]
        neg = [g for g in ts if _f(g.get("amp_mv",0.0)) < 0]
        if not pos or not neg: continue
        dom = max(ts, key=lambda g: abs(_f(g.get("amp_mv",0.0))))
        min_gate = max(0.15, 0.30*abs(_f(dom.get("amp_mv",0.0))))
        pmax_g = max(pos, key=lambda g: abs(_f(g["amp_mv"])))
        nmax_g = max(neg, key=lambda g: abs(_f(g["amp_mv"])))
        pmax, nmax = abs(_f(pmax_g["amp_mv"])), abs(_f(nmax_g["amp_mv"]))
        if pmax < min_gate or nmax < min_gate: continue
        if min(pmax,nmax)/max(pmax,nmax) < 0.30: continue
        if abs(_f(pmax_g["center_ms"]) - _f(nmax_g["center_ms"])) < 35.0: continue
        first = pmax_g if _f(pmax_g["center_ms"]) < _f(nmax_g["center_ms"]) else nmax_g
        patterns[lead] = "positive" if _f(first["amp_mv"]) > 0 else "negative"
        cand.append(lead)
    groups = _keep_significant_groups(cand, 2)
    if not groups: return {}
    best = max(groups, key=len)
    desc = f"Biphasic T-waves in {format_lead_list(best)} (lobe ratio \u226530%, sep \u226535 ms)"
    if any(l in ("V2","V3") for l in best):
        desc += ", possible Wellens-type pattern if clinical context fits"
    return {"t_wave_biphasic": {"present": True, "description": desc,
            "pattern": {k:v for k,v in patterns.items() if k in best}}}

# 6. Prolonged duration (tighten: >360 ms)
def t_wave_prolonged_duration(ctx: FeatureContext) -> Dict[str, Any]:
    suspects = []
    for lead in ctx.leads:
        ts = get_t_wave_gaussians(ctx.lead_fits, lead)
        if not ts: continue
        peak_abs, t_start, t_end, dur = _t_amp_and_duration(ts)
        if dur is not None and _f(dur) > 360.0 and peak_abs >= 0.10:
            suspects.append(lead)
    groups = _keep_significant_groups(suspects, 2)
    if not groups: return {}
    flat = sorted({L for g in groups for L in g})
    max_dur = max(_t_amp_and_duration(get_t_wave_gaussians(ctx.lead_fits, L))[3] for L in flat)
    return {"t_wave_prolonged": {"present": True,
            "description": f"Prolonged T-wave duration in {format_lead_list(flat)} (up to {float(max_dur):.0f} ms)"}}

# 7. Axis wandering (unchanged)
def t_wave_axis_wandering(ctx: FeatureContext) -> Dict[str, Any]:
    axes = []
    by_beat_I = ctx.lead_fits.get("I", {}).get("by_beat", [])
    for i in range(min(20, len(by_beat_I))):
        t_I = get_t_wave_gaussians(ctx.lead_fits, "I", i)
        t_aVF = get_t_wave_gaussians(ctx.lead_fits, "aVF", i)
        if t_I and t_aVF:
            amp_I = sum(_f(g["amp_mv"]) for g in t_I)
            amp_aVF = sum(_f(g["amp_mv"]) for g in t_aVF)
            axis = float(np.degrees(np.arctan2(amp_aVF, amp_I)))
            axes.append(axis)
    if len(axes) >= 5:
        axis_var = float(np.std(axes))
        if axis_var > 45.0:
            return {"t_wave_axis_wandering": {"present": True,
                    "description": f"T-wave axis instability ({axis_var:.0f}\u00b0 SD)" }}
    return {}

# 8. Hyperacute (unchanged)
def t_wave_hyperacute(ctx: FeatureContext) -> Dict[str, Any]:
    leads = []
    for L in ctx.leads:
        ts = get_t_wave_gaussians(ctx.lead_fits, L)
        if not ts: continue
        for g in ts:
            if _f(g.get("amp_mv",0.0)) > 1.2 and _f(g.get("sigma_ms",0.0)) < 50:
                leads.append(L); break
    prec = [L for L in leads if L in ["V1","V2","V3","V4","V5","V6"]]
    if len(prec) >= 2:
        max_amp = 0.0
        for L in prec:
            for g in get_t_wave_gaussians(ctx.lead_fits, L):
                max_amp = max(max_amp, _f(g.get("amp_mv",0.0)))
        return {"t_wave_hyperacute": {"present": True,
                "description": f"Hyperacute T-waves in {format_lead_list(prec)} (max {max_amp:.2f} mV)",
                "max_amplitude": max_amp}}
    return {}

# 9. Hidden (unchanged)
def t_wave_hidden(ctx: FeatureContext) -> Dict[str, Any]:
    rr = ctx.rr_intervals
    mean_rr = float(np.mean(rr)) if isinstance(rr, (list,tuple,np.ndarray)) and len(rr) else None
    if mean_rr is None or mean_rr > 800.0: return {}
    ok = []
    for L in ctx.leads:
        ts = get_t_wave_gaussians(ctx.lead_fits, L)
        if not ts: continue
        t_peak = max(abs(_f(g.get("amp_mv",0.0))) for g in ts)
        if t_peak < 0.10: continue
        t_start = min(_f(g["center_ms"]) - 2.0*max(1.0,_f(g.get("sigma_ms",1.0))) for g in ts)
        qrs = get_qrs_gaussians(ctx.lead_fits, L)
        if not qrs: continue
        q_end = max(_f(g["center_ms"]) + 2.0*max(1.0,_f(g.get("sigma_ms",1.0))) for g in qrs)
        if t_start < q_end + 10.0:
            ok.append(L)
    if len(ok) < 3: return {}
    return {"t_wave_hidden": {"present": True,
            "description": f"T onset encroaches on QRS in {format_lead_list(ok)} (tachycardia, RR\u2264800 ms)",
            "affected_leads": ok}}

# 10. Lead discordance (tighten T>=0.20, QRS>=0.7)
def t_wave_lead_discordance(ctx: FeatureContext) -> Dict[str, Any]:
    cand = []
    for L in ctx.leads:
        if L == "aVR" or L not in ctx.lead_fits: continue
        ts = get_t_wave_gaussians(ctx.lead_fits, L)
        qrs = get_qrs_gaussians(ctx.lead_fits, L)
        if not ts or not qrs: continue
        t_dom = max(ts, key=lambda g: abs(_f(g.get("amp_mv",0.0))))
        q_dom = max(qrs, key=lambda g: abs(_f(g.get("amp_mv",0.0))))
        if abs(_f(t_dom["amp_mv"])) < 0.20 or abs(_f(q_dom["amp_mv"])) < 0.7:  # tightened
            continue
        if (_f(t_dom["amp_mv"]) > 0) != (_f(q_dom["amp_mv"]) > 0):
            cand.append(L)
    groups = _keep_significant_groups(cand, 2)
    if not groups: return {}
    best = max(groups, key=len)
    return {"t_wave_lead_discordance": {"present": True,
            "description": f"T/QRS discordance in {format_lead_list(best)} (|T|\u22650.20 mV & |QRS|\u22650.7 mV)",
            "affected_leads": best}}

# 11. Fusion (tighten: >0.65*RR)
def t_wave_fusion(ctx: FeatureContext) -> Dict[str, Any]:
    if not ctx.rr_intervals: return {}
    mean_rr = float(np.mean(ctx.rr_intervals))
    for L in ctx.leads:
        ts = get_t_wave_gaussians(ctx.lead_fits, L)
        if not ts: continue
        latest = max(_f(g["center_ms"]) + 2.0*max(1.0,_f(g.get("sigma_ms",1.0))) for g in ts)
        if latest > 0.65 * mean_rr:
            return {"t_wave_fusion": {"present": True,
                    "description": "T-wave fusion with subsequent P (T extends >65% of RR)"}}
    return {}

# 12. U-on-T (tighten: separation>100 ms, ratio<0.4) — and require >=2 contiguous leads
def t_wave_u_on_t(ctx: FeatureContext) -> Dict[str, Any]:
    ok = []
    for L in ctx.leads:
        ts = get_t_wave_gaussians(ctx.lead_fits, L)
        if len(ts) < 2: continue
        s = sorted(ts, key=lambda x: _f(x["center_ms"]))
        main_t, late = s[0], s[-1]
        sep = _f(late["center_ms"]) - _f(main_t["center_ms"])
        ratio = abs(_f(late["amp_mv"]))/max(1e-9, abs(_f(main_t["amp_mv"])))
        if sep > 100.0 and ratio < 0.4:
            ok.append(L)
    groups = _keep_significant_groups(ok, 2)
    if not groups: return {}
    best = max(groups, key=len)
    return {"t_wave_u_on_t": {"present": True,
            "description": f"U-on-T in {format_lead_list(best)}"}}

# 13. Abrupt offset (unchanged)
def t_wave_abrupt_offset(ctx: FeatureContext) -> Dict[str, Any]:
    leads = []
    for L in ctx.leads:
        ts = get_t_wave_gaussians(ctx.lead_fits, L)
        if not ts: continue
        for g in ts:
            if _f(g.get("sigma_ms", 0.0)) < 20.0:
                leads.append(L); break
    if not leads: return {}
    return {"t_wave_abrupt_offset": {"present": True,
            "description": f"Abrupt T-wave termination in {format_lead_list(leads)}"}}

# 14. Prolonged isoelectric ST (unchanged)
def t_wave_prolonged_isoelectric_st(ctx: FeatureContext) -> Dict[str, Any]:
    try:
        mean_rr = float(np.mean(ctx.rr_intervals)) if ctx.rr_intervals else None
    except Exception:
        mean_rr = None
    if mean_rr is not None and mean_rr < 800.0: return {}
    suspects = []
    for L in ctx.leads:
        ts = get_t_wave_gaussians(ctx.lead_fits, L)
        qrs = get_qrs_gaussians(ctx.lead_fits, L)
        if not (ts and qrs): continue
        peak_abs, t_start, _, _ = _t_amp_and_duration(ts)
        q_end = max(_f(g["center_ms"]) + _f(g.get("sigma_ms", 1.0)) for g in qrs)
        st_flat = (t_start - q_end) if (t_start is not None) else 0.0
        if st_flat > 200.0 and peak_abs >= 0.10 and L in {"V2","V3","V4","V5"}:
            suspects.append(L)
    groups = _keep_significant_groups(suspects, 2)
    if not groups: return {}
    flat = sorted({L for g in groups for L in g})
    return {"prolonged_st_segment": {"present": True,
            "description": f"Prolonged isoelectric ST in {format_lead_list(flat)}"}}

# 15. Peaked (unchanged)
def t_wave_peaked(ctx: FeatureContext) -> Dict[str, Any]:
    leads = []
    for L in ctx.leads:
        ts = get_t_wave_gaussians(ctx.lead_fits, L)
        if not ts: continue
        for g in ts:
            if _f(g.get("amp_mv",0.0)) > 0.5 and _f(g.get("sigma_ms",0.0)) < 40.0:
                leads.append(L); break
    if not leads: return {}
    return {"t_wave_peaked": {"present": True,
            "description": f"Peaked T-waves in {format_lead_list(leads)}"}}

# 16. Asymmetric (tighten: width ratio > 3:1)
def t_wave_asymmetric(ctx: FeatureContext) -> Dict[str, Any]:
    asym = []
    for L in ctx.leads:
        ts = get_t_wave_gaussians(ctx.lead_fits, L)
        if len(ts) >= 2:
            sig = [max(1.0,_f(g.get("sigma_ms",1.0))) for g in ts]
            if max(sig)/min(sig) > 3.0:
                asym.append(L)
    groups = _keep_significant_groups(asym, 2)
    if not groups: return {}
    best = max(groups, key=len)
    return {"t_wave_asymmetric": {"present": True,
            "description": f"Asymmetric T-waves in {format_lead_list(best)}"}}

# 17. Repolarization dispersion (unchanged)
def t_wave_repol_dispersion(ctx: FeatureContext) -> Dict[str, Any]:
    t_peaks = {}
    for L in ctx.leads:
        ts = get_t_wave_gaussians(ctx.lead_fits, L)
        if ts:
            dom = max(ts, key=lambda x: abs(_f(x.get("amp_mv",0.0))))
            t_peaks[L] = _f(dom.get("center_ms", 0.0))
    if len(t_peaks) >= 6:
        disp = float(np.std(list(t_peaks.values())))
        if disp > 50.0:
            return {"t_wave_dispersion": {"present": True,
                    "description": f"Increased T-wave dispersion ({disp:.0f} ms)"}}
    return {}

# 18. Early termination (unchanged)
def t_wave_early_termination(ctx: FeatureContext) -> Dict[str, Any]:
    if not ctx.rr_intervals: return {}
    mean_rr = float(np.mean(ctx.rr_intervals))
    suspects = []
    for L in ctx.leads:
        if L == "aVR": continue
        ts = get_t_wave_gaussians(ctx.lead_fits, L)
        if not ts: continue
        peak_abs, _, _, dur = _t_amp_and_duration(ts)
        if dur < 0.12*mean_rr and peak_abs >= 0.10 and L != "V1":
            suspects.append(L)
    groups = _keep_significant_groups(suspects, 2)
    if not groups: return {}
    flat = sorted({L for g in groups for L in g})
    return {"t_wave_early_termination": {"present": True,
            "description": f"Abbreviated T-wave duration in {format_lead_list(flat)}"}}

# 19. Split (tighten: each >=0.12 mV, separation >=50 ms)
def t_wave_split(ctx: FeatureContext) -> Dict[str, Any]:
    ok = []
    for L in ctx.leads:
        ts = get_t_wave_gaussians(ctx.lead_fits, L)
        if len(ts) < 2: continue
        top2 = sorted(ts, key=lambda g: abs(_f(g.get("amp_mv",0.0))), reverse=True)[:2]
        if min(abs(_f(top2[0]["amp_mv"])), abs(_f(top2[1]["amp_mv"]))) < 0.12: continue
        if abs(_f(top2[0]["center_ms"]) - _f(top2[1]["center_ms"])) < 50.0: continue
        ok.append(L)
    groups = _keep_significant_groups(ok, 2)
    if not groups: return {}
    best = max(groups, key=len)
    return {"t_wave_split": {"present": True,
            "description": f"Split T-waves in {format_lead_list(best)} (lobes \u22650.12 mV, sep \u226550 ms)",
            "affected_leads": best}}

# 20. Plateau (unchanged)
def t_wave_plateau(ctx: FeatureContext) -> Dict[str, Any]:
    leads_ok = []
    for L in ctx.leads:
        ts = get_t_wave_gaussians(ctx.lead_fits, L)
        if len(ts) < 2: continue
        top2 = sorted(ts, key=lambda g: abs(_f(g.get("amp_mv",0.0))), reverse=True)[:2]
        a1, a2 = abs(_f(top2[0]["amp_mv"])), abs(_f(top2[1]["amp_mv"]))
        s1, s2 = _f(top2[0].get("sigma_ms",0.0)), _f(top2[1].get("sigma_ms",0.0))
        if max(a1,a2) < 0.12: continue
        if abs(a1 - a2) >= 0.03: continue
        if s1 < 50.0 or s2 < 50.0: continue
        leads_ok.append(L)
    groups = _keep_significant_groups(leads_ok, 2)
    if not groups: return {}
    best = max(groups, key=len)
    return {"t_wave_plateau": {"present": True,
            "description": f"Plateau-like T-waves in {format_lead_list(best)}"}}

# 21. Pseudo-normalization (unchanged)
def t_wave_pseudo_normalization(ctx: FeatureContext) -> Dict[str, Any]:
    baseline = ctx.baseline_lead_fits
    if baseline is None: return {}
    changed = []
    for L in ["V1","V2","V3"]:
        if L not in ctx.lead_fits or L not in baseline: continue
        cur = get_t_wave_gaussians(ctx.lead_fits, L)
        prev = get_t_wave_gaussians(baseline, L)
        if not (cur and prev): continue
        neg_prev = sum(_f(g["amp_mv"]) for g in prev if _f(g["amp_mv"]) < 0)
        pos_prev = sum(_f(g["amp_mv"]) for g in prev if _f(g["amp_mv"]) > 0)
        neg_cur  = sum(_f(g["amp_mv"]) for g in cur  if _f(g["amp_mv"]) < 0)
        pos_cur  = sum(_f(g["amp_mv"]) for g in cur  if _f(g["amp_mv"]) > 0)
        peak_prev = max(abs(_f(g["amp_mv"])) for g in prev)
        peak_cur  = max(abs(_f(g["amp_mv"])) for g in cur)
        if (abs(neg_prev) > abs(pos_prev) and pos_cur > abs(neg_cur) and (peak_cur - peak_prev) >= 0.15):
            changed.append(L)
    if len(changed) >= 2:
        return {"t_wave_pseudo_normalization": {"present": True,
                "description": f"T-wave pseudo-normalization in {format_lead_list(sorted(changed))}"}}
    return {}


# ============================================================================
# REGISTRY OF ALL T-WAVE FEATURES (original set preserved)
# ============================================================================

T_WAVE_FEATURES = [
    t_wave_alternans,
    t_wave_inverted,
    t_wave_flattened,
    t_wave_notched,
    t_wave_biphasic,
    t_wave_prolonged_duration,
    t_wave_axis_wandering,
    t_wave_hyperacute,
    t_wave_hidden,
    t_wave_lead_discordance,
    t_wave_fusion,
    t_wave_u_on_t,
    t_wave_abrupt_offset,
    t_wave_prolonged_isoelectric_st,
    t_wave_peaked,
    t_wave_asymmetric,
    t_wave_repol_dispersion,
    t_wave_early_termination,
    t_wave_split,
    t_wave_plateau,
    t_wave_pseudo_normalization,
]
