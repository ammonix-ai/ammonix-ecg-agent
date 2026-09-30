"""
Feature extraction registry -- domain-ordered semantic feature extraction.

Source: Cell 15 of q_psi_ai_for_ecg_Feb14_Adele.ipynb (lines 454-705)

Key functions:
    extract_semantic_features(): Runs all registered extractors in EXTRACTION_ORDER
    postprocess_semantic_flags(): Post-pass harmonization (AF gating, QRS descriptors)

Dependencies (all Layer 0-3):
    features/t_wave: T_WAVE_FEATURES (21 extractors)
    features/p_wave: P_WAVE_FEATURES (20 extractors)
    features/qrs_complex: QRS_FEATURES (21 extractors), inject_qrs_descriptors
    features/rhythm: RHYTHM_FEATURES (21 extractors), rhythm_absolute_arrhythmia, _rr_features
    features/global_features: GLOBAL_FEATURES (10 extractors)
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, Dict, List, Optional, Set

from qpsi.constants import logger
from qpsi.features.context import FeatureContext
from qpsi.features.global_features import GLOBAL_FEATURES
from qpsi.features.p_wave import P_WAVE_FEATURES
from qpsi.features.qrs_complex import QRS_FEATURES, inject_qrs_descriptors
from qpsi.features.rhythm import RHYTHM_FEATURES, _rr_features, rhythm_absolute_arrhythmia
from qpsi.features.t_wave import T_WAVE_FEATURES

# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

DOMAIN_EXTRACTORS: Dict[str, List[Callable]] = {
    "T": T_WAVE_FEATURES,
    "P": P_WAVE_FEATURES,
    "QRS": QRS_FEATURES,
    "rhythm": RHYTHM_FEATURES,
    "global": GLOBAL_FEATURES,
}

# Gotcha #17: explicit ordering so iteration is deterministic.
EXTRACTION_ORDER: List[str] = ["T", "P", "QRS", "rhythm", "global"]


# ---------------------------------------------------------------------------
# Main extraction entry-point
# ---------------------------------------------------------------------------

def extract_semantic_features(
    ctx: FeatureContext,
    *,
    record_json: Optional[dict] = None,
    exclude_domains: Optional[Set[str]] = None,
    debug: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Run every registered extractor in :data:`EXTRACTION_ORDER`.

    Parameters
    ----------
    ctx:
        Populated :class:`FeatureContext` with lead data, segments, etc.
    record_json:
        Unused -- retained for backward compatibility with older callers.
    exclude_domains:
        Domain names to skip (e.g. ``{"P"}`` to omit P-wave features).
    debug:
        When *True* emit per-function logger.debug output.

    Returns
    -------
    dict
        ``{domain: {feature_name: value, ...}, ...}`` for every domain
        that produced at least one non-empty result.
    """
    exclude: Set[str] = exclude_domains or set()
    semantic_flags: Dict[str, Dict[str, Any]] = {}

    for domain in EXTRACTION_ORDER:
        extractor_list = DOMAIN_EXTRACTORS.get(domain)
        if extractor_list is None:
            continue

        if domain in exclude:
            if debug:
                logger.debug("Skipping domain '%s' (excluded)", domain)
            continue

        out_dom: Dict[str, Any] = {}
        if debug:
            logger.debug("Processing domain: %s", domain)

        for fn in extractor_list:
            try:
                result = fn(ctx)
                if debug:
                    if result:
                        logger.debug("    %s -> %s", fn.__name__, result)
                    else:
                        logger.debug("    %s -> (empty)", fn.__name__)
                if result:
                    out_dom.update(result)
            except Exception as exc:
                logger.warning("Error in %s: %s", fn.__name__, exc)
                if debug:
                    logger.debug(traceback.format_exc())
                continue

        if out_dom:
            semantic_flags[domain] = out_dom
            if debug:
                logger.debug(
                    "  Added %d %s-domain features", len(out_dom), domain
                )

    if debug:
        logger.debug("Summary of extracted domains:")
        for key, val in semantic_flags.items():
            logger.debug("  %s: %d features", key, len(val))

    return semantic_flags


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def postprocess_semantic_flags(
    semantic_flags: Optional[Dict[str, Any]],
    ctx: FeatureContext,
) -> Dict[str, Any]:
    """Harmonize the feature set after initial extraction.

    Steps:
        1. Add ``absolute_arrhythmia`` if present.
        2. Drop P-domain findings when absolute arrhythmia is detected.
        3. Drop T-P fusion when no stable P.
        4. Remove ``reduced_hrv`` under irregular rhythms.
        5. Add QRS morphology descriptors.
    """
    sf: Dict[str, Any] = dict(semantic_flags or {})
    sf.setdefault("rhythm", {})
    sf.setdefault("P", {})
    sf.setdefault("T", {})

    # 1) Absolute arrhythmia detection
    aa = rhythm_absolute_arrhythmia(ctx)
    if aa:
        sf["rhythm"].update(aa)

    # 2) Remove reduced-HRV when rhythm is irregular (meaningless label)
    rr = getattr(ctx, "rr_intervals", []) or []
    feats = _rr_features(rr)
    irregular_now = (feats.get("cv") or 0.0) > 0.12
    if "reduced_hrv" in sf["rhythm"] and (aa or irregular_now):
        sf["rhythm"].pop("reduced_hrv", None)

    # 3) Gate P and T items when no stable P
    have_aa = bool(
        sf["rhythm"].get("absolute_arrhythmia", {}).get("present")
    )
    if have_aa:
        sf["P"].clear()
        sf["T"].pop("t_wave_fusion", None)
        if "overall_impression" in sf.get("global", {}):
            comp = sf["global"]["overall_impression"].get("components", [])
            sf["global"]["overall_impression"]["components"] = [
                c for c in comp if "regular rhythm" not in c.lower()
            ]

    # 4) QRS morphology descriptors + neutralize wording
    sf = inject_qrs_descriptors(sf, getattr(ctx, "lead_fits", {}) or {})

    return sf
