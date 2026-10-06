"""Case assembly for the ECG Agent: everything the model is told, and nothing else.

One function, :func:`build_case`, runs the frozen pipeline for a recording and
packs the result into a :class:`CaseBrief`:

    signals.load_record  ->  inference.extract_features  ->  inference.classify
                         ->  inference.top_feature_contributions
                         ->  projection.place_point (+ universe row lookup)
                         ->  skills.match_skills

:func:`render_brief` turns that into the prompt text, and :data:`SYSTEM_PROMPT`
carries the framing that matters: **the agent verifies, it does not diagnose.**
The diagnosis is already made — by the frozen classifier, by the fixed universe
of 63,256 labelled neighbours, and by the domain skills. The model's job is to
hold those calls up against the image and say what the trace does and does not
support.

Nothing here fits anything. The classifier, the thresholds and the embedding are
read as shipped.

Cache: analysing a record costs ~1.4 s with the Rust kernels and ~21 s without,
on top of an 8 s one-time model load, so the last :data:`CACHE_SIZE` cases are
kept in memory. The classifier is frozen, so a cached case can never go stale.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from services import inference, projection, signals, skills as skills_mod
from services.universe_rows import UniverseRow, row_by_display_id, row_by_recording_id

logger = logging.getLogger(__name__)

CACHE_SIZE = 8
DEFAULT_NEIGHBORS = 6
TOP_FEATURE_COUNT = 10
#: How many of the leading calls get a SHAP feature breakdown in the brief.
TOP_FEATURE_DIAGNOSES = 3

#: Lead whose QRS beat count gives the rate estimate, with fallbacks.
RATE_LEADS = ("II", "V2", "I", "V5")
RATE_FEATURE = "ml_direct::{lead}::QRS::_beat_clusters::n_beats"


SYSTEM_PROMPT = """\
You are the ECG analysis system, speaking to a clinician about a recording this
system has already analysed. You speak as one system, in one voice.

The brief below is your OWN prior analysis of this recording: the diagnoses and
their probabilities against per-diagnosis thresholds, the waveform features that
drove them, the labelled recordings nearest to this one, and the domain rules
that matched. Treat all of it as settled and as yours.

VOICE — this matters as much as the content
- Never name or attribute internal components. Do not say "the classifier", "the
  model", "the universe", "the embedding", "the skills", "the system flagged",
  or refer to yourself in the third person. There is no committee here.
- Say "this ECG shows", "the analysis places this recording near…", "the finding
  rests on…". Own the analysis; do not report on it from outside.
- Weave the relevant findings from the brief into your answer as your own
  statements. Do not preface them as coming from somewhere else.

HOW TO ANSWER
- Lead with the conclusion. Give the answer first, then the evidence that
  supports it.
- Do NOT narrate your reasoning process, list what you checked lead by lead, or
  think out loud. Consider the trace carefully, then state what you found. The
  clinician wants the finding, not the search.
- Name the evidence concretely: lead, wave, interval, amplitude, morphology.
- Use the clinical language a cardiologist would use. Be concise.
- Answer the question actually asked. Do not restate the brief back to them.

WHAT YOU MAY AND MAY NOT CLAIM
- Skill checklists describe evidence to look for, not findings already observed
  in this recording. In particular, an "Against:" clause is conditional: do not
  repeat it as a patient finding unless the available evidence supports it.
- The diagnoses in the brief are settled. Your task is to explain them, and to
  check them against the trace — confirming, or reporting honestly where the
  trace does not support them.
- Do not overturn a diagnosis on your own authority or substitute an independent
  read. If the trace appears to contradict the analysis, say so plainly as a
  discrepancy worth review, and explain what you see.
- If you notice something the analysis did not raise, offer it as an
  observation worth checking — not as the diagnosis.
- Never invent measurements. Quote numbers only from the brief; anything read
  off the image is an estimate and must be described as one.
- Probabilities are meaningful only against each diagnosis's own threshold. 0.20
  against a 0.15 threshold is a positive call; 0.45 against a 0.54 threshold is
  not. Reason in ratio-to-threshold terms, never in bare probability.
- The nearest recordings belong to other patients. They are not this patient's
  history.
- Some labels cannot be read from a surface ECG at all. Say so rather than
  pretending to see them.
- If the evidence is not visible — poor baseline, missing lead, artefact — say
  that. "The trace does not show this" is a useful answer.
"""

IMAGE_NOTE = (
    "\nA rendered 12-lead ECG of this recording is attached to the clinician's "
    "message — all 12 leads, drawn from the signal.\n"
    "\nThe image may carry MARKS THE CLINICIAN DREW: a coloured ring on a "
    "waveform, sometimes with a short text label beside it, on the lead they "
    "were looking at. Those marks are the clinician pointing at something and "
    "asking about it — they are not part of the recording and not a finding of "
    "this analysis. When a mark is present, treat it as the subject of the "
    "question: say what the waveform is at that point on that lead, and how it "
    "bears on the findings. Never describe a mark as if it were part of the "
    "ECG signal, and do not read the clinician's own label back to them as "
    "though the trace showed it.\n"
    "\nRead the trace. Where what you see does not match the analysis, report "
    "the discrepancy plainly — do not smooth it over.\n"
)

NO_IMAGE_NOTE = (
    "\nNo ECG image is attached to this conversation. You can still verify the "
    "call against the numbers in the brief, but say plainly that you have not "
    "seen the trace, and do not describe waveform morphology as if you had.\n"
)


# ---------------------------------------------------------------------------
# feature names
# ---------------------------------------------------------------------------

_WAVE_NAMES = {"P": "P wave", "QRS": "QRS complex", "T": "T wave", "PQ": "PQ segment",
               "ST": "ST segment", "TP": "TP segment"}
_STAT_NAMES = {
    "mean": "mean", "median": "median", "std": "standard deviation", "mad": "median abs deviation",
    "iqr": "interquartile range", "min": "minimum", "max": "maximum", "rms": "RMS",
    "range": "range", "skew": "skewness", "kurtosis": "kurtosis", "cv": "coefficient of variation",
    "alternation_index": "beat-to-beat alternation", "n_beats": "beats detected",
    "trend_slope": "trend across beats",
}


def humanize_feature(name: str) -> str:
    """Readable gloss for a QPSI feature name; the raw name is always kept too.

    ``ml_direct::V4::T::extr_2_amp_mv::acf_lag_8``
        -> ``lead V4, T wave, extr_2_amp_mv, autocorrelation lag 8``
    """
    parts = name.split("::")
    if len(parts) < 4:
        return name
    track, where, wave, measure = parts[0], parts[1], parts[2], parts[3]
    stat = parts[4] if len(parts) > 4 else ""
    scope = f"lead {where}" if track == "ml_direct" else f"{where} plane"
    bits = [scope, _WAVE_NAMES.get(wave, wave), measure]
    if stat:
        if stat.startswith("acf_lag_"):
            bits.append(f"autocorrelation lag {stat.rsplit('_', 1)[-1]}")
        elif stat.startswith("alternation_index_p"):
            bits.append(f"alternation index (period {stat[-1]})")
        else:
            bits.append(_STAT_NAMES.get(stat, stat))
    return ", ".join(bits)


# ---------------------------------------------------------------------------
# the brief
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaseBrief:
    """Everything the agent is allowed to know about one recording."""

    recording_id: str
    display_id: str | None
    source: str | None
    sampling_rate: int | None
    duration_sec: float | None
    rate_bpm: float | None

    probabilities: dict[str, float]
    thresholds: dict[str, float]
    predictions: list[str]
    top_prediction: str | None
    max_score: float | None
    fold_probabilities: dict[str, list[float]]
    score_kind: str
    threshold_kind: str | None
    feature_count: int | None
    model_version: str | None

    top_features: dict[str, list[dict[str, Any]]]
    neighbors: list[dict[str, Any]]
    placement: dict[str, Any]
    skills: list[dict[str, Any]]
    stored: dict[str, Any] | None
    in_universe: bool
    provenance: str                       # "live" | "client-supplied"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordingId": self.recording_id,
            "displayId": self.display_id,
            "source": self.source,
            "samplingRate": self.sampling_rate,
            "durationSec": self.duration_sec,
            "rateBpmEstimate": self.rate_bpm,
            "scoreKind": self.score_kind,
            "thresholdKind": self.threshold_kind,
            "featureCount": self.feature_count,
            "modelVersion": self.model_version,
            "predictions": list(self.predictions),
            "topPrediction": self.top_prediction,
            "maxScore": self.max_score,
            "probabilities": dict(self.probabilities),
            "thresholds": dict(self.thresholds),
            "topFeatures": {k: list(v) for k, v in self.top_features.items()},
            "neighbors": list(self.neighbors),
            "projection": {
                k: self.placement.get(k) for k in ("pca", "umap", "tsne")
            },
            "tsnePlacement": self.placement.get("tsnePlacement"),
            "skills": list(self.skills),
            "stored": self.stored,
            "inUniverse": self.in_universe,
            "provenance": self.provenance,
            "notes": list(self.notes),
        }


_CACHE: "OrderedDict[str, CaseBrief]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def cached_case(recording_id: str) -> CaseBrief | None:
    with _CACHE_LOCK:
        brief = _CACHE.get(recording_id)
        if brief is not None:
            _CACHE.move_to_end(recording_id)
        return brief


def _store(brief: CaseBrief) -> CaseBrief:
    with _CACHE_LOCK:
        _CACHE[brief.recording_id] = brief
        _CACHE.move_to_end(brief.recording_id)
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return brief


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def cache_info() -> dict[str, Any]:
    with _CACHE_LOCK:
        return {"size": len(_CACHE), "capacity": CACHE_SIZE, "recordings": list(_CACHE)}


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def _rate_estimate(features: Mapping[str, float], duration_sec: float | None) -> float | None:
    """Ventricular rate from the QRS beat count over the recording window.

    The private sinus gate read an R-R interval feature; this feature space has
    none, so the rate comes from the beat detector's count. It is an estimate and
    the brief says so.
    """
    if not duration_sec or duration_sec <= 0:
        return None
    for lead in RATE_LEADS:
        n_beats = features.get(RATE_FEATURE.format(lead=lead))
        if n_beats is None:
            continue
        try:
            beats = float(n_beats)
        except (TypeError, ValueError):
            continue
        if beats <= 0:
            continue
        return beats / duration_sec * 60.0
    return None


def _enrich_neighbors(
    raw: Sequence[Mapping[str, Any]],
    *,
    exclude_display_id: str | None,
    k: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Attach each neighbour's reference labels; drop the record's own point.

    A record that is already in the universe is its own nearest neighbour at
    distance 0, which wastes a slot and tells the agent nothing.
    """
    out: list[dict[str, Any]] = []
    self_excluded = False
    for item in raw:
        display_id = str(item.get("displayId"))
        if exclude_display_id and display_id == exclude_display_id:
            self_excluded = True
            continue
        entry: dict[str, Any] = {
            "displayId": display_id,
            "distance": float(item.get("distance", 0.0)),
            "primary": item.get("primary"),
        }
        row: UniverseRow | None = row_by_display_id(display_id)
        if row is not None:
            entry.update({
                "labels": list(row.labels),
                "source": row.source,
                "storedTopPrediction": row.top_prediction,
                "storedMaxScore": row.max_score,
            })
            if row.xgb_correct is not None:
                # Set equality between the stored prediction set and the labels;
                # see UniverseRow.to_dict for why the name is spelled out.
                entry["storedPredictionSetExact"] = row.xgb_correct
        out.append(entry)
        if len(out) >= k:
            break
    return out, self_excluded


def analyze_record(
    recording_id: str,
    *,
    k: int = DEFAULT_NEIGHBORS,
    top_feature_diagnoses: int = TOP_FEATURE_DIAGNOSES,
) -> CaseBrief:
    """Run the frozen pipeline over a shipped recording. Blocking and CPU-bound.

    Call it from a worker thread (``asyncio.to_thread``) — it holds the GIL only
    while numpy/XGBoost are not in their own C loops, but it takes seconds.
    """
    cached = cached_case(recording_id)
    if cached is not None:
        return cached

    lead_signals = signals.load_record(recording_id)
    features = inference.extract_features(
        lead_signals.signal, lead_signals.sampling_rate, recording_id=recording_id,
    )
    result = inference.classify(features)
    probabilities: dict[str, float] = dict(result["probabilities"])
    thresholds: dict[str, float] = dict(result["thresholds"])

    # Explain the calls, not the whole ranking: a SHAP pass costs real time and a
    # below-threshold diagnosis has nothing to verify. With no call at all, the
    # single highest-scoring diagnosis is the only thing worth explaining.
    ranked = [dx for dx, _ in inference.rank_probabilities(probabilities, 1)]
    wanted = (list(result.get("predictions") or ()) or ranked)[: max(1, top_feature_diagnoses)]
    top_features: dict[str, list[dict[str, Any]]] = {}
    for dx in wanted:
        try:
            contributions = inference.top_feature_contributions(
                features, dx, k=TOP_FEATURE_COUNT,
            )
        except Exception as exc:  # a SHAP failure must not sink the chat
            logger.warning("Feature attribution failed for %r: %s", dx, exc)
            continue
        top_features[dx] = [
            {**item, "readable": humanize_feature(str(item.get("name", "")))}
            for item in contributions
        ]

    stored_row = row_by_recording_id(recording_id)
    placement = projection.place_point(probabilities, k=k + 1)
    neighbors, self_excluded = _enrich_neighbors(
        placement.get("neighbors", ()),
        exclude_display_id=stored_row.display_id if stored_row else None,
        k=k,
    )

    rate_bpm = _rate_estimate(features, lead_signals.duration_sec)
    matched = skills_mod.match_skills(probabilities, thresholds, rate_bpm=rate_bpm)

    notes: list[str] = []
    if self_excluded:
        notes.append(
            "This recording is itself one of the labelled reference recordings; its own "
            "point was dropped from the neighbour list."
        )
    if stored_row is not None:
        notes.append(
            "This is a shipped test recording. Most of them were in the training pool for "
            "at least some diagnoses, and the score is the mean of all five folds including "
            "the folds that trained on it — so it is not an out-of-sample number. The "
            "per-fold spread shows how much the folds disagree."
        )
    if placement.get("umapPlacement") == projection.KNN_APPROX:
        notes.append(str(placement.get("umapNote") or "UMAP placed by k-NN approximation."))

    brief = CaseBrief(
        recording_id=recording_id,
        display_id=(stored_row.display_id if stored_row else lead_signals.display_id),
        source=(stored_row.source if stored_row else lead_signals.source),
        sampling_rate=lead_signals.sampling_rate,
        duration_sec=lead_signals.duration_sec,
        rate_bpm=rate_bpm,
        probabilities=probabilities,
        thresholds=thresholds,
        predictions=list(result.get("predictions", [])),
        top_prediction=result.get("topPrediction"),
        max_score=result.get("maxScore"),
        fold_probabilities={k_: list(v) for k_, v in (result.get("foldProbabilities") or {}).items()},
        score_kind=str(result.get("scoreKind", inference.ENSEMBLE_SCORE_KIND)),
        threshold_kind=result.get("thresholdKind"),
        feature_count=result.get("featureCount"),
        model_version=result.get("modelVersion"),
        top_features=top_features,
        neighbors=neighbors,
        placement=placement,
        skills=[s.to_dict() for s in matched],
        stored=(
            {
                "topPrediction": stored_row.top_prediction,
                "maxScore": stored_row.max_score,
                "labels": list(stored_row.labels),
                "primary": stored_row.primary,
                "scoreKind": inference.STORED_SCORE_KIND,
            }
            if stored_row else None
        ),
        in_universe=stored_row is not None,
        provenance="live",
        notes=notes,
    )
    return _store(brief)


def brief_from_analysis(analysis: Mapping[str, Any], recording_id: str) -> CaseBrief:
    """Build a brief from a client-supplied ``POST /api/analyze`` body.

    Lets the chat reuse an analysis the page already ran instead of paying for it
    twice. The provenance is recorded so the agent's context says where the
    numbers came from; anything the client left out is simply absent.
    """
    probabilities = {str(k): float(v) for k, v in (analysis.get("probabilities") or {}).items()}
    thresholds = {str(k): float(v) for k, v in (analysis.get("thresholds") or {}).items()}
    if probabilities and not thresholds:
        try:
            thresholds = dict(inference.load_runtime().thresholds)
        except Exception:  # model package absent: fall back to what was sent
            thresholds = {}

    stored_row = row_by_recording_id(recording_id)
    raw_neighbors = analysis.get("neighbors") or []
    neighbors, self_excluded = _enrich_neighbors(
        raw_neighbors,
        exclude_display_id=stored_row.display_id if stored_row else None,
        k=DEFAULT_NEIGHBORS,
    )
    top_features_in = analysis.get("topFeatures") or []
    top_prediction = analysis.get("topPrediction") or (
        (analysis.get("predictions") or [None])[0]
    )
    top_features: dict[str, list[dict[str, Any]]] = {}
    if isinstance(top_features_in, list) and top_prediction:
        top_features[str(top_prediction)] = [
            {**item, "readable": humanize_feature(str(item.get("name", "")))}
            for item in top_features_in
            if isinstance(item, Mapping)
        ]
    elif isinstance(top_features_in, Mapping):
        top_features = {
            str(dx): [
                {**item, "readable": humanize_feature(str(item.get("name", "")))}
                for item in items if isinstance(item, Mapping)
            ]
            for dx, items in top_features_in.items()
        }

    matched = skills_mod.match_skills(probabilities, thresholds)
    notes = ["These scores came with the request; they were not recomputed for this conversation."]
    if self_excluded:
        notes.append("The recording's own reference point was dropped from the neighbour list.")

    return CaseBrief(
        recording_id=recording_id,
        display_id=(stored_row.display_id if stored_row else analysis.get("displayId")),
        source=(stored_row.source if stored_row else analysis.get("source")),
        sampling_rate=analysis.get("samplingRate"),
        duration_sec=analysis.get("durationSec"),
        rate_bpm=None,
        probabilities=probabilities,
        thresholds=thresholds,
        predictions=[str(p) for p in (analysis.get("predictions") or [])],
        top_prediction=(str(top_prediction) if top_prediction else None),
        max_score=analysis.get("maxScore"),
        fold_probabilities={
            str(k): [float(x) for x in v]
            for k, v in (analysis.get("foldProbabilities") or {}).items()
        },
        score_kind=str(analysis.get("scoreKind") or inference.ENSEMBLE_SCORE_KIND),
        threshold_kind=analysis.get("thresholdKind"),
        feature_count=analysis.get("featureCount"),
        model_version=analysis.get("modelVersion"),
        top_features=top_features,
        neighbors=neighbors,
        placement={
            "tsnePlacement": analysis.get("tsnePlacement"),
            **(analysis.get("projection") or {}),
        },
        skills=[s.to_dict() for s in matched],
        stored=(
            {
                "topPrediction": stored_row.top_prediction,
                "maxScore": stored_row.max_score,
                "labels": list(stored_row.labels),
                "primary": stored_row.primary,
                "scoreKind": inference.STORED_SCORE_KIND,
            }
            if stored_row else None
        ),
        in_universe=stored_row is not None,
        provenance="client-supplied",
        notes=notes,
    )


def build_case(
    recording_id: str,
    *,
    analysis: Mapping[str, Any] | None = None,
) -> CaseBrief:
    """A brief for this recording: from the client's analysis if it sent one,
    otherwise by running the frozen pipeline here."""
    if analysis:
        return brief_from_analysis(analysis, recording_id)
    return analyze_record(recording_id)


# ---------------------------------------------------------------------------
# prompt rendering
# ---------------------------------------------------------------------------

def _call_line(brief: CaseBrief, dx: str) -> str:
    prob = brief.probabilities.get(dx, 0.0)
    threshold = brief.thresholds.get(dx)
    folds = brief.fold_probabilities.get(dx)
    if threshold and threshold > 0:
        ratio = prob / threshold
        verdict = "ABOVE" if prob >= threshold else "below"
        line = f"  {dx}: p={prob:.3f} vs threshold {threshold:.3f} — {ratio:.2f}x, {verdict}"
    else:
        line = f"  {dx}: p={prob:.3f}"
    if folds:
        line += f" [per-fold {', '.join(f'{f:.2f}' for f in folds)}]"
    return line


def render_brief(brief: CaseBrief) -> str:
    """The case brief as prompt text."""
    lines: list[str] = ["CASE BRIEF (this system's own prior analysis of this recording)"]

    where = ", ".join(
        part for part in (
            f"source {brief.source}" if brief.source else "",
            f"{brief.duration_sec:.1f} s" if brief.duration_sec else "",
            f"{brief.sampling_rate} Hz" if brief.sampling_rate else "",
        ) if part
    )
    lines.append(f"RECORDING: {brief.recording_id}" + (f" ({where})" if where else ""))
    if brief.display_id:
        lines.append(f"REFERENCE ID: {brief.display_id}")
    if brief.rate_bpm:
        lines.append(
            f"RATE ESTIMATE: ~{brief.rate_bpm:.0f}/min (QRS complexes detected over the "
            "recording window; approximate)"
        )
    lines.append(
        f"ANALYSIS VERSION: {brief.model_version or 'frozen package'}, "
        f"{brief.feature_count or 'n'} features, score = {brief.score_kind}"
        + (f", thresholds = {brief.threshold_kind}" if brief.threshold_kind else "")
    )

    lines.append("")
    if brief.predictions:
        lines.append("CALLS ABOVE THRESHOLD (this is the diagnosis; your job is to verify it):")
        for dx in brief.predictions:
            lines.append(_call_line(brief, dx))
    else:
        lines.append(
            "CALLS ABOVE THRESHOLD: none. No diagnosis reached its decision threshold."
        )

    near = [
        dx for dx, p in sorted(brief.probabilities.items(), key=lambda kv: -kv[1])
        if dx not in brief.predictions
        and p > 0.2 * (brief.thresholds.get(dx) or 0.5)
    ][:6]
    if near:
        lines.append("")
        lines.append("IN THE DIFFERENTIAL BUT BELOW THRESHOLD:")
        for dx in near:
            lines.append(_call_line(brief, dx))

    for dx, items in brief.top_features.items():
        if not items:
            continue
        lines.append("")
        lines.append(
            f"TOP FEATURE CONTRIBUTIONS for '{dx}' "
            "(mean SHAP over the 5 folds, log-odds; + pushes towards the call):"
        )
        for item in items:
            contribution = float(item.get("contribution") or 0.0)
            value = float(item.get("value") or 0.0)
            sign = "+" if contribution >= 0 else "-"
            lines.append(
                f"  {sign}{abs(contribution):.3f}  {item.get('readable') or item.get('name')}"
                f" = {value:.4g}   [{item.get('name')}]"
            )

    if brief.neighbors:
        lines.append("")
        lines.append(
            f"NEAREST LABELLED RECORDINGS ({len(brief.neighbors)} of 63,256, Euclidean in "
            "32-D probability space) — other patients, not this one:"
        )
        for i, n in enumerate(brief.neighbors, start=1):
            labels = ", ".join(n.get("labels") or []) or (n.get("primary") or "unlabelled")
            agreement = ""
            if "storedPredictionSetExact" in n:
                agreement = (
                    " [stored prediction set exactly equalled its labels]"
                    if n["storedPredictionSetExact"]
                    else " [stored prediction set differed from its labels]"
                )
            lines.append(
                f"  {i}. {n['displayId']} d={n['distance']:.3f} "
                f"[{n.get('source') or 'unknown source'}] labels: {labels}{agreement}"
            )
        agreement = _neighbour_agreement(brief)
        if agreement:
            lines.append(agreement)

    if brief.stored:
        lines.append("")
        lines.append(
            "STORED REFERENCE ENTRY FOR THIS RECORDING: "
            f"top prediction {brief.stored['topPrediction']} at {brief.stored['maxScore']:.3f}; "
            f"reference labels: {', '.join(brief.stored['labels']) or 'none'}."
        )
        lines.append(
            "  Reference labels are the dataset's own annotations. Do not treat them as the "
            "truth you are verifying against — verify against the trace."
        )

    lines.append("")
    lines.append("WHAT TO CHECK ON THE TRACE (and why):")
    lines.append(skills_mod.render_skills(brief.skills))

    if brief.notes:
        lines.append("")
        lines.append("CAVEATS THAT ATTACH TO THESE NUMBERS:")
        for note in brief.notes:
            lines.append(f"  - {note}")

    return "\n".join(lines)


def _neighbour_agreement(brief: CaseBrief) -> str:
    """One line on how the neighbourhood relates to the leading call.

    The reduced form of ``ammonix.chatbot.analyze_neighborhood_agreement``: the
    open universe rows carry labels and a stored correctness flag, which is
    enough to say whether this corner of the universe is a place the classifier
    reads well.
    """
    top = brief.top_prediction
    if not top or not brief.neighbors:
        return ""
    total = len(brief.neighbors)
    carrying = sum(1 for n in brief.neighbors if top in (n.get("labels") or []))
    known = [n for n in brief.neighbors if "storedPredictionSetExact" in n]
    exact = sum(1 for n in known if n["storedPredictionSetExact"])
    parts = [
        f"  AGREEMENT: {carrying}/{total} neighbours carry '{top}' among their labels"
    ]
    if known:
        parts.append(
            f"; on {exact}/{len(known)} of them the whole prediction set exactly "
            "equalled the label set (a strict all-or-nothing test — one extra call counts as "
            "a miss)"
        )
    if carrying * 2 <= total:
        parts.append(
            " — the neighbourhood does not back this call, so weigh the trace more heavily"
        )
    return "".join(parts) + "."


# ---------------------------------------------------------------------------
# message assembly
# ---------------------------------------------------------------------------

DEFAULT_QUESTION = (
    "Explain the leading findings on this ECG and check them against the trace."
)
HISTORY_TOKEN_BUDGET = 2000


def _estimate_tokens(text: str) -> int:
    """Rough char->token estimate (~3 chars/token), as the private client used."""
    return len(text) // 3


def truncate_history(
    messages: Sequence[Mapping[str, Any]],
    *,
    budget: int = HISTORY_TOKEN_BUDGET,
) -> list[dict[str, str]]:
    """Keep the most recent turns that fit the budget, oldest-first on return."""
    kept: list[dict[str, str]] = []
    total = 0
    for message in reversed(list(messages)):
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role not in ("user", "assistant") or not content:
            continue
        cost = _estimate_tokens(content)
        if total + cost > budget:
            break
        kept.append({"role": role, "content": content})
        total += cost
    kept.reverse()
    return kept


def build_messages(
    brief: CaseBrief | None,
    messages: Sequence[Mapping[str, Any]],
    *,
    image_data_url: str | None = None,
    unavailable_context: str | None = None,
) -> list[dict[str, Any]]:
    """System prompt + prior turns + one final turn carrying brief, image, question.

    The brief travels with the image and the question in the same user turn:
    vision models attend to them together, and it keeps the model from answering
    a stale question against a fresh brief.
    """
    from services.llm import user_message  # local import: keeps services.case importable
    #                                        without httpx installed (scripts, tests)

    system = SYSTEM_PROMPT + (IMAGE_NOTE if image_data_url else NO_IMAGE_NOTE)

    history = truncate_history(messages)
    question = DEFAULT_QUESTION
    if history and history[-1]["role"] == "user":
        question = history.pop()["content"]

    if brief is not None:
        context_text = render_brief(brief)
    else:
        context_text = (
            "CASE BRIEF: unavailable.\n"
            + (unavailable_context or "No recording was supplied with this conversation.")
            + "\nNo prior analysis of this recording is available. Say so, answer only what "
            "can be answered without it, and do not invent findings."
        )

    final = f"{context_text}\n\n---\nCLINICIAN'S QUESTION: {question}"
    return [
        {"role": "system", "content": system},
        *history,
        user_message(final, image_data_url),
    ]
