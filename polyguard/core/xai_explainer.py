"""
polyguard.core.xai_explainer
=============================
Explainable AI module for PolyGuard.  Answers "WHY did the system flag this?"
at every pipeline step.

WHERE XAI IS USED (and why each matters clinically):

  Step 3 — Severity Scoring
    WHY: A doctor cannot act on "SEVERE" alone. They need the specific
         phrases that drove that score and the model's uncertainty.
    HOW: Feature attribution (LR coef × TF-IDF weight) + confidence bars.

  Step 4 — Organ Distribution
    WHY: "Cardiovascular risk" is vague. The clinician needs to know which
         words in the interaction description implied cardiac involvement.
    HOW: Per-organ top features from OvR classifier coefficients.

  Step 5 — Patient Adjustment
    WHY: A ×2.25 multiplier is a black box. The clinician needs to see
         exactly which lab values / conditions contributed how much.
    HOW: Waterfall chart decomposing base → each factor → final score.

  Step 6 — Cascade Detection
    WHY: A polypharmacy cascade alarm means nothing without knowing which
         drugs, which organ, and what the cumulative exposure mechanism is.
    HOW: Cascade attribution linking each drug pair to the shared organ.

  Step 7 — Clinical Report
    WHY: The report is what the pharmacist or physician reads and acts on.
         XAI makes it auditable and explainable to patients/regulators.
    HOW: Counterfactuals ("if eGFR improved to 65, renal risk drops 40%"),
         waterfall charts, and a plain-English XAI summary section.

Public API
----------
    from polyguard.core.xai_explainer import (
        explain_severity,
        explain_organ,
        explain_all_organs,
        waterfall_for_organ,
        counterfactuals_for_organ,
        generate_xai_report,
        print_xai_report,
        severity_confidence_bars,
        organ_confidence_bars,
    )
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

# sys.path is pre-populated by polyguard/core/__init__.py so these bare-module
# imports resolve to the flat files at the project root.
from nlp_engine import get_models, preprocess, ALL_ORGANS  # type: ignore[import]
from evidence_based_weights import (  # type: ignore[import]
    AGE_VULNERABILITY_MULTIPLIERS,
    COMORBIDITY_MULTIPLIERS,
    LAB_VALUE_THRESHOLDS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ev(src) -> float:
    return src.value if hasattr(src, "value") else float(src)


def _severity_label(score: int) -> str:
    if   score >= 50: return "CRITICAL"
    elif score >= 35: return "SEVERE"
    elif score >= 20: return "MODERATE"
    elif score >= 10: return "MILD"
    else:             return "MINIMAL"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 XAI — Feature Attribution for Severity Scoring
# ─────────────────────────────────────────────────────────────────────────────

def explain_severity(text: str, top_n: int = 8) -> Dict:
    """
    STEP 3 XAI: Which words drove the severity prediction?

    Technique: signed feature attribution = LR_coef[class][feature] × tfidf_weight.
    This is exact for linear models — no approximation required.

    Parameters
    ----------
    text  : interaction description string
    top_n : number of top supporting / opposing terms to return

    Returns
    -------
    dict with keys:
        predicted_class  : severity bucket string
        confidence       : float 0-1
        tier_confidence  : dict[str, float] — probability per bucket
        supporting       : list[(term, score)] — terms that pushed toward this class
        opposing         : list[(term, score)] — terms that pushed away
        is_negated       : bool
        explanation      : plain-English summary sentence
    """
    models = get_models()
    processed, is_negated, _ = preprocess(text)

    pipeline = models.severity._pipeline
    if pipeline is None:
        raise RuntimeError("Severity model pipeline not initialised. Call get_models() first.")

    tfidf   = pipeline.named_steps["tfidf"]
    clf     = pipeline.named_steps["clf"]
    classes = list(clf.classes_)

    X          = tfidf.transform([processed])
    proba      = clf.predict_proba(X)[0]
    pred       = clf.predict(X)[0]
    pred_idx   = classes.index(pred)
    confidence = float(proba[pred_idx])

    coef_row    = clf.coef_[pred_idx]
    tfidf_dense = X.toarray()[0]
    feat_names  = tfidf.get_feature_names_out()
    nonzero     = tfidf_dense.nonzero()[0]

    attrs = [(feat_names[i], float(coef_row[i] * tfidf_dense[i])) for i in nonzero]
    attrs.sort(key=lambda x: x[1], reverse=True)

    supporting = [(t, round(s, 3)) for t, s in attrs if s > 0][:top_n]
    opposing   = [(t, round(s, 3)) for t, s in attrs if s < 0][: top_n // 2]
    tier_conf  = dict(zip(classes, [round(float(p), 3) for p in proba]))

    top_terms   = ", ".join(f'"{t}"' for t, _ in supporting[:3]) if supporting else "no dominant features"
    explanation = f"Predicted {pred} ({confidence:.0%} confidence) because of: {top_terms}."
    if is_negated:
        explanation += " Warning: negation detected — score discounted 70%."

    return {
        "predicted_class": pred,
        "confidence":      round(confidence, 3),
        "tier_confidence": tier_conf,
        "supporting":      supporting,
        "opposing":        opposing,
        "is_negated":      is_negated,
        "explanation":     explanation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 XAI — Feature Attribution for Organ Classification
# ─────────────────────────────────────────────────────────────────────────────

def explain_organ(text: str, organ: str, top_n: int = 6) -> Dict:
    """
    STEP 4 XAI: Which words caused this organ system to be predicted?

    Uses the One-vs-Rest binary classifier's coefficient for the target organ.

    Parameters
    ----------
    text   : interaction description
    organ  : organ key e.g. 'CARDIOVASCULAR', 'HEPATIC'
    top_n  : number of top features to return

    Returns
    -------
    dict with keys: organ, probability, top_features, explanation
    """
    models    = get_models()
    processed, _, _ = preprocess(text)

    vectoriser = models.organs._vectoriser
    clf        = models.organs._clf

    if clf is None or vectoriser is None:
        raise RuntimeError("Organ classifier not initialised. Call get_models() first.")

    organ_idx  = ALL_ORGANS.index(organ)
    estimator  = clf.estimators_[organ_idx]

    X           = vectoriser.transform([processed])
    probability = float(estimator.predict_proba(X)[0][1])
    feat_names  = vectoriser.get_feature_names_out()
    coef        = estimator.coef_[0]
    tfidf_dense = X.toarray()[0]
    nonzero     = tfidf_dense.nonzero()[0]

    attrs = [(feat_names[i], float(coef[i] * tfidf_dense[i])) for i in nonzero]
    attrs.sort(key=lambda x: x[1], reverse=True)
    top_features = [(t, round(s, 3)) for t, s in attrs if s > 0][:top_n]

    organ_display = organ.replace("_", " ").title()
    terms = ", ".join(f'"{t}"' for t, _ in top_features[:3]) if top_features else "generalisation from training"
    explanation = f"{organ_display} flagged ({probability:.0%} confidence) because of: {terms}."

    return {
        "organ":        organ,
        "probability":  round(probability, 3),
        "top_features": top_features,
        "explanation":  explanation,
    }


def explain_all_organs(text: str, threshold: float = 0.10, top_n: int = 5) -> List[Dict]:
    """
    Return organ attributions for all organs above *threshold* probability,
    sorted by probability descending.
    """
    models = get_models()
    proba  = models.organs.organ_probability_vector(text)
    return [
        explain_organ(text, organ, top_n)
        for organ in sorted(ALL_ORGANS, key=lambda o: proba[o], reverse=True)
        if proba[organ] >= threshold
    ]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 XAI — Waterfall Score Decomposition
# ─────────────────────────────────────────────────────────────────────────────

def waterfall_for_organ(system: Dict, patient_data: Optional[Dict]) -> Dict:
    """
    STEP 5 XAI: Show exactly how base score became adjusted score.

    Example printed output::

        Base score (NLP)                          21
        + Age 72 yrs             ×1.28           +6   -> 27
        + Hypertension           ×1.45           +10  -> 37
        + Atrial Fibrillation    ×1.52           +10  -> 47
        ─────────────────────────────────────────────────────
        Final adjusted score [SEVERE]             47

    Parameters
    ----------
    system       : organ system dict from the pipeline (must include base_score / score)
    patient_data : optional patient context dict

    Returns
    -------
    dict with keys: organ, base_score, final_score, total_delta, steps, text, bar_chart
    """
    base  = system.get("base_score", system.get("score", 0))
    final = system.get("adjusted_score", base)
    organ = system.get("organ_key", "")

    steps: List[Dict] = [
        {"label": "Base score (NLP severity model)", "delta": 0, "running": base, "type": "base"}
    ]

    if patient_data and organ:
        steps += _decompose_multiplier(organ, patient_data, base)

    steps.append({
        "label":   f"Final adjusted score [{system.get('severity', '?')}]",
        "delta":   final - base,
        "running": final,
        "type":    "total",
    })

    return {
        "organ":       organ.replace("_", " ").title() or system.get("system", ""),
        "base_score":  base,
        "final_score": final,
        "total_delta": final - base,
        "steps":       steps,
        "text":        _steps_to_text(steps),
        "bar_chart":   _steps_to_bars(steps),
    }


def _decompose_multiplier(organ: str, patient_data: Dict, base: int) -> List[Dict]:
    """Break the overall multiplier into per-factor additive steps."""
    age        = patient_data.get("age", 0)
    conditions = " ".join(c.lower() for c in patient_data.get("conditions", []))
    labs       = patient_data.get("lab_values", {})
    steps: List[Dict] = []
    running = base

    def add(label: str, multiplier: float) -> None:
        nonlocal running
        delta    = int(base * (multiplier - 1.0))
        running += delta
        steps.append({
            "label":   label,
            "delta":   delta,
            "running": running,
            "type":    "increase" if delta >= 0 else "decrease",
        })

    # Age
    for (lo, hi), ev_src in AGE_VULNERABILITY_MULTIPLIERS.items():
        if lo <= age <= hi and _ev(ev_src) != 1.0:
            add(f"Age {age} yrs  (×{_ev(ev_src):.2f})", _ev(ev_src))
            break

    # Organ-specific comorbidities
    if organ == "CARDIOVASCULAR":
        cv = COMORBIDITY_MULTIPLIERS["cardiovascular"]
        if "hypertension" in conditions:
            add(f"Hypertension  (×{_ev(cv['hypertension']):.2f})", _ev(cv["hypertension"]))
        if any(t in conditions for t in ("heart failure", "chf")):
            add(f"Heart failure  (×{_ev(cv['heart_failure']):.2f})", _ev(cv["heart_failure"]))
        if any(t in conditions for t in ("atrial fibrillation", "afib")):
            add(
                f"Atrial fibrillation  (×{_ev(cv['atrial_fibrillation']):.2f})",
                _ev(cv["atrial_fibrillation"]),
            )

    elif organ == "RENAL":
        egfr = labs.get("eGFR", 100)
        for (lo, hi), ev_src in LAB_VALUE_THRESHOLDS["eGFR"].items():
            if lo <= egfr < hi:
                add(f"eGFR {egfr} mL/min  (×{_ev(ev_src):.2f})", _ev(ev_src))
                break

    elif organ == "HEPATIC":
        alt = labs.get("ALT", 0)
        if alt > 40:
            for (lo, hi), ev_src in LAB_VALUE_THRESHOLDS["ALT"].items():
                if lo <= alt < hi:
                    add(f"ALT {alt} U/L  (×{_ev(ev_src):.2f})", _ev(ev_src))
                    break
        for child, key in [
            ("c", "cirrhosis_child_c"),
            ("b", "cirrhosis_child_b"),
            ("a", "cirrhosis_child_a"),
        ]:
            if any(t in conditions for t in (f"child-pugh {child}", f"cirrhosis child {child}")):
                ev = COMORBIDITY_MULTIPLIERS["hepatic"][key]
                add(f"Cirrhosis Child-Pugh {child.upper()}  (×{_ev(ev):.2f})", _ev(ev))
                break

    elif organ == "HEMATOLOGIC":
        plt = labs.get("platelet_count", 200)
        if plt < 150:
            for (lo, hi), ev_src in LAB_VALUE_THRESHOLDS["platelet_count"].items():
                if lo <= plt < hi:
                    add(f"Platelets {plt}k/uL  (×{_ev(ev_src):.2f})", _ev(ev_src))
                    break
        inr = labs.get("INR", 0)
        if inr >= 3.0:
            for (lo, hi), ev_src in LAB_VALUE_THRESHOLDS["INR"].items():
                if lo <= inr < hi:
                    add(f"INR {inr}  (×{_ev(ev_src):.2f})", _ev(ev_src))
                    break

    elif organ == "ENDOCRINE":
        if "diabetes" in conditions:
            ev = COMORBIDITY_MULTIPLIERS["diabetes"]
            add(f"Diabetes mellitus  (×{_ev(ev):.2f})", _ev(ev))
        gluc = labs.get("blood_glucose", 100)
        for (lo, hi), ev_src in LAB_VALUE_THRESHOLDS["blood_glucose"].items():
            if lo <= gluc < hi:
                add(f"Blood glucose {gluc} mg/dL  (×{_ev(ev_src):.2f})", _ev(ev_src))
                break

    elif organ == "RESPIRATORY":
        if any(t in conditions for t in ("copd", "asthma")):
            ev = COMORBIDITY_MULTIPLIERS["copd_asthma"]
            add(f"COPD/Asthma  (×{_ev(ev):.2f})", _ev(ev))

    return steps


def _steps_to_text(steps: List[Dict]) -> str:
    lines = []
    for s in steps:
        if s["type"] == "base":
            lines.append(f"  {'Base score (NLP)':<42} {s['running']:>4}")
        elif s["type"] == "total":
            lines.append(f"  {'─' * 48}")
            lines.append(f"  {s['label']:<42} {s['running']:>4}")
        else:
            sign = "+" if s["delta"] >= 0 else ""
            lines.append(f"  {s['label']:<42} {sign}{s['delta']:>3}  ->  {s['running']}")
    return "\n".join(lines)


def _steps_to_bars(steps: List[Dict]) -> str:
    max_val = max((s["running"] for s in steps), default=1)
    bars = []
    for s in steps:
        filled = int(28 * s["running"] / max(max_val, 1))
        bar    = "|" * filled + "." * (28 - filled)
        sign   = f'+{s["delta"]}' if s["delta"] > 0 else (str(s["delta"]) if s["delta"] < 0 else "")
        bars.append(f"  {s['label'][:32]:<32} [{bar}] {s['running']:>3}  {sign}")
    return "\n".join(bars)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 XAI — Cascade Attribution
# ─────────────────────────────────────────────────────────────────────────────

def explain_cascade(cascade: Dict, interactions_list: List[Dict], base_scores: Dict) -> Dict:
    """
    STEP 6 XAI: Which drug pairs contributed to this organ cascade,
    and what is the compounding mechanism?

    Parameters
    ----------
    cascade           : cascade dict from detect_polypharmacy_cascades
    interactions_list : raw interactions passed to the engine
    base_scores       : output of calculate_interaction_score_robust

    Returns
    -------
    dict with keys: organ, alert_level, contributing_pairs, mechanism_summary, num_contributors
    """
    organ_key = cascade.get("organ_system", "").upper().replace(" ", "_")
    models    = get_models()

    contributing: List[Dict] = []
    for idx, ix in enumerate(interactions_list):
        desc       = ix.get("description", "")
        proba      = models.organs.organ_probability_vector(desc)
        organ_prob = proba.get(organ_key, 0.0)
        if organ_prob < 0.15:
            continue
        ix_score = base_scores["detailed_breakdown"][idx]["score"]
        contributing.append({
            "drugs":             f"{ix.get('drug_a', '?')} <-> {ix.get('drug_b', '?')}",
            "description":       desc[:90] + "..." if len(desc) > 90 else desc,
            "organ_confidence":  round(organ_prob, 3),
            "interaction_score": ix_score,
            "contribution":      round(ix_score * organ_prob, 1),
        })

    contributing.sort(key=lambda x: x["contribution"], reverse=True)
    organ_display = cascade.get("organ_system", organ_key.replace("_", " ").title())

    if len(contributing) >= 2:
        pairs     = " and ".join(c["drugs"] for c in contributing[:2])
        mechanism = (
            f"Polypharmacy cascade on {organ_display}: {len(contributing)} independent "
            f"drug pairs ({pairs}) each stress the same organ system. "
            f"Cumulative exposure compounds risk beyond safe tolerance. "
            f"Evidence: Masnoon 2017 (5+ meds = 88% higher ADR risk, n=138 studies)."
        )
    else:
        mechanism = f"Single interaction pathway identified for {organ_display}."

    return {
        "organ":              organ_display,
        "alert_level":        cascade.get("alert_level", ""),
        "contributing_pairs": contributing,
        "mechanism_summary":  mechanism,
        "num_contributors":   len(contributing),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 + 7 XAI — Counterfactual Analysis
# ─────────────────────────────────────────────────────────────────────────────

def counterfactuals_for_organ(system: Dict, patient_data: Dict) -> Dict:
    """
    STEP 5/7 XAI: What lab or condition change would most reduce this organ risk?

    Generates specific, actionable recommendations. For each abnormal lab value
    or active condition, the function recomputes what the adjusted score *would*
    be if that factor were normalised, and ranks interventions by score saving.

    Parameters
    ----------
    system       : organ system dict (must include organ_key and adjusted_score)
    patient_data : patient context dict

    Returns
    -------
    dict with keys: organ, organ_key, current_score, base_score, counterfactuals, top_action
    """
    base       = system.get("base_score", system.get("score", 0))
    adj        = system.get("adjusted_score", base)
    organ      = system.get("organ_key", "")
    labs       = patient_data.get("lab_values", {})
    conditions = patient_data.get("conditions", [])

    _NORMAL = {
        "eGFR":          90,
        "ALT":           30,
        "AST":           30,
        "platelet_count":200,
        "blood_glucose": 110,
        "INR":           2.0,
    }
    _ABNORMAL = {
        "eGFR":          lambda v: v < 60,
        "ALT":           lambda v: v > 40,
        "AST":           lambda v: v > 40,
        "platelet_count":lambda v: v < 150,
        "blood_glucose": lambda v: v < 70 or v > 180,
        "INR":           lambda v: v >= 3.0,
    }

    cfs: List[Dict] = []

    for lab_key, normal_val in _NORMAL.items():
        current = labs.get(lab_key)
        if current is None or not _ABNORMAL[lab_key](current):
            continue
        hypo_patient = {**patient_data, "lab_values": {**labs, lab_key: normal_val}}
        hypo_score   = int(base * _recompute_multiplier(organ, hypo_patient))
        saving       = adj - hypo_score
        if saving >= 5:
            from_sev = _severity_label(adj)
            to_sev   = _severity_label(hypo_score)
            tier_txt = f" ({from_sev} -> {to_sev})" if from_sev != to_sev else ""
            cfs.append({
                "type":        "lab",
                "variable":    lab_key,
                "current":     current,
                "target":      normal_val,
                "score_saving":saving,
                "from_score":  adj,
                "to_score":    hypo_score,
                "narrative": (
                    f"Normalise {lab_key} ({current} -> {normal_val}): "
                    f"{organ.replace('_', ' ').title()} risk "
                    f"{adj} -> {hypo_score} (-{saving} pts{tier_txt})"
                ),
            })

    for cond in conditions:
        hypo_patient = {
            **patient_data,
            "conditions": [c for c in conditions if c != cond],
        }
        hypo_score = int(base * _recompute_multiplier(organ, hypo_patient))
        saving     = adj - hypo_score
        if saving >= 5:
            from_sev = _severity_label(adj)
            to_sev   = _severity_label(hypo_score)
            tier_txt = f" ({from_sev} -> {to_sev})" if from_sev != to_sev else ""
            cfs.append({
                "type":        "condition",
                "variable":    cond,
                "current":     "present",
                "target":      "controlled",
                "score_saving":saving,
                "from_score":  adj,
                "to_score":    hypo_score,
                "narrative": (
                    f"Control '{cond}': "
                    f"{organ.replace('_', ' ').title()} risk "
                    f"{adj} -> {hypo_score} (-{saving} pts{tier_txt})"
                ),
            })

    cfs.sort(key=lambda x: x["score_saving"], reverse=True)

    return {
        "organ":           organ.replace("_", " ").title(),
        "organ_key":       organ,
        "current_score":   adj,
        "base_score":      base,
        "counterfactuals": cfs,
        "top_action":      cfs[0]["narrative"] if cfs else "No modifiable risk factors identified.",
    }


def _recompute_multiplier(organ: str, patient_data: Dict) -> float:
    """Recompute the vulnerability multiplier for one organ from scratch."""
    age        = patient_data.get("age", 0)
    conditions = " ".join(c.lower() for c in patient_data.get("conditions", []))
    labs       = patient_data.get("lab_values", {})
    mult       = 1.0

    for (lo, hi), ev_src in AGE_VULNERABILITY_MULTIPLIERS.items():
        if lo <= age <= hi:
            mult += _ev(ev_src) - 1.0
            break

    if organ == "CARDIOVASCULAR":
        cv = COMORBIDITY_MULTIPLIERS["cardiovascular"]
        if "hypertension" in conditions:
            mult += _ev(cv["hypertension"]) - 1.0
        if any(t in conditions for t in ("heart failure", "chf")):
            mult += _ev(cv["heart_failure"]) - 1.0
        if any(t in conditions for t in ("atrial fibrillation", "afib")):
            mult += _ev(cv["atrial_fibrillation"]) - 1.0
    elif organ == "RENAL":
        egfr = labs.get("eGFR", 100)
        for (lo, hi), ev_src in LAB_VALUE_THRESHOLDS["eGFR"].items():
            if lo <= egfr < hi:
                mult += _ev(ev_src) - 1.0
                break
    elif organ == "HEPATIC":
        alt = labs.get("ALT", 0)
        if alt > 40:
            for (lo, hi), ev_src in LAB_VALUE_THRESHOLDS["ALT"].items():
                if lo <= alt < hi:
                    mult += _ev(ev_src) - 1.0
                    break
    elif organ == "HEMATOLOGIC":
        plt = labs.get("platelet_count", 200)
        if plt < 150:
            for (lo, hi), ev_src in LAB_VALUE_THRESHOLDS["platelet_count"].items():
                if lo <= plt < hi:
                    mult += _ev(ev_src) - 1.0
                    break
        inr = labs.get("INR", 0)
        if inr >= 3.0:
            for (lo, hi), ev_src in LAB_VALUE_THRESHOLDS["INR"].items():
                if lo <= inr < hi:
                    mult += _ev(ev_src) - 1.0
                    break
    elif organ == "ENDOCRINE":
        if "diabetes" in conditions:
            mult += _ev(COMORBIDITY_MULTIPLIERS["diabetes"]) - 1.0
        gluc = labs.get("blood_glucose", 100)
        for (lo, hi), ev_src in LAB_VALUE_THRESHOLDS["blood_glucose"].items():
            if lo <= gluc < hi:
                mult += _ev(ev_src) - 1.0
                break
    elif organ == "RESPIRATORY":
        if any(t in conditions for t in ("copd", "asthma")):
            mult += _ev(COMORBIDITY_MULTIPLIERS["copd_asthma"]) - 1.0

    return max(1.0, round(mult, 3))


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE DISPLAY  (Steps 3 & 4)
# ─────────────────────────────────────────────────────────────────────────────

_SEV_ORDER = ["CRITICAL", "SEVERE", "MODERATE", "MILD", "MINIMAL"]
_SEV_ICONS = {
    "CRITICAL": "🔴", "SEVERE": "🟠", "MODERATE": "🟡",
    "MILD": "🟢",    "MINIMAL": "⚪",
}


def severity_confidence_bars(severity_proba: Dict[str, float]) -> str:
    """Render a text confidence bar chart for severity tier probabilities."""
    lines = ["  Severity confidence:"]
    for tier in _SEV_ORDER:
        prob   = severity_proba.get(tier, 0.0)
        filled = round(prob * 20)
        bar    = "█" * filled + "░" * (20 - filled)
        lines.append(f"    {_SEV_ICONS.get(tier, '⚪')} {tier:<10} {bar} {prob:5.1%}")
    return "\n".join(lines)


def organ_confidence_bars(organ_proba: Dict[str, float], threshold: float = 0.10) -> str:
    """Render a text confidence bar chart for organ system probabilities."""
    relevant = {o: p for o, p in organ_proba.items() if p >= threshold}
    if not relevant:
        return "  No organs above threshold."
    lines = ["  Organ system confidence:"]
    for organ, prob in sorted(relevant.items(), key=lambda x: x[1], reverse=True):
        filled = round(prob * 20)
        bar    = "█" * filled + "░" * (20 - filled)
        label  = organ.replace("_", " ").title()
        lines.append(f"    {label:<28} {bar} {prob:5.1%}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MASTER XAI REPORT  (called from analyser.py / Step 7)
# ─────────────────────────────────────────────────────────────────────────────

def generate_xai_report(
    base_scores:       Dict,
    organ_analysis:    Dict,
    patient_adj:       Dict,
    cascade_detection: Dict,
    interactions_list: List[Dict],
    patient_data:      Optional[Dict] = None,
) -> Dict:
    """
    Generate the complete XAI report covering all pipeline steps.

    Called from ``PolyGuardAnalyser.analyse()`` (Step 7).

    Returns
    -------
    dict with keys:
        step3_interaction_explanations  list — per-interaction severity + organ XAI
        step5_waterfalls                list — base → adjusted score waterfalls
        step5_counterfactuals           list — actionable risk-reduction scenarios
        step6_cascade_attributions      list — which pairs drive each cascade
        has_patient_xai                 bool
        has_counterfactuals             bool
        has_cascade_xai                 bool
        methodology                     str
    """

    # ── Steps 3 & 4: per-interaction severity + organ attribution ─────────────
    step3_explanations: List[Dict] = []
    for idx, ix in enumerate(interactions_list):
        desc = ix.get("description", "")
        bd   = base_scores["detailed_breakdown"][idx]

        try:
            sev_xai = explain_severity(desc)
        except Exception as exc:
            logger.warning("Severity attribution failed for '%s': %s", desc[:40], exc)
            sev_xai = {
                "explanation":     f"Attribution unavailable: {exc}",
                "supporting":      [],
                "opposing":        [],
                "tier_confidence": {},
                "is_negated":      False,
            }

        try:
            organ_xais = explain_all_organs(desc, threshold=0.15, top_n=4)
        except Exception as exc:
            logger.warning("Organ attribution failed: %s", exc)
            organ_xais = []

        step3_explanations.append({
            "drugs":         bd.get("drugs", ""),
            "score":         bd.get("score", 0),
            "severity":      bd.get("severity", ""),
            "severity_xai":  sev_xai,
            "organ_xais":    organ_xais[:3],
            "severity_bars": severity_confidence_bars(bd.get("severity_proba", {})),
            "organ_bars":    organ_confidence_bars(bd.get("organ_proba_vec", {})),
            "nearest_refs":  bd.get("nearest_refs", []),
        })

    # ── Step 5: waterfall + counterfactuals per organ ─────────────────────────
    adjusted_systems      = patient_adj.get("adjusted_systems", [])
    step5_waterfalls:     List[Dict] = []
    step5_counterfactuals:List[Dict] = []

    for sys in adjusted_systems:
        base = sys.get("base_score", sys.get("score", 0))
        adj  = sys.get("adjusted_score", base)
        if adj - base >= 5:
            step5_waterfalls.append(waterfall_for_organ(sys, patient_data))
            if patient_data:
                try:
                    cf = counterfactuals_for_organ(sys, patient_data)
                    if cf["counterfactuals"]:
                        step5_counterfactuals.append(cf)
                except Exception as exc:
                    logger.warning("Counterfactual generation failed: %s", exc)

    step5_counterfactuals.sort(
        key=lambda x: x["current_score"] - x["base_score"], reverse=True
    )

    # ── Step 6: cascade attribution ───────────────────────────────────────────
    step6_cascade_xai: List[Dict] = []
    for cascade in cascade_detection.get("cascades", []):
        try:
            step6_cascade_xai.append(
                explain_cascade(cascade, interactions_list, base_scores)
            )
        except Exception as exc:
            logger.warning("Cascade XAI failed: %s", exc)

    return {
        "step3_interaction_explanations": step3_explanations,
        "step5_waterfalls":               step5_waterfalls,
        "step5_counterfactuals":          step5_counterfactuals,
        "step6_cascade_attributions":     step6_cascade_xai,
        "has_patient_xai":                len(step5_waterfalls) > 0,
        "has_counterfactuals":            len(step5_counterfactuals) > 0,
        "has_cascade_xai":                len(step6_cascade_xai) > 0,
        "methodology": (
            "XAI: Step 3 — LR coef × TF-IDF weight (exact linear attribution); "
            "Step 4 — OvR organ classifier per-feature attribution; "
            "Step 5 — Waterfall score decomposition + counterfactual lab/condition analysis; "
            "Step 6 — Cascade attribution linking drug pairs to shared organ systems."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_xai_report(xai: Dict) -> None:
    """Pretty-print the full XAI report to stdout."""
    W = 72
    _ICON = {
        "CRITICAL": "🔴", "SEVERE": "🟠", "MODERATE": "🟡",
        "MILD": "🟢",    "MINIMAL": "⚪",
    }

    print(f"\n{'=' * W}")
    print(f"  XAI — WHY DID POLYGUARD FLAG THESE RISKS?")
    print(f"{'=' * W}")

    for ex in xai["step3_interaction_explanations"]:
        icon = _ICON.get(ex["severity"], "⚪")
        print(f"\n  STEP 3 — {ex['drugs']}  {icon} [{ex['severity']}] score={ex['score']}")

        sxai = ex["severity_xai"]
        print(f"    {sxai.get('explanation', '')}")

        sup = sxai.get("supporting", [])
        if sup:
            print("    Supporting: " + ", ".join(f'"{t}"({s:+.2f})' for t, s in sup[:5]))
        opp = sxai.get("opposing", [])
        if opp:
            print("    Opposing:   " + ", ".join(f'"{t}"({s:.2f})' for t, s in opp[:3]))

        print(f"\n{ex['severity_bars']}")

        if ex.get("organ_xais"):
            print(f"\n  STEP 4 — Organ attribution:")
            for oxai in ex["organ_xais"]:
                print(f"    * {oxai['explanation']}")
                feats = oxai.get("top_features", [])
                if feats:
                    print("      Features: " + ", ".join(f'"{t}"({s:+.2f})' for t, s in feats[:4]))

        print(f"\n{ex['organ_bars']}")

        if ex.get("nearest_refs"):
            print(f"\n    Nearest training references:")
            for ref in ex["nearest_refs"][:2]:
                print(f"      {ref}")

    if xai["step5_waterfalls"]:
        print(f"\n  {'─' * W}")
        print(f"  STEP 5 — SCORE WATERFALL  (NLP base score → patient-adjusted)")
        print(f"  {'─' * W}")
        for wf in xai["step5_waterfalls"]:
            print(f"\n  {wf['organ']}  (base={wf['base_score']} → adjusted={wf['final_score']}  delta=+{wf['total_delta']})")
            print(wf["bar_chart"])

    if xai["step5_counterfactuals"]:
        print(f"\n  {'─' * W}")
        print(f"  STEP 5/7 — COUNTERFACTUALS  (what interventions would reduce risk?)")
        print(f"  {'─' * W}")
        for cf in xai["step5_counterfactuals"]:
            print(f"\n  {cf['organ']}  (current score: {cf['current_score']})")
            for item in cf["counterfactuals"][:3]:
                print(f"    >> {item['narrative']}")

    if xai["step6_cascade_attributions"]:
        print(f"\n  {'─' * W}")
        print(f"  STEP 6 — CASCADE ATTRIBUTION")
        print(f"  {'─' * W}")
        for cxai in xai["step6_cascade_attributions"]:
            print(f"\n  [CASCADE] {cxai['organ']}  [{cxai['alert_level']}]")
            print(f"     {cxai['mechanism_summary']}")
            for pair in cxai["contributing_pairs"][:3]:
                print(
                    f"     * {pair['drugs']}  confidence={pair['organ_confidence']:.0%}  "
                    f"contribution={pair['contribution']:.1f}"
                )

    print(f"\n  Methodology: {xai['methodology']}")
    print(f"{'=' * W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST  (python -m polyguard.core.xai_explainer)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Path already on sys.path via core/__init__.py when run as a module.
    # If run directly as a script, inject it manually.
    import sys
    from pathlib import Path
    _root = str(Path(__file__).resolve().parent.parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from polyguard_engine_evidence_based import (  # type: ignore[import]
        calculate_interaction_score_robust,
        analyze_biological_impact,
        adjust_for_patient_context,
        detect_polypharmacy_cascades,
    )

    interactions = [
        {
            "drug_a": "Warfarin",  "drug_b": "Aspirin",
            "description": (
                "Risk of bleeding and haemorrhage significantly increased. "
                "Platelet coagulation impaired."
            ),
        },
        {
            "drug_a": "Isoniazid", "drug_b": "Rifampicin",
            "description": (
                "Severe hepatotoxicity and acute liver failure. "
                "ALT and AST markedly elevated."
            ),
        },
        {
            "drug_a": "Morphine",  "drug_b": "Diazepam",
            "description": "Severe respiratory depression and apnea. Pulmonary failure risk.",
        },
    ]
    patient = {
        "age":        72,
        "conditions": ["Hypertension", "Atrial Fibrillation", "Diabetes Type 2"],
        "lab_values": {
            "eGFR": 42, "ALT": 85, "platelet_count": 110,
            "INR": 3.2, "blood_glucose": 195,
        },
    }

    print("Running self-test pipeline…")
    base   = calculate_interaction_score_robust(interactions)
    organs = analyze_biological_impact(interactions, base)
    adj    = adjust_for_patient_context(organs["affected_organ_systems"], patient)
    casc   = detect_polypharmacy_cascades(adj["adjusted_systems"], interactions, 3)

    xai = generate_xai_report(
        base_scores       = base,
        organ_analysis    = organs,
        patient_adj       = adj,
        cascade_detection = casc,
        interactions_list = interactions,
        patient_data      = patient,
    )
    print_xai_report(xai)
    print(f"Step 3 explanations : {len(xai['step3_interaction_explanations'])}")
    print(f"Step 5 waterfalls   : {len(xai['step5_waterfalls'])}")
    print(f"Step 5 counterfacts : {len(xai['step5_counterfactuals'])}")
    print(f"Step 6 cascade XAI  : {len(xai['step6_cascade_attributions'])}")