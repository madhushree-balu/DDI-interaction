# polyguard_engine_evidence_based.py
"""
PolyGuard Engine — NLP + Evidence-Based Scoring Pipeline (Steps 3-7).

Steps 3 & 4 now use a real NLP pipeline (nlp_engine.py):
  • TF-IDF + Logistic Regression severity scorer
  • TF-IDF + Multi-label One-vs-Rest organ classifier
  • Negation-aware tokeniser / lemmatiser
  • TF-IDF cosine semantic similarity for nearest-neighbour explainability

Steps 5-7 use evidence-based multipliers from peer-reviewed literature.
"""

import re
from typing import List, Dict
from collections import defaultdict

from evidence_based_weights import (
    ORGAN_SEVERITY_WEIGHTS,
    KEYWORD_SEVERITY_SCORES,
    AGE_VULNERABILITY_MULTIPLIERS,
    COMORBIDITY_MULTIPLIERS,
    LAB_VALUE_THRESHOLDS,
    CASCADE_DETECTION_THRESHOLD,
)

from nlp_engine import analyse_interaction_batch, analyse_interaction_text


# ─── Evidence helpers ─────────────────────────────────────────────────────────

def _ev(src) -> float:
    return src.value if hasattr(src, 'value') else float(src)

def _cite(src) -> str:
    return src.citation if hasattr(src, 'citation') else 'No citation'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — NLP INTERACTION SCORING
# ─────────────────────────────────────────────────────────────────────────────

def calculate_interaction_score_robust(interactions_list: List[Dict]) -> Dict:
    """
    STEP 3 — Score each interaction using the NLP severity classifier.

    Replaces hand-crafted regex with:
      • TF-IDF + Logistic Regression (trained on 99 labelled DrugBank sentences)
      • Negation-aware preprocessing (reduces score to 30% when cue detected)
      • Confidence-weighted score within severity bucket

    Returns total score, per-interaction breakdown, and overall risk level.
    """
    if not interactions_list:
        return {
            'total_score':        0,
            'risk_level':         'MINIMAL',
            'risk_color':         '⚪',
            'recommendation':     '✓  No interactions to analyse',
            'num_interactions':   0,
            'detailed_breakdown': [],
            'methodology':        'NLP: TF-IDF + Logistic Regression severity scorer',
        }

    # Run NLP batch analysis
    enriched = analyse_interaction_batch(interactions_list)

    total_score = 0
    breakdown   = []

    for item in enriched:
        nlp   = item['nlp']
        score = nlp['score']

        # Evidence citation for the severity tier
        tier_key_map = {
            'CRITICAL': 'fatal', 'SEVERE': 'severe',
            'MODERATE': 'moderate', 'MILD': 'mild', 'MINIMAL': 'mild',
        }
        tier_key = tier_key_map.get(nlp['severity'], 'mild')
        citation = _cite(KEYWORD_SEVERITY_SCORES.get(tier_key, 'mild'))

        # Nearest references for explainability
        ref_texts = [
            f"sim={r['similarity']:.2f}: {r['text'][:60]}…"
            for r in nlp['nearest_refs']
        ]

        breakdown.append({
            'drugs':              f"{item.get('drug_a','?')} ↔ {item.get('drug_b','?')}",
            'description':        item.get('description', ''),
            'score':              score,
            'severity':           nlp['severity'],
            'icon':               _severity_icon(nlp['severity']),
            'has_negation':       nlp['is_negated'],
            'processed_text':     nlp['processed_text'],
            'severity_proba':     nlp['severity_proba'],     # model confidence
            'organ_proba_vec':    nlp['organ_proba_vec'],    # per-organ probabilities
            'nearest_refs':       ref_texts,                 # explainability
            'evidence_citations': [citation],
            'matched_keywords':   [                          # kept for backward compat
                f"NLP: {nlp['severity']} (score {score})"
                + (" [NEGATED → ×0.30]" if nlp['is_negated'] else "")
            ],
        })
        total_score += score

    risk = _overall_risk(total_score, len(interactions_list))

    return {
        'total_score':        total_score,
        'risk_level':         risk['level'],
        'risk_color':         risk['color'],
        'recommendation':     risk['action'],
        'num_interactions':   len(interactions_list),
        'detailed_breakdown': breakdown,
        'methodology': (
            'NLP Step 3: TF-IDF (word n-grams 1-3) + Logistic Regression severity scorer '
            'trained on 99 labelled DrugBank/MedDRA sentences. '
            'Negation-aware preprocessing (score ×0.30 when negation detected). '
            'Scores mapped to evidence-based severity tiers (FDA MedDRA v26.0).'
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — NLP ORGAN SYSTEM DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def analyze_biological_impact(interactions_list: List[Dict], base_scores: Dict) -> Dict:
    """
    STEP 4 — Map interaction scores onto organ systems using the NLP multi-label
    organ classifier (TF-IDF + One-vs-Rest Logistic Regression).

    Replaces hand-crafted regex organ patterns with a trained classifier that:
      • Outputs per-organ probability vectors
      • Handles multi-organ interactions (e.g. bleeding → CARDIOVASCULAR + HEMATOLOGIC)
      • Weighs organ contributions by evidence-based severity weights from literature
      • Uses classifier confidence as a scaling factor for the score contribution
    """
    acc = defaultdict(lambda: {
        'score': 0.0, 'interaction_count': 0, 'evidence_source': None,
        'confidence_sum': 0.0,
    })

    for idx, ix in enumerate(interactions_list):
        description = ix.get('description', '')
        ix_score    = base_scores['detailed_breakdown'][idx]['score']

        # ── NLP organ prediction ──────────────────────────────────────────────
        # Use the pre-computed proba vector from Step 3 if available,
        # otherwise run a fresh analysis.
        bd  = base_scores['detailed_breakdown'][idx]
        if 'organ_proba_vec' in bd:
            organ_proba = bd['organ_proba_vec']
        else:
            organ_proba = analyse_interaction_text(description)['organ_proba_vec']

        for organ, confidence in organ_proba.items():
            if confidence < 0.15:          # ignore near-zero probability organs
                continue

            ev_src = ORGAN_SEVERITY_WEIGHTS.get(organ)
            weight = _ev(ev_src) if ev_src else 1.0

            # Score contribution = base_score × 0.4 × organ_weight × classifier_confidence
            contrib = (ix_score * 0.4) * weight * confidence

            acc[organ]['score']             += contrib
            acc[organ]['interaction_count'] += 1
            acc[organ]['confidence_sum']    += confidence
            acc[organ]['evidence_source']    = ev_src

    systems = []
    for organ, data in acc.items():
        score = int(data['score'])
        if score == 0:
            continue
        ev  = data['evidence_source']
        sev = _severity_label(score)
        systems.append({
            'system':             organ.replace('_', ' ').title(),
            'organ_key':          organ,
            'score':              score,
            'severity':           sev,
            'icon':               _severity_icon(sev),
            'interaction_count':  data['interaction_count'],
            'nlp_confidence':     round(data['confidence_sum'] / max(data['interaction_count'], 1), 3),
            'severity_weight':    _ev(ev) if ev else 1.0,
            'evidence_citation':  _cite(ev) if ev else 'N/A',
            'evidence_rationale': ev.rationale if ev else 'Based on clinical practice',
            'evidence_level':     ev.evidence_level if ev else 'N/A',
        })

    # Sort by NLP confidence first (unbiased model signal) then by score as tiebreaker.
    # Previously sorted by score alone, which always promoted CARDIOVASCULAR because its
    # organ severity weight (1.52) is the highest in the table, regardless of NLP confidence.
    systems.sort(key=lambda x: (x['nlp_confidence'], x['score']), reverse=True)

    return {
        'affected_organ_systems': systems,
        'num_organs_affected':    len(systems),
        'highest_risk_organ':     systems[0] if systems else None,
        'methodology': (
            'NLP Step 4: TF-IDF + One-vs-Rest Logistic Regression multi-label organ classifier '
            '(10 classes, threshold=0.15). Score contribution = base_score × 0.4 × '
            'organ_severity_weight × classifier_confidence. '
            'Organ weights from peer-reviewed mortality/hospitalisation studies.'
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — PATIENT CONTEXT ADJUSTMENT
# ─────────────────────────────────────────────────────────────────────────────

def adjust_for_patient_context(organ_systems: List[Dict], patient_data: Dict) -> Dict:
    """
    STEP 5 — Apply evidence-based vulnerability multipliers per patient profile.
    """
    if not patient_data:
        return {'status': 'NO_PATIENT_DATA', 'adjusted_systems': organ_systems}

    age        = patient_data.get('age', 0)
    conditions = ' '.join(c.lower() for c in patient_data.get('conditions', []))
    labs       = patient_data.get('lab_values', {})

    adjusted = []

    for sys in organ_systems:
        organ    = sys['organ_key']
        base     = sys['score']
        mult     = 1.0
        factors  = []
        citations= []

        # ── Age ──────────────────────────────────────────────────────────────
        for rng, ev in AGE_VULNERABILITY_MULTIPLIERS.items():
            if rng[0] <= age <= rng[1]:
                mult += _ev(ev) - 1.0
                factors.append(f"Age {age} yrs")
                citations.append(_cite(ev))
                break

        # ── Organ-specific comorbidities ──────────────────────────────────
        cv = COMORBIDITY_MULTIPLIERS['cardiovascular']
        if organ == 'CARDIOVASCULAR':
            if 'hypertension' in conditions:
                ev = cv['hypertension']
                mult += _ev(ev) - 1.0; factors.append("Hypertension"); citations.append(_cite(ev))
            if any(t in conditions for t in ('heart failure', 'chf', 'cardiac failure')):
                ev = cv['heart_failure']
                mult += _ev(ev) - 1.0; factors.append("Heart failure"); citations.append(_cite(ev))
            if any(t in conditions for t in ('atrial fibrillation', 'afib', 'af')):
                ev = cv['atrial_fibrillation']
                mult += _ev(ev) - 1.0; factors.append("Atrial fibrillation"); citations.append(_cite(ev))

        elif organ == 'RENAL':
            egfr = labs.get('eGFR', labs.get('egfr', 100))
            for (lo, hi), ev in LAB_VALUE_THRESHOLDS['eGFR'].items():
                if lo <= egfr < hi:
                    mult += _ev(ev) - 1.0
                    factors.append(f"eGFR {egfr} mL/min")
                    citations.append(_cite(ev))
                    break

        elif organ == 'HEPATIC':
            alt = labs.get('ALT', labs.get('alt', 0))
            if alt > 40:
                for (lo, hi), ev in LAB_VALUE_THRESHOLDS['ALT'].items():
                    if lo <= alt < hi:
                        mult += _ev(ev) - 1.0
                        factors.append(f"Elevated ALT {alt} U/L")
                        citations.append(_cite(ev))
                        break
            if any(t in conditions for t in ('cirrhosis child c', 'child-pugh c')):
                ev = COMORBIDITY_MULTIPLIERS['hepatic']['cirrhosis_child_c']
                mult += _ev(ev) - 1.0; factors.append("Cirrhosis Child-Pugh C"); citations.append(_cite(ev))
            elif any(t in conditions for t in ('cirrhosis child b', 'child-pugh b')):
                ev = COMORBIDITY_MULTIPLIERS['hepatic']['cirrhosis_child_b']
                mult += _ev(ev) - 1.0; factors.append("Cirrhosis Child-Pugh B"); citations.append(_cite(ev))
            elif any(t in conditions for t in ('cirrhosis', 'child-pugh a', 'cirrhosis child a')):
                ev = COMORBIDITY_MULTIPLIERS['hepatic']['cirrhosis_child_a']
                mult += _ev(ev) - 1.0; factors.append("Cirrhosis Child-Pugh A"); citations.append(_cite(ev))

        elif organ == 'HEMATOLOGIC':
            plt = labs.get('platelet_count', labs.get('platelets', 200))
            if plt < 150:
                for (lo, hi), ev in LAB_VALUE_THRESHOLDS['platelet_count'].items():
                    if lo <= plt < hi:
                        mult += _ev(ev) - 1.0
                        factors.append(f"Platelets {plt}k")
                        citations.append(_cite(ev))
                        break
            inr = labs.get('INR', labs.get('inr', 0))
            if inr >= 3.0:
                for (lo, hi), ev in LAB_VALUE_THRESHOLDS['INR'].items():
                    if lo <= inr < hi:
                        mult += _ev(ev) - 1.0
                        factors.append(f"INR {inr}")
                        citations.append(_cite(ev))
                        break

        elif organ == 'ENDOCRINE':
            if 'diabetes' in conditions:
                ev = COMORBIDITY_MULTIPLIERS['diabetes']
                mult += _ev(ev) - 1.0; factors.append("Diabetes mellitus"); citations.append(_cite(ev))
            gluc = labs.get('blood_glucose', labs.get('glucose', 100))
            for (lo, hi), ev in LAB_VALUE_THRESHOLDS['blood_glucose'].items():
                if lo <= gluc < hi:
                    mult += _ev(ev) - 1.0
                    factors.append(f"Blood glucose {gluc} mg/dL")
                    citations.append(_cite(ev))
                    break

        elif organ == 'RESPIRATORY':
            if any(t in conditions for t in ('copd', 'asthma', 'chronic obstructive')):
                ev = COMORBIDITY_MULTIPLIERS['copd_asthma']
                mult += _ev(ev) - 1.0; factors.append("COPD/Asthma"); citations.append(_cite(ev))

        adj_score    = int(base * mult)
        score_delta  = adj_score - base
        warning      = None
        if score_delta > 20:
            urgency = "🚨 IMMEDIATE ATTENTION" if adj_score >= 50 else "⚠️  URGENT"
            warning = (f"{urgency} — {sys['system']} risk ↑{score_delta} pts due to: "
                       + ", ".join(factors))

        entry = sys.copy()
        entry.update({
            'base_score':              base,
            'adjusted_score':          adj_score,
            'score_increase':          score_delta,
            'vulnerability_multiplier':round(mult, 2),
            'risk_factors':            factors,
            'evidence_citations':      list(set(citations)),
            'patient_specific_warning':warning,
            'severity':                _severity_label(adj_score),
            'icon':                    _severity_icon(_severity_label(adj_score)),
        })
        adjusted.append(entry)

    # Same fix as Step 4: use nlp_confidence as primary sort key so that
    # weight-inflated adjusted scores don't always push CARDIOVASCULAR to the top.
    adjusted.sort(key=lambda x: (x['nlp_confidence'], x['adjusted_score']), reverse=True)

    return {
        'status':           'ADJUSTED',
        'adjusted_systems': adjusted,
        'methodology':      'Patient adjustments: Mangoni 2004, KDIGO 2023, ADA 2023, ACC/AHA guidelines',
        'patient_profile':  {
            'age':          age,
            'num_conditions':len(patient_data.get('conditions', [])),
            'has_lab_data': bool(labs),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — POLYPHARMACY CASCADE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_polypharmacy_cascades(
    adjusted_systems: List[Dict],
    interactions_list: List[Dict],
    num_drugs: int,
) -> Dict:
    """
    STEP 6 — Flag organ systems hit by multiple overlapping interactions.
    """
    cum_thr  = CASCADE_DETECTION_THRESHOLD['cumulative']
    crit_thr = CASCADE_DETECTION_THRESHOLD['critical_score']
    ev       = CASCADE_DETECTION_THRESHOLD['evidence']

    cascades = []
    for sys in adjusted_systems:
        n     = sys.get('interaction_count', 0)
        score = sys.get('adjusted_score', sys.get('score', 0))

        is_cascade    = n >= cum_thr and score >= crit_thr
        is_cumulative = n >= cum_thr and score >= 25

        if is_cascade or is_cumulative:
            cascades.append({
                'organ_system':    sys['system'],
                'alert_level':     'CASCADE' if is_cascade else 'CUMULATIVE',
                'cumulative_score':score,
                'severity':        sys['severity'],
                'icon':            sys['icon'],
                'num_interactions':n,
                'evidence_citation':_cite(ev),
                'evidence_rationale':ev.rationale,
                'threshold_used':  f"{n} interactions ≥ {cum_thr} (evidence threshold)",
            })

    cascades.sort(key=lambda x: x['cumulative_score'], reverse=True)

    return {
        'has_cascades':               len(cascades) > 0,
        'num_cascades':               len(cascades),
        'cascades':                   cascades,
        'methodology':                f'Cascade thresholds: {_cite(ev)}',
        'polypharmacy_risk_multiplier':_ev(ev),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — CLINICAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_clinical_report(
    base_scores:       Dict,
    organ_analysis:    Dict,
    patient_adj:       Dict,
    cascade_detection: Dict,
    patient_data:      Dict = None,
) -> Dict:
    """
    STEP 7 — Assemble the final evidence-referenced clinical report,
    including an Explainable AI section that surfaces the model's reasoning.
    """
    systems   = patient_adj.get('adjusted_systems') or organ_analysis.get('affected_organ_systems', [])
    max_organ = max((s.get('adjusted_score', s.get('score', 0)) for s in systems), default=0)
    total     = base_scores['total_score']
    n_casc    = cascade_detection.get('num_cascades', 0)

    overall = _final_risk(total, max_organ, n_casc)

    # Collect all citations
    citations: set = set()
    for item in base_scores['detailed_breakdown']:
        citations.update(item.get('evidence_citations', []))
    for s in systems:
        if s.get('evidence_citation'): citations.add(s['evidence_citation'])
        citations.update(s.get('evidence_citations', []))
    for c in cascade_detection.get('cascades', []):
        if c.get('evidence_citation'): citations.add(c['evidence_citation'])

    # ── XAI SECTION ──────────────────────────────────────────────────────────
    xai = _build_xai_section(base_scores, organ_analysis, systems)

    return {
        'summary': {
            'overall_risk_level':  overall['level'],
            'risk_color':          overall['color'],
            'risk_icon':           overall['icon'],
            'primary_action':      overall['action'],
            'total_interaction_score': total,
            'num_interactions':    base_scores['num_interactions'],
            'num_organs_affected': len(systems),
            'num_cascades':        n_casc,
        },
        'interaction_analysis':          base_scores,
        'organ_system_analysis':         organ_analysis,
        'patient_specific_analysis':     patient_adj,
        'polypharmacy_cascade_analysis': cascade_detection,
        'explainability':                xai,          # ← NEW
        'evidence_base': {
            'all_citations': sorted(c for c in citations if c and c != 'No citation'),
            'methodology_summary': (
                'Severity scored by TF-IDF + Logistic Regression (99-example training corpus). '
                'Organ classification by TF-IDF + One-vs-Rest multi-label classifier (10 classes). '
                'Negation detection reduces scores to 30%. '
                'Organ weights from peer-reviewed mortality/hospitalisation studies. '
                'Patient adjustments from pharmacokinetic research and clinical guidelines.'
            ),
            'key_sources': [
                'FDA MedDRA Terminology v26.0 (2023)',
                'Mangoni AA, Jackson SH. Br J Clin Pharmacol. 2004;57(1):6-14',
                'KDIGO Clinical Practice Guidelines 2012–2023',
                'Tisdale JE, et al. Circulation. 2020;142(15):e214-e233',
                'Masnoon N, et al. BMC Geriatr. 2017;17(1):230',
            ],
        },
        'report_metadata': {
            'analysis_version':  'PolyGuard v2.1 NLP+Evidence-Based',
            'nlp_model':         'TF-IDF + LR severity | TF-IDF + OvR multi-label organ classifier',
            'training_examples': 99,
            'organ_classes':     10,
            'validation_status': 'Evidence-derived parameters from peer-reviewed literature',
        },
    }


def _build_xai_section(
    base_scores:    Dict,
    organ_analysis: Dict,
    systems:        List[Dict],
) -> Dict:
    """
    Build the Explainable AI section of the report.

    For every interaction, surfaces:
      • The model's confidence distribution across severity tiers
      • The 3 nearest training sentences (semantic similarity) with their known scores
      • Whether negation was detected and what it changed
      • The full organ probability vector (what organs the model considered)

    For the overall prediction:
      • Dominant evidence sources that drove the highest-scoring interactions
      • Low-confidence warnings (when top-bucket prob < 0.5)
    """
    per_interaction = []
    low_confidence_warnings = []

    for item in base_scores['detailed_breakdown']:
        sev_proba  = item.get('severity_proba', {})
        organ_prob = item.get('organ_proba_vec', {})
        neg        = item.get('has_negation', False)
        score      = item['score']
        severity   = item['severity']

        # Confidence of the predicted bucket
        confidence = sev_proba.get(severity, 0.0)

        # Build severity confidence bar
        sev_order = ['CRITICAL', 'SEVERE', 'MODERATE', 'MILD', 'MINIMAL']
        sev_dist  = {k: round(sev_proba.get(k, 0.0), 3) for k in sev_order}

        # Top-3 organs predicted by classifier
        top_organs = sorted(organ_prob.items(), key=lambda x: x[1], reverse=True)[:3]
        top_organs = [(o, round(p, 3)) for o, p in top_organs if p > 0.05]

        # Nearest reference sentences
        refs = item.get('nearest_refs', [])

        # Negation explanation
        neg_note = None
        if neg:
            neg_note = f"Negation detected — raw score discounted to 30% (score would be ~{score*3} without negation)"

        entry = {
            'drugs':           item['drugs'],
            'predicted':       severity,
            'score':           score,
            'model_confidence': round(confidence, 3),
            'severity_distribution': sev_dist,
            'top_predicted_organs':  top_organs,
            'nearest_training_refs': refs,
            'negation_detected':     neg,
            'negation_note':         neg_note,
            'processed_text':        item.get('processed_text', ''),
        }
        per_interaction.append(entry)

        # Flag low-confidence predictions for the clinician
        if confidence < 0.45 and score > 10:
            low_confidence_warnings.append(
                f"{item['drugs']}: model confidence {confidence:.0%} for {severity} — "
                f"manual review recommended"
            )

    # Overall model transparency summary
    avg_confidence = (
        sum(e['model_confidence'] for e in per_interaction) / len(per_interaction)
        if per_interaction else 0.0
    )

    # Which organs had the strongest NLP signal across all interactions
    organ_signal = {}
    for item in base_scores['detailed_breakdown']:
        for organ, prob in item.get('organ_proba_vec', {}).items():
            organ_signal[organ] = organ_signal.get(organ, 0.0) + prob
    top_signal_organs = sorted(organ_signal.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        'per_interaction':   per_interaction,
        'model_transparency': {
            'average_confidence':       round(avg_confidence, 3),
            'low_confidence_warnings':  low_confidence_warnings,
            'top_signal_organs':        [(o, round(p, 3)) for o, p in top_signal_organs],
            'negated_interactions':     sum(1 for e in per_interaction if e['negation_detected']),
            'nlp_methodology': (
                'Severity: TF-IDF(word,1-3gram) → Logistic Regression '
                '(5-class: MINIMAL/MILD/MODERATE/SEVERE/CRITICAL). '
                'Organs: TF-IDF(word,1-3gram) → One-vs-Rest LR (10 binary classifiers). '
                'Negation: detected by lexical cue scanning; reduces severity score to 30%. '
                'Nearest neighbours: TF-IDF cosine similarity against 99 labelled training examples.'
            ),
        },
        'how_to_read': (
            'severity_distribution shows model probability across all 5 severity tiers. '
            'top_predicted_organs are the organ systems the classifier assigned highest probability. '
            'nearest_training_refs are the most semantically similar labelled examples '
            'from the training corpus — they explain what the model compared this text against. '
            'low_confidence_warnings flag predictions where model certainty < 45%.'
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _severity_label(score: int) -> str:
    if   score >= 50: return 'CRITICAL'
    elif score >= 35: return 'SEVERE'
    elif score >= 20: return 'MODERATE'
    elif score >= 10: return 'MILD'
    else:             return 'MINIMAL'

def _severity_icon(sev: str) -> str:
    return {'CRITICAL':'🔴','SEVERE':'🟠','MODERATE':'🟡','MILD':'🟢','MINIMAL':'⚪'}.get(sev,'⚪')

def _overall_risk(total: int, n: int) -> Dict:
    avg = total / n if n else 0
    if   total >= 100 or avg >= 50: return {'level':'CRITICAL','color':'🔴','action':'❌ DO NOT COMBINE — Immediate medical consultation required'}
    elif total >=  60 or avg >= 35: return {'level':'SEVERE',  'color':'🟠','action':'⚠️  Use with extreme caution — Physician approval needed'}
    elif total >=  30 or avg >= 20: return {'level':'MODERATE','color':'🟡','action':'⚡ Monitor closely — Regular follow-up required'}
    else:                            return {'level':'MILD',    'color':'🟢','action':'✓  Generally safe — Routine monitoring'}

def _final_risk(total: int, max_organ: int, n_casc: int) -> Dict:
    if   max_organ >= 70 or n_casc >= 2 or total >= 100: return {'level':'CRITICAL','color':'🔴','icon':'🚨','action':'DO NOT COMBINE — Immediate medical intervention required'}
    elif max_organ >= 50 or n_casc >= 1 or total >= 60:  return {'level':'SEVERE',  'color':'🟠','icon':'⚠️', 'action':'Use with extreme caution — Physician consultation mandatory'}
    elif max_organ >= 30 or total >= 35:                  return {'level':'MODERATE','color':'🟡','icon':'⚡','action':'Monitor closely — Regular follow-up required'}
    else:                                                  return {'level':'MILD',    'color':'🟢','icon':'✓', 'action':'Generally safe — Routine monitoring'}


__all__ = [
    'calculate_interaction_score_robust',
    'analyze_biological_impact',
    'adjust_for_patient_context',
    'detect_polypharmacy_cascades',
    'generate_clinical_report',
]