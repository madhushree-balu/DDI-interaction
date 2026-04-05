"""
polyguard.core.analyser
=======================
``PolyGuardAnalyser`` is the single entry-point for the full PolyGuard pipeline.

It accepts plain Python dicts / lists so it can be called from:
  • FastAPI route handlers  (api layer)
  • CLI scripts             (scripts layer)
  • Jupyter notebooks
  • Unit tests

All I/O (CSV loading, JSON report writing) is kept outside this class.

Typical usage::

    from polyguard.core import PolyGuardAnalyser
    from polyguard.core.data_loader import DataLoader

    loader   = DataLoader("./datasets").load()
    analyser = PolyGuardAnalyser(loader)

    
    brands = analyser.search_brands("Aug")

    
    ings = analyser.get_ingredients("Augmentin 625 Duo Tablet")

    
    result = analyser.analyse(
        brand_names=["Augmentin 625 Duo Tablet", "Ascoril LS Syrup"],
        patient_data={"age": 72, "conditions": ["Hypertension"], "lab_values": {"eGFR": 42}},
    )
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .data_loader import DataLoader
from .models import (
    AnalysisResult,
    BrandSearchResult,
    CascadeAlert,
    Counterfactual,
    IngredientsResult,
    InteractionResult,
    OrganSystemRisk,
    ReportSummary,
    SeverityBreakdown,
    WaterfallStep,
    Waterfall,
    XAIEntry,
    XAIReport,
)

logger = logging.getLogger(__name__)


class PolyGuardAnalyser:
    """
    Orchestrates all pipeline steps (1-7) and returns structured Pydantic models.
    """

    def __init__(self, loader: DataLoader) -> None:
        if not loader._loaded:
            raise ValueError(
                "DataLoader must be loaded before passing to PolyGuardAnalyser.")
        self._loader = loader
        self._engine = None
        self._xai = None

    def search_brands(self, prefix: str, limit: int = 10) -> BrandSearchResult:

        results = self._loader.search_brands(prefix, limit)
        return BrandSearchResult(prefix=prefix, results=results, total_found=len(results))

    def get_ingredients(self, brand_name: str) -> IngredientsResult:

        ings = self._loader.get_ingredients(brand_name)
        return IngredientsResult(
            brand_name=brand_name,
            ingredients=ings,
            found=bool(ings),
        )

    def analyse(
        self,
        brand_names: List[str],
        patient_data: Optional[Dict] = None,
        explain: bool = True,
    ) -> AnalysisResult:
        self._ensure_engine()

        brand_ingredient_map: Dict[str, List[str]] = {}
        all_ingredients: List[str] = []

        for brand in brand_names:
            ings = self._loader.get_ingredients(brand)
            brand_ingredient_map[brand] = ings
            all_ingredients.extend(ings)

        unique_ingredients = list(dict.fromkeys(all_ingredients))
        raw_interactions = self._loader.lookup_interactions(unique_ingredients)

        if not raw_interactions:
            return AnalysisResult(
                status="NO_INTERACTIONS",
                message="No known interactions found between these medications.",
                brand_names=brand_names,
                brand_ingredient_map=brand_ingredient_map,
                all_ingredients=unique_ingredients,
            )

        engine = self._engine
        base_scores = engine.calculate_interaction_score_robust(
            raw_interactions)

        organ_analysis = engine.analyze_biological_impact(
            raw_interactions, base_scores)

        if patient_data:
            patient_adj = engine.adjust_for_patient_context(
                organ_analysis["affected_organ_systems"], patient_data
            )
        else:
            patient_adj = {
                "status": "NO_PATIENT_DATA",
                "adjusted_systems": organ_analysis["affected_organ_systems"],
            }

        cascade_detection = engine.detect_polypharmacy_cascades(
            patient_adj.get("adjusted_systems", []),
            raw_interactions,
            len(unique_ingredients),
        )

        clinical_report = engine.generate_clinical_report(
            base_scores, organ_analysis, patient_adj, cascade_detection, patient_data
        )

        xai_result: Optional[XAIReport] = None
        if explain:
            try:
                from polyguard.core.xai_explainer import generate_xai_report
                raw_xai = generate_xai_report(
                    base_scores=base_scores,
                    organ_analysis=organ_analysis,
                    patient_adj=patient_adj,
                    cascade_detection=cascade_detection,
                    interactions_list=raw_interactions,
                    patient_data=patient_data,
                )
                xai_result = self._map_xai(raw_xai)
            except Exception as exc:
                logger.warning("XAI generation failed: %s", exc)

        return self._build_result(
            brand_names=brand_names,
            brand_ingredient_map=brand_ingredient_map,
            all_ingredients=unique_ingredients,
            raw_interactions=raw_interactions,
            base_scores=base_scores,
            organ_analysis=organ_analysis,
            patient_adj=patient_adj,
            cascade_detection=cascade_detection,
            clinical_report=clinical_report,
            xai_result=xai_result,
            patient_adjusted=patient_data is not None,
        )

    def _ensure_engine(self) -> None:
        if self._engine is not None:
            return
        import polyguard_engine_evidence_based as _eng
        self._engine = _eng

    def _build_result(
        self,
        brand_names: List[str],
        brand_ingredient_map: Dict,
        all_ingredients: List[str],
        raw_interactions: List[Dict],
        base_scores: Dict,
        organ_analysis: Dict,
        patient_adj: Dict,
        cascade_detection: Dict,
        clinical_report: Dict,
        xai_result: Optional[XAIReport],
        patient_adjusted: bool,
    ) -> AnalysisResult:
        summary_raw = clinical_report.get("summary", {})
        systems = patient_adj.get("adjusted_systems") or organ_analysis.get(
            "affected_organ_systems", [])

        interactions = [InteractionResult(**i) for i in raw_interactions]

        breakdown = [
            SeverityBreakdown(
                drugs=b.get("drugs", ""),
                score=b.get("score", 0),
                severity=b.get("severity", ""),
                icon=b.get("icon", ""),
                description=b.get("description", ""),
                mechanism=b.get("mechanism"),
                has_negation=b.get("has_negation", False),
                severity_proba=b.get("severity_proba", {}),
                nearest_refs=b.get("nearest_refs", []),
            )
            for b in base_scores.get("detailed_breakdown", [])
        ]

        organ_systems = [
            OrganSystemRisk(
                system=s.get("system", ""),
                organ_key=s.get("organ_key", ""),
                score=s.get("score", 0),
                adjusted_score=s.get("adjusted_score"),
                severity=s.get("severity", ""),
                icon=s.get("icon", ""),
                nlp_confidence=s.get("nlp_confidence"),
                vulnerability_multiplier=s.get("vulnerability_multiplier"),
                risk_factors=s.get("risk_factors", []),
                evidence_citation=s.get("evidence_citation"),
            )
            for s in systems
        ]

        cascades = [
            CascadeAlert(
                organ_system=c.get("organ_system", ""),
                alert_level=c.get("alert_level", ""),
                cumulative_score=c.get("cumulative_score", 0),
                severity=c.get("severity", ""),
                num_interactions=c.get("num_interactions", 0),
                evidence_rationale=c.get("evidence_rationale", ""),
            )
            for c in cascade_detection.get("cascades", [])
        ]

        summary = ReportSummary(
            overall_risk_level=summary_raw.get(
                "overall_risk_level", "MINIMAL"),
            risk_color=summary_raw.get("risk_color", "⚪"),
            risk_icon=summary_raw.get("risk_icon", ""),
            primary_action=summary_raw.get("primary_action", ""),
            total_interaction_score=summary_raw.get(
                "total_interaction_score", 0),
            num_interactions=summary_raw.get("num_interactions", 0),
            num_organs_affected=summary_raw.get("num_organs_affected", 0),
            num_cascades=summary_raw.get("num_cascades", 0),
        ) if summary_raw else None

        highest = organ_analysis.get("highest_risk_organ")
        highest_name = highest["system"] if highest else None

        return AnalysisResult(
            status="INTERACTIONS_FOUND",
            brand_names=brand_names,
            brand_ingredient_map=brand_ingredient_map,
            all_ingredients=all_ingredients,
            interactions_found=interactions,
            total_score=base_scores.get("total_score", 0),
            risk_level=base_scores.get("risk_level", "MINIMAL"),
            risk_color=base_scores.get("risk_color", "⚪"),
            severity_breakdown=breakdown,
            organ_systems=organ_systems,
            num_organs_affected=organ_analysis.get("num_organs_affected", 0),
            highest_risk_organ=highest_name,
            patient_adjusted=patient_adjusted,
            cascades=cascades,
            summary=summary,
            xai=xai_result,
        )

    def _map_xai(self, raw: Dict) -> XAIReport:
        """Map the raw xai dict from xai_explainer into a typed XAIReport."""
        entries: List[XAIEntry] = []
        for ex in raw.get("step3_interaction_explanations", []):
            sxai = ex.get("severity_xai", {})
            organ_xais = []
            for ox in ex.get("organ_xais", []):
                organ_xais.append({
                    "organ":       ox.get("organ", ""),
                    "probability": ox.get("probability", 0.0),
                    "explanation": ox.get("explanation", ""),
                    "top_features": ox.get("top_features", []),
                })
            entries.append(XAIEntry(
                drugs=ex.get("drugs", ""),
                score=ex.get("score", 0),
                severity=ex.get("severity", ""),
                explanation=sxai.get("explanation", ""),
                supporting_terms=sxai.get("supporting", []),
                organ_explanations=organ_xais,
                severity_bars=ex.get("severity_bars", ""),
                organ_bars=ex.get("organ_bars", ""),
                nearest_refs=ex.get("nearest_refs", []),
            ))

        waterfalls: List[Waterfall] = []
        for wf in raw.get("step5_waterfalls", []):
            steps = [WaterfallStep(**s) for s in wf.get("steps", [])]
            waterfalls.append(Waterfall(
                organ=wf.get("organ", ""),
                base_score=wf.get("base_score", 0),
                final_score=wf.get("final_score", 0),
                total_delta=wf.get("total_delta", 0),
                steps=steps,
                text=wf.get("text", ""),
                bar_chart=wf.get("bar_chart", ""),
            ))

        cfs: List[Counterfactual] = []
        for cf in raw.get("step5_counterfactuals", []):
            cfs.append(Counterfactual(
                organ=cf.get("organ", ""),
                current_score=cf.get("current_score", 0),
                top_action=cf.get("top_action", ""),
                items=cf.get("counterfactuals", []),
            ))

        return XAIReport(
            step3_interaction_explanations=entries,
            step5_waterfalls=waterfalls,
            step5_counterfactuals=cfs,
            step6_cascade_attributions=raw.get(
                "step6_cascade_attributions", []),
            has_patient_xai=raw.get("has_patient_xai", False),
            has_counterfactuals=raw.get("has_counterfactuals", False),
            has_cascade_xai=raw.get("has_cascade_xai", False),
            methodology=raw.get("methodology", ""),
        )
