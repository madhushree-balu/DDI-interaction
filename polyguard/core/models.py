"""
polyguard.core.models
=====================
Pydantic data-models for every request / response object in the pipeline.

Using Pydantic v2 (pydantic>=2.0).  All models are:
  • Serialisable to / from JSON with .model_dump() / .model_validate()
  • Validated on construction (type-safe)
  • Used both by the core engine AND the FastAPI layer (zero duplication)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


# REQUEST MODELS

class PatientData(BaseModel):
    """Optional patient context passed to the full analysis pipeline."""

    age: Optional[int] = Field(None, ge=0, le=130, description="Patient age in years")
    gender: Optional[str] = Field(None, description="Patient gender")
    conditions: List[str] = Field(
        default_factory=list,
        description="Active diagnoses e.g. ['Hypertension', 'Diabetes Type 2']",
    )
    lab_values: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Lab results keyed by test name. "
            "Recognised keys: eGFR, ALT, AST, platelet_count, INR, blood_glucose"
        ),
    )

    model_config = {"json_schema_extra": {
        "example": {
            "age": 72,
            "gender": "Female",
            "conditions": ["Hypertension", "Diabetes Type 2", "Atrial Fibrillation"],
            "lab_values": {"eGFR": 42, "ALT": 85, "platelet_count": 110, "INR": 3.2},
        }
    }}


class AnalysisRequest(BaseModel):
    """Request body for POST /analyse."""

    brand_names: List[str] = Field(
        ...,
        min_length=1,
        description="One or more brand medication names to analyse",
    )
    patient_data: Optional[PatientData] = Field(
        None,
        description="Optional patient profile for risk adjustment (Step 5)",
    )
    explain: bool = Field(
        True,
        description="Include XAI explanations in the response",
    )

    @field_validator("brand_names")
    @classmethod
    def brand_names_not_empty(cls, v: List[str]) -> List[str]:
        cleaned = [b.strip() for b in v if b.strip()]
        if not cleaned:
            raise ValueError("brand_names must contain at least one non-empty string")
        return cleaned

    model_config = {"json_schema_extra": {
        "example": {
            "brand_names": ["Augmentin 625 Duo Tablet", "Ascoril LS Syrup"],
            "patient_data": {
                "age": 65,
                "conditions": ["Hypertension"],
                "lab_values": {"eGFR": 58},
            },
            "explain": True,
        }
    }}


class BrandSearchRequest(BaseModel):
    """Request body for GET /brands/search."""

    prefix: str = Field(..., min_length=1, description="Brand name prefix to search")
    limit: int = Field(10, ge=1, le=50, description="Maximum results to return")


# RESPONSE MODELS

class BrandSearchResult(BaseModel):
    """Response from brand name prefix search."""

    prefix: str
    results: List[str]
    total_found: int


class IngredientsResult(BaseModel):
    """Response from ingredient lookup for a single brand."""

    brand_name: str
    ingredients: List[str]
    found: bool


class InteractionResult(BaseModel):
    """A single pairwise drug-drug interaction."""

    drug_a: str
    drug_b: str
    description: str
    severity: str
    mechanism: str
    source: str


class SeverityBreakdown(BaseModel):
    """NLP severity prediction for one interaction."""

    drugs: str
    score: int
    severity: str
    icon: str
    description: str
    mechanism: Optional[str] = None
    has_negation: bool = False
    severity_proba: Dict[str, float] = Field(default_factory=dict)
    nearest_refs: List[str] = Field(default_factory=list)


class OrganSystemRisk(BaseModel):
    """Risk assessment for a single organ system."""

    system: str
    organ_key: str
    score: int
    adjusted_score: Optional[int] = None
    severity: str
    icon: str
    nlp_confidence: Optional[float] = None
    vulnerability_multiplier: Optional[float] = None
    risk_factors: List[str] = Field(default_factory=list)
    evidence_citation: Optional[str] = None


class CascadeAlert(BaseModel):
    """Polypharmacy cascade detected on one organ system."""

    organ_system: str
    alert_level: str
    cumulative_score: int
    severity: str
    num_interactions: int
    evidence_rationale: str


class ReportSummary(BaseModel):
    """High-level summary block in the clinical report."""

    overall_risk_level: str
    risk_color: str
    risk_icon: str
    primary_action: str
    total_interaction_score: int
    num_interactions: int
    num_organs_affected: int
    num_cascades: int


class XAIEntry(BaseModel):
    """XAI explanation for a single interaction (Steps 3 & 4)."""

    drugs: str
    score: int
    severity: str
    explanation: str
    supporting_terms: List[Tuple[str, float]] = Field(default_factory=list)
    organ_explanations: List[Dict[str, Any]] = Field(default_factory=list)
    severity_bars: str = ""
    organ_bars: str = ""
    nearest_refs: List[str] = Field(default_factory=list)


class WaterfallStep(BaseModel):
    """One step in a waterfall score decomposition."""

    label: str
    delta: int
    running: int
    type: str  # 'base' | 'increase' | 'decrease' | 'total'


class Waterfall(BaseModel):
    """Full waterfall chart for one organ system (Step 5 XAI)."""

    organ: str
    base_score: int
    final_score: int
    total_delta: int
    steps: List[WaterfallStep]
    text: str
    bar_chart: str


class Counterfactual(BaseModel):
    """What-if risk reduction scenario (Step 5/7 XAI)."""

    organ: str
    current_score: int
    top_action: str
    items: List[Dict[str, Any]] = Field(default_factory=list)


class XAIReport(BaseModel):
    """Complete XAI report covering all pipeline steps."""

    step3_interaction_explanations: List[XAIEntry] = Field(default_factory=list)
    step5_waterfalls: List[Waterfall] = Field(default_factory=list)
    step5_counterfactuals: List[Counterfactual] = Field(default_factory=list)
    step6_cascade_attributions: List[Dict[str, Any]] = Field(default_factory=list)
    has_patient_xai: bool = False
    has_counterfactuals: bool = False
    has_cascade_xai: bool = False
    methodology: str = ""


class AnalysisResult(BaseModel):
    """
    Full response from POST /analyse.

    status is one of:
      'NO_INTERACTIONS'  — no pairwise interactions found in the database
      'INTERACTIONS_FOUND' — one or more interactions detected
      'ERROR'            — pipeline error (see message)
    """

    status: str
    message: Optional[str] = None

    # Step 1 & 2
    brand_names: List[str] = Field(default_factory=list)
    brand_ingredient_map: Dict[str, List[str]] = Field(default_factory=dict)
    all_ingredients: List[str] = Field(default_factory=list)
    interactions_found: List[InteractionResult] = Field(default_factory=list)

    # Step 3
    total_score: int = 0
    risk_level: str = "MINIMAL"
    risk_color: str = "⚪"
    severity_breakdown: List[SeverityBreakdown] = Field(default_factory=list)

    # Step 4
    organ_systems: List[OrganSystemRisk] = Field(default_factory=list)
    num_organs_affected: int = 0
    highest_risk_organ: Optional[str] = None

    # Step 5
    patient_adjusted: bool = False

    # Step 6
    cascades: List[CascadeAlert] = Field(default_factory=list)

    # Step 7
    summary: Optional[ReportSummary] = None

    # XAI
    xai: Optional[XAIReport] = None

    model_config = {"json_schema_extra": {
        "example": {
            "status": "INTERACTIONS_FOUND",
            "brand_names": ["Augmentin 625 Duo Tablet"],
            "total_score": 42,
            "risk_level": "MODERATE",
        }
    }}