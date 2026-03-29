"""
polyguard/api/blueprint.py
==========================
Flask Blueprint covering every endpoint the PolyGuard UI needs.

UI flow
-------
  Step 1 — Search
    User types a tablet name → typeahead calls GET /api/brands/search
    Results populate a dropdown.

  Step 2 — Select & preview
    User picks a brand → GET /api/brands/<brand>/ingredients
    shows the active ingredients before they commit.
    User can add more brands; each triggers another ingredient call.
    GET /api/brands/batch-ingredients lets the UI do one call for several
    already-selected brands at once (page refresh / restore state).

  Step 3 — (Optional) patient context
    User fills an optional form; data is sent with the analysis request.

  Step 4 — Analyse
    POST /api/analyse  with the final brand list + optional patient data.
    Returns the full structured result in one JSON response.

  Step 5 — Display results
    GET /api/results/<result_id>  fetches a previously-run result by ID
    (result_id is returned by POST /analyse so the UI can bookmark/share).

  Step 6 — Download
    GET /api/results/<result_id>/download  returns a clean JSON report
    with Content-Disposition: attachment so the browser saves it.

Registering the blueprint
-------------------------
In your Flask app factory (e.g. app.py at the project root)::

    from flask import Flask
    from polyguard.api.blueprint import api_bp, init_analyser

    app = Flask(__name__)
    init_analyser(data_dir="./datasets")   # loads datasets once
    app.register_blueprint(api_bp, url_prefix="/api")

    if __name__ == "__main__":
        app.run(debug=True, port=5000)

Environment variables
---------------------
POLYGUARD_DATA_DIR   path to datasets directory (default: ./datasets)
POLYGUARD_LOG_LEVEL  logging level            (default: INFO)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional

from flask import Blueprint, Response, jsonify, request

from polyguard.core.analyser import PolyGuardAnalyser
from polyguard.core.data_loader import DataLoader

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETONS
# Initialised once via init_analyser(); shared across all requests.
# ─────────────────────────────────────────────────────────────────────────────

_analyser: Optional[PolyGuardAnalyser] = None

# In-memory result store: result_id -> result dict
# Replace with Redis / DB in production.
_result_store: Dict[str, Dict[str, Any]] = {}


def init_analyser(data_dir: str = "./datasets") -> None:
    """
    Load datasets and initialise the analyser singleton.
    Call this once from your Flask app factory before the first request.
    """
    global _analyser
    log_level = os.getenv("POLYGUARD_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
    data_dir = os.getenv("POLYGUARD_DATA_DIR", data_dir)
    logger.info("PolyGuard: loading datasets from %s", data_dir)
    loader    = DataLoader(data_dir=data_dir).load()
    _analyser = PolyGuardAnalyser(loader)
    logger.info("PolyGuard: analyser ready.")


# ─────────────────────────────────────────────────────────────────────────────
# BLUEPRINT
# ─────────────────────────────────────────────────────────────────────────────

api_bp = Blueprint("api", __name__)


# ── Guard decorator ───────────────────────────────────────────────────────────

def _require_analyser(fn):
    """Return 503 if init_analyser() has not been called yet."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if _analyser is None:
            return jsonify({"error": "Service not ready. Call init_analyser() at startup."}), 503
        return fn(*args, **kwargs)
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.get("/health")
def health():
    """
    Liveness probe.

    Response 200
    ------------
    {
        "status": "ok",
        "version": "2.1.0",
        "analyser_ready": true
    }
    """
    return jsonify({
        "status":         "ok",
        "version":        "2.1.0",
        "analyser_ready": _analyser is not None,
    })


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — BRAND SEARCH  (typeahead)
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.get("/brands/search")
@_require_analyser
def search_brands():
    """
    Typeahead search — returns brand names starting with *q*.
    Called on every keystroke (debounce on the frontend, min 2 chars).

    Query params
    ------------
    q      : str  — search prefix (required, min 2 chars)
    limit  : int  — max results to return (default 10, max 50)

    Response 200
    ------------
    {
        "query":       "Aug",
        "results":     ["Augmentin 625 Duo Tablet", "Augmentin 375 Tablet", ...],
        "total_found": 3
    }

    Response 400  — q missing or too short
    """
    q     = request.args.get("q", "").strip()
    limit = min(int(request.args.get("limit", 10)), 50)

    if len(q) < 2:
        return jsonify({"error": "Query param 'q' must be at least 2 characters."}), 400

    result = _analyser.search_brands(prefix=q, limit=limit)
    return jsonify({
        "query":       q,
        "results":     result.results,
        "total_found": result.total_found,
    })


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — INGREDIENT PREVIEW  (single brand)
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.get("/brands/<path:brand_name>/ingredients")
@_require_analyser
def get_ingredients(brand_name: str):
    """
    Resolve one brand name to its active ingredients.
    Called when the user selects a brand from the search dropdown.

    Path param
    ----------
    brand_name : exact brand name (URL-encoded if it contains spaces)

    Response 200
    ------------
    {
        "brand_name":  "Augmentin 625 Duo Tablet",
        "ingredients": ["Amoxycillin", "Clavulanic Acid"],
        "found":       true
    }

    Response 404  — brand not in database
    """
    result = _analyser.get_ingredients(brand_name)
    if not result.found:
        return jsonify({
            "brand_name":  brand_name,
            "ingredients": [],
            "found":       False,
            "message":     f"Brand '{brand_name}' not found in the database.",
        }), 404

    return jsonify({
        "brand_name":  result.brand_name,
        "ingredients": result.ingredients,
        "found":       result.found,
    })


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2b — BATCH INGREDIENT PREVIEW  (multiple brands at once)
# Used when the UI restores a saved selection or the user pastes a list.
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.post("/brands/batch-ingredients")
@_require_analyser
def batch_ingredients():
    """
    Resolve several brand names to their ingredients in one request.
    Use this instead of N sequential GET calls when restoring UI state.

    Request body (JSON)
    -------------------
    {
        "brands": ["Augmentin 625 Duo Tablet", "Ascoril LS Syrup"]
    }

    Response 200
    ------------
    {
        "results": [
            {
                "brand_name":  "Augmentin 625 Duo Tablet",
                "ingredients": ["Amoxycillin", "Clavulanic Acid"],
                "found":       true
            },
            {
                "brand_name":  "Ascoril LS Syrup",
                "ingredients": ["Ambroxol", "Guaifenesin", "Levosalbutamol"],
                "found":       true
            }
        ],
        "all_ingredients": ["Amoxycillin", "Clavulanic Acid", "Ambroxol", ...]
    }

    Response 400  — body missing or brands list empty
    """
    body   = request.get_json(silent=True) or {}
    brands = body.get("brands", [])

    if not brands or not isinstance(brands, list):
        return jsonify({"error": "'brands' must be a non-empty list."}), 400

    results        = []
    all_ingredients = []
    seen            = set()

    for brand in brands:
        r = _analyser.get_ingredients(str(brand).strip())
        results.append({
            "brand_name":  r.brand_name,
            "ingredients": r.ingredients,
            "found":       r.found,
        })
        for ing in r.ingredients:
            if ing not in seen:
                all_ingredients.append(ing)
                seen.add(ing)

    return jsonify({
        "results":         results,
        "all_ingredients": all_ingredients,
    })


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — RUN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.post("/analyse")
@_require_analyser
def analyse():
    """
    Run the full PolyGuard pipeline (Steps 1-7) and store the result.

    Request body (JSON)
    -------------------
    {
        "brand_names": ["Augmentin 625 Duo Tablet", "Ascoril LS Syrup"],

        // optional — include for patient-adjusted risk scoring
        "patient_data": {
            "age":        72,
            "gender":     "Female",
            "conditions": ["Hypertension", "Diabetes Type 2"],
            "lab_values": {
                "eGFR":          42,
                "ALT":           85,
                "platelet_count": 110,
                "INR":           3.2,
                "blood_glucose": 195
            }
        },

        // set false to skip XAI explanations and respond faster
        "explain": true
    }

    Response 200
    ------------
    {
        "result_id":   "a3f2...",       // use for GET /results/<id>
        "status":      "INTERACTIONS_FOUND",
        "risk_level":  "MODERATE",
        "risk_color":  "🟡",
        "total_score": 42,
        "num_interactions": 3,
        "num_organs_affected": 4,
        "num_cascades": 1,
        "primary_action": "Monitor closely — Regular follow-up required",

        "brand_ingredient_map": {
            "Augmentin 625 Duo Tablet": ["Amoxycillin", "Clavulanic Acid"],
            ...
        },
        "interactions": [
            {
                "drug_a":      "Amoxycillin",
                "drug_b":      "Clarithromycin",
                "severity":    "MODERATE",
                "score":       20,
                "icon":        "🟡",
                "description": "...",
                "mechanism":   "..."
            },
            ...
        ],
        "organ_systems": [
            {
                "system":        "Hepatic",
                "severity":      "MODERATE",
                "score":         18,
                "adjusted_score": 26,
                "icon":          "🟡",
                "nlp_confidence": 0.82,
                "risk_factors":  ["Age 72 yrs", "Elevated ALT 85 U/L"]
            },
            ...
        ],
        "cascades": [
            {
                "organ_system":     "Hepatic",
                "alert_level":      "CASCADE",
                "cumulative_score": 38,
                "num_interactions": 2
            }
        ],
        "xai": { ... }   // only if explain=true
    }

    Response 400  — brand_names missing or empty
    Response 500  — pipeline error
    """
    body = request.get_json(silent=True) or {}

    brand_names  = body.get("brand_names", [])
    patient_data = body.get("patient_data")
    explain      = bool(body.get("explain", True))

    if not brand_names or not isinstance(brand_names, list):
        return jsonify({"error": "'brand_names' must be a non-empty list."}), 400

    brand_names = [str(b).strip() for b in brand_names if str(b).strip()]
    if not brand_names:
        return jsonify({"error": "'brand_names' contained no valid entries."}), 400

    try:
        result = _analyser.analyse(
            brand_names  = brand_names,
            patient_data = patient_data,
            explain      = explain,
        )
    except Exception as exc:
        logger.exception("Analysis pipeline failed")
        return jsonify({"error": str(exc)}), 500

    result_dict = result.model_dump()

    # Store for later retrieval
    result_id = str(uuid.uuid4())
    _result_store[result_id] = {
        "result_id":  result_id,
        "created_at": datetime.utcnow().isoformat(),
        "request":    {
            "brand_names":   brand_names,
            "has_patient":   patient_data is not None,
            "explain":       explain,
        },
        "data": result_dict,
    }

    # Build the flat summary response the UI needs
    summary = result_dict.get("summary") or {}
    return jsonify({
        "result_id":         result_id,
        "status":            result_dict["status"],
        "risk_level":        result_dict.get("risk_level", "MINIMAL"),
        "risk_color":        result_dict.get("risk_color", "⚪"),
        "total_score":       result_dict.get("total_score", 0),
        "primary_action":    summary.get("primary_action", ""),
        "num_interactions":  len(result_dict.get("interactions_found", [])),
        "num_organs_affected": result_dict.get("num_organs_affected", 0),
        "num_cascades":      len(result_dict.get("cascades", [])),

        "brand_ingredient_map": result_dict.get("brand_ingredient_map", {}),
        "all_ingredients":      result_dict.get("all_ingredients", []),

        "interactions": [
            {
                "drug_a":      ix.get("drug_a"),
                "drug_b":      ix.get("drug_b"),
                "severity":    bd.get("severity"),
                "score":       bd.get("score"),
                "icon":        bd.get("icon"),
                "description": ix.get("description"),
                "mechanism":   ix.get("mechanism"),
            }
            for ix, bd in zip(
                result_dict.get("interactions_found", []),
                result_dict.get("severity_breakdown", []),
            )
        ],

        "organ_systems": [
            {
                "system":             s.get("system"),
                "severity":           s.get("severity"),
                "score":              s.get("adjusted_score") or s.get("score"),
                "base_score":         s.get("score"),
                "icon":               s.get("icon"),
                "nlp_confidence":     s.get("nlp_confidence"),
                "vulnerability_multiplier": s.get("vulnerability_multiplier"),
                "risk_factors":       s.get("risk_factors", []),
            }
            for s in result_dict.get("organ_systems", [])
        ],

        "cascades": [
            {
                "organ_system":     c.get("organ_system"),
                "alert_level":      c.get("alert_level"),
                "cumulative_score": c.get("cumulative_score"),
                "severity":         c.get("severity"),
                "num_interactions": c.get("num_interactions"),
            }
            for c in result_dict.get("cascades", [])
        ],

        "xai": result_dict.get("xai") if explain else None,
    })


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — FETCH A STORED RESULT
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.get("/results/<result_id>")
def get_result(result_id: str):
    """
    Fetch a previously-run analysis result by ID.
    The result_id is returned by POST /analyse.
    Use this to restore the results page after navigation or share a link.

    Response 200  — full result dict (same shape as POST /analyse response)
    Response 404  — result_id not found or expired
    """
    stored = _result_store.get(result_id)
    if not stored:
        return jsonify({
            "error":     f"Result '{result_id}' not found.",
            "hint":      "Results are held in memory and lost on server restart.",
        }), 404

    data = stored["data"]
    summary = data.get("summary") or {}
    return jsonify({
        "result_id":         result_id,
        "created_at":        stored["created_at"],
        "request":           stored["request"],
        "status":            data["status"],
        "risk_level":        data.get("risk_level", "MINIMAL"),
        "risk_color":        data.get("risk_color", "⚪"),
        "total_score":       data.get("total_score", 0),
        "primary_action":    summary.get("primary_action", ""),
        "num_interactions":  len(data.get("interactions_found", [])),
        "num_organs_affected": data.get("num_organs_affected", 0),
        "num_cascades":      len(data.get("cascades", [])),
        "brand_ingredient_map": data.get("brand_ingredient_map", {}),
        "all_ingredients":   data.get("all_ingredients", []),
        "interactions": [
            {
                "drug_a":      ix.get("drug_a"),
                "drug_b":      ix.get("drug_b"),
                "severity":    bd.get("severity"),
                "score":       bd.get("score"),
                "icon":        bd.get("icon"),
                "description": ix.get("description"),
                "mechanism":   ix.get("mechanism"),
            }
            for ix, bd in zip(
                data.get("interactions_found", []),
                data.get("severity_breakdown", []),
            )
        ],
        "organ_systems": [
            {
                "system":            s.get("system"),
                "severity":          s.get("severity"),
                "score":             s.get("adjusted_score") or s.get("score"),
                "base_score":        s.get("score"),
                "icon":              s.get("icon"),
                "nlp_confidence":    s.get("nlp_confidence"),
                "vulnerability_multiplier": s.get("vulnerability_multiplier"),
                "risk_factors":      s.get("risk_factors", []),
            }
            for s in data.get("organ_systems", [])
        ],
        "cascades": [
            {
                "organ_system":     c.get("organ_system"),
                "alert_level":      c.get("alert_level"),
                "cumulative_score": c.get("cumulative_score"),
                "severity":         c.get("severity"),
                "num_interactions": c.get("num_interactions"),
            }
            for c in data.get("cascades", [])
        ],
        "xai": data.get("xai"),
    })


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — DOWNLOAD REPORT
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.get("/results/<result_id>/download")
def download_result(result_id: str):
    """
    Download the full raw analysis result as a JSON file.
    The browser will prompt "Save As" due to Content-Disposition: attachment.

    Response 200  — application/json file download
    Response 404  — result_id not found
    """
    stored = _result_store.get(result_id)
    if not stored:
        return jsonify({"error": f"Result '{result_id}' not found."}), 404

    payload = {
        "result_id":  result_id,
        "created_at": stored["created_at"],
        "request":    stored["request"],
        "result":     stored["data"],
    }
    filename = f"polyguard_report_{result_id[:8]}.json"
    return Response(
        json.dumps(payload, indent=2, default=str),
        mimetype     = "application/json",
        headers      = {"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXTRAS — small helpers the UI may call
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.get("/ingredients/search")
@_require_analyser
def search_ingredients():
    """
    Search by ingredient name directly (bypass brand lookup).
    Useful if the user wants to type a generic drug name instead of a brand.

    Query params
    ------------
    q     : ingredient name prefix (min 2 chars)
    limit : max results (default 10)

    Response 200
    ------------
    {
        "query":   "amox",
        "results": ["Amoxycillin", "Amoxicillin Trihydrate", ...]
    }
    """
    q     = request.args.get("q", "").strip()
    limit = min(int(request.args.get("limit", 10)), 50)

    if len(q) < 2:
        return jsonify({"error": "Query param 'q' must be at least 2 characters."}), 400

    # Reuse the loader's pharma_db directly for ingredient search
    loader = _analyser._loader
    if loader._pharma_db.empty:
        return jsonify({"query": q, "results": []})

    col = "primary_ingredient"
    if col not in loader._pharma_db.columns:
        return jsonify({"query": q, "results": []})

    mask    = loader._pharma_db[col].str.lower().str.startswith(q.lower(), na=False)
    results = (
        loader._pharma_db[mask][col]
        .dropna()
        .drop_duplicates()
        .head(limit)
        .tolist()
    )
    return jsonify({"query": q, "results": results})


@api_bp.post("/ingredients/analyse")
@_require_analyser
def analyse_by_ingredients():
    """
    Run analysis directly from ingredient names — no brand lookup step.
    Use when the user has searched by ingredient instead of brand name.

    Request body (JSON)
    -------------------
    {
        "ingredients": ["Amoxycillin", "Clarithromycin", "Omeprazole"],
        "patient_data": { ... },   // optional
        "explain": true            // optional, default true
    }

    Response — same shape as POST /analyse
    """
    body        = request.get_json(silent=True) or {}
    ingredients = body.get("ingredients", [])
    patient_data = body.get("patient_data")
    explain      = bool(body.get("explain", True))

    if not ingredients or not isinstance(ingredients, list):
        return jsonify({"error": "'ingredients' must be a non-empty list."}), 400

    ingredients = [str(i).strip() for i in ingredients if str(i).strip()]

    # Build synthetic brand names so the result shape is consistent
    synthetic_brands = [f"[ingredient] {i}" for i in ingredients]

    try:
        # Directly look up interactions without the brand→ingredient resolution step
        loader = _analyser._loader
        raw    = loader.lookup_interactions(ingredients)

        if not raw:
            return jsonify({
                "result_id":        None,
                "status":           "NO_INTERACTIONS",
                "message":          "No known interactions found between these ingredients.",
                "all_ingredients":  ingredients,
            })

        _analyser._ensure_engine()
        engine      = _analyser._engine
        base        = engine.calculate_interaction_score_robust(raw)
        organ       = engine.analyze_biological_impact(raw, base)
        patient_adj = (
            engine.adjust_for_patient_context(organ["affected_organ_systems"], patient_data)
            if patient_data
            else {"status": "NO_PATIENT_DATA", "adjusted_systems": organ["affected_organ_systems"]}
        )
        cascade = engine.detect_polypharmacy_cascades(
            patient_adj.get("adjusted_systems", []), raw, len(ingredients)
        )
        report = engine.generate_clinical_report(base, organ, patient_adj, cascade, patient_data)

        result = _analyser._build_result(
            brand_names          = synthetic_brands,
            brand_ingredient_map = {b: [i] for b, i in zip(synthetic_brands, ingredients)},
            all_ingredients      = ingredients,
            raw_interactions     = raw,
            base_scores          = base,
            organ_analysis       = organ,
            patient_adj          = patient_adj,
            cascade_detection    = cascade,
            clinical_report      = report,
            xai_result           = None,
            patient_adjusted     = patient_data is not None,
        )
    except Exception as exc:
        logger.exception("Ingredient analysis failed")
        return jsonify({"error": str(exc)}), 500

    result_dict = result.model_dump()
    result_id   = str(uuid.uuid4())
    _result_store[result_id] = {
        "result_id":  result_id,
        "created_at": datetime.utcnow().isoformat(),
        "request":    {"ingredients": ingredients, "has_patient": patient_data is not None},
        "data":       result_dict,
    }

    summary = result_dict.get("summary") or {}
    return jsonify({
        "result_id":         result_id,
        "status":            result_dict["status"],
        "risk_level":        result_dict.get("risk_level", "MINIMAL"),
        "risk_color":        result_dict.get("risk_color", "⚪"),
        "total_score":       result_dict.get("total_score", 0),
        "primary_action":    summary.get("primary_action", ""),
        "num_interactions":  len(result_dict.get("interactions_found", [])),
        "num_organs_affected": result_dict.get("num_organs_affected", 0),
        "num_cascades":      len(result_dict.get("cascades", [])),
        "all_ingredients":   ingredients,
        "interactions": [
            {
                "drug_a":      ix.get("drug_a"),
                "drug_b":      ix.get("drug_b"),
                "severity":    bd.get("severity"),
                "score":       bd.get("score"),
                "icon":        bd.get("icon"),
                "description": ix.get("description"),
                "mechanism":   ix.get("mechanism"),
            }
            for ix, bd in zip(
                result_dict.get("interactions_found", []),
                result_dict.get("severity_breakdown", []),
            )
        ],
        "organ_systems": [
            {
                "system":            s.get("system"),
                "severity":          s.get("severity"),
                "score":             s.get("adjusted_score") or s.get("score"),
                "icon":              s.get("icon"),
                "nlp_confidence":    s.get("nlp_confidence"),
                "risk_factors":      s.get("risk_factors", []),
            }
            for s in result_dict.get("organ_systems", [])
        ],
        "cascades": [
            {
                "organ_system":     c.get("organ_system"),
                "alert_level":      c.get("alert_level"),
                "cumulative_score": c.get("cumulative_score"),
                "num_interactions": c.get("num_interactions"),
            }
            for c in result_dict.get("cascades", [])
        ],
    })