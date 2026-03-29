"""
polyguard.api.app
=================
FastAPI application for the PolyGuard drug interaction analysis system.

Endpoints
---------
GET  /health                    — liveness probe
GET  /brands/search             — brand name prefix search
GET  /brands/{brand_name}/ingredients — ingredient lookup
POST /analyse                   — full pipeline (Steps 1-7)
GET  /bibliography              — literature sources as JSON
GET  /bibliography/markdown     — literature sources as Markdown

Environment variables
---------------------
POLYGUARD_DATA_DIR   Path to the datasets directory (default: ./datasets)
POLYGUARD_LOG_LEVEL  Logging level (default: INFO)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from polyguard.core.analyser import PolyGuardAnalyser
from polyguard.core.bibliography import BibliographyGenerator
from polyguard.core.data_loader import DataLoader
from polyguard.core.models import (
    AnalysisRequest,
    AnalysisResult,
    BrandSearchResult,
    IngredientsResult,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION STATE  (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────

_loader:   DataLoader | None   = None
_analyser: PolyGuardAnalyser | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Load heavy datasets once on startup; release on shutdown."""
    global _loader, _analyser

    log_level = os.getenv("POLYGUARD_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

    data_dir = os.getenv("POLYGUARD_DATA_DIR", "./datasets")
    logger.info("PolyGuard API starting up — data_dir=%s", data_dir)

    _loader   = DataLoader(data_dir=data_dir).load()
    _analyser = PolyGuardAnalyser(_loader)

    logger.info("PolyGuard API ready.")
    yield

    logger.info("PolyGuard API shutting down.")
    _loader   = None
    _analyser = None


# ─────────────────────────────────────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title       = "PolyGuard API",
        description = "Evidence-based drug interaction analysis with XAI.",
        version     = "2.1.0",
        lifespan    = lifespan,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins  = ["*"],   # tighten in production
        allow_methods  = ["*"],
        allow_headers  = ["*"],
    )

    _register_routes(app)
    return app


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY
# ─────────────────────────────────────────────────────────────────────────────

def get_analyser() -> PolyGuardAnalyser:
    if _analyser is None:
        raise HTTPException(status_code=503, detail="Service not ready — analyser not initialised.")
    return _analyser


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

def _register_routes(app: FastAPI) -> None:

    @app.get("/health", tags=["System"])
    def health() -> dict:
        """Liveness probe — confirms the service is running."""
        return {"status": "ok", "version": "2.1.0"}

    # ── Brand search ──────────────────────────────────────────────────────────

    @app.get(
        "/brands/search",
        response_model = BrandSearchResult,
        tags           = ["Brands"],
        summary        = "Search brand names by prefix",
    )
    def search_brands(
        prefix:   str = Query(..., min_length=1, description="Brand name prefix"),
        limit:    int = Query(10, ge=1, le=50, description="Max results"),
        analyser: PolyGuardAnalyser = Depends(get_analyser),
    ) -> BrandSearchResult:
        """Return brand names from the Indian pharma database that start with *prefix*."""
        return analyser.search_brands(prefix=prefix, limit=limit)

    # ── Ingredient lookup ─────────────────────────────────────────────────────

    @app.get(
        "/brands/{brand_name}/ingredients",
        response_model = IngredientsResult,
        tags           = ["Brands"],
        summary        = "Get active ingredients for a brand",
    )
    def get_ingredients(
        brand_name: str,
        analyser:   PolyGuardAnalyser = Depends(get_analyser),
    ) -> IngredientsResult:
        """Resolve a brand name to its active pharmaceutical ingredient(s)."""
        return analyser.get_ingredients(brand_name)

    # ── Full analysis pipeline ────────────────────────────────────────────────

    @app.post(
        "/analyse",
        response_model = AnalysisResult,
        tags           = ["Analysis"],
        summary        = "Run full drug interaction analysis (Steps 1–7)",
    )
    def analyse(
        body:     AnalysisRequest,
        analyser: PolyGuardAnalyser = Depends(get_analyser),
    ) -> AnalysisResult:
        """
        Run the complete PolyGuard pipeline for the supplied brand medications.

        - **brand_names**: one or more brand medication names
        - **patient_data**: optional profile for risk adjustment (Step 5)
        - **explain**: include XAI feature attributions, waterfalls, counterfactuals
        """
        try:
            patient_dict = body.patient_data.model_dump() if body.patient_data else None
            return analyser.analyse(
                brand_names  = body.brand_names,
                patient_data = patient_dict,
                explain      = body.explain,
            )
        except Exception as exc:
            logger.exception("Analysis failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ── Bibliography ──────────────────────────────────────────────────────────

    @app.get(
        "/bibliography",
        tags    = ["Reference"],
        summary = "Literature sources as JSON",
    )
    def bibliography_json() -> dict:
        """Return the full evidence base as a JSON object."""
        from polyguard.core.bibliography import LITERATURE_SOURCES
        return LITERATURE_SOURCES

    @app.get(
        "/bibliography/markdown",
        response_class = PlainTextResponse,
        tags           = ["Reference"],
        summary        = "Literature sources as Markdown",
    )
    def bibliography_markdown() -> str:
        """Return the full evidence base formatted as a Markdown document."""
        return BibliographyGenerator().to_markdown()

    # ── Convenience: analyse a single free-text ingredient list ───────────────

    @app.get(
        "/analyse/quick",
        response_model = AnalysisResult,
        tags           = ["Analysis"],
        summary        = "Quick analysis by ingredient names (no brand lookup)",
    )
    def analyse_quick(
        ingredients: List[str] = Query(..., description="List of ingredient names"),
        analyser:    PolyGuardAnalyser = Depends(get_analyser),
    ) -> AnalysisResult:
        """
        Bypass brand → ingredient lookup and analyse ingredient names directly.
        Useful for testing or when you already know the active ingredients.
        """
        try:
            raw = analyser._loader.lookup_interactions(ingredients)
            if not raw:
                return AnalysisResult(
                    status       = "NO_INTERACTIONS",
                    message      = "No known interactions found.",
                    all_ingredients = ingredients,
                )
            analyser._ensure_engine()
            engine      = analyser._engine
            base        = engine.calculate_interaction_score_robust(raw)
            organ       = engine.analyze_biological_impact(raw, base)
            patient_adj = {"status": "NO_PATIENT_DATA", "adjusted_systems": organ["affected_organ_systems"]}
            cascade     = engine.detect_polypharmacy_cascades(organ["affected_organ_systems"], raw, len(ingredients))
            report      = engine.generate_clinical_report(base, organ, patient_adj, cascade, None)
            return analyser._build_result(
                brand_names          = [],
                brand_ingredient_map = {},
                all_ingredients      = ingredients,
                raw_interactions     = raw,
                base_scores          = base,
                organ_analysis       = organ,
                patient_adj          = patient_adj,
                cascade_detection    = cascade,
                clinical_report      = report,
                xai_result           = None,
                patient_adjusted     = False,
            )
        except Exception as exc:
            logger.exception("Quick analysis failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc