"""
examples/usage.py
==================
Complete usage guide for 

Run this file directly to see live output (you need datasets/ populated):

    python examples/usage.py

Each section is an independent, copy-paste-ready snippet.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. ONE-TIME SETUP  (shared across all examples)
# ─────────────────────────────────────────────────────────────────────────────

from polyguard.core import PolyGuardAnalyser
from polyguard.core.data_loader import DataLoader

# Load datasets once — this is the only I/O; everything else is pure Python.
loader   = DataLoader(data_dir="./datasets").load()
analyser = PolyGuardAnalyser(loader)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BRAND SEARCH
# ─────────────────────────────────────────────────────────────────────────────

result = analyser.search_brands("Aug", limit=5)
print(f"Brands matching 'Aug*': {result.results}")
# BrandSearchResult(prefix='Aug', results=['Augmentin 625 Duo Tablet', ...], total_found=4)


# ─────────────────────────────────────────────────────────────────────────────
# 2. INGREDIENT LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

ing = analyser.get_ingredients("Augmentin 625 Duo Tablet")
print(f"Ingredients: {ing.ingredients}")
# IngredientsResult(brand_name='Augmentin 625 Duo Tablet',
#                   ingredients=['Amoxycillin', 'Clavulanic Acid'], found=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. BASIC ANALYSIS  (no patient data)
# ─────────────────────────────────────────────────────────────────────────────

result = analyser.analyse(
    brand_names=["Augmentin 625 Duo Tablet", "Ascoril LS Syrup"],
)

print(result.status)          # 'NO_INTERACTIONS' or 'INTERACTIONS_FOUND'
print(result.risk_level)      # 'MINIMAL' | 'MILD' | 'MODERATE' | 'SEVERE' | 'CRITICAL'
print(result.total_score)     # int — overall interaction severity score

for ix in result.interactions_found:
    print(f"  {ix.drug_a} ↔ {ix.drug_b}  [{ix.severity}]")

for organ in result.organ_systems:
    score = organ.adjusted_score or organ.score
    print(f"  {organ.icon} {organ.system}: {score}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. FULL PIPELINE  (with patient context + XAI)
# ─────────────────────────────────────────────────────────────────────────────

from polyguard.core.models import PatientData

patient = PatientData(
    age       = 72,
    gender    = "Female",
    conditions= ["Hypertension", "Diabetes Type 2", "Atrial Fibrillation", "COPD"],
    lab_values= {
        "eGFR":          42,
        "ALT":           85,
        "platelet_count":110,
        "INR":           3.2,
        "blood_glucose": 195,
    },
)

result = analyser.analyse(
    brand_names  = ["Augmentin 625 Duo Tablet", "Azithral 500 Tablet", "Ascoril LS Syrup"],
    patient_data = patient.model_dump(),   # convert Pydantic → plain dict
    explain      = True,
)

# ── Summary ───────────────────────────────────────────────────────────────────
if result.summary:
    s = result.summary
    print(f"\nOverall risk : {s.overall_risk_level} {s.risk_color}")
    print(f"Action       : {s.primary_action}")
    print(f"Score        : {s.total_interaction_score}")
    print(f"Cascades     : {s.num_cascades}")

# ── Organ systems (patient-adjusted) ─────────────────────────────────────────
for organ in result.organ_systems:
    adj   = organ.adjusted_score or organ.score
    mult  = organ.vulnerability_multiplier or 1.0
    print(f"  {organ.icon} {organ.system}: {adj}  (×{mult:.2f})")
    for rf in organ.risk_factors:
        print(f"     ↳ {rf}")

# ── Cascade alerts ────────────────────────────────────────────────────────────
for c in result.cascades:
    print(f"\n🔗 CASCADE: {c.organ_system} [{c.alert_level}]  score={c.cumulative_score}")

# ── XAI explanations ──────────────────────────────────────────────────────────
if result.xai:
    xai = result.xai

    print("\n── Step 3 XAI: WHY each interaction was scored ──")
    for ex in xai.step3_interaction_explanations:
        print(f"  {ex.drugs}  [{ex.severity}]  score={ex.score}")
        print(f"    {ex.explanation}")
        for term, weight in ex.supporting_terms[:3]:
            print(f"    key term: '{term}' (+{weight:.3f})")

    print("\n── Step 5 XAI: Score waterfalls (base → adjusted) ──")
    for wf in xai.step5_waterfalls:
        print(f"\n  {wf.organ}  (base={wf.base_score} → adjusted={wf.final_score})")
        print(wf.bar_chart)

    print("\n── Step 5/7 XAI: Counterfactuals (what would reduce risk) ──")
    for cf in xai.step5_counterfactuals:
        print(f"\n  {cf.organ}  (current score: {cf.current_score})")
        print(f"  Best action: {cf.top_action}")
        for item in cf.items[:2]:
            print(f"    >> {item['narrative']}")

    print("\n── Step 6 XAI: Cascade attribution ──")
    for cx in xai.step6_cascade_attributions:
        print(f"\n  [CASCADE] {cx['organ']}  [{cx['alert_level']}]")
        print(f"  {cx['mechanism_summary']}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SERIALISE TO JSON  (for saving / API response)
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path

report_json = result.model_dump()                     # full nested dict
json_str    = json.dumps(report_json, indent=2, default=str)
Path("polyguard_report.json").write_text(json_str)
print("\nReport saved to polyguard_report.json")

# Deserialise back
from polyguard.core.models import AnalysisResult
loaded = AnalysisResult.model_validate(report_json)
assert loaded.status == result.status


# ─────────────────────────────────────────────────────────────────────────────
# 6. BIBLIOGRAPHY GENERATION
# ─────────────────────────────────────────────────────────────────────────────

from polyguard.core.bibliography import BibliographyGenerator

gen = BibliographyGenerator()

# Render as Markdown string
md = gen.to_markdown()
print(md[:500])

# Save to file
saved = gen.save("PolyGuard_Bibliography.md")
print(f"\nBibliography saved to {saved}")

# Get a flat list of APA citations
citations = gen.to_apa()
print(f"\n{len(citations)} citations:")
for c in citations:
    print(f"  • {c[:80]}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. USE AS A FASTAPI DEPENDENCY  (inside a route handler)
# ─────────────────────────────────────────────────────────────────────────────
#
# In your FastAPI app, use the factory + dependency pattern:
#
#   from api.app import create_app
#   app = create_app()
#
# Or reference the ready-made ASGI entry point:
#
#   uvicorn asgi:app --reload
#
# The analyser is injected into routes via FastAPI's Depends():
#
#   from api.app import get_analyser
#   @app.post("/analyse")
#   def my_route(body: AnalysisRequest, analyser = Depends(get_analyser)):
#       return analyser.analyse(body.brand_names, ...)


# ─────────────────────────────────────────────────────────────────────────────
# 8. CALLING THE API WITH httpx (once the server is running)
# ─────────────────────────────────────────────────────────────────────────────
#
#   import httpx
#
#   BASE = "http://localhost:8000"
#
#   # Brand search
#   r = httpx.get(f"{BASE}/brands/search", params={"prefix": "Aug"})
#   print(r.json())
#
#   # Full analysis
#   r = httpx.post(f"{BASE}/analyse", json={
#       "brand_names": ["Augmentin 625 Duo Tablet", "Ascoril LS Syrup"],
#       "patient_data": {"age": 72, "conditions": ["Hypertension"],
#                        "lab_values": {"eGFR": 42}},
#       "explain": True,
#   })
#   data = r.json()
#   print(data["summary"]["overall_risk_level"])
#   print(data["cascades"])