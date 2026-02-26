# main.py
"""
PolyGuard — Complete Drug Interaction Analysis Pipeline
Datasets used:
  • indian_pharmaceutical_products_clean.csv  — brand→ingredient mapping
  • drugbank_interactions.csv                 — raw interaction descriptions
  • ddi_complete.csv                          — labeled interactions with severity & mechanism
  • drugbank_drugs.csv                        — drug name ↔ DrugBank ID lookup
"""

import json
import re
import pandas as pd
from itertools import combinations
from typing import List, Dict, Optional

from utils import load_data
from polyguard_engine_evidence_based import (
    calculate_interaction_score_robust,
    analyze_biological_impact,
    adjust_for_patient_context,
    detect_polypharmacy_cascades,
    generate_clinical_report,
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

print("Loading datasets…")

pharma_db: pd.DataFrame      = load_data('./datasets/indian_pharmaceutical_products_clean.csv')
drugbank_raw: pd.DataFrame   = load_data('./datasets/drugbank_interactions.csv')
ddi_labeled: pd.DataFrame    = load_data('./datasets/ddi_labeled.csv')
ddi_complete: pd.DataFrame   = load_data('./datasets/ddi_complete.csv')
drugbank_drugs: pd.DataFrame = load_data('./datasets/drugbank_drugs.csv')

# ── Normalise column names to lowercase ──────────────────────────────────────
for _df in [pharma_db, drugbank_raw, ddi_labeled, ddi_complete, drugbank_drugs]:
    _df.columns = [c.strip().lower() for c in _df.columns]

# ── Build merged interactions table (ddi_complete preferred; fallback raw) ───
def _build_interactions_table() -> pd.DataFrame:
    """
    Merge all interaction sources into one canonical DataFrame with columns:
        drug1_name, drug2_name, description, severity, mechanism
    """
    frames = []

    # 1. ddi_complete — richest source (has severity + mechanism)
    if not ddi_complete.empty and 'drug1_name' in ddi_complete.columns:
        df = ddi_complete[['drug1_name','drug2_name','description',
                            'severity','mechanism']].copy()
        df['source'] = 'ddi_complete'
        frames.append(df)

    # 2. ddi_labeled — also has severity
    if not ddi_labeled.empty and 'drug1_name' in ddi_labeled.columns:
        cols = ['drug1_name','drug2_name','description','severity']
        df   = ddi_labeled[[c for c in cols if c in ddi_labeled.columns]].copy()
        df['mechanism'] = ddi_labeled.get('mechanism', 'Unknown')
        df['source']    = 'ddi_labeled'
        frames.append(df)

    # 3. drugbank_interactions — raw descriptions, no severity label
    if not drugbank_raw.empty and 'drug1_name' in drugbank_raw.columns:
        df = drugbank_raw[['drug1_name','drug2_name','description']].copy()
        df['severity']  = 'Unknown'
        df['mechanism'] = 'Unknown'
        df['source']    = 'drugbank_raw'
        frames.append(df)

    if not frames:
        print("⚠️  No interaction databases loaded.")
        return pd.DataFrame(columns=['drug1_name','drug2_name','description','severity','mechanism','source'])

    merged = pd.concat(frames, ignore_index=True)

    # Normalise drug names
    merged['drug1_name'] = merged['drug1_name'].str.strip().str.lower()
    merged['drug2_name'] = merged['drug2_name'].str.strip().str.lower()
    merged['description'] = merged['description'].fillna('No description available')

    # De-duplicate: prefer ddi_complete over ddi_labeled over raw
    source_prio = {'ddi_complete': 0, 'ddi_labeled': 1, 'drugbank_raw': 2}
    merged['_prio'] = merged['source'].map(source_prio).fillna(9)
    merged.sort_values('_prio', inplace=True)
    merged.drop_duplicates(subset=['drug1_name','drug2_name'], keep='first', inplace=True)
    merged.drop(columns='_prio', inplace=True)
    merged.reset_index(drop=True, inplace=True)

    print(f"   Interaction table: {len(merged):,} unique pairs")
    return merged


interactions_table = _build_interactions_table()

# ── Build ingredient → DrugBank ID lookup ────────────────────────────────────
def _build_ingredient_id_map() -> Dict[str, str]:
    """
    Map lowercase ingredient name → DrugBank ID using drugbank_drugs.csv.
    Falls back to name-based search if IDs unavailable.
    """
    if drugbank_drugs.empty:
        return {}
    name_col = next((c for c in drugbank_drugs.columns if 'name' in c), None)
    id_col   = next((c for c in drugbank_drugs.columns if 'id' in c), None)
    if not (name_col and id_col):
        return {}
    return dict(zip(
        drugbank_drugs[name_col].str.strip().str.lower(),
        drugbank_drugs[id_col].str.strip()
    ))

ingredient_id_map = _build_ingredient_id_map()
print(f"   DrugBank ID map: {len(ingredient_id_map):,} entries")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — BRAND NAME SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def search_brand_name(prefix: str, limit: int = 10) -> pd.Series:
    """
    Return brand names starting with `prefix` (case-insensitive).

    Parameters:
        prefix  : search prefix string
        limit   : max results to return (default 10)

    Returns:
        pd.Series of matching brand names
    """
    if pharma_db.empty or 'brand_name' not in pharma_db.columns:
        print("⚠️  Pharma database not loaded.")
        return pd.Series(dtype=str)

    mask    = pharma_db['brand_name'].str.lower().str.startswith(prefix.lower(), na=False)
    results = pharma_db[mask]['brand_name'].drop_duplicates()
    return results.head(limit)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1b — INGREDIENT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def get_ingredients_by_brand_name(brand_name: str) -> List[str]:
    """
    Return all unique active ingredients for a given brand name.

    Parameters:
        brand_name : exact brand name string

    Returns:
        List of ingredient names (strings)
    """
    if pharma_db.empty:
        return []

    rows = pharma_db[pharma_db['brand_name'].str.lower() == brand_name.lower()]

    if rows.empty:
        # Fuzzy fallback: contains search
        rows = pharma_db[pharma_db['brand_name'].str.lower().str.contains(
            re.escape(brand_name.lower()), na=False)]

    if rows.empty:
        print(f"   ⚠️  Brand '{brand_name}' not found in database.")
        return []

    result: set = set()

    # Primary ingredient column
    if 'primary_ingredient' in rows.columns:
        for val in rows['primary_ingredient'].dropna():
            result.add(str(val).strip())

    # Active ingredients JSON column
    if 'active_ingredients' in rows.columns:
        for raw in rows['active_ingredients'].dropna():
            try:
                parsed = eval(raw) if isinstance(raw, str) else raw
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and 'name' in item:
                            result.add(str(item['name']).strip())
            except Exception:
                pass  # malformed JSON — skip

    return sorted(result)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — INTERACTION LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def get_drug_interactions(ingredients: List[str]) -> List[Dict]:
    """
    Look up all pairwise drug-drug interactions for the given ingredient list.

    Strategy (in priority order):
      1. Exact name match in interaction table
      2. DrugBank ID match (via ingredient_id_map)
      3. Partial / contains match as fallback

    Parameters:
        ingredients : list of ingredient name strings

    Returns:
        List of interaction detail dicts
    """
    if interactions_table.empty:
        print("   ⚠️  No interaction database available.")
        return []

    norm = {ing: ing.lower().strip() for ing in ingredients}
    pairs = list(combinations(list(norm.values()), 2))
    print(f"   Checking {len(pairs)} pair(s) across {len(interactions_table):,} known interactions…")

    found: List[Dict] = []

    for drug_a, drug_b in pairs:
        # ── Exact match both directions ────────────────────────────────────
        mask = (
            ((interactions_table['drug1_name'] == drug_a) & (interactions_table['drug2_name'] == drug_b)) |
            ((interactions_table['drug1_name'] == drug_b) & (interactions_table['drug2_name'] == drug_a))
        )
        matches = interactions_table[mask]

        # ── Partial / contains match (fallback) ───────────────────────────
        if matches.empty:
            mask = (
                (interactions_table['drug1_name'].str.contains(drug_a, na=False, regex=False) &
                 interactions_table['drug2_name'].str.contains(drug_b, na=False, regex=False)) |
                (interactions_table['drug1_name'].str.contains(drug_b, na=False, regex=False) &
                 interactions_table['drug2_name'].str.contains(drug_a, na=False, regex=False))
            )
            matches = interactions_table[mask]

        for _, row in matches.iterrows():
            found.append({
                'drug_a':       drug_a.title(),
                'drug_b':       drug_b.title(),
                'description':  row.get('description', 'No description available'),
                'severity':     row.get('severity', 'Unknown'),
                'mechanism':    row.get('mechanism', 'Unknown'),
                'source':       row.get('source', 'DrugBank'),
            })
            print(f"   ⚠️  {drug_a.title()} ↔ {drug_b.title()} [{row.get('severity','?')}]")

    if not found:
        print("   ✅ No interactions found between these ingredients.")
    else:
        print(f"   📊 {len(found)} interaction(s) detected.")

    return found


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED BRAND → INTERACTIONS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def get_interactions_for_multiple_brands(brand_names: List[str]) -> Dict:
    """
    Extract ingredients from multiple brands and look up all pairwise interactions.

    Parameters:
        brand_names : list of brand name strings

    Returns:
        Dict with brand→ingredient map, all ingredients, and interactions found
    """
    brand_ingredient_map: Dict[str, List[str]] = {}
    all_ingredients: List[str] = []

    for brand in brand_names:
        ings = get_ingredients_by_brand_name(brand)
        brand_ingredient_map[brand] = ings
        all_ingredients.extend(ings)

    unique_ingredients = list(dict.fromkeys(all_ingredients))  # preserve order, deduplicate

    print(f"\n{'='*72}")
    print(f"  POLYPHARMACY ANALYSIS")
    print(f"{'='*72}")
    print(f"  Brands    : {len(brand_names)}")
    print(f"  Ingredients: {len(unique_ingredients)} unique")
    for brand, ings in brand_ingredient_map.items():
        print(f"    {brand}: {', '.join(ings) if ings else 'not found'}")
    print(f"{'='*72}\n")

    interactions = get_drug_interactions(unique_ingredients)

    return {
        'brand_names':           brand_names,
        'brand_ingredient_map':  brand_ingredient_map,
        'all_ingredients':       unique_ingredients,
        'interactions_found':    interactions,
        'num_interactions':      len(interactions),
        'requires_cascade_analysis': len(unique_ingredients) >= 3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE  (Steps 1 → 7)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_interactions_with_context(
    brand_names:  List[str],
    patient_data: Optional[Dict] = None,
    save_report:  Optional[str]  = None,
) -> Dict:
    """
    Run the complete PolyGuard pipeline for a list of brand medications.

    Parameters:
        brand_names  : list of brand medication names
        patient_data : optional dict with keys:
                         age        (int)
                         gender     (str)
                         conditions (list[str])  e.g. ['Hypertension', 'Diabetes Type 2']
                         lab_values (dict)       e.g. {'eGFR': 45, 'ALT': 52, 'platelets': 180}
        save_report  : optional filepath to write the JSON report

    Returns:
        Full analysis dict including clinical_report
    """

    # ── Steps 1 & 2 ──────────────────────────────────────────────────────────
    interaction_data = get_interactions_for_multiple_brands(brand_names)

    if interaction_data['num_interactions'] == 0:
        print("\n✅ No known interactions found between these medications.\n")
        return {
            'status': 'NO_INTERACTIONS',
            'message': 'No known interactions found between these medications.',
            'data': interaction_data,
        }

    interactions_list = interaction_data['interactions_found']
    num_drugs         = len(interaction_data['all_ingredients'])

    # ── Step 3 ───────────────────────────────────────────────────────────────
    print("\n[Step 3] Scoring interaction severity…")
    base_scores = calculate_interaction_score_robust(interactions_list)
    print(f"         Total score: {base_scores['total_score']} | "
          f"Risk: {base_scores['risk_level']} {base_scores['risk_color']}")

    # ── Step 4 ───────────────────────────────────────────────────────────────
    print("\n[Step 4] Mapping risk to organ systems…")
    organ_analysis = analyze_biological_impact(interactions_list, base_scores)
    print(f"         {organ_analysis['num_organs_affected']} organ system(s) affected")
    if organ_analysis['highest_risk_organ']:
        top = organ_analysis['highest_risk_organ']
        print(f"         Highest: {top['system']} — {top['severity']} {top['icon']}")

    # ── Step 5 ───────────────────────────────────────────────────────────────
    if patient_data:
        print("\n[Step 5] Applying patient-specific risk adjustments…")
        patient_adj = adjust_for_patient_context(
            organ_analysis['affected_organ_systems'], patient_data
        )
        for sys in patient_adj['adjusted_systems']:
            if sys.get('patient_specific_warning'):
                print(f"\n         {sys['patient_specific_warning']}")
    else:
        print("\n[Step 5] No patient data — skipping context adjustment.")
        patient_adj = {
            'status': 'NO_PATIENT_DATA',
            'adjusted_systems': organ_analysis['affected_organ_systems'],
        }

    adjusted_systems = patient_adj.get('adjusted_systems', [])

    # ── Step 6 ───────────────────────────────────────────────────────────────
    print("\n[Step 6] Detecting polypharmacy cascades…")
    cascade_detection = detect_polypharmacy_cascades(
        adjusted_systems, interactions_list, num_drugs
    )
    if cascade_detection['has_cascades']:
        print(f"         ⚠️  {cascade_detection['num_cascades']} cascade(s) detected!")
        for c in cascade_detection['cascades']:
            print(f"            • {c['organ_system']}: {c['alert_level']} "
                  f"(score {c['cumulative_score']}, {c['num_interactions']} interactions)")
    else:
        print("         ✅ No cascades detected.")

    # ── Step 7 ───────────────────────────────────────────────────────────────
    print("\n[Step 7] Generating clinical report…")
    clinical_report = generate_clinical_report(
        base_scores, organ_analysis, patient_adj, cascade_detection, patient_data
    )

    _print_report(clinical_report)

    result = {
        'status':            'INTERACTIONS_FOUND',
        'basic_data':        interaction_data,
        'patient_data':      patient_data,
        'base_scores':       base_scores,
        'organ_analysis':    organ_analysis,
        'patient_adjustments': patient_adj,
        'cascade_detection': cascade_detection,
        'clinical_report':   clinical_report,
    }

    if save_report:
        try:
            with open(save_report, 'w') as f:
                json.dump(clinical_report, f, indent=2, default=str)
            print(f"\n   💾 Report saved → {save_report}")
        except Exception as e:
            print(f"\n   ⚠️  Could not save report: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(report: Dict) -> None:
    """Pretty-print the final PolyGuard clinical report."""
    s  = report['summary']
    ev = report['evidence_base']
    W  = 72

    print(f"\n{'='*W}")
    print(f"  POLYGUARD CLINICAL REPORT  {s['risk_icon']}  {s['overall_risk_level']}")
    print(f"{'='*W}")
    print(f"  Overall Risk    : {s['risk_color']} {s['overall_risk_level']}")
    print(f"  Action Required : {s['primary_action']}")
    print(f"  Total Score     : {s['total_interaction_score']}")
    print(f"  Interactions    : {s['num_interactions']}")
    print(f"  Organs at Risk  : {s['num_organs_affected']}")
    print(f"  Cascades        : {s['num_cascades']}")

    # ── Interaction breakdown ─────────────────────────────────────────────
    breakdown = report['interaction_analysis'].get('detailed_breakdown', [])
    if breakdown:
        print(f"\n  {'─'*W}")
        print(f"  INTERACTION DETAILS")
        print(f"  {'─'*W}")
        for item in breakdown:
            print(f"  {item['icon']}  {item['drugs']}  [{item['severity']}] +{item['score']}")
            desc = item['description']
            if desc and desc != 'No description available':
                print(f"       {desc[:110]}{'…' if len(desc) > 110 else ''}")
            if item.get('mechanism') and item['mechanism'] not in ('Unknown', ''):
                print(f"       Mechanism: {item['mechanism']}")

    # ── Organ system breakdown ────────────────────────────────────────────
    systems = (
        report['patient_specific_analysis'].get('adjusted_systems') or
        report['organ_system_analysis'].get('affected_organ_systems', [])
    )
    if systems:
        print(f"\n  {'─'*W}")
        print(f"  ORGAN SYSTEM RISK BREAKDOWN")
        print(f"  {'─'*W}")
        for sys in systems[:8]:
            score  = sys.get('adjusted_score', sys.get('score', 0))
            base   = sys.get('base_score', score)
            mult   = sys.get('vulnerability_multiplier', 1.0)
            bar    = '█' * min(int(score / 4), 20)
            print(f"  {sys['icon']}  {sys['system']:<28} {score:>4}  {bar}")
            if mult != 1.0:
                print(f"       base {base} × {mult:.2f}x  [{sys['severity']}]")
            for rf in sys.get('risk_factors', []):
                print(f"       ↳ {rf}")

    # ── Cascade alerts ────────────────────────────────────────────────────
    cascades = report['polypharmacy_cascade_analysis'].get('cascades', [])
    if cascades:
        print(f"\n  {'─'*W}")
        print(f"  POLYPHARMACY CASCADE ALERTS")
        print(f"  {'─'*W}")
        for c in cascades:
            print(f"  🔗  {c['organ_system']}  [{c['alert_level']}]")
            print(f"       Score {c['cumulative_score']}  across {c['num_interactions']} interactions")
            print(f"       {c['evidence_rationale'][:90]}")

    # ── Evidence base ─────────────────────────────────────────────────────
    print(f"\n  {'─'*W}")
    print(f"  EVIDENCE BASE")
    print(f"  {'─'*W}")
    for src in ev['key_sources']:
        print(f"    • {src}")
    print(f"\n  Analysis: {report['report_metadata']['analysis_version']}")
    print(f"{'='*W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT — Demo / smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("\n" + "=" * 72)
    print("  POLYGUARD — DRUG INTERACTION ANALYSIS SYSTEM  v2.0")
    print("=" * 72)

    # ── Demo 1: Brand search ───────────────────────────────────────────────
    print("\n── Demo 1: Brand search ──────────────────────────────────")
    for prefix in ['Aug', 'Cal', 'Asp', 'Met']:
        results = search_brand_name(prefix, limit=5)
        print(f"  '{prefix}*'  →  {results.tolist()}")

    # ── Demo 2: Ingredient extraction ─────────────────────────────────────
    print("\n── Demo 2: Ingredient extraction ────────────────────────")
    test_brands = ['Augmentin 625 Duo Tablet', 'Ascoril LS Syrup', 'Allegra 120mg Tablet']
    for b in test_brands:
        ings = get_ingredients_by_brand_name(b)
        print(f"  {b}  →  {ings}")

    # ── Demo 3: Analysis without patient data ─────────────────────────────
    print("\n── Demo 3: Multi-drug analysis (no patient data) ────────")
    meds_simple = ['Augmentin 625 Duo Tablet', 'Ascoril LS Syrup']
    r3 = analyze_interactions_with_context(brand_names=meds_simple)
    print(f"  Status: {r3['status']}")

    # ── Demo 4: Full pipeline with patient context ────────────────────────
    print("\n── Demo 4: Full pipeline with patient context ───────────")

    patient = {
        'age':       72,
        'gender':    'Female',
        'conditions': [
            'Hypertension',
            'Diabetes Type 2',
            'Atrial Fibrillation',
            'COPD',
        ],
        'lab_values': {
            'eGFR':          42,       # Stage 3 CKD
            'ALT':           85,       # Elevated
            'platelet_count':110,      # Mild thrombocytopenia
            'INR':           3.2,      # Slightly supra-therapeutic
            'blood_glucose': 195,      # Moderate hyperglycaemia
        },
    }

    # Use brands that are likely to have DrugBank interactions
    # (Amoxycillin/Clavulanic Acid from Augmentin, plus others)
    meds_full = [
        'Augmentin 625 Duo Tablet',
        'Azithral 500 Tablet',
        'Ascoril LS Syrup',
    ]

    r4 = analyze_interactions_with_context(
        brand_names=meds_full,
        patient_data=patient,
        save_report='polyguard_report.json',
    )

    print(f"\n  Pipeline status : {r4['status']}")
    if r4['status'] == 'INTERACTIONS_FOUND':
        cr = r4['clinical_report']['summary']
        print(f"  Overall risk    : {cr['overall_risk_level']} {cr['risk_color']}")
        print(f"  Cascades found  : {cr['num_cascades']}")