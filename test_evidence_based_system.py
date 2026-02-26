# test_evidence_based_system.py
"""
Test the evidence-based PolyGuard system
"""

from polyguard_engine_evidence_based import (
    calculate_interaction_score_robust,
    analyze_biological_impact,
    adjust_for_patient_context,
    detect_polypharmacy_cascades,
    generate_clinical_report
)

# Sample interactions
interactions = [
    {
        'drug_a': 'Warfarin',
        'drug_b': 'Aspirin',
        'description': 'The risk or severity of bleeding and hemorrhage can be increased when Aspirin is combined with Warfarin.'
    },
    {
        'drug_a': 'Lisinopril',
        'drug_b': 'Ibuprofen',
        'description': 'May increase risk of renal dysfunction and decreased antihypertensive efficacy.'
    }
]

# Patient data
patient = {
    'age': 72,
    'conditions': ['Hypertension', 'Chronic Kidney Disease Stage 3'],
    'lab_values': {
        'eGFR': 48,
        'platelet_count': 145
    }
}

# Run analysis
print("="*80)
print("EVIDENCE-BASED POLYGUARD ANALYSIS")
print("="*80)

base_scores = calculate_interaction_score_robust(interactions)
print(f"\n✅ Base Scores Calculated")
print(f"   Methodology: {base_scores['methodology']}")

organ_analysis = analyze_biological_impact(interactions, base_scores)
print(f"\n✅ Organ Analysis Complete")
print(f"   Methodology: {organ_analysis['methodology']}")

patient_adjustments = adjust_for_patient_context(
    organ_analysis['affected_organ_systems'],
    patient
)
print(f"\n✅ Patient Adjustments Applied")
print(f"   Methodology: {patient_adjustments['methodology']}")

cascades = detect_polypharmacy_cascades(
    patient_adjustments['adjusted_systems'],
    interactions,
    num_drugs=4
)
print(f"\n✅ Cascade Detection Complete")
print(f"   Methodology: {cascades['methodology']}")

report = generate_clinical_report(
    base_scores,
    organ_analysis,
    patient_adjustments,
    cascades,
    patient
)

# Print evidence base
print(f"\n{'='*80}")
print("Report Summary:")
print(report)
print(f"\n{'='*80}")
print("EVIDENCE BASE")
print(f"{'='*80}")
print("\nKey Citations:")
for citation in report['evidence_base']['key_sources']:
    print(f"  • {citation}")

print(f"\nAll Citations Used: {len(report['evidence_base']['all_citations'])}")

print("\n" + "="*80)
print("✅ EVIDENCE-BASED ANALYSIS COMPLETE")
print("="*80)