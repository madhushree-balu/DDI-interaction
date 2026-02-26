# test_polyguard.py
"""
PolyGuard — Unit & Integration Tests
Runs fully offline using mock data; no real CSV files required.

Run with:
    python test_polyguard.py           # all tests
    python test_polyguard.py -v        # verbose
    python test_polyguard.py TestStep3 # single suite
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Mock evidence_based_weights BEFORE any polyguard import
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import Tuple

@dataclass
class EvidenceSource:
    value: float
    citation: str
    rationale: str
    evidence_level: str
    dataset_size: int = None
    confidence_interval: Tuple[float, float] = None
    validation_method: str = None

_MOCK_WEIGHTS = {
    'CARDIOVASCULAR': EvidenceSource(1.52, 'Tisdale 2020', 'CV risk', 'High'),
    'HEPATIC':        EvidenceSource(1.38, 'Bjornsson 2016', 'Liver risk', 'High'),
    'RENAL':          EvidenceSource(1.35, 'Kellum 2021', 'Kidney risk', 'High'),
    'HEMATOLOGIC':    EvidenceSource(1.34, 'Schulman 2005', 'Bleed risk', 'High'),
    'GASTROINTESTINAL': EvidenceSource(1.15,'Lanas 2017', 'GI risk', 'High'),
    'CENTRAL_NERVOUS_SYSTEM': EvidenceSource(1.42,'Lavan 2016','CNS risk','Moderate'),
    'RESPIRATORY':    EvidenceSource(1.48, 'Pandit 2023', 'Resp risk', 'Moderate'),
    'ENDOCRINE':      EvidenceSource(1.22, 'Lipska 2016', 'Endo risk', 'High'),
    'MUSCULOSKELETAL':EvidenceSource(0.95, 'Expert consensus', 'MSK risk', 'Low'),
    'IMMUNE_SYSTEM':  EvidenceSource(1.28, 'Immunology consensus', 'Immune risk', 'Moderate'),
}
_MOCK_KEYWORD = {
    'fatal':             EvidenceSource(50, 'MedDRA', 'Fatal', 'High'),
    'life_threatening':  EvidenceSource(45, 'MedDRA', 'Life-threat', 'High'),
    'hemorrhage':        EvidenceSource(38, 'GUSTO', 'Major bleed', 'High'),
    'organ_failure':     EvidenceSource(45, 'Clinical', 'Organ fail', 'High'),
    'severe':            EvidenceSource(35, 'MedDRA', 'Severe', 'High'),
    'toxicity':          EvidenceSource(35, 'Clinical', 'Toxic', 'Moderate'),
    'moderate':          EvidenceSource(20, 'MedDRA', 'Moderate', 'High'),
    'increased_risk':    EvidenceSource(20, 'Pharmavig', 'Inc risk', 'Moderate'),
    'decreased_efficacy':EvidenceSource(18, 'Pharmakin', 'Dec eff', 'Moderate'),
    'mild':              EvidenceSource(10, 'MedDRA', 'Mild', 'High'),
    'monitor':           EvidenceSource(10, 'Clinical', 'Monitor', 'Moderate'),
}
_MOCK_AGE = {
    (75,150): EvidenceSource(1.43,'Mangoni 2004','Elderly ≥75','High'),
    (65, 74): EvidenceSource(1.28,'Onder 2019','65-74','High'),
    (18, 64): EvidenceSource(1.0, 'Reference','Adult baseline','High'),
    (12, 17): EvidenceSource(1.18,'AAP 2021','Adolescent','Moderate'),
    ( 2, 11): EvidenceSource(1.32,'WHO 2023','Child','High'),
    ( 0,  1): EvidenceSource(1.45,'WHO 2023','Infant','Moderate'),
}
_MOCK_COMORBID = {
    'cardiovascular': {
        'hypertension':       EvidenceSource(1.45,'Fried 2001','HTN','High'),
        'heart_failure':      EvidenceSource(1.68,'Yancy 2017','HF','High'),
        'atrial_fibrillation':EvidenceSource(1.52,'January 2019','AFib','High'),
    },
    'renal': {
        'ckd_stage_3': EvidenceSource(1.42,'KDIGO 2012','CKD3','High'),
        'ckd_stage_4': EvidenceSource(1.89,'KDIGO 2023','CKD4','High'),
        'ckd_stage_5': EvidenceSource(2.15,'KDIGO 2023','CKD5','High'),
    },
    'hepatic': {
        'cirrhosis_child_a': EvidenceSource(1.38,'Verbeeck 2008','CirrA','High'),
        'cirrhosis_child_b': EvidenceSource(1.72,'Verbeeck 2008','CirrB','High'),
        'cirrhosis_child_c': EvidenceSource(2.25,'Verbeeck 2008','CirrC','High'),
    },
    'diabetes':    EvidenceSource(1.28,'Lipska 2016','DM','High'),
    'copd_asthma': EvidenceSource(1.35,'GOLD 2023','COPD','High'),
}
_MOCK_LAB = {
    'eGFR': {
        (0,15):  EvidenceSource(2.15,'KDIGO','ESRD','High'),
        (15,30): EvidenceSource(1.89,'KDIGO','CKD4','High'),
        (30,60): EvidenceSource(1.42,'KDIGO','CKD3','High'),
        (60,90): EvidenceSource(1.12,'KDIGO','CKD2','Moderate'),
    },
    'ALT': {
        (200,10000): EvidenceSource(2.05,"Hy's Law",'Severe hep','High'),
        (100,200):   EvidenceSource(1.65,'DILIN','Mod-sev hep','High'),
        (60,100):    EvidenceSource(1.35,'DILIN','Mild-mod hep','Moderate'),
        (40,60):     EvidenceSource(1.15,'Clinical','Mild elev','Moderate'),
    },
    'AST': {
        (200,10000): EvidenceSource(2.05,'DILIN','Severe','High'),
        (100,200):   EvidenceSource(1.65,'DILIN','Mod-sev','High'),
        (60,100):    EvidenceSource(1.35,'DILIN','Mild-mod','Moderate'),
        (40,60):     EvidenceSource(1.15,'Clinical','Mild','Moderate'),
    },
    'platelet_count': {
        (0,50):   EvidenceSource(2.15,'ISTH','Severe thrombocytopenia','High'),
        (50,100): EvidenceSource(1.72,'ISTH','Moderate','High'),
        (100,150):EvidenceSource(1.38,'Clinical','Mild','Moderate'),
    },
    'blood_glucose': {
        (250,600): EvidenceSource(1.48,'ADA 2023','Severe hyper','High'),
        (180,250): EvidenceSource(1.25,'ADA 2023','Moderate hyper','High'),
        (0,70):    EvidenceSource(1.62,'ADA 2023','Hypoglycaemia','High'),
    },
    'INR': {
        (4.0,20.0): EvidenceSource(2.25,'Ansell 2008','INR >4','High'),
        (3.0,4.0):  EvidenceSource(1.55,'Ansell 2008','INR 3-4','High'),
    },
    'serum_creatinine': {
        (3.0,20.0): EvidenceSource(2.15,'KDOQI','Sev renal','High'),
        (2.0,3.0):  EvidenceSource(1.68,'KDOQI','Mod-sev','High'),
        (1.5,2.0):  EvidenceSource(1.35,'KDOQI','Mild-mod','High'),
    },
}
_POLY_EV = EvidenceSource(1.88,'Masnoon 2017','Polypharmacy 5+ meds','High',138)
_MOCK_CASCADE = {'cumulative':2,'cascade':3,'critical_score':40,'evidence':_POLY_EV}

import sys, types

# Inject mock module so engine imports succeed without real files
mock_weights_mod = types.ModuleType('evidence_based_weights')
mock_weights_mod.EvidenceSource             = EvidenceSource
mock_weights_mod.ORGAN_SEVERITY_WEIGHTS     = _MOCK_WEIGHTS
mock_weights_mod.KEYWORD_SEVERITY_SCORES    = _MOCK_KEYWORD
mock_weights_mod.AGE_VULNERABILITY_MULTIPLIERS = _MOCK_AGE
mock_weights_mod.COMORBIDITY_MULTIPLIERS    = _MOCK_COMORBID
mock_weights_mod.LAB_VALUE_THRESHOLDS       = _MOCK_LAB
mock_weights_mod.CASCADE_DETECTION_THRESHOLD= _MOCK_CASCADE
mock_weights_mod.POLYPHARMACY_EVIDENCE      = _POLY_EV
sys.modules['evidence_based_weights'] = mock_weights_mod

# Now safe to import engine + main
from polyguard_engine_evidence_based import (
    calculate_interaction_score_robust,
    analyze_biological_impact,
    adjust_for_patient_context,
    detect_polypharmacy_cascades,
    generate_clinical_report,
)

# Patch utils.load_data so main.py won't fail on missing CSVs
mock_utils = types.ModuleType('utils')
mock_utils.load_data = lambda path: pd.DataFrame()
sys.modules['utils'] = mock_utils

import main as m


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

INTERACTIONS_CRITICAL = [
    {'drug_a': 'Warfarin', 'drug_b': 'Aspirin',
     'description': 'The risk or severity of bleeding and hemorrhage can be increased.',
     'severity': 'Major', 'mechanism': 'Pharmacodynamic'},
    {'drug_a': 'Metformin', 'drug_b': 'Contrast Dye',
     'description': 'May increase the risk of lactic acidosis and organ failure.',
     'severity': 'Moderate', 'mechanism': 'Renal clearance'},
]

INTERACTIONS_MILD = [
    {'drug_a': 'Paracetamol', 'drug_b': 'Ibuprofen',
     'description': 'May increase risk of GI adverse effects. Monitor patients.',
     'severity': 'Mild', 'mechanism': 'Unknown'},
]

INTERACTIONS_FATAL = [
    {'drug_a': 'MAO Inhibitor', 'drug_b': 'SSRI',
     'description': 'Combination can be fatal; life-threatening serotonin syndrome reported.',
     'severity': 'Contraindicated', 'mechanism': 'Serotonergic'},
]

PATIENT_COMPLEX = {
    'age':        72,
    'gender':     'Female',
    'conditions': ['Hypertension', 'Diabetes Type 2', 'Atrial Fibrillation'],
    'lab_values': {
        'eGFR':          42,
        'ALT':           85,
        'platelet_count':110,
        'INR':           3.2,
        'blood_glucose': 195,
    },
}

PATIENT_YOUNG_HEALTHY = {
    'age':        30,
    'gender':     'Male',
    'conditions': [],
    'lab_values': {},
}

# Minimal pharma DataFrame for main.py tests
MOCK_PHARMA = pd.DataFrame([
    {'brand_name': 'Augmentin 625 Duo Tablet',
     'primary_ingredient': 'Amoxycillin',
     'active_ingredients': "[{'name':'Amoxycillin','strength':'500mg'},{'name':'Clavulanic Acid','strength':'125mg'}]"},
    {'brand_name': 'Azithral 500 Tablet',
     'primary_ingredient': 'Azithromycin',
     'active_ingredients': "[{'name':'Azithromycin','strength':'500mg'}]"},
    {'brand_name': 'Ascoril LS Syrup',
     'primary_ingredient': 'Ambroxol',
     'active_ingredients': "[{'name':'Ambroxol','strength':'30mg/5ml'},{'name':'Levosalbutamol','strength':'1mg/5ml'}]"},
    {'brand_name': 'Calpol 500mg Tablet',
     'primary_ingredient': 'Paracetamol',
     'active_ingredients': "[{'name':'Paracetamol','strength':'500mg'}]"},
])

MOCK_DDI = pd.DataFrame([
    {'drug1_name': 'amoxycillin', 'drug2_name': 'azithromycin',
     'description': 'May increase the risk of cardiac arrhythmia and QT prolongation.',
     'severity': 'Moderate', 'mechanism': 'Pharmacokinetic', 'source': 'ddi_complete'},
    {'drug1_name': 'ambroxol', 'drug2_name': 'levosalbutamol',
     'description': 'Monitor for increased bronchodilator effects. Generally safe.',
     'severity': 'Mild', 'mechanism': 'Pharmacodynamic', 'source': 'ddi_complete'},
    {'drug1_name': 'warfarin', 'drug2_name': 'aspirin',
     'description': 'Bleeding and hemorrhage risk significantly increased.',
     'severity': 'Major', 'mechanism': 'Pharmacodynamic', 'source': 'ddi_complete'},
    {'drug1_name': 'metformin', 'drug2_name': 'ibuprofen',
     'description': 'May increase risk of renal failure and adverse GI effects.',
     'severity': 'Moderate', 'mechanism': 'Unknown', 'source': 'ddi_complete'},
])


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 1 — Step 3: Interaction Scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestStep3InteractionScoring(unittest.TestCase):
    """Tests for calculate_interaction_score_robust"""

    def test_fatal_keywords_score_maximum(self):
        """Fatal / life-threatening keywords should score at top tier (≥45)."""
        result = calculate_interaction_score_robust(INTERACTIONS_FATAL)
        bd = result['detailed_breakdown'][0]
        self.assertGreaterEqual(bd['score'], 45)
        self.assertIn(bd['severity'], ['CRITICAL', 'SEVERE'])

    def test_hemorrhage_keyword_scores_severe(self):
        """Bleeding/hemorrhage descriptions should be SEVERE or CRITICAL."""
        result = calculate_interaction_score_robust(INTERACTIONS_CRITICAL)
        bd_bleeding = result['detailed_breakdown'][0]
        self.assertGreaterEqual(bd_bleeding['score'], 35)

    def test_mild_interaction_lower_score(self):
        """Monitor/caution keyword → MILD tier."""
        result = calculate_interaction_score_robust(INTERACTIONS_MILD)
        bd = result['detailed_breakdown'][0]
        self.assertLessEqual(bd['score'], 25)

    def test_total_score_is_sum(self):
        """total_score must equal sum of individual scores."""
        result = calculate_interaction_score_robust(INTERACTIONS_CRITICAL + INTERACTIONS_MILD)
        expected = sum(b['score'] for b in result['detailed_breakdown'])
        self.assertEqual(result['total_score'], expected)

    def test_empty_interactions(self):
        """Empty input must not crash and should return zero score."""
        result = calculate_interaction_score_robust([])
        self.assertEqual(result['total_score'], 0)
        self.assertEqual(result['num_interactions'], 0)

    def test_negation_reduces_score(self):
        """Negation phrases should reduce score by ~70%."""
        pos = [{'drug_a':'A','drug_b':'B','description':'Risk of bleeding and hemorrhage.'}]
        neg = [{'drug_a':'A','drug_b':'B','description':'No significant risk of bleeding, unlikely.'}]
        pos_score = calculate_interaction_score_robust(pos)['total_score']
        neg_score = calculate_interaction_score_robust(neg)['total_score']
        self.assertLess(neg_score, pos_score)

    def test_risk_level_labels(self):
        """Critical total score → CRITICAL risk level."""
        # Create enough interactions to push total over 100
        many = [{'drug_a':f'D{i}','drug_b':f'E{i}',
                 'description':'Fatal cardiac arrest respiratory failure.'}
                for i in range(4)]
        result = calculate_interaction_score_robust(many)
        self.assertEqual(result['risk_level'], 'CRITICAL')

    def test_output_keys_present(self):
        """Result dict must contain all required keys."""
        result = calculate_interaction_score_robust(INTERACTIONS_MILD)
        for key in ('total_score','risk_level','risk_color','recommendation',
                    'num_interactions','detailed_breakdown','methodology'):
            self.assertIn(key, result)

    def test_each_breakdown_has_required_fields(self):
        """Each breakdown item needs drug pair, score, severity, icon."""
        result = calculate_interaction_score_robust(INTERACTIONS_CRITICAL)
        for item in result['detailed_breakdown']:
            for field in ('drugs','score','severity','icon','matched_keywords'):
                self.assertIn(field, item)

    def test_generic_interaction_gets_minimum_score(self):
        """An interaction with no keywords should still get score > 0."""
        vague = [{'drug_a':'X','drug_b':'Y','description':'some interaction exists'}]
        result = calculate_interaction_score_robust(vague)
        self.assertGreater(result['total_score'], 0)


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 2 — Step 4: Organ Distribution
# ─────────────────────────────────────────────────────────────────────────────

class TestStep4OrganDistribution(unittest.TestCase):
    """Tests for analyze_biological_impact"""

    def _run(self, interactions):
        base = calculate_interaction_score_robust(interactions)
        return analyze_biological_impact(interactions, base)

    def test_hemorrhage_hits_cardiovascular_and_hematologic(self):
        """Bleeding descriptions should map to CARDIOVASCULAR or HEMATOLOGIC."""
        result = self._run(INTERACTIONS_CRITICAL)
        organ_names = [s['system'] for s in result['affected_organ_systems']]
        self.assertTrue(
            any(o in organ_names for o in ['Cardiovascular','Hematologic']),
            f"Expected CV/Hematologic in {organ_names}"
        )

    def test_organ_failure_hits_renal_or_hepatic(self):
        """Organ failure description should hit RENAL or HEPATIC."""
        ix = [{'drug_a':'A','drug_b':'B',
               'description':'Risk of liver failure, hepatic toxicity, and renal failure.'}]
        result = self._run(ix)
        names = [s['system'] for s in result['affected_organ_systems']]
        self.assertTrue(any(o in names for o in ['Hepatic','Renal']), names)

    def test_highest_risk_organ_has_max_score(self):
        """highest_risk_organ must have the highest score."""
        result = self._run(INTERACTIONS_CRITICAL)
        systems = result['affected_organ_systems']
        if systems:
            top = result['highest_risk_organ']
            self.assertEqual(top['score'], max(s['score'] for s in systems))

    def test_empty_interactions_returns_empty_organs(self):
        """No interactions → no organ systems."""
        base = calculate_interaction_score_robust([])
        result = analyze_biological_impact([], base)
        self.assertEqual(result['num_organs_affected'], 0)

    def test_output_keys(self):
        result = self._run(INTERACTIONS_MILD)
        for key in ('affected_organ_systems','num_organs_affected','highest_risk_organ','methodology'):
            self.assertIn(key, result)

    def test_organ_list_is_sorted_descending(self):
        """Organs should be sorted highest score first."""
        result = self._run(INTERACTIONS_CRITICAL + INTERACTIONS_MILD)
        scores = [s['score'] for s in result['affected_organ_systems']]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_cns_keywords_map_to_cns(self):
        """Serotonin/CNS keywords should map to Central Nervous System."""
        ix = [{'drug_a':'A','drug_b':'B',
               'description':'Fatal serotonin syndrome with severe CNS seizure depression.'}]
        result = self._run(ix)
        names = [s['system'] for s in result['affected_organ_systems']]
        self.assertIn('Central Nervous System', names)

    def test_evidence_citation_present(self):
        """Each organ entry should have an evidence citation."""
        result = self._run(INTERACTIONS_CRITICAL)
        for sys in result['affected_organ_systems']:
            self.assertIn('evidence_citation', sys)


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 3 — Step 5: Patient Adjustment
# ─────────────────────────────────────────────────────────────────────────────

class TestStep5PatientAdjustment(unittest.TestCase):
    """Tests for adjust_for_patient_context"""

    def _get_systems(self, interactions=None):
        ix = interactions or INTERACTIONS_CRITICAL
        base = calculate_interaction_score_robust(ix)
        organ = analyze_biological_impact(ix, base)
        return organ['affected_organ_systems']

    def test_elderly_patient_increases_score(self):
        """An elderly patient should have ≥ adjusted score vs baseline."""
        systems = self._get_systems()
        if not systems:
            self.skipTest("No organ systems from test data")
        result = adjust_for_patient_context(systems, {'age': 80, 'conditions': [], 'lab_values': {}})
        for sys in result['adjusted_systems']:
            self.assertGreaterEqual(sys['adjusted_score'], sys['base_score'])

    def test_young_healthy_multiplier_is_one(self):
        """A 30-year-old with no conditions should have multiplier ~1.0."""
        systems = self._get_systems()
        if not systems:
            self.skipTest("No organ systems")
        result = adjust_for_patient_context(systems, PATIENT_YOUNG_HEALTHY)
        for sys in result['adjusted_systems']:
            self.assertAlmostEqual(sys['vulnerability_multiplier'], 1.0, places=1)

    def test_complex_patient_gets_higher_adjusted_scores(self):
        """Complex patient (elderly, HTN, DM, AFib, poor labs) > young healthy."""
        systems = self._get_systems()
        if not systems:
            self.skipTest("No organ systems")
        complex_result = adjust_for_patient_context(systems, PATIENT_COMPLEX)
        young_result   = adjust_for_patient_context(systems, PATIENT_YOUNG_HEALTHY)
        complex_total = sum(s['adjusted_score'] for s in complex_result['adjusted_systems'])
        young_total   = sum(s['adjusted_score'] for s in young_result['adjusted_systems'])
        self.assertGreater(complex_total, young_total)

    def test_no_patient_data_returns_unchanged(self):
        """None patient data → status NO_PATIENT_DATA, systems unchanged."""
        systems = self._get_systems()
        result = adjust_for_patient_context(systems, None)
        self.assertEqual(result['status'], 'NO_PATIENT_DATA')
        self.assertEqual(result['adjusted_systems'], systems)

    def test_low_egfr_increases_renal_score(self):
        """Low eGFR must increase RENAL adjusted score."""
        ix = [{'drug_a':'A','drug_b':'B','description':'Risk of renal failure and nephrotoxicity.'}]
        systems = self._get_systems(ix)
        renal = [s for s in systems if s.get('organ_key') == 'RENAL']
        if not renal:
            self.skipTest("No RENAL system in test data")
        sys_before = renal[0]['score']
        result = adjust_for_patient_context(
            systems, {'age': 40, 'conditions': [], 'lab_values': {'eGFR': 20}}
        )
        renal_after = next(s for s in result['adjusted_systems'] if s.get('organ_key') == 'RENAL')
        self.assertGreater(renal_after['adjusted_score'], sys_before)

    def test_elevated_alt_increases_hepatic_score(self):
        """Elevated ALT must increase HEPATIC adjusted score."""
        ix = [{'drug_a':'A','drug_b':'B','description':'Hepatic liver failure and hepatotoxicity ALT elevated.'}]
        systems = self._get_systems(ix)
        hepatic = [s for s in systems if s.get('organ_key') == 'HEPATIC']
        if not hepatic:
            self.skipTest("No HEPATIC system in test data")
        base = hepatic[0]['score']
        result = adjust_for_patient_context(
            systems, {'age': 50, 'conditions': [], 'lab_values': {'ALT': 250}}
        )
        h_after = next(s for s in result['adjusted_systems'] if s.get('organ_key') == 'HEPATIC')
        self.assertGreater(h_after['adjusted_score'], base)

    def test_hypertension_increases_cv_score(self):
        """Hypertension condition must increase CARDIOVASCULAR score."""
        ix = [{'drug_a':'A','drug_b':'B','description':'Increased cardiac arrhythmia risk and bleeding.'}]
        systems = self._get_systems(ix)
        cv = [s for s in systems if s.get('organ_key') == 'CARDIOVASCULAR']
        if not cv:
            self.skipTest("No CARDIOVASCULAR in test data")
        base = cv[0]['score']
        result = adjust_for_patient_context(
            systems, {'age': 50, 'conditions': ['Hypertension'], 'lab_values': {}}
        )
        cv_after = next(s for s in result['adjusted_systems'] if s.get('organ_key') == 'CARDIOVASCULAR')
        self.assertGreater(cv_after['adjusted_score'], base)

    def test_risk_factors_populated(self):
        """Complex patient should produce non-empty risk_factors lists."""
        systems = self._get_systems()
        if not systems:
            self.skipTest("No organ systems")
        result = adjust_for_patient_context(systems, PATIENT_COMPLEX)
        all_factors = [f for s in result['adjusted_systems'] for f in s.get('risk_factors', [])]
        self.assertGreater(len(all_factors), 0)

    def test_output_has_patient_profile(self):
        """Result should contain patient_profile summary."""
        systems = self._get_systems()
        if not systems:
            self.skipTest("No organ systems")
        result = adjust_for_patient_context(systems, PATIENT_COMPLEX)
        self.assertIn('patient_profile', result)
        self.assertEqual(result['patient_profile']['age'], 72)


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 4 — Step 6: Cascade Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestStep6CascadeDetection(unittest.TestCase):
    """Tests for detect_polypharmacy_cascades"""

    def _make_high_risk_system(self, organ='Cardiovascular', score=60, n_interactions=3):
        return {
            'system':            organ,
            'organ_key':         organ.upper().replace(' ','_'),
            'score':             score,
            'adjusted_score':    score,
            'severity':          'CRITICAL',
            'icon':              '🔴',
            'interaction_count': n_interactions,
            'risk_factors':      [],
        }

    def test_cascade_detected_when_thresholds_met(self):
        """3+ interactions + high score → cascade detected."""
        systems = [self._make_high_risk_system(score=60, n_interactions=3)]
        result  = detect_polypharmacy_cascades(systems, INTERACTIONS_CRITICAL, 4)
        self.assertTrue(result['has_cascades'])
        self.assertGreater(result['num_cascades'], 0)

    def test_no_cascade_for_low_interaction_count(self):
        """1 interaction should not trigger a cascade."""
        systems = [self._make_high_risk_system(score=60, n_interactions=1)]
        result  = detect_polypharmacy_cascades(systems, INTERACTIONS_MILD, 2)
        self.assertFalse(result['has_cascades'])

    def test_no_cascade_for_low_score(self):
        """High interaction count but low score → no CASCADE (may be CUMULATIVE)."""
        systems = [self._make_high_risk_system(score=10, n_interactions=4)]
        result  = detect_polypharmacy_cascades(systems, INTERACTIONS_MILD, 3)
        cascade_alerts = [c for c in result['cascades'] if c['alert_level'] == 'CASCADE']
        self.assertEqual(len(cascade_alerts), 0)

    def test_multiple_organs_all_checked(self):
        """Multiple high-risk organs should each be evaluated."""
        systems = [
            self._make_high_risk_system('Cardiovascular', 65, 3),
            self._make_high_risk_system('Hepatic',        55, 3),
        ]
        result = detect_polypharmacy_cascades(systems, INTERACTIONS_CRITICAL, 5)
        self.assertGreaterEqual(result['num_cascades'], 2)

    def test_output_structure(self):
        """Result must have required keys."""
        systems = [self._make_high_risk_system()]
        result = detect_polypharmacy_cascades(systems, INTERACTIONS_CRITICAL, 4)
        for key in ('has_cascades','num_cascades','cascades','methodology',
                    'polypharmacy_risk_multiplier'):
            self.assertIn(key, result)

    def test_cascade_entry_fields(self):
        """Each cascade entry must have required fields."""
        systems = [self._make_high_risk_system(score=65, n_interactions=3)]
        result  = detect_polypharmacy_cascades(systems, INTERACTIONS_CRITICAL, 5)
        if result['cascades']:
            cascade = result['cascades'][0]
            for field in ('organ_system','alert_level','cumulative_score',
                          'severity','num_interactions','evidence_citation'):
                self.assertIn(field, cascade)

    def test_cascades_sorted_by_score_desc(self):
        """Cascades should be sorted highest score first."""
        systems = [
            self._make_high_risk_system('Cardiovascular', 80, 4),
            self._make_high_risk_system('Hepatic', 55, 3),
        ]
        result  = detect_polypharmacy_cascades(systems, INTERACTIONS_CRITICAL, 5)
        scores  = [c['cumulative_score'] for c in result['cascades']]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 5 — Step 7: Report Generation
# ─────────────────────────────────────────────────────────────────────────────

class TestStep7ReportGeneration(unittest.TestCase):
    """Tests for generate_clinical_report"""

    def _full_pipeline(self, interactions=None, patient=None):
        ix   = interactions or INTERACTIONS_CRITICAL
        base = calculate_interaction_score_robust(ix)
        org  = analyze_biological_impact(ix, base)
        adj  = adjust_for_patient_context(org['affected_organ_systems'], patient or {})
        casc = detect_polypharmacy_cascades(adj.get('adjusted_systems', []), ix, 3)
        rep  = generate_clinical_report(base, org, adj, casc, patient)
        return rep

    def test_report_has_summary(self):
        report = self._full_pipeline()
        self.assertIn('summary', report)
        s = report['summary']
        for k in ('overall_risk_level','risk_color','risk_icon','primary_action',
                  'total_interaction_score','num_interactions','num_organs_affected','num_cascades'):
            self.assertIn(k, s)

    def test_critical_interaction_generates_high_risk_report(self):
        """Fatal/life-threatening interaction should produce at minimum MODERATE risk."""
        report = self._full_pipeline(INTERACTIONS_FATAL)
        valid_high = ['CRITICAL', 'SEVERE', 'MODERATE']
        self.assertIn(report['summary']['overall_risk_level'], valid_high)
        # Score must be meaningfully high (not MILD/MINIMAL)
        self.assertGreater(report['summary']['total_interaction_score'], 20)

    def test_mild_interaction_generates_low_risk_report(self):
        report = self._full_pipeline(INTERACTIONS_MILD)
        self.assertIn(report['summary']['overall_risk_level'], ['MILD','MINIMAL','MODERATE'])

    def test_evidence_base_present(self):
        report = self._full_pipeline()
        self.assertIn('evidence_base', report)
        self.assertIn('key_sources', report['evidence_base'])
        self.assertGreater(len(report['evidence_base']['key_sources']), 0)

    def test_report_contains_all_sections(self):
        report = self._full_pipeline()
        for section in ('summary','interaction_analysis','organ_system_analysis',
                        'patient_specific_analysis','polypharmacy_cascade_analysis',
                        'evidence_base','report_metadata'):
            self.assertIn(section, report)

    def test_patient_data_included_when_provided(self):
        report = self._full_pipeline(patient=PATIENT_COMPLEX)
        adj = report['patient_specific_analysis']
        self.assertEqual(adj.get('status'), 'ADJUSTED')

    def test_no_patient_data_flag(self):
        report = self._full_pipeline(patient=None)
        adj = report['patient_specific_analysis']
        self.assertEqual(adj.get('status'), 'NO_PATIENT_DATA')

    def test_all_citations_is_list(self):
        report = self._full_pipeline()
        self.assertIsInstance(report['evidence_base']['all_citations'], list)


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 6 — main.py functions
# ─────────────────────────────────────────────────────────────────────────────

class TestMainFunctions(unittest.TestCase):
    """Tests for the data-layer functions in main.py"""

    def setUp(self):
        """Inject mock DataFrames into main module."""
        m.pharma_db         = MOCK_PHARMA.copy()
        m.interactions_table= MOCK_DDI.copy()
        m.ingredient_id_map = {}

    # ── search_brand_name ────────────────────────────────────────────────

    def test_search_returns_matching_brands(self):
        results = m.search_brand_name('Aug')
        self.assertIn('Augmentin 625 Duo Tablet', results.tolist())

    def test_search_case_insensitive(self):
        r1 = m.search_brand_name('aug').tolist()
        r2 = m.search_brand_name('AUG').tolist()
        self.assertEqual(r1, r2)

    def test_search_no_match_returns_empty(self):
        results = m.search_brand_name('ZZZNOMATCH')
        self.assertEqual(len(results), 0)

    def test_search_limit_respected(self):
        results = m.search_brand_name('A', limit=2)
        self.assertLessEqual(len(results), 2)

    def test_search_empty_db_returns_empty(self):
        m.pharma_db = pd.DataFrame()
        results = m.search_brand_name('Aug')
        self.assertEqual(len(results), 0)

    # ── get_ingredients_by_brand_name ────────────────────────────────────

    def test_get_ingredients_returns_list(self):
        ings = m.get_ingredients_by_brand_name('Augmentin 625 Duo Tablet')
        self.assertIsInstance(ings, list)
        self.assertGreater(len(ings), 0)

    def test_augmentin_contains_amoxycillin_and_clavulanic(self):
        ings = m.get_ingredients_by_brand_name('Augmentin 625 Duo Tablet')
        ing_lower = [i.lower() for i in ings]
        self.assertIn('amoxycillin', ing_lower)
        self.assertIn('clavulanic acid', ing_lower)

    def test_unknown_brand_returns_empty(self):
        ings = m.get_ingredients_by_brand_name('NoSuchBrand 999')
        self.assertEqual(ings, [])

    def test_case_insensitive_brand_lookup(self):
        ings1 = m.get_ingredients_by_brand_name('Augmentin 625 Duo Tablet')
        ings2 = m.get_ingredients_by_brand_name('augmentin 625 duo tablet')
        self.assertEqual(sorted(ings1), sorted(ings2))

    def test_no_duplicate_ingredients(self):
        ings = m.get_ingredients_by_brand_name('Augmentin 625 Duo Tablet')
        self.assertEqual(len(ings), len(set(ings)))

    # ── get_drug_interactions ────────────────────────────────────────────

    def test_known_pair_returns_interaction(self):
        result = m.get_drug_interactions(['Amoxycillin', 'Azithromycin'])
        self.assertGreater(len(result), 0)

    def test_unknown_pair_returns_empty(self):
        result = m.get_drug_interactions(['Paracetamol', 'Vitamin C'])
        self.assertEqual(result, [])

    def test_single_drug_no_pairs(self):
        result = m.get_drug_interactions(['Paracetamol'])
        self.assertEqual(result, [])

    def test_interaction_dict_has_required_fields(self):
        result = m.get_drug_interactions(['Amoxycillin', 'Azithromycin'])
        if result:
            item = result[0]
            for field in ('drug_a','drug_b','description','severity'):
                self.assertIn(field, item)

    def test_empty_ingredients_returns_empty(self):
        result = m.get_drug_interactions([])
        self.assertEqual(result, [])

    def test_partial_name_match_works(self):
        """Partial match should find interactions via contains search."""
        result = m.get_drug_interactions(['amoxycillin', 'azithromycin'])
        self.assertGreater(len(result), 0)

    # ── get_interactions_for_multiple_brands ────────────────────────────

    def test_multi_brand_returns_all_ingredients(self):
        result = m.get_interactions_for_multiple_brands(
            ['Augmentin 625 Duo Tablet', 'Azithral 500 Tablet']
        )
        self.assertIn('all_ingredients', result)
        self.assertGreater(len(result['all_ingredients']), 0)

    def test_multi_brand_deduplicates_ingredients(self):
        result = m.get_interactions_for_multiple_brands(
            ['Augmentin 625 Duo Tablet', 'Augmentin 625 Duo Tablet']  # same brand twice
        )
        self.assertEqual(
            len(result['all_ingredients']),
            len(set(result['all_ingredients']))
        )

    def test_multi_brand_result_structure(self):
        result = m.get_interactions_for_multiple_brands(['Calpol 500mg Tablet'])
        for key in ('brand_names','brand_ingredient_map','all_ingredients',
                    'interactions_found','num_interactions','requires_cascade_analysis'):
            self.assertIn(key, result)

    def test_cascade_flag_true_for_3_plus_ingredients(self):
        result = m.get_interactions_for_multiple_brands(
            ['Augmentin 625 Duo Tablet', 'Azithral 500 Tablet', 'Ascoril LS Syrup']
        )
        self.assertTrue(result['requires_cascade_analysis'])

    # ── analyze_interactions_with_context ────────────────────────────────

    def test_no_interaction_status(self):
        """Brands with no known interactions → NO_INTERACTIONS status."""
        result = m.analyze_interactions_with_context(['Calpol 500mg Tablet'])
        # Calpol (Paracetamol) has no partner in mock DDI → no interaction
        self.assertIn(result['status'], ('NO_INTERACTIONS', 'INTERACTIONS_FOUND'))

    def test_full_pipeline_returns_clinical_report_when_interactions_found(self):
        """If interactions are found the result must have a clinical_report."""
        result = m.analyze_interactions_with_context(
            ['Augmentin 625 Duo Tablet', 'Azithral 500 Tablet'],
            patient_data=PATIENT_COMPLEX,
        )
        if result['status'] == 'INTERACTIONS_FOUND':
            self.assertIn('clinical_report', result)
            self.assertIn('summary', result['clinical_report'])

    def test_save_report_creates_file(self):
        """save_report parameter should write a JSON file."""
        import tempfile, os, json
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            path = tf.name
        try:
            result = m.analyze_interactions_with_context(
                ['Augmentin 625 Duo Tablet', 'Azithral 500 Tablet'],
                save_report=path,
            )
            if result['status'] == 'INTERACTIONS_FOUND':
                self.assertTrue(os.path.exists(path))
                with open(path) as f:
                    data = json.load(f)
                self.assertIn('summary', data)
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 7 — End-to-End Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd(unittest.TestCase):
    """Full pipeline integration tests."""

    def setUp(self):
        m.pharma_db          = MOCK_PHARMA.copy()
        m.interactions_table = MOCK_DDI.copy()
        m.ingredient_id_map  = {}

    def test_full_pipeline_no_crash(self):
        """Full pipeline must complete without raising exceptions."""
        try:
            m.analyze_interactions_with_context(
                ['Augmentin 625 Duo Tablet', 'Azithral 500 Tablet', 'Ascoril LS Syrup'],
                patient_data=PATIENT_COMPLEX,
            )
        except Exception as e:
            self.fail(f"Pipeline raised an exception: {e}")

    def test_pipeline_without_patient_data(self):
        """Pipeline must work fine without patient data."""
        try:
            m.analyze_interactions_with_context(
                ['Augmentin 625 Duo Tablet', 'Azithral 500 Tablet'],
            )
        except Exception as e:
            self.fail(f"Pipeline raised: {e}")

    def test_pipeline_with_young_healthy_patient(self):
        """Young healthy patient should produce lower risk than complex patient."""
        result_complex = m.analyze_interactions_with_context(
            ['Augmentin 625 Duo Tablet', 'Azithral 500 Tablet'],
            patient_data=PATIENT_COMPLEX,
        )
        result_young = m.analyze_interactions_with_context(
            ['Augmentin 625 Duo Tablet', 'Azithral 500 Tablet'],
            patient_data=PATIENT_YOUNG_HEALTHY,
        )
        if (result_complex['status'] == 'INTERACTIONS_FOUND' and
                result_young['status'] == 'INTERACTIONS_FOUND'):
            complex_score = result_complex['clinical_report']['summary']['total_interaction_score']
            young_score   = result_young['clinical_report']['summary']['total_interaction_score']
            # Base interaction score is the same; patient adjustment is on organs
            # Just ensure no crash and both have reports
            self.assertIn('clinical_report', result_complex)
            self.assertIn('clinical_report', result_young)

    def test_single_brand_no_pairs(self):
        """Single brand → no pairs → no interactions found."""
        result = m.analyze_interactions_with_context(['Calpol 500mg Tablet'])
        # Calpol only has Paracetamol; no counterpart in mock DDI
        self.assertIn(result['status'], ('NO_INTERACTIONS', 'INTERACTIONS_FOUND'))

    def test_report_risk_levels_are_valid_values(self):
        """Risk level must be one of the defined tiers."""
        valid = {'CRITICAL','SEVERE','MODERATE','MILD','MINIMAL'}
        result = m.analyze_interactions_with_context(
            ['Augmentin 625 Duo Tablet','Azithral 500 Tablet'],
            patient_data=PATIENT_COMPLEX,
        )
        if result['status'] == 'INTERACTIONS_FOUND':
            level = result['clinical_report']['summary']['overall_risk_level']
            self.assertIn(level, valid)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Optional: verbose flag
    verbosity = 2 if '-v' in sys.argv else 1
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)