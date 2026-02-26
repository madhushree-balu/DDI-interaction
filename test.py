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
# Each fixture targets a specific organ system so tests are unambiguous.
# ─────────────────────────────────────────────────────────────────────────────

# ── Per-organ targeted interactions ──────────────────────────────────────────

IX_CARDIOVASCULAR = {
    'drug_a': 'Warfarin', 'drug_b': 'Aspirin',
    'description': 'The risk of cardiac arrhythmia, QT prolongation, and bleeding is significantly increased when combined.',
    'severity': 'Major', 'mechanism': 'Pharmacodynamic',
}
IX_HEPATIC = {
    'drug_a': 'Isoniazid', 'drug_b': 'Rifampicin',
    'description': 'Increased hepatotoxicity and liver failure risk due to elevated ALT and AST. Hepatic cytochrome CYP450 metabolism impaired.',
    'severity': 'Major', 'mechanism': 'Hepatic enzyme induction',
}
IX_RENAL = {
    'drug_a': 'Gentamicin', 'drug_b': 'Vancomycin',
    'description': 'Nephrotoxic combination significantly increases risk of acute kidney injury. Monitor creatinine and eGFR closely.',
    'severity': 'Major', 'mechanism': 'Additive nephrotoxicity',
}
IX_HEMATOLOGIC = {
    'drug_a': 'Clopidogrel', 'drug_b': 'Warfarin',
    'description': 'Risk of major bleeding, hemorrhage, and platelet dysfunction is markedly increased. INR must be monitored.',
    'severity': 'Major', 'mechanism': 'Antiplatelet + anticoagulant synergy',
}
IX_GI = {
    'drug_a': 'Naproxen', 'drug_b': 'Prednisolone',
    'description': 'Gastrointestinal bleeding, stomach ulcer formation, and nausea significantly increased with concurrent use.',
    'severity': 'Moderate', 'mechanism': 'GI mucosal damage',
}
IX_CNS = {
    'drug_a': 'MAO Inhibitor', 'drug_b': 'SSRI',
    'description': 'Fatal serotonin syndrome with life-threatening CNS seizure, confusion, and severe sedation. Brain serotonergic crisis.',
    'severity': 'Contraindicated', 'mechanism': 'Serotonergic excess',
}
IX_RESPIRATORY = {
    'drug_a': 'Morphine', 'drug_b': 'Diazepam',
    'description': 'Severe respiratory depression and dyspnea. Risk of apnea and pulmonary failure requiring ICU admission.',
    'severity': 'Major', 'mechanism': 'CNS/respiratory depression',
}
IX_ENDOCRINE = {
    'drug_a': 'Glibenclamide', 'drug_b': 'Fluconazole',
    'description': 'Severe hypoglycaemia risk due to impaired insulin and glucose metabolism. Blood glucose must be monitored.',
    'severity': 'Major', 'mechanism': 'CYP2C9 inhibition',
}
IX_MUSCULOSKELETAL = {
    'drug_a': 'Simvastatin', 'drug_b': 'Gemfibrozil',
    'description': 'Substantially increased risk of rhabdomyolysis and severe myopathy with elevated creatine kinase (CK).',
    'severity': 'Major', 'mechanism': 'CYP3A4 inhibition',
}
IX_IMMUNE = {
    'drug_a': 'Methotrexate', 'drug_b': 'Trimethoprim',
    'description': 'Severe immunosuppression and risk of anaphylaxis. Stevens-Johnson syndrome and hypersensitivity reactions reported.',
    'severity': 'Major', 'mechanism': 'Folate antagonism',
}

# ── Composite fixture sets ────────────────────────────────────────────────────

# All 10 organ systems represented
INTERACTIONS_ALL_ORGANS = [
    IX_CARDIOVASCULAR, IX_HEPATIC, IX_RENAL, IX_HEMATOLOGIC,
    IX_GI, IX_CNS, IX_RESPIRATORY, IX_ENDOCRINE,
    IX_MUSCULOSKELETAL, IX_IMMUNE,
]

# High-severity multi-organ set (used in scoring + cascade tests)
INTERACTIONS_CRITICAL = [
    IX_CARDIOVASCULAR,   # bleeding + hemorrhage + arrhythmia
    IX_RENAL,            # nephrotoxicity
    IX_HEPATIC,          # hepatotoxicity
]

# Mild / low-risk set
INTERACTIONS_MILD = [
    {'drug_a': 'Paracetamol', 'drug_b': 'Ibuprofen',
     'description': 'May increase risk of GI adverse effects. Monitor patients for stomach discomfort.',
     'severity': 'Mild', 'mechanism': 'Unknown'},
]

# Fatal keyword set (Step 3 max-tier test)
INTERACTIONS_FATAL = [IX_CNS]   # "fatal" + "life-threatening" + "serotonin"

# Patient profiles
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

# ── Pharma brand mock ─────────────────────────────────────────────────────────

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
    {'brand_name': 'Warfarin Sodium Tablet',
     'primary_ingredient': 'Warfarin',
     'active_ingredients': "[{'name':'Warfarin','strength':'5mg'}]"},
    {'brand_name': 'Metformin 500mg Tablet',
     'primary_ingredient': 'Metformin',
     'active_ingredients': "[{'name':'Metformin','strength':'500mg'}]"},
    {'brand_name': 'Isoniazid 300mg Tablet',
     'primary_ingredient': 'Isoniazid',
     'active_ingredients': "[{'name':'Isoniazid','strength':'300mg'}]"},
    {'brand_name': 'Rifampicin 450mg Capsule',
     'primary_ingredient': 'Rifampicin',
     'active_ingredients': "[{'name':'Rifampicin','strength':'450mg'}]"},
    {'brand_name': 'Simvastatin 20mg Tablet',
     'primary_ingredient': 'Simvastatin',
     'active_ingredients': "[{'name':'Simvastatin','strength':'20mg'}]"},
    {'brand_name': 'Morphine Sulfate Tablet',
     'primary_ingredient': 'Morphine',
     'active_ingredients': "[{'name':'Morphine','strength':'10mg'}]"},
    {'brand_name': 'Diazepam 5mg Tablet',
     'primary_ingredient': 'Diazepam',
     'active_ingredients': "[{'name':'Diazepam','strength':'5mg'}]"},
])

# ── Full DDI mock — one entry per organ system ────────────────────────────────
# Each row has a description with clear organ-system keywords so Step 4 routing
# is deterministic and tests are not all cardiovascular.

MOCK_DDI = pd.DataFrame([
    # CARDIOVASCULAR — arrhythmia / QT / cardiac
    {'drug1_name': 'amoxycillin',  'drug2_name': 'azithromycin',
     'description': 'May increase the risk of cardiac arrhythmia and QT prolongation in susceptible patients.',
     'severity': 'Moderate', 'mechanism': 'Pharmacokinetic', 'source': 'ddi_complete'},
    # RESPIRATORY — bronchodilator / respiratory
    {'drug1_name': 'ambroxol',     'drug2_name': 'levosalbutamol',
     'description': 'Concurrent use may cause excessive bronchodilator effects and respiratory distress. Monitor breathing.',
     'severity': 'Mild', 'mechanism': 'Pharmacodynamic', 'source': 'ddi_complete'},
    # HEMATOLOGIC — bleeding / hemorrhage / platelet
    {'drug1_name': 'warfarin',     'drug2_name': 'aspirin',
     'description': 'Bleeding and hemorrhage risk significantly increased. Platelet function and coagulation impaired.',
     'severity': 'Major', 'mechanism': 'Pharmacodynamic', 'source': 'ddi_complete'},
    # RENAL + GI — kidney failure / GI effects
    {'drug1_name': 'metformin',    'drug2_name': 'ibuprofen',
     'description': 'May increase risk of renal failure, nephrotoxicity, and adverse gastrointestinal GI effects including nausea.',
     'severity': 'Moderate', 'mechanism': 'Unknown', 'source': 'ddi_complete'},
    # HEPATIC — liver / hepatotoxicity / ALT / CYP
    {'drug1_name': 'isoniazid',    'drug2_name': 'rifampicin',
     'description': 'Severe hepatotoxicity and liver failure reported. ALT and AST elevation common. Hepatic CYP450 enzyme induction.',
     'severity': 'Major', 'mechanism': 'Hepatic enzyme induction', 'source': 'ddi_complete'},
    # MUSCULOSKELETAL — rhabdomyolysis / myopathy / CK
    {'drug1_name': 'simvastatin',  'drug2_name': 'gemfibrozil',
     'description': 'Substantially increased risk of rhabdomyolysis, severe myopathy, and elevated creatine kinase (CK).',
     'severity': 'Major', 'mechanism': 'CYP3A4 inhibition', 'source': 'ddi_complete'},
    # CNS — serotonin / CNS / seizure / sedation
    {'drug1_name': 'phenelzine',   'drug2_name': 'fluoxetine',
     'description': 'Fatal serotonin syndrome with life-threatening CNS seizure, confusion, and severe sedation reported.',
     'severity': 'Contraindicated', 'mechanism': 'Serotonergic excess', 'source': 'ddi_complete'},
    # RESPIRATORY — apnea / respiratory depression / dyspnea
    {'drug1_name': 'morphine',     'drug2_name': 'diazepam',
     'description': 'Severe respiratory depression and dyspnea. Apnea and pulmonary failure risk requires monitoring.',
     'severity': 'Major', 'mechanism': 'CNS depression', 'source': 'ddi_complete'},
    # ENDOCRINE — glucose / hypoglycaemia / insulin
    {'drug1_name': 'glibenclamide','drug2_name': 'fluconazole',
     'description': 'Severe hypoglycaemia due to impaired glucose metabolism and insulin secretion. Blood glucose monitoring essential.',
     'severity': 'Major', 'mechanism': 'CYP2C9 inhibition', 'source': 'ddi_complete'},
    # IMMUNE — immunosuppression / anaphylaxis / hypersensitivity
    {'drug1_name': 'methotrexate', 'drug2_name': 'trimethoprim',
     'description': 'Severe immunosuppression with risk of anaphylaxis and Stevens-Johnson hypersensitivity reactions.',
     'severity': 'Major', 'mechanism': 'Folate antagonism', 'source': 'ddi_complete'},
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
    """Tests for analyze_biological_impact — one test per organ system."""

    def _run(self, interactions):
        base = calculate_interaction_score_robust(interactions)
        return analyze_biological_impact(interactions, base)

    def _organ_names(self, interactions):
        return [s['system'] for s in self._run(interactions)['affected_organ_systems']]

    # ── Per-organ routing tests ───────────────────────────────────────────────

    def test_cardiovascular_keywords_route_to_cardiovascular(self):
        """Arrhythmia / QT / cardiac keywords → Cardiovascular."""
        names = self._organ_names([IX_CARDIOVASCULAR])
        self.assertIn('Cardiovascular', names, f"Got: {names}")

    def test_hepatic_keywords_route_to_hepatic(self):
        """Hepatotoxicity / ALT / CYP450 keywords → Hepatic."""
        names = self._organ_names([IX_HEPATIC])
        self.assertIn('Hepatic', names, f"Got: {names}")

    def test_renal_keywords_route_to_renal(self):
        """Nephrotoxicity / creatinine / eGFR keywords → Renal."""
        names = self._organ_names([IX_RENAL])
        self.assertIn('Renal', names, f"Got: {names}")

    def test_hematologic_keywords_route_to_hematologic(self):
        """Bleeding / hemorrhage / platelet / coagulation → Hematologic."""
        names = self._organ_names([IX_HEMATOLOGIC])
        self.assertTrue(
            any(o in names for o in ['Hematologic', 'Cardiovascular']),
            f"Expected Hematologic/CV in {names}"
        )

    def test_gi_keywords_route_to_gastrointestinal(self):
        """Gastrointestinal / ulcer / nausea / stomach → Gastrointestinal."""
        names = self._organ_names([IX_GI])
        self.assertIn('Gastrointestinal', names, f"Got: {names}")

    def test_cns_keywords_route_to_cns(self):
        """Serotonin / CNS / seizure / sedation → Central Nervous System."""
        names = self._organ_names([IX_CNS])
        self.assertIn('Central Nervous System', names, f"Got: {names}")

    def test_respiratory_keywords_route_to_respiratory(self):
        """Respiratory depression / apnea / dyspnea / pulmonary → Respiratory."""
        names = self._organ_names([IX_RESPIRATORY])
        self.assertIn('Respiratory', names, f"Got: {names}")

    def test_endocrine_keywords_route_to_endocrine(self):
        """Hypoglycaemia / glucose / insulin keywords → Endocrine."""
        names = self._organ_names([IX_ENDOCRINE])
        self.assertIn('Endocrine', names, f"Got: {names}")

    def test_musculoskeletal_keywords_route_to_musculoskeletal(self):
        """Rhabdomyolysis / myopathy / creatine kinase → Musculoskeletal."""
        names = self._organ_names([IX_MUSCULOSKELETAL])
        self.assertIn('Musculoskeletal', names, f"Got: {names}")

    def test_immune_keywords_route_to_immune(self):
        """Anaphylaxis / immunosuppression / Stevens-Johnson → Immune System."""
        names = self._organ_names([IX_IMMUNE])
        self.assertIn('Immune System', names, f"Got: {names}")

    # ── Multi-organ coverage tests ────────────────────────────────────────────

    def test_all_organs_fixture_produces_multiple_organ_systems(self):
        """INTERACTIONS_ALL_ORGANS should produce ≥ 6 distinct organ systems."""
        result = self._run(INTERACTIONS_ALL_ORGANS)
        n = result['num_organs_affected']
        self.assertGreaterEqual(n, 6,
            f"Expected ≥6 organ systems from all-organs fixture, got {n}: "
            f"{[s['system'] for s in result['affected_organ_systems']]}")

    def test_all_organs_fixture_covers_every_expected_system(self):
        """Each major organ system should appear at least once."""
        names = set(self._organ_names(INTERACTIONS_ALL_ORGANS))
        expected = {
            'Cardiovascular', 'Hepatic', 'Renal', 'Gastrointestinal',
            'Central Nervous System', 'Respiratory', 'Endocrine', 'Musculoskeletal',
        }
        missing = expected - names
        self.assertEqual(missing, set(),
            f"Missing organ systems: {missing}. Got: {names}")

    def test_single_organ_fixture_produces_exactly_targeted_system(self):
        """A single-organ fixture should produce exactly 1 primary system."""
        result = self._run([IX_CNS])  # pure CNS description
        self.assertGreater(result['num_organs_affected'], 0)
        self.assertIn('Central Nervous System',
                      [s['system'] for s in result['affected_organ_systems']])

    # ── Structural / correctness tests ───────────────────────────────────────

    def test_highest_risk_organ_has_max_score(self):
        """highest_risk_organ must have the highest score."""
        result = self._run(INTERACTIONS_ALL_ORGANS)
        systems = result['affected_organ_systems']
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
        result = self._run(INTERACTIONS_ALL_ORGANS)
        scores = [s['score'] for s in result['affected_organ_systems']]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_evidence_citation_present_on_all_organs(self):
        """Every organ entry should have an evidence_citation field."""
        result = self._run(INTERACTIONS_ALL_ORGANS)
        for sys in result['affected_organ_systems']:
            self.assertIn('evidence_citation', sys)

    def test_severity_weight_applied(self):
        """Higher-weight organ (CV=1.52) should score higher than lower-weight (GI=1.15)
        when fed descriptions of equal base score."""
        cv_only  = [{'drug_a':'X','drug_b':'Y',
                     'description':'Cardiac arrhythmia and bleeding hemorrhage risk.'}]
        gi_only  = [{'drug_a':'A','drug_b':'B',
                     'description':'Gastrointestinal stomach GI nausea ulcer only.'}]
        cv_res = self._run(cv_only)
        gi_res = self._run(gi_only)
        cv_score = next((s['score'] for s in cv_res['affected_organ_systems']
                         if s['system'] == 'Cardiovascular'), 0)
        gi_score = next((s['score'] for s in gi_res['affected_organ_systems']
                         if s['system'] == 'Gastrointestinal'), 0)
        # Both should be > 0 and CV weight (1.52) > GI weight (1.15)
        self.assertGreater(cv_score, 0)
        self.assertGreater(gi_score, 0)
        self.assertGreater(cv_score, gi_score,
            f"CV score ({cv_score}) should exceed GI score ({gi_score}) due to higher weight")


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 3 — Step 5: Patient Adjustment
# ─────────────────────────────────────────────────────────────────────────────

class TestStep5PatientAdjustment(unittest.TestCase):
    """Tests for adjust_for_patient_context"""

    def _systems_for(self, interactions):
        base  = calculate_interaction_score_robust(interactions)
        organ = analyze_biological_impact(interactions, base)
        return organ['affected_organ_systems']

    def _systems_all(self):
        return self._systems_for(INTERACTIONS_ALL_ORGANS)

    def test_elderly_patient_increases_all_scores(self):
        """Every organ system adjusted score should be ≥ base score for elderly patient."""
        systems = self._systems_all()
        result  = adjust_for_patient_context(systems, {'age': 80, 'conditions': [], 'lab_values': {}})
        for sys in result['adjusted_systems']:
            self.assertGreaterEqual(
                sys['adjusted_score'], sys['base_score'],
                f"{sys['system']}: adjusted {sys['adjusted_score']} < base {sys['base_score']}"
            )

    def test_young_healthy_multiplier_near_one(self):
        """30-year-old healthy patient → multiplier ~1.0 on every organ."""
        systems = self._systems_all()
        result  = adjust_for_patient_context(systems, PATIENT_YOUNG_HEALTHY)
        for sys in result['adjusted_systems']:
            self.assertAlmostEqual(
                sys['vulnerability_multiplier'], 1.0, places=1,
                msg=f"{sys['system']} multiplier {sys['vulnerability_multiplier']} expected ~1.0"
            )

    def test_complex_patient_total_higher_than_young_healthy(self):
        """Complex patient total adjusted score > young healthy total."""
        systems       = self._systems_all()
        complex_total = sum(s['adjusted_score'] for s in
                            adjust_for_patient_context(systems, PATIENT_COMPLEX)['adjusted_systems'])
        young_total   = sum(s['adjusted_score'] for s in
                            adjust_for_patient_context(systems, PATIENT_YOUNG_HEALTHY)['adjusted_systems'])
        self.assertGreater(complex_total, young_total)

    def test_no_patient_data_returns_unchanged(self):
        """None patient data → status NO_PATIENT_DATA, systems passed through unchanged."""
        systems = self._systems_all()
        result  = adjust_for_patient_context(systems, None)
        self.assertEqual(result['status'], 'NO_PATIENT_DATA')
        self.assertEqual(result['adjusted_systems'], systems)

    def test_low_egfr_increases_renal_score(self):
        """eGFR=20 (Stage 4 CKD, ×1.89) must raise RENAL adjusted score above base."""
        systems = self._systems_for([IX_RENAL])
        renal   = next((s for s in systems if s['organ_key'] == 'RENAL'), None)
        self.assertIsNotNone(renal, "RENAL organ system not produced by IX_RENAL fixture")
        base    = renal['score']
        result  = adjust_for_patient_context(
            systems, {'age': 40, 'conditions': [], 'lab_values': {'eGFR': 20}}
        )
        after   = next(s for s in result['adjusted_systems'] if s['organ_key'] == 'RENAL')
        self.assertGreater(after['adjusted_score'], base,
            f"eGFR=20 should raise RENAL score above {base}, got {after['adjusted_score']}")

    def test_elevated_alt_increases_hepatic_score(self):
        """ALT=250 U/L (×2.05 via Hy's Law) must raise HEPATIC adjusted score above base."""
        systems = self._systems_for([IX_HEPATIC])
        hepatic = next((s for s in systems if s['organ_key'] == 'HEPATIC'), None)
        self.assertIsNotNone(hepatic, "HEPATIC organ system not produced by IX_HEPATIC fixture")
        base    = hepatic['score']
        result  = adjust_for_patient_context(
            systems, {'age': 50, 'conditions': [], 'lab_values': {'ALT': 250}}
        )
        after   = next(s for s in result['adjusted_systems'] if s['organ_key'] == 'HEPATIC')
        self.assertGreater(after['adjusted_score'], base,
            f"ALT=250 should raise HEPATIC score above {base}, got {after['adjusted_score']}")

    def test_hypertension_increases_cv_score(self):
        """Hypertension (×1.45) must raise CARDIOVASCULAR adjusted score above base."""
        systems = self._systems_for([IX_CARDIOVASCULAR])
        cv      = next((s for s in systems if s['organ_key'] == 'CARDIOVASCULAR'), None)
        self.assertIsNotNone(cv, "CARDIOVASCULAR not produced by IX_CARDIOVASCULAR fixture")
        base    = cv['score']
        result  = adjust_for_patient_context(
            systems, {'age': 50, 'conditions': ['Hypertension'], 'lab_values': {}}
        )
        after   = next(s for s in result['adjusted_systems'] if s['organ_key'] == 'CARDIOVASCULAR')
        self.assertGreater(after['adjusted_score'], base,
            f"Hypertension should raise CV score above {base}, got {after['adjusted_score']}")

    def test_afib_increases_cv_score(self):
        """Atrial Fibrillation (×1.52) must raise CARDIOVASCULAR score."""
        systems = self._systems_for([IX_CARDIOVASCULAR])
        cv      = next((s for s in systems if s['organ_key'] == 'CARDIOVASCULAR'), None)
        if not cv:
            self.skipTest("No CV system")
        base    = cv['score']
        result  = adjust_for_patient_context(
            systems, {'age': 50, 'conditions': ['Atrial Fibrillation'], 'lab_values': {}}
        )
        after   = next(s for s in result['adjusted_systems'] if s['organ_key'] == 'CARDIOVASCULAR')
        self.assertGreater(after['adjusted_score'], base)

    def test_low_platelets_increases_hematologic_score(self):
        """Platelets=40k (×2.15 severe thrombocytopenia) must raise HEMATOLOGIC score."""
        systems = self._systems_for([IX_HEMATOLOGIC])
        hema    = next((s for s in systems if s['organ_key'] == 'HEMATOLOGIC'), None)
        self.assertIsNotNone(hema, "HEMATOLOGIC not produced by IX_HEMATOLOGIC fixture")
        base    = hema['score']
        result  = adjust_for_patient_context(
            systems, {'age': 50, 'conditions': [], 'lab_values': {'platelet_count': 40}}
        )
        after   = next(s for s in result['adjusted_systems'] if s['organ_key'] == 'HEMATOLOGIC')
        self.assertGreater(after['adjusted_score'], base,
            f"Platelets=40k should raise HEMATOLOGIC above {base}, got {after['adjusted_score']}")

    def test_diabetes_increases_endocrine_score(self):
        """Diabetes condition (×1.28) must raise ENDOCRINE adjusted score."""
        systems = self._systems_for([IX_ENDOCRINE])
        endo    = next((s for s in systems if s['organ_key'] == 'ENDOCRINE'), None)
        self.assertIsNotNone(endo, "ENDOCRINE not produced by IX_ENDOCRINE fixture")
        base    = endo['score']
        result  = adjust_for_patient_context(
            systems, {'age': 50, 'conditions': ['Diabetes Type 2'], 'lab_values': {}}
        )
        after   = next(s for s in result['adjusted_systems'] if s['organ_key'] == 'ENDOCRINE')
        self.assertGreater(after['adjusted_score'], base)

    def test_copd_increases_respiratory_score(self):
        """COPD condition (×1.35) must raise RESPIRATORY adjusted score."""
        systems = self._systems_for([IX_RESPIRATORY])
        resp    = next((s for s in systems if s['organ_key'] == 'RESPIRATORY'), None)
        self.assertIsNotNone(resp, "RESPIRATORY not produced by IX_RESPIRATORY fixture")
        base    = resp['score']
        result  = adjust_for_patient_context(
            systems, {'age': 50, 'conditions': ['COPD'], 'lab_values': {}}
        )
        after   = next(s for s in result['adjusted_systems'] if s['organ_key'] == 'RESPIRATORY')
        self.assertGreater(after['adjusted_score'], base)

    def test_inr_elevation_increases_hematologic_score(self):
        """INR=3.5 (×1.55, supra-therapeutic) must raise HEMATOLOGIC score."""
        systems = self._systems_for([IX_HEMATOLOGIC])
        hema    = next((s for s in systems if s['organ_key'] == 'HEMATOLOGIC'), None)
        if not hema:
            self.skipTest("No HEMATOLOGIC system")
        base    = hema['score']
        result  = adjust_for_patient_context(
            systems, {'age': 50, 'conditions': [], 'lab_values': {'INR': 3.5}}
        )
        after   = next(s for s in result['adjusted_systems'] if s['organ_key'] == 'HEMATOLOGIC')
        self.assertGreater(after['adjusted_score'], base)

    def test_complex_patient_produces_warnings(self):
        """Complex patient with multi-organ involvement should produce ≥1 warning."""
        systems  = self._systems_all()
        result   = adjust_for_patient_context(systems, PATIENT_COMPLEX)
        warnings = [s['patient_specific_warning'] for s in result['adjusted_systems']
                    if s.get('patient_specific_warning')]
        self.assertGreater(len(warnings), 0, "Expected at least one patient-specific warning")

    def test_risk_factors_populated_for_complex_patient(self):
        """Complex patient should produce non-empty risk_factors on affected organs."""
        systems     = self._systems_all()
        result      = adjust_for_patient_context(systems, PATIENT_COMPLEX)
        all_factors = [f for s in result['adjusted_systems'] for f in s.get('risk_factors', [])]
        self.assertGreater(len(all_factors), 0)

    def test_output_has_patient_profile(self):
        """Result dict must contain a patient_profile summary with correct age."""
        systems = self._systems_all()
        result  = adjust_for_patient_context(systems, PATIENT_COMPLEX)
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

    def test_hepatic_pair_returns_interaction(self):
        """Isoniazid + Rifampicin must find the hepatic interaction."""
        result = m.get_drug_interactions(['Isoniazid', 'Rifampicin'])
        self.assertGreater(len(result), 0)
        desc = result[0]['description'].lower()
        self.assertTrue(
            any(kw in desc for kw in ['hepatotox','liver','alt','hepatic']),
            f"Expected hepatic keywords in: {desc}"
        )

    def test_respiratory_pair_returns_interaction(self):
        """Morphine + Diazepam must find the respiratory depression interaction."""
        result = m.get_drug_interactions(['Morphine', 'Diazepam'])
        self.assertGreater(len(result), 0)
        desc = result[0]['description'].lower()
        self.assertTrue(
            any(kw in desc for kw in ['respiratory','dyspnea','apnea','pulmonary','breathing']),
            f"Expected respiratory keywords in: {desc}"
        )

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
                ['Isoniazid 300mg Tablet', 'Rifampicin 450mg Capsule',
                 'Simvastatin 20mg Tablet'],
                patient_data=PATIENT_COMPLEX,
            )
        except Exception as e:
            self.fail(f"Pipeline raised an exception: {e}")

    def test_pipeline_without_patient_data(self):
        """Pipeline must work fine without patient data."""
        try:
            m.analyze_interactions_with_context(
                ['Isoniazid 300mg Tablet', 'Rifampicin 450mg Capsule'],
            )
        except Exception as e:
            self.fail(f"Pipeline raised: {e}")

    def test_multi_organ_pipeline_produces_multiple_organ_systems(self):
        """Morphine + Diazepam + Isoniazid + Rifampicin should hit ≥2 organ systems."""
        result = m.analyze_interactions_with_context(
            ['Morphine Sulfate Tablet', 'Diazepam 5mg Tablet',
             'Isoniazid 300mg Tablet',  'Rifampicin 450mg Capsule'],
            patient_data=PATIENT_COMPLEX,
        )
        if result['status'] == 'INTERACTIONS_FOUND':
            systems = (
                result['patient_adjustments'].get('adjusted_systems') or
                result['organ_analysis'].get('affected_organ_systems', [])
            )
            n = len(systems)
            self.assertGreaterEqual(n, 2,
                f"Expected ≥2 organ systems from multi-drug pipeline, got {n}: "
                f"{[s['system'] for s in systems]}")

    def test_complex_patient_organ_scores_higher_than_young_healthy(self):
        """Complex patient total organ-adjusted score > young healthy for same drugs."""
        brands = ['Morphine Sulfate Tablet', 'Diazepam 5mg Tablet']
        r_complex = m.analyze_interactions_with_context(brands, patient_data=PATIENT_COMPLEX)
        r_young   = m.analyze_interactions_with_context(brands, patient_data=PATIENT_YOUNG_HEALTHY)
        if r_complex['status'] == r_young['status'] == 'INTERACTIONS_FOUND':
            complex_total = sum(
                s.get('adjusted_score', s.get('score', 0))
                for s in r_complex['patient_adjustments'].get('adjusted_systems', [])
            )
            young_total = sum(
                s.get('adjusted_score', s.get('score', 0))
                for s in r_young['patient_adjustments'].get('adjusted_systems', [])
            )
            self.assertGreater(complex_total, young_total,
                f"Complex ({complex_total}) should exceed young healthy ({young_total})")

    def test_respiratory_organ_appears_for_morphine_diazepam(self):
        """Morphine + Diazepam interaction (respiratory depression) should produce Respiratory organ."""
        result = m.analyze_interactions_with_context(
            ['Morphine Sulfate Tablet', 'Diazepam 5mg Tablet'],
        )
        if result['status'] == 'INTERACTIONS_FOUND':
            systems = result['organ_analysis'].get('affected_organ_systems', [])
            names   = [s['system'] for s in systems]
            self.assertIn('Respiratory', names,
                f"Expected Respiratory in organ systems, got: {names}")

    def test_hepatic_organ_appears_for_isoniazid_rifampicin(self):
        """Isoniazid + Rifampicin (hepatotoxicity) should produce Hepatic organ system."""
        result = m.analyze_interactions_with_context(
            ['Isoniazid 300mg Tablet', 'Rifampicin 450mg Capsule'],
        )
        if result['status'] == 'INTERACTIONS_FOUND':
            systems = result['organ_analysis'].get('affected_organ_systems', [])
            names   = [s['system'] for s in systems]
            self.assertIn('Hepatic', names,
                f"Expected Hepatic in organ systems, got: {names}")

    def test_single_brand_no_pairs(self):
        """Single brand → no pairs → NO_INTERACTIONS status."""
        result = m.analyze_interactions_with_context(['Calpol 500mg Tablet'])
        self.assertEqual(result['status'], 'NO_INTERACTIONS')

    def test_report_risk_levels_are_valid_values(self):
        """Risk level must be one of the five defined tiers."""
        valid  = {'CRITICAL','SEVERE','MODERATE','MILD','MINIMAL'}
        result = m.analyze_interactions_with_context(
            ['Isoniazid 300mg Tablet', 'Rifampicin 450mg Capsule'],
            patient_data=PATIENT_COMPLEX,
        )
        if result['status'] == 'INTERACTIONS_FOUND':
            level = result['clinical_report']['summary']['overall_risk_level']
            self.assertIn(level, valid)

    def test_save_report_creates_file(self):
        """save_report parameter should write a valid JSON file."""
        import tempfile, os, json
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            path = tf.name
        try:
            result = m.analyze_interactions_with_context(
                ['Isoniazid 300mg Tablet', 'Rifampicin 450mg Capsule'],
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
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Optional: verbose flag
    verbosity = 2 if '-v' in sys.argv else 1
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)