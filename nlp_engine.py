
"""
PolyGuard NLP Engine
=====================
Real NLP pipeline replacing all hand-crafted regex with:

  1. NLPPreprocessor       — tokenisation, lemmatisation, stopword removal,
                             negation scope marking
  2. SeverityScorer        — TF-IDF + Logistic Regression trained on labelled
                             interaction sentences → 0-100 severity score
  3. OrganClassifier       — TF-IDF + multi-label One-vs-Rest classifier
                             trained on organ-annotated sentences
  4. SemanticSimilarity    — cosine similarity between TF-IDF vectors for
                             nearest-neighbour lookup of unknown descriptions
  5. NLPInteractionAnalyser— top-level orchestrator (replaces Steps 3 & 4)

No external models are downloaded. Everything is trained from the embedded
labelled corpus (TRAINING_CORPUS) which covers all organ systems and all
severity tiers using realistic DrugBank-style sentences.
"""

from __future__ import annotations

import re
import math
import string
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline








TRAINING_CORPUS: List[Tuple[str, int, List[str]]] = [

    # ── CRITICAL (score 50) ──────────────────────────────────────────────────
    ("Combination is fatal and results in death due to cardiac arrest.", 50, ["CARDIOVASCULAR"]),
    ("Life-threatening respiratory failure requiring immediate ICU intervention.", 50, ["RESPIRATORY"]),
    ("Fatal serotonin syndrome reported with this combination in multiple case reports.", 50, ["CENTRAL_NERVOUS_SYSTEM"]),
    ("Lethal QT prolongation leading to torsades de pointes and cardiac arrest.", 50, ["CARDIOVASCULAR"]),
    ("Anaphylaxis and anaphylactic shock can result in death.", 50, ["IMMUNE_SYSTEM"]),
    ("Risk of fatal hepatic necrosis and acute liver failure.", 50, ["HEPATIC"]),
    ("Life-threatening bone marrow suppression and aplastic anemia.", 50, ["HEMATOLOGIC"]),
    ("This combination results in fatal CNS depression and respiratory arrest.", 50, ["CENTRAL_NERVOUS_SYSTEM", "RESPIRATORY"]),
    ("Fatal rhabdomyolysis with myoglobinuria and acute renal failure.", 50, ["MUSCULOSKELETAL", "RENAL"]),
    ("Lethal hypoglycaemia leading to coma and death in diabetic patients.", 50, ["ENDOCRINE"]),

    # ── SEVERE (score 35-45) ─────────────────────────────────────────────────

    # CARDIOVASCULAR
    ("The risk or severity of bleeding and haemorrhage is markedly increased.", 38, ["CARDIOVASCULAR", "HEMATOLOGIC"]),
    ("Severe cardiac arrhythmia including ventricular fibrillation reported.", 35, ["CARDIOVASCULAR"]),
    ("QT interval prolongation and torsades de pointes risk significantly elevated.", 35, ["CARDIOVASCULAR"]),
    ("Major haemorrhagic stroke risk increased when combining anticoagulants.", 38, ["CARDIOVASCULAR", "HEMATOLOGIC"]),
    ("Severe bradycardia and complete heart block requiring pacemaker.", 35, ["CARDIOVASCULAR"]),
    ("Anticoagulant activities are significantly increased causing dangerous bleeding.", 38, ["HEMATOLOGIC", "CARDIOVASCULAR"]),
    ("Risk of myocardial infarction and coronary thrombosis substantially increased.", 35, ["CARDIOVASCULAR"]),

    # HEPATIC
    ("Severe hepatotoxicity and drug-induced liver injury (DILI) reported.", 35, ["HEPATIC"]),
    ("Acute liver failure with markedly elevated ALT and AST enzymes.", 38, ["HEPATIC"]),
    ("Hepatocellular damage and cholestatic jaundice with bilirubin elevation.", 35, ["HEPATIC"]),
    ("Cytochrome P450 CYP3A4 inhibition causes severe drug accumulation and liver toxicity.", 35, ["HEPATIC"]),
    ("Fulminant hepatic failure requiring liver transplant evaluation.", 45, ["HEPATIC"]),
    ("Rifampicin-induced hepatic enzyme induction dramatically alters metabolism.", 35, ["HEPATIC"]),

    # RENAL
    ("Nephrotoxicity and acute kidney injury (AKI) significantly increased.", 35, ["RENAL"]),
    ("Severe renal failure with creatinine elevation and eGFR reduction.", 38, ["RENAL"]),
    ("Aminoglycoside nephrotoxicity potentiated, requiring creatinine monitoring.", 35, ["RENAL"]),
    ("Risk of tubulointerstitial nephritis and renal tubular acidosis.", 35, ["RENAL"]),
    ("Acute tubular necrosis and oliguric renal failure in high-risk patients.", 38, ["RENAL"]),

    # HEMATOLOGIC
    ("Major bleeding with platelet dysfunction and coagulation impairment.", 38, ["HEMATOLOGIC"]),
    ("Severe thrombocytopenia and risk of spontaneous haemorrhage.", 35, ["HEMATOLOGIC"]),
    ("INR substantially elevated increasing major bleeding risk.", 38, ["HEMATOLOGIC"]),
    ("Antiplatelet effects combined with anticoagulation cause serious haemorrhage.", 38, ["HEMATOLOGIC", "CARDIOVASCULAR"]),
    ("Disseminated intravascular coagulation (DIC) risk substantially increased.", 45, ["HEMATOLOGIC"]),

    # CENTRAL NERVOUS SYSTEM
    ("Severe serotonin syndrome with hyperthermia, clonus, and agitation.", 38, ["CENTRAL_NERVOUS_SYSTEM"]),
    ("CNS depression causing profound sedation, confusion, and loss of consciousness.", 35, ["CENTRAL_NERVOUS_SYSTEM"]),
    ("Seizure threshold significantly lowered, risk of generalised tonic-clonic seizures.", 35, ["CENTRAL_NERVOUS_SYSTEM"]),
    ("Neuroleptic malignant syndrome with hyperthermia and muscle rigidity.", 38, ["CENTRAL_NERVOUS_SYSTEM"]),
    ("Severe encephalopathy with cognitive impairment and disorientation.", 35, ["CENTRAL_NERVOUS_SYSTEM"]),

    # RESPIRATORY
    ("Severe respiratory depression and apnea requiring ventilatory support.", 38, ["RESPIRATORY"]),
    ("Pulmonary oedema and severe dyspnea with hypoxia.", 35, ["RESPIRATORY"]),
    ("Bronchospasm and severe asthma exacerbation with this combination.", 35, ["RESPIRATORY"]),
    ("Respiratory muscle paralysis causing ventilatory failure.", 45, ["RESPIRATORY"]),

    # ENDOCRINE
    ("Severe hypoglycaemia including loss of consciousness and seizures.", 35, ["ENDOCRINE"]),
    ("Profound hyperglycaemia and diabetic ketoacidosis (DKA) risk increased.", 35, ["ENDOCRINE"]),
    ("Thyroid storm precipitated by drug interaction in hyperthyroid patients.", 38, ["ENDOCRINE"]),
    ("Adrenal insufficiency crisis triggered by abrupt steroid withdrawal interaction.", 38, ["ENDOCRINE"]),

    # MUSCULOSKELETAL
    ("Rhabdomyolysis with severe myopathy and markedly elevated creatine kinase (CK).", 38, ["MUSCULOSKELETAL"]),
    ("Statin-fibrate interaction causes severe myositis and muscle necrosis.", 35, ["MUSCULOSKELETAL"]),
    ("Severe muscle breakdown with myoglobin release and brown urine.", 38, ["MUSCULOSKELETAL"]),
    ("Acute compartment syndrome following severe drug-induced rhabdomyolysis.", 35, ["MUSCULOSKELETAL"]),

    # IMMUNE SYSTEM
    ("Stevens-Johnson syndrome and toxic epidermal necrolysis risk increased.", 38, ["IMMUNE_SYSTEM"]),
    ("Severe anaphylaxis with urticaria and bronchospasm.", 38, ["IMMUNE_SYSTEM", "RESPIRATORY"]),
    ("Severe immunosuppression with opportunistic infections.", 35, ["IMMUNE_SYSTEM"]),
    ("Drug reaction with eosinophilia and systemic symptoms (DRESS) reported.", 35, ["IMMUNE_SYSTEM"]),

    # GASTROINTESTINAL
    ("Severe gastrointestinal bleeding and gastric ulcer perforation risk.", 35, ["GASTROINTESTINAL"]),
    ("Serious GI haemorrhage when NSAID combined with anticoagulant.", 38, ["GASTROINTESTINAL", "HEMATOLOGIC"]),
    ("Acute pancreatitis risk substantially increased with combination therapy.", 35, ["GASTROINTESTINAL"]),
    ("Severe colitis and intestinal perforation risk with this drug pair.", 38, ["GASTROINTESTINAL"]),

    # ── MODERATE (score 20-29) ── one representative per organ ───────────────
    ("Increased risk of cardiac adverse events. Close ECG monitoring recommended.", 20, ["CARDIOVASCULAR"]),
    ("Elevated ALT enzyme levels indicating moderate hepatic stress.", 20, ["HEPATIC"]),
    ("Moderate increase in serum creatinine. Renal function monitoring advised.", 20, ["RENAL"]),
    ("Elevated INR requiring dose adjustment. Regular coagulation checks needed.", 20, ["HEMATOLOGIC"]),
    ("Moderate CNS adverse effects including drowsiness and cognitive impairment.", 20, ["CENTRAL_NERVOUS_SYSTEM"]),
    ("Moderate respiratory adverse effects including mild dyspnea and cough.", 20, ["RESPIRATORY"]),
    ("Blood glucose levels moderately affected. Diabetic patients should monitor.", 20, ["ENDOCRINE"]),
    ("Mild to moderate myalgia and muscle aches with elevated CK.", 20, ["MUSCULOSKELETAL"]),
    ("Moderate GI adverse effects including nausea, vomiting, and stomach discomfort.", 20, ["GASTROINTESTINAL"]),
    ("Moderate hypersensitivity reaction with skin rash and mild urticaria.", 20, ["IMMUNE_SYSTEM"]),
    ("Hypertension may be worsened. Blood pressure monitoring recommended.", 20, ["CARDIOVASCULAR"]),
    ("Reduced antibiotic efficacy when taken together. Treatment failure risk.", 18, ["IMMUNE_SYSTEM"]),
    ("Drug levels may be increased or decreased. Therapeutic drug monitoring required.", 20, ["HEPATIC"]),
    ("Moderate QT prolongation observed. ECG monitoring recommended.", 20, ["CARDIOVASCULAR"]),
    ("Decreased efficacy of antihypertensive therapy observed.", 18, ["CARDIOVASCULAR"]),
    ("Plasma concentration elevated due to CYP450 inhibition. Monitor closely.", 20, ["HEPATIC"]),
    ("Concurrent use increases renal clearance, dose adjustment may be needed.", 20, ["RENAL"]),
    ("Platelet aggregation moderately impaired. Caution with surgical procedures.", 20, ["HEMATOLOGIC"]),
    ("Moderate hypoglycaemic effect enhanced with insulin sensitisers.", 20, ["ENDOCRINE"]),
    ("Moderate bronchodilator interaction. Pulse-oximetry monitoring advised.", 20, ["RESPIRATORY"]),

    # ── MILD / MINIMAL (score 5-19) ── distributed across all organs ─────────
    ("May slightly increase the risk of adverse cardiac effects. Routine monitoring.", 10, ["CARDIOVASCULAR"]),
    ("Minor hepatic pharmacokinetic interaction. Clinical significance is low.", 5, ["HEPATIC"]),
    ("Caution advised when combining. Renal function routine monitoring sufficient.", 10, ["RENAL"]),
    ("Small increase in drug exposure. Generally well tolerated. Hematologic monitoring.", 8, ["HEMATOLOGIC"]),
    ("Mild nausea and GI discomfort may occur when taken together. Administer with food.", 10, ["GASTROINTESTINAL"]),
    ("Slight reduction in drug absorption possible. Separate by 2 hours.", 8, ["GASTROINTESTINAL"]),
    ("Minor CNS effects such as mild headache or dizziness possible.", 10, ["CENTRAL_NERVOUS_SYSTEM"]),
    ("Low respiratory interaction risk. Patient counselling regarding symptom monitoring.", 8, ["RESPIRATORY"]),
    ("Mild drowsiness may be enhanced by CNS combination. Avoid driving if affected.", 10, ["CENTRAL_NERVOUS_SYSTEM"]),
    ("Generally safe combination. Routine endocrine follow-up recommended.", 5, ["ENDOCRINE"]),
    ("No significant musculoskeletal interaction expected at standard doses.", 5, ["MUSCULOSKELETAL"]),
    ("Minor immune interaction with limited clinical impact.", 5, ["IMMUNE_SYSTEM"]),
    ("Interaction unlikely to cause clinically significant immune effects.", 5, ["IMMUNE_SYSTEM"]),
    ("Monitor for minor changes in hepatic drug levels. No dose adjustment needed.", 8, ["HEPATIC"]),
    ("Mild increase in bronchodilator effects. No clinical pulmonary action needed.", 8, ["RESPIRATORY"]),
    ("Minor effect on blood glucose unlikely to require intervention.", 8, ["ENDOCRINE"]),
    ("Mild muscle aches reported rarely. No dose change required.", 8, ["MUSCULOSKELETAL"]),
    ("Minor GI discomfort when taken on an empty stomach.", 5, ["GASTROINTESTINAL"]),

    # ── NEGATED / NO-INTERACTION ── distributed across all organs ────────────
    ("No significant cardiovascular drug interaction has been documented.", 5, ["CARDIOVASCULAR"]),
    ("No increased cardiac risk observed in clinical trials.", 5, ["CARDIOVASCULAR"]),
    ("No clinically relevant hepatic interaction expected under normal circumstances.", 5, ["HEPATIC"]),
    ("No evidence of hepatotoxicity was found in controlled studies.", 5, ["HEPATIC"]),
    ("No renal toxicity reported. eGFR unaffected in pharmacokinetic studies.", 5, ["RENAL"]),
    ("No clinically significant renal interaction expected.", 5, ["RENAL"]),
    ("No significant haematologic interaction. INR unaffected in pharmacokinetic studies.", 5, ["HEMATOLOGIC"]),
    ("No CNS interaction documented. No dose adjustment required.", 5, ["CENTRAL_NERVOUS_SYSTEM"]),
    ("No respiratory adverse effects expected from this combination.", 5, ["RESPIRATORY"]),
    ("No significant endocrine interaction. Blood glucose unaffected.", 5, ["ENDOCRINE"]),
    ("No musculoskeletal interaction. CK levels unaffected in studies.", 5, ["MUSCULOSKELETAL"]),
    ("No immune hypersensitivity interaction documented.", 5, ["IMMUNE_SYSTEM"]),
    ("No GI interaction. Bioavailability unaffected when co-administered.", 5, ["GASTROINTESTINAL"]),
]


ALL_ORGANS = [
    'CARDIOVASCULAR', 'HEPATIC', 'RENAL', 'HEMATOLOGIC',
    'GASTROINTESTINAL', 'CENTRAL_NERVOUS_SYSTEM', 'RESPIRATORY',
    'ENDOCRINE', 'MUSCULOSKELETAL', 'IMMUNE_SYSTEM',
]







_MEDICAL_STOPWORDS = {
    'the','a','an','of','in','is','are','was','be','to','and','or','this',
    'that','with','for','on','at','by','from','as','may','can','could',
    'should','when','if','which','these','those','it','its','than','more',
    'also','both','such','any','all','some','other','been','has','have',
    'had','not','but','so','do','does','did','would','will','use','used',
    'between','during','after','while','however','therefore','thus','hence',
}


_NEGATION_CUES = {
    'no','not','never','neither','without','unlikely','rarely','minimal',
    'negligible','absence','absent','lack','lacking','none','non',
}


_LEMMA_RULES = [
    (r'ations?$', 'ation'), (r'ities$', 'ity'), (r'nesses$', 'ness'),
    (r'ings?$', 'ing'),     (r'ments?$', 'ment'), (r'ical$', 'ic'),
    (r'ically$', 'ic'),     (r'toxic$', 'toxic'), (r'ities$', 'ity'),
    (r'haemorrhag\w*', 'hemorrhage'), (r'hemor+hag\w*', 'hemorrhage'),
    (r'hepatoto\w+', 'hepatotoxic'),  (r'nephrotox\w+', 'nephrotoxic'),
    (r'hypoglycae?m\w+', 'hypoglycemia'),
    (r'hypoglycae?mic', 'hypoglycemia'),
    (r'arrhythm\w+', 'arrhythmia'),
    (r'rhabdomyolys\w+', 'rhabdomyolysis'),
    (r'thrombocytopen\w+', 'thrombocytopenia'),
    # Narrowed: previously 'cardio\w+' → 'cardiovascular' collapsed ALL cardiac
    # vocabulary (cardiac, cardiomyopathy, cardiotoxic, etc.) into one token,
    # artificially inflating CARDIOVASCULAR classifier confidence for any text
    # mentioning heart-related terms even when labelled as another organ.
    (r'cardiovascular', 'cardiovascular'),
    (r'cardiomyopath\w+', 'cardiomyopathy'),
    (r'cardiac', 'cardiac'),
]



def preprocess(text: str) -> Tuple[str, bool, List[str]]:
    """
    Tokenise, lemmatise, remove stopwords, detect negation scope.

    Returns:
        processed_text : cleaned, lemmatised string for vectoriser
        is_negated     : True if negation cue found in sentence
        tokens         : raw token list (for debugging)
    """
    
    text_clean = text.lower()
    text_clean = re.sub(r'[^\w\s\-]', ' ', text_clean)
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()

    tokens = text_clean.split()

    
    is_negated = any(t in _NEGATION_CUES for t in tokens)

    
    processed = []
    for tok in tokens:
        if tok in _MEDICAL_STOPWORDS:
            continue
        lemma = tok
        for pattern, replacement in _LEMMA_RULES:
            if re.fullmatch(pattern, tok):
                lemma = replacement
                break
        processed.append(lemma)

    return ' '.join(processed), is_negated, tokens







class SeverityScorer:
    """
    Trained text classifier: interaction description → severity score (int 0-50).

    Model: TF-IDF (char n-grams + word n-grams) → Logistic Regression.
    Training data: TRAINING_CORPUS (embedded, no file I/O needed).
    """

    
    _BUCKETS = {
        'MINIMAL':  (0,  9),
        'MILD':     (10, 19),
        'MODERATE': (20, 29),
        'SEVERE':   (35, 44),
        'CRITICAL': (45, 50),
    }

    def __init__(self):
        self._pipeline: Optional[Pipeline] = None
        self._classes: Optional[np.ndarray] = None
        self._is_trained = False

    def train(self, corpus: List[Tuple[str, int, List[str]]]) -> 'SeverityScorer':
        """Train from list of (text, score, organs) tuples."""
        texts  = [preprocess(t)[0] for t, _, _ in corpus]
        
        labels = [self._score_to_bucket(s) for _, s, _ in corpus]

        self._pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 3),
                min_df=1,
                sublinear_tf=True,
                max_features=3000,
            )),
            ('clf', LogisticRegression(
                C=2.0,
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
            )),
        ])

        self._pipeline.fit(texts, labels)
        self._classes = self._pipeline.classes_
        self._is_trained = True
        return self

    def predict_score(self, text: str, is_negated: bool = False) -> Tuple[int, str, np.ndarray]:
        """
        Predict severity score for a description.

        Returns:
            score      : int 0-50
            bucket     : 'MINIMAL' | 'MILD' | 'MODERATE' | 'SEVERE' | 'CRITICAL'
            proba_vec  : probability vector over buckets (for calibration)
        """
        if not self._is_trained:
            raise RuntimeError("SeverityScorer not trained. Call .train() first.")

        processed, _, _ = preprocess(text)
        if not self._pipeline:
            raise RuntimeError("Pipeline not initialized.")
        
        proba = self._pipeline.predict_proba([processed])[0]
        bucket = self._pipeline.predict([processed])[0]

        if bucket not in self._BUCKETS:
            raise RuntimeError(f"Unexpected bucket: {bucket}")
        
        
        lo, hi = self._BUCKETS[bucket]
        raw_score = int((lo + hi) / 2)

        
        bucket_idx = list(self._classes).index(bucket)
        confidence = float(proba[bucket_idx])
        score = int(lo + (hi - lo) * confidence)
        score = max(lo, min(hi, score))

        
        if is_negated:
            score = max(5, int(score * 0.30))
            bucket = self._score_to_bucket(score)

        return score, bucket, proba

    @staticmethod
    def _score_to_bucket(score: int) -> str:
        if score >= 45: return 'CRITICAL'
        if score >= 35: return 'SEVERE'
        if score >= 20: return 'MODERATE'
        if score >= 10: return 'MILD'
        return 'MINIMAL'







class OrganClassifier:
    """
    Multi-label trained classifier: description → set of affected organ systems.

    Model: TF-IDF (word + char n-grams) → One-vs-Rest Logistic Regression.
    Each organ has its own binary classifier. Multiple organs can be predicted.
    """

    def __init__(self):
        self._vectoriser: Optional[TfidfVectorizer] = None
        self._clf: Optional[OneVsRestClassifier]    = None
        self._mlb: Optional[MultiLabelBinarizer]    = None
        self._is_trained = False

    def train(self, corpus: List[Tuple[str, int, List[str]]]) -> 'OrganClassifier':
        texts  = [preprocess(t)[0] for t, _, _ in corpus]
        labels = [organs for _, _, organs in corpus]

        self._mlb = MultiLabelBinarizer(classes=ALL_ORGANS)
        Y = self._mlb.fit_transform(labels)

        self._vectoriser = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
            max_features=4000,
        )
        X = self._vectoriser.fit_transform(texts)

        self._clf = OneVsRestClassifier(
            LogisticRegression(
                C=1.5,
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
            )
        )
        self._clf.fit(X, Y)
        self._is_trained = True
        return self

    def predict_organs(
        self,
        text: str,
        threshold: float = 0.20,
    ) -> List[Tuple[str, float]]:
        """
        Predict organ systems and confidence scores.

        Args:
            text      : interaction description
            threshold : minimum probability to include an organ (default 0.20)

        Returns:
            List of (organ_name, probability) sorted by probability desc.
            Falls back to top-1 organ if nothing exceeds threshold.
        """
        if not self._is_trained:
            raise RuntimeError("OrganClassifier not trained.")

        processed, _, _ = preprocess(text)
        X = self._vectoriser.transform([processed])

        
        proba_matrix = np.array([
            est.predict_proba(X)[0][1]
            for est in self._clf.estimators_
        ])

        results = []
        for organ, prob in zip(self._mlb.classes_, proba_matrix):
            if prob >= threshold:
                results.append((organ, float(prob)))

        
        if not results:
            best_idx = int(np.argmax(proba_matrix))
            results  = [(self._mlb.classes_[best_idx], float(proba_matrix[best_idx]))]

        return sorted(results, key=lambda x: x[1], reverse=True)

    def organ_probability_vector(self, text: str) -> Dict[str, float]:
        """Return full probability dict for all organs (useful for debugging)."""
        processed, _, _ = preprocess(text)
        X = self._vectoriser.transform([processed])
        proba_matrix = np.array([
            est.predict_proba(X)[0][1]
            for est in self._clf.estimators_
        ])
        return dict(zip(self._mlb.classes_, proba_matrix.tolist()))






class SemanticSimilarity:
    """
    TF-IDF cosine similarity for nearest-neighbour lookup.

    Use case: given an unseen interaction description, find the most similar
    labelled description in the training corpus to explain the prediction.
    """

    def __init__(self):
        self._vectoriser: Optional[TfidfVectorizer] = None
        self._matrix     = None
        self._corpus_texts: List[str] = []
        self._corpus_scores: List[int] = []
        self._corpus_organs: List[List[str]] = []

    def fit(self, corpus: List[Tuple[str, int, List[str]]]) -> 'SemanticSimilarity':
        self._corpus_texts  = [preprocess(t)[0] for t, _, _ in corpus]
        self._corpus_scores = [s for _, s, _ in corpus]
        self._corpus_organs = [o for _, _, o in corpus]

        self._vectoriser = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self._matrix = self._vectoriser.fit_transform(self._corpus_texts)
        return self

    def find_nearest(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict]:
        """Return top-k most similar training sentences with their labels."""
        processed, _, _ = preprocess(query)
        q_vec = self._vectoriser.transform([processed])
        sims  = cosine_similarity(q_vec, self._matrix)[0]
        top_k_idx = np.argsort(sims)[::-1][:top_k]

        return [
            {
                'text':       self._corpus_texts[i],
                'similarity': float(sims[i]),
                'score':      self._corpus_scores[i],
                'organs':     self._corpus_organs[i],
            }
            for i in top_k_idx
        ]






class _ModelRegistry:
    """Lazy-initialised singleton holding all trained NLP models."""

    _instance: Optional['_ModelRegistry'] = None

    def __init__(self):
        self._severity_scorer   : Optional[SeverityScorer]   = None
        self._organ_classifier  : Optional[OrganClassifier]   = None
        self._semantic_sim      : Optional[SemanticSimilarity] = None
        self._trained           = False

    @classmethod
    def get(cls) -> '_ModelRegistry':
        if cls._instance is None:
            cls._instance = _ModelRegistry()
        if not cls._instance._trained:
            cls._instance._train()
        return cls._instance

    def _train(self):
        print("   [NLP] Training models on embedded corpus…", end=' ', flush=True)
        self._severity_scorer  = SeverityScorer().train(TRAINING_CORPUS)
        self._organ_classifier = OrganClassifier().train(TRAINING_CORPUS)
        self._semantic_sim     = SemanticSimilarity().fit(TRAINING_CORPUS)
        self._trained = True
        print(f"done ({len(TRAINING_CORPUS)} examples, {len(ALL_ORGANS)} organ classes)")

    @property
    def severity(self) -> SeverityScorer:
        return self._severity_scorer

    @property
    def organs(self) -> OrganClassifier:
        return self._organ_classifier

    @property
    def similarity(self) -> SemanticSimilarity:
        return self._semantic_sim


def get_models() -> _ModelRegistry:
    """Return the singleton trained model registry."""
    return _ModelRegistry.get()






def analyse_interaction_text(description: str) -> Dict:
    """
    Full NLP analysis of a single interaction description.

    Returns a rich dict consumed by the rest of the PolyGuard pipeline:
      score          : int 0-50
      severity       : MINIMAL | MILD | MODERATE | SEVERE | CRITICAL
      is_negated     : bool — negation detected in text
      organs         : list of (organ_name, confidence) tuples
      nearest_refs   : 3 most similar labelled training examples
      processed_text : cleaned/lemmatised text fed to models
      organ_proba_vec: full probability dict for all 10 organs
    """
    models = get_models()

    processed, is_negated, tokens = preprocess(description)
    score, severity, proba_vec    = models.severity.predict_score(description, is_negated)
    organs                         = models.organs.predict_organs(description)
    organ_proba_vec                = models.organs.organ_probability_vector(description)
    nearest                        = models.similarity.find_nearest(description, top_k=3)

    return {
        'score':           score,
        'severity':        severity,
        'is_negated':      is_negated,
        'organs':          organs,          
        'nearest_refs':    nearest,
        'processed_text':  processed,
        'tokens':          tokens,
        'organ_proba_vec': organ_proba_vec,
        'severity_proba':  dict(zip(models.severity._classes, proba_vec.tolist())),
    }


def analyse_interaction_batch(interactions: List[Dict]) -> List[Dict]:
    """
    Analyse a list of interaction dicts (each must have 'description' key).
    Returns the input list enriched with NLP analysis fields.
    """
    enriched = []
    for ix in interactions:
        desc   = ix.get('description', '')
        result = analyse_interaction_text(desc)
        enriched.append({**ix, 'nlp': result})
    return enriched






if __name__ == '__main__':
    print("=" * 70)
    print("  PolyGuard NLP Engine — Self-Test")
    print("=" * 70)

    test_cases = [
        ("Fatal serotonin syndrome reported. Life-threatening CNS depression.", ['CENTRAL_NERVOUS_SYSTEM'], 'CRITICAL'),
        ("Severe hepatotoxicity with acute liver failure and elevated ALT.", ['HEPATIC'],                   'SEVERE'),
        ("Nephrotoxicity risk significantly increased. Monitor creatinine.", ['RENAL'],                     'SEVERE'),
        ("Rhabdomyolysis and severe myopathy with CK elevation.",             ['MUSCULOSKELETAL'],           'SEVERE'),
        ("Severe respiratory depression. Apnea and pulmonary failure risk.",  ['RESPIRATORY'],              'SEVERE'),
        ("Major bleeding and haemorrhage risk from anticoagulant combination.",['HEMATOLOGIC', 'CARDIOVASCULAR'], 'SEVERE'),
        ("Severe hypoglycaemia with loss of consciousness.",                   ['ENDOCRINE'],                'SEVERE'),
        ("Stevens-Johnson syndrome and severe immunosuppression.",             ['IMMUNE_SYSTEM'],            'SEVERE'),
        ("GI bleeding and gastric ulcer perforation risk.",                    ['GASTROINTESTINAL'],         'SEVERE'),
        ("Mild nausea may occur. Monitor patients routinely.",                 ['GASTROINTESTINAL'],         'MILD'),
        ("No significant interaction documented. No action required.",         ['CARDIOVASCULAR'],           'MINIMAL'),
    ]

    models = get_models()
    print(f"\n{'TEXT':<55} {'PRED_SEV':<10} {'EXP_SEV':<10} {'TOP_ORGAN':<28} {'NEG'}")
    print("-" * 115)

    correct_sev   = 0
    correct_organ = 0

    for text, exp_organs, exp_sev in test_cases:
        result     = analyse_interaction_text(text)
        pred_sev   = result['severity']
        pred_organ = result['organs'][0][0] if result['organs'] else 'NONE'
        neg        = '✓' if result['is_negated'] else ''

        sev_ok   = pred_sev   == exp_sev
        organ_ok = any(pred_organ == eo for eo in exp_organs)

        correct_sev   += int(sev_ok)
        correct_organ += int(organ_ok)

        sev_mark   = '✓' if sev_ok   else '✗'
        organ_mark = '✓' if organ_ok else '✗'

        print(f"  {text[:52]:<55} {pred_sev+sev_mark:<10} {exp_sev:<10} "
              f"{pred_organ+organ_mark:<28} {neg}")

    n = len(test_cases)
    print(f"\n  Severity accuracy : {correct_sev}/{n} ({100*correct_sev//n}%)")
    print(f"  Organ accuracy    : {correct_organ}/{n} ({100*correct_organ//n}%)")

    
    print("\n  Negation test:")
    pos = analyse_interaction_text("Severe bleeding and haemorrhage risk increased.")
    neg = analyse_interaction_text("No significant bleeding risk. Unlikely to cause haemorrhage.")
    print(f"    Positive: score={pos['score']}  negated={pos['is_negated']}")
    print(f"    Negated : score={neg['score']}  negated={neg['is_negated']}")
    assert neg['score'] < pos['score'], "Negation must reduce score"
    print("    ✓ Negation correctly reduces score")

    print("\n  Nearest-neighbour lookup:")
    refs = models.similarity.find_nearest("Acute liver toxicity and enzyme elevation.", top_k=2)
    for r in refs:
        print(f"    sim={r['similarity']:.3f}  score={r['score']}  organs={r['organs']}")
        print(f"    → '{r['text'][:70]}'")

    print(f"\n{'='*70}")
    print("  Self-test complete.")
    print(f"{'='*70}")