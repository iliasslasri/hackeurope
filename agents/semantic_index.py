"""
semantic_index.py
-----------------
Semantic similarity for medical symptom matching.

BACKENDS (auto-selected at startup)
=====================================
1. HuggingFace Inference API  -- NeuML/pubmedbert-base-embeddings
   PubMedBERT fine-tuned as a sentence transformer on medical text.
   Requires: HF_TOKEN env var + network access.

2. Local sentence-transformers -- same model, loaded locally.
   Requires: pip install sentence-transformers torch

3. TF-IDF + medical synonyms  -- offline, zero extra dependencies.
   Hand-curated synonym dict bridges clinical ↔ lay language:
   dyspnea=shortness of breath, pyrexia=fever, diaphoresis=sweating, etc.

UPGRADE PATH
============
When network becomes available, set HF_TOKEN and restart -- the system
automatically upgrades to PubMedBERT embeddings with no code changes.
"""

from __future__ import annotations
import os, re, json
import numpy as np
import urllib.request, urllib.error
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from agents.scorer import _normalise

HIGH_THRESHOLD = 0.82
LOW_THRESHOLD  = 0.58

HF_MODEL   = "NeuML/pubmedbert-base-embeddings"
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"

# ---------------------------------------------------------------------------
# Medical synonym dictionary  (clinical term -> lay equivalents and vice versa)
# ---------------------------------------------------------------------------
SYNONYMS: Dict[str, List[str]] = {
    "dyspnea":                ["shortness of breath","difficulty breathing","breathlessness","hard to breathe"],
    "shortness of breath":    ["dyspnea","breathlessness","difficulty breathing"],
    "tachypnea":              ["rapid breathing","fast breathing","breathing fast"],
    "hemoptysis":             ["blood in sputum","coughing blood","blood when coughing"],
    "rhinorrhea":             ["runny nose","nasal discharge","nose running"],
    "epistaxis":              ["nosebleed","nasal bleeding"],
    "chest tightness":        ["chest pain","chest discomfort","chest pressure","tight chest"],
    "chest pressure":         ["chest pain","chest tightness","chest discomfort","angina"],
    "angina":                 ["chest pain","chest tightness","chest discomfort"],
    "palpitations":           ["heart racing","rapid heart rate","fast heartbeat","heart pounding"],
    "tachycardia":            ["rapid heart rate","fast pulse","palpitations","heart racing"],
    "bradycardia":            ["slow heart rate","slow pulse"],
    "syncope":                ["fainting","passing out","loss of consciousness","blackout"],
    "edema":                  ["leg swelling","swelling","fluid retention"],
    "emesis":                 ["vomiting","throwing up","being sick"],
    "dysphagia":              ["difficulty swallowing","trouble swallowing","can't swallow"],
    "melena":                 ["blood in stool","dark stool","black stool"],
    "hematuria":              ["blood in urine","bloody urine"],
    "polyuria":               ["frequent urination","urinating a lot","excessive urination"],
    "polydipsia":             ["excessive thirst","increased thirst","very thirsty"],
    "pyrexia":                ["fever","high fever","elevated temperature","temperature"],
    "hyperthermia":           ["fever","high fever","overheating"],
    "myalgia":                ["muscle aches","muscle pain","body aches","sore muscles"],
    "arthralgia":             ["joint pain","joint ache","sore joints"],
    "cephalalgia":            ["headache","head pain"],
    "vertigo":                ["dizziness","lightheadedness","room spinning","spinning sensation"],
    "malaise":                ["fatigue","tiredness","general unwell feeling","feeling unwell"],
    "fatigue":                ["tiredness","exhaustion","weakness","lethargy","no energy","worn out"],
    "rigors":                 ["chills","shivering","shaking with cold"],
    "chills":                 ["rigors","shivering","feeling cold","cold sweats"],
    "anorexia":               ["loss of appetite","not hungry","reduced appetite","no appetite"],
    "pruritus":               ["itching","itchy skin","itch"],
    "diaphoresis":            ["sweating","night sweats","excessive sweating","profuse sweating"],
    "pallor":                 ["pale skin","paleness","looking pale","white skin"],
    "alopecia":               ["hair loss","losing hair","baldness"],
    "photophobia":            ["sensitivity to light","light hurts eyes","eyes hurt in light"],
    "phonophobia":            ["sensitivity to sound","noise sensitivity","sound hurts"],
    "radiating arm pain":     ["pain radiating to arm","arm pain","left arm pain","arm radiating pain"],
    "mucus":                  ["mucus production","phlegm","sputum","secretions"],
    "wheezing":               ["whistling breathing","noisy breathing","wheeze"],
    "cough":                  ["coughing","dry cough","wet cough","persistent cough"],
    "sore throat":            ["throat pain","throat ache","painful throat","scratchy throat"],
    "nasal congestion":       ["congestion","stuffy nose","blocked nose","nose blocked"],
    "weight loss":            ["unexplained weight loss","losing weight","weight dropping"],
    "weight gain":            ["gaining weight","putting on weight"],
    "insomnia":               ["sleep disturbance","can't sleep","trouble sleeping","sleep problems"],
    "depression":             ["persistent sadness","feeling depressed","low mood","sadness"],
    "anxiety":                ["excessive worry","anxious","nervousness","panic"],
    "pelvic pain":            ["lower abdominal pain","lower belly pain","pelvic ache"],
    "dysuria":                ["burning urination","painful urination","burning when urinating"],
    "urgency":                ["urgency to urinate","need to urinate urgently","sudden urge to urinate"],
}

def _expand(text: str) -> str:
    """Append synonym expansions to enrich TF-IDF representation."""
    norm = _normalise(text)
    extras = set()
    for key, syns in SYNONYMS.items():
        if key in norm:
            extras.update(syns)
        for s in syns:
            if s in norm:
                extras.add(key)
                extras.update(syns)
    return (norm + " " + " ".join(extras)).strip() if extras else norm


# ---------------------------------------------------------------------------
# Backend: HuggingFace Inference API
# ---------------------------------------------------------------------------
class _HFBackend:
    def __init__(self, token: str):
        self._token = token
        self._cache: Dict[str, np.ndarray] = {}
        self.name = f"PubMedBERT via HuggingFace API ({HF_MODEL})"

    def embed(self, texts: List[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            hdrs = {"Content-Type": "application/json"}
            if self._token:
                hdrs["Authorization"] = f"Bearer {self._token}"
            body = json.dumps({"inputs": missing, "options": {"wait_for_model": True}}).encode()
            req = urllib.request.Request(HF_API_URL, data=body, headers=hdrs)
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.loads(r.read())
            for t, v in zip(missing, data):
                arr = np.array(v, dtype=np.float32)
                self._cache[t] = arr / (np.linalg.norm(arr) + 1e-9)
        return np.stack([self._cache[t] for t in texts])

    @classmethod
    def probe(cls, token: str) -> Optional["_HFBackend"]:
        try:
            b = cls(token); b.embed(["test"]); return b
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Backend: local sentence-transformers
# ---------------------------------------------------------------------------
class _STBackend:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(HF_MODEL)
        self._cache: Dict[str, np.ndarray] = {}
        self.name = f"PubMedBERT via sentence-transformers (local)"

    def embed(self, texts: List[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            vecs = self._model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            for t, v in zip(missing, vecs):
                self._cache[t] = v.astype(np.float32)
        return np.stack([self._cache[t] for t in texts])

    @classmethod
    def probe(cls) -> Optional["_STBackend"]:
        try:
            return cls()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Backend: TF-IDF + medical synonym expansion (offline fallback)
# ---------------------------------------------------------------------------
class _TFIDFBackend:
    def __init__(self, corpus: List[str]):
        self.name = "TF-IDF + medical synonyms (offline)"
        self._cache: Dict[str, np.ndarray] = {}
        expanded = [_expand(t) for t in corpus]
        self._vect = TfidfVectorizer(analyzer="word", ngram_range=(1, 3),
                                     min_df=1, sublinear_tf=True)
        self._vect.fit(expanded)

    def embed(self, texts: List[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            mat = self._vect.transform([_expand(t) for t in missing]).toarray().astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat /= (norms + 1e-9)
            for t, v in zip(missing, mat):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])


# ---------------------------------------------------------------------------
# Public SemanticIndex
# ---------------------------------------------------------------------------
class SemanticIndex:
    """
    Semantic similarity index over all disease symptom/RF phrases.
    Auto-selects best available backend and pre-embeds all known phrases.
    """

    def __init__(self, phrases: List[str], hf_token: str = "", verbose: bool = True):
        self._phrases = [_normalise(p) for p in phrases]
        token = hf_token or os.environ.get("HF_TOKEN", "")
        self._backend = self._select(phrases, token, verbose)
        if verbose:
            print(f"  [SemanticIndex] {self._backend.name}")
        try:
            self._vecs = self._backend.embed(self._phrases) if self._phrases else np.array([])
        except Exception:
            self._vecs = None
        self._idx = {p: i for i, p in enumerate(self._phrases)}

    def _select(self, phrases, token, verbose):
        b = _HFBackend.probe(token)
        if b:
            if verbose: print("  [SemanticIndex] Connected to HuggingFace Inference API ✓")
            return b
        b = _STBackend.probe()
        if b:
            if verbose: print("  [SemanticIndex] Loaded local sentence-transformers ✓")
            return b
        if verbose:
            print("  [SemanticIndex] No LLM — using TF-IDF + medical synonyms")
            print("                  → For PubMedBERT: export HF_TOKEN=<your_token>")
            print("                    or: pip install sentence-transformers torch")
        return _TFIDFBackend(phrases)

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def _vec_for(self, phrase: str) -> np.ndarray:
        """Get embedding for a known phrase (fast) or embed fresh (slow)."""
        n = _normalise(phrase)
        if n in self._idx and self._vecs is not None:
            return self._vecs[self._idx[n]]
        return self._backend.embed([n])[0]

    def semantic_overlap(
        self,
        reported_tokens: List[str],
        reference_phrases: List[str],
    ) -> Tuple[float, float, int]:
        """
        For each reference phrase find best semantic match among reported tokens.
        Returns (eff_match, eff_miss, observed).
        """
        if not reported_tokens or not reference_phrases:
            return 0.0, float(len(reference_phrases)), 0

        try:
            rep_vecs = self._backend.embed([_normalise(t) for t in reported_tokens])
            ref_vecs = np.stack([self._vec_for(r) for r in reference_phrases])
            sim_mat  = ref_vecs @ rep_vecs.T        # (n_ref, n_rep)
            best     = sim_mat.max(axis=1)          # (n_ref,)
        except Exception:
            return 0.0, float(len(reference_phrases)), 0

        eff_match = 0.0; eff_miss = 0.0; observed = 0
        for s in best:
            if s >= HIGH_THRESHOLD:
                eff_match += 1.0; observed += 1
            elif s >= LOW_THRESHOLD:
                eff_match += 0.5; observed += 1
            else:
                eff_miss  += 1.0
        return eff_match, eff_miss, observed

    def similarity(self, a: str, b: str) -> float:
        va = self._vec_for(a); vb = self._vec_for(b)
        return float(np.dot(va, vb))

    def top_matches(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:
        if self._vecs is None or len(self._vecs) == 0:
            return []
        qv = self._backend.embed([_normalise(query)])[0]
        sims = self._vecs @ qv
        idxs = np.argsort(-sims)[:k]
        out = []
        for i in idxs:
            s = float(sims[i])
            mtype = "exact" if s >= HIGH_THRESHOLD else "partial" if s >= LOW_THRESHOLD else "miss"
            out.append((self._phrases[i], s, mtype))
        return out
    def extract_symptoms_from_question(
        self,
        question_text: str,
        k: int = 6,
        threshold: float = 0.45,
    ) -> List[Tuple[str, float]]:
        """
        Given a question text, return the symptom phrases it is asking about.

        Uses semantic similarity: embed the full question, then find all
        known symptom phrases whose cosine similarity to the question
        exceeds `threshold`.  Returns up to `k` phrases ranked by similarity.

        Examples
        --------
        "Do you have chest pain or shortness of breath?"
          → [("chest pain", 0.91), ("shortness of breath", 0.87)]

        "Tell me about your breathing and any coughing"
          → [("shortness of breath", 0.82), ("cough", 0.79), ("wheezing", 0.61)]

        "How is your energy level lately, any fatigue?"
          → [("fatigue", 0.88), ("malaise", 0.65)]

        "When did this start?"           (temporal / open question)
          → []   (no symptom above threshold)

        Parameters
        ----------
        question_text : raw question string (not yet normalised)
        k             : max number of symptoms to return
        threshold     : min cosine similarity (0.45 works well for TF-IDF,
                        0.55 recommended with PubMedBERT)

        Returns
        -------
        List of (symptom_phrase, similarity_score) sorted by score descending.
        """
        if self._vecs is None or len(self._vecs) == 0:
            return []

        # Embed the question as a whole
        q_norm = _normalise(question_text)
        try:
            q_vec  = self._backend.embed([q_norm])[0]
        except Exception:
            return []

        # Cosine similarity against every known symptom phrase
        sims = self._vecs @ q_vec   # (N,)

        # Collect all above threshold, take top-k
        hits = [
            (self._phrases[i], float(sims[i]))
            for i in range(len(self._phrases))
            if sims[i] >= threshold
        ]
        hits.sort(key=lambda x: -x[1])
        return hits[:k]