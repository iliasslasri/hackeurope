"""
scorer.py
---------
Probabilistic scoring engine. Uses SemanticIndex for symptom matching
instead of substring overlap -- "chest tightness" correctly matches
"chest pain", "difficulty breathing" matches "shortness of breath", etc.
Falls back gracefully to string matching if SemanticIndex is unavailable.
"""

from __future__ import annotations
import re, numpy as np, pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.semantic_index import SemanticIndex

RNG        = np.random.default_rng(42)
MC_SAMPLES = 5_000
W_SYMPTOMS = 0.65
W_RISK     = 0.20
W_PRIOR    = 0.15

# ---------------------------------------------------------------------------
# Text helpers (exported so other modules can import them)
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def _tokenise(text: str) -> set[str]:
    tokens = set()
    for part in re.split(r"[,;.\n]", text):
        part = _normalise(part)
        if part:
            tokens.add(part)
            for w in part.split():
                if len(w) > 3:
                    tokens.add(w)
    return tokens

def _overlap_score(reported: set[str], reference: List[str]) -> Tuple[int,int,int,int]:
    """Legacy string overlap -- used when SemanticIndex is unavailable."""
    exact = partial = misses = observed = 0
    for ref in reference:
        rn = _normalise(ref); rt = set(rn.split())
        if rn in reported:
            exact += 1; observed += 1; continue
        hit = any(r in rn or rn in r or any(w in rt for w in r.split())
                  for r in reported)
        if hit: partial += 1; observed += 1
        else:   misses  += 1
    return exact, partial, misses, observed

# ---------------------------------------------------------------------------
# CandidateDiagnosis
# ---------------------------------------------------------------------------

@dataclass
class CandidateDiagnosis:
    disease: str
    symptom_match_exact:   int
    symptom_match_partial: int
    symptom_total:         int
    symptom_observed:      int
    risk_factor_hits:      int
    risk_factor_total:     int
    risk_factor_observed:  int
    prior:                 float
    probability:           float = 0.0
    probability_raw:       float = 0.0
    prob_ci_lo:            float = 0.0
    prob_ci_hi:            float = 0.0
    prob_std:              float = 0.0
    confidence:            float = 0.0
    evidence_tokens:       int   = 0
    top_matches:           List[Tuple[str,float]] = field(default_factory=list)
    samples:               np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def probability_label(self):
        p = self.probability
        if p >= 0.20: return "Very High"
        if p >= 0.10: return "High"
        if p >= 0.05: return "Moderate"
        if p >= 0.02: return "Low"
        return "Very Low"

    @property
    def confidence_label(self):
        c = self.confidence
        if c >= 0.80: return "High"
        if c >= 0.50: return "Moderate"
        if c >= 0.25: return "Low"
        return "Very Low"

    @property
    def risk_flag(self):
        if self.probability >= 0.08 and self.confidence < 0.50:
            return "⚠️  HIGH-PROB / LOW-CONF"
        return ""

# ---------------------------------------------------------------------------
# DiagnosisScorer
# ---------------------------------------------------------------------------

class DiagnosisScorer:
    def __init__(self, dataset: pd.DataFrame,
                 semantic_index: Optional["SemanticIndex"] = None):
        self.dataset = dataset
        self.sem     = semantic_index

    def score(self, symptoms_text: str, anamnesis_text: str,
              top_k: int = 10) -> List[CandidateDiagnosis]:
        rep_syms = list(_tokenise(symptoms_text))
        rep_risk = list(_tokenise(anamnesis_text))
        candidates = []

        for _, row in self.dataset.iterrows():
            disease  = row["disease"]
            sym_list = [_normalise(s) for s in row["symptoms"]]
            rf_list  = [_normalise(r) for r in row["risk_factors"]]
            prior    = float(row["base_prevalence"])

            if self.sem:
                s_match, s_miss, s_obs = self.sem.semantic_overlap(rep_syms, sym_list)
                r_match, r_miss, r_obs = self.sem.semantic_overlap(rep_risk, rf_list)
                # collect best semantic matches for display
                top_m = []
                for tok in rep_syms[:6]:
                    for ph, sim, mt in self.sem.top_matches(tok, k=2):
                        if mt != "miss" and ph in sym_list:
                            top_m.append((f"{tok} → {ph}", round(sim, 2)))
                top_m.sort(key=lambda x: -x[1])
            else:
                rep_set = set(rep_syms)
                se, sp, _, s_obs = _overlap_score(rep_set, sym_list)
                re_, rp, _, r_obs = _overlap_score(set(rep_risk), rf_list)
                s_match = se + 0.5 * sp
                s_miss  = max(len(sym_list) - s_match, 0)
                r_match = re_ + 0.5 * rp
                r_miss  = max(len(rf_list)  - r_match, 0)
                top_m   = []

            cand = CandidateDiagnosis(
                disease=disease,
                symptom_match_exact=int(round(s_match)),
                symptom_match_partial=0,
                symptom_total=max(len(sym_list), 1),
                symptom_observed=s_obs,
                risk_factor_hits=int(round(r_match)),
                risk_factor_total=max(len(rf_list), 1),
                risk_factor_observed=r_obs,
                prior=prior,
                top_matches=top_m[:4],
            )
            self._compute_prob(cand, s_match, s_miss, r_match, r_miss)
            self._compute_conf(cand)
            candidates.append(cand)

        total = sum(c.probability_raw for c in candidates) or 1.0
        for c in candidates:
            c.probability = c.probability_raw / total
            c.prob_ci_lo /= total; c.prob_ci_hi /= total; c.prob_std /= total

        candidates.sort(key=lambda c: (-c.probability, -c.confidence))
        return candidates[:top_k]

    @staticmethod
    def _compute_prob(cand, s_match, s_miss, r_match, r_miss):
        sa = s_match + 1.0;  sb = max(s_miss, 0) + 1.0
        ra = r_match + 1.0;  rb = max(r_miss,  0) + 1.0
        ss = RNG.beta(sa, sb, size=MC_SAMPLES)
        rs = RNG.beta(ra, rb, size=MC_SAMPLES)
        pn = np.clip(RNG.normal(cand.prior, cand.prior*0.1, MC_SAMPLES), 1e-6, 1)
        combined = W_SYMPTOMS*ss + W_RISK*rs + W_PRIOR*pn
        cand.samples         = combined
        cand.probability_raw = float(np.mean(combined))
        cand.prob_std        = float(np.std(combined))
        cand.prob_ci_lo      = float(np.percentile(combined, 5))
        cand.prob_ci_hi      = float(np.percentile(combined, 95))

    @staticmethod
    def _compute_conf(cand):
        """
        Confidenza basata SOLO su osservazioni attive.

        I miss passivi (sintomi nel profilo mai chiesti) NON sono evidenza:
        non sappiamo se il paziente non li ha o semplicemente non li ha menzionati.

        Usiamo solo:
          - symptom_match_exact/partial  (riportati dal paziente -> yes attivo)
          - symptom_observed             (quanti ne abbiamo effettivamente valutati)

        I miss attivi = symptom_observed - match_effettivi.
        """
        n = cand.symptom_observed + cand.risk_factor_observed
        cand.evidence_tokens = n

        # Osservazioni ATTIVE: solo i sintomi che abbiamo effettivamente valutato
        s_active_match = cand.symptom_match_exact + 0.5 * cand.symptom_match_partial
        s_active_miss  = max(cand.symptom_observed - s_active_match, 0)
        rf_active_match = cand.risk_factor_hits
        rf_active_miss  = max(cand.risk_factor_observed - rf_active_match, 0)

        cand.confidence = beta_variance_confidence(
            sym_alpha = s_active_match + 1.0,
            sym_beta  = s_active_miss  + 1.0,
            rf_alpha  = rf_active_match + 1.0,
            rf_beta   = rf_active_miss  + 1.0,
        )


# ---------------------------------------------------------------------------
# Beta-variance confidence (replaces n/(n+k) heuristic)
# ---------------------------------------------------------------------------

def beta_variance(alpha: float, beta: float) -> float:
    """
    Varianza analitica esatta del posterior Beta(alpha, beta):

        Var[theta] = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))

    Decresce monotonicamente all'aumentare dell'evidenza:
      - Beta(1,1)   -> Var = 1/12  ≈ 0.0833  (prior uniforme, massima incertezza)
      - Beta(5,5)   -> Var = 1/44  ≈ 0.0227
      - Beta(10,10) -> Var = 1/84  ≈ 0.0119
      - Beta(50,2)  -> Var ≈ 0.0016 (molto evidenza, distribuzione stretta)
    """
    s = alpha + beta
    return (alpha * beta) / (s * s * (s + 1.0))


def beta_variance_confidence(sym_alpha: float, sym_beta: float,
                              rf_alpha: float,  rf_beta: float) -> float:
    """
    Confidenza basata sulla varianza analitica del posterior Beta.

    La likelihood combinata e':
        lambda_d = w_s * theta_sym + w_r * theta_rf + w_p * pi_d

    Poiche' theta_sym e theta_rf sono indipendenti, la varianza di lambda_d
    (ignorando il termine prior che ha varianza fissa piccola) e':

        Var[lambda_d] ≈ w_s^2 * Var[theta_sym] + w_r^2 * Var[theta_rf]

    Normalizziamo rispetto al massimo teorico (entrambi Beta(1,1)):

        Var_max = w_s^2 * 1/12 + w_r^2 * 1/12
                = (w_s^2 + w_r^2) / 12
                = (0.65^2 + 0.20^2) / 12
                = 0.4625 / 12
                ≈ 0.03854

    Confidenza:
        C_d = 1 - Var[lambda_d] / Var_max  ∈ [0, 1]

    Con Beta(1,1) -> C = 0  (nessuna evidenza)
    All'aumentare di alpha e beta -> Var -> 0 -> C -> 1
    """
    var_sym = beta_variance(sym_alpha, sym_beta)
    var_rf  = beta_variance(rf_alpha,  rf_beta)

    var_combined = W_SYMPTOMS**2 * var_sym + W_RISK**2 * var_rf

    # Massimo teorico: entrambi Beta(1,1) -> Var = 1/12
    var_max = (W_SYMPTOMS**2 + W_RISK**2) / 12.0   # ≈ 0.03854

    return float(1.0 - var_combined / var_max)