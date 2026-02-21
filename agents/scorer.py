"""
scorer.py
---------
Probabilistic scoring engine for candidate disease ranking.

KEY DESIGN PRINCIPLE
====================
**Probability** and **Confidence** are INDEPENDENT quantities:

  Probability  = How likely is this disease given the symptoms and anamnesis?
                 → Driven by what the evidence *says* (match quality)

  Confidence   = How much do we trust that probability estimate?
                 → Driven by how much evidence we *have* (evidence quantity)

A disease can be:
  • High probability + High confidence : many symptoms match, strong evidence
  • High probability + Low confidence  : few symptoms reported but they all match
                                         a rare/unusual disease — we should worry
                                         but can't be sure
  • Low probability + High confidence  : many symptoms checked, few match — fairly
                                         sure it's not this
  • Low probability + Low confidence   : little data either way — unknown

Approach
========
1. **Probability** — posterior mean of a Beta-based likelihood model:
   Beta(α = eff_matches + 1,  β = eff_misses + 1)
   Combined across symptoms and risk factors with weighted average.
   Normalised across all diseases so values sum to 1.0 (true posterior).

2. **Confidence** — derived entirely from *evidence quantity*, not score size:
   confidence = 1 − 1 / (1 + total_evidence_tokens)
   where total_evidence_tokens = sym_observed + rf_observed
   (how many of the disease's known symptoms/RFs we actually had data about)
   This is *independent* of whether those tokens matched or not.

   Additionally the Beta posterior width (std) gives a probabilistic credible
   interval that captures per-disease uncertainty in the probability estimate.
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple

RNG = np.random.default_rng(42)
MC_SAMPLES = 5_000
W_SYMPTOMS = 0.65
W_RISK     = 0.20
W_PRIOR    = 0.15


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenise(text: str) -> set[str]:
    """Expand a free-text string into a flat set of lowercase tokens/phrases."""
    tokens = set()
    for part in re.split(r"[,;.\n]", text):
        part = _normalise(part)
        if part:
            tokens.add(part)
            for w in part.split():
                if len(w) > 3:
                    tokens.add(w)
    return tokens


def _overlap_score(reported: set[str], reference: List[str]) -> Tuple[int, int, int, int]:
    """
    Returns (exact_matches, partial_matches, misses, observed).

    - exact_matches  : reference symptom phrase found verbatim in reported tokens
    - partial_matches: at least one word overlaps
    - misses         : no overlap — reported but NOT present
    - observed       : how many reference items we had *any* data about
                       (matches + misses; i.e. we can say something about them)
    """
    exact = partial = misses = observed = 0
    for ref_symptom in reference:
        ref_norm   = _normalise(ref_symptom)
        ref_tokens = set(ref_norm.split())
        if ref_norm in reported:
            exact    += 1
            observed += 1
            continue
        hit = any(
            r in ref_norm or ref_norm in r or
            any(rt in ref_tokens for rt in r.split())
            for r in reported
        )
        if hit:
            partial  += 1
            observed += 1
        else:
            # We have no information about this symptom — don't count as observed
            misses += 1
    return exact, partial, misses, observed


# ---------------------------------------------------------------------------
# Dataclass for a single candidate result
# ---------------------------------------------------------------------------

@dataclass
class CandidateDiagnosis:
    disease: str

    # Raw match counts
    symptom_match_exact: int
    symptom_match_partial: int
    symptom_total: int
    symptom_observed: int        # how many disease symptoms we had data on

    risk_factor_hits: int
    risk_factor_total: int
    risk_factor_observed: int    # how many RFs we had data on

    prior: float

    # ---- PROBABILITY (what does the evidence say?) ----
    probability: float = 0.0          # normalised posterior, sums to 1 across diseases
    probability_raw: float = 0.0      # unnormalised likelihood mean
    prob_ci_lo: float = 0.0           # 5th percentile of MC posterior
    prob_ci_hi: float = 0.0           # 95th percentile of MC posterior
    prob_std: float = 0.0             # std of MC posterior

    # ---- CONFIDENCE (how much evidence did we have?) ----
    confidence: float = 0.0           # 0–1, based on evidence quantity only
    evidence_tokens: int = 0          # total observed tokens driving confidence

    samples: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def probability_label(self) -> str:
        p = self.probability
        if p >= 0.20:  return "Very High"
        if p >= 0.10:  return "High"
        if p >= 0.05:  return "Moderate"
        if p >= 0.02:  return "Low"
        return "Very Low"

    @property
    def confidence_label(self) -> str:
        c = self.confidence
        if c >= 0.80:  return "High"
        if c >= 0.50:  return "Moderate"
        if c >= 0.25:  return "Low"
        return "Very Low"

    @property
    def risk_flag(self) -> str:
        """
        Highlights the dangerous quadrant: high probability but low confidence.
        This means the evidence points strongly at the disease but we don't have
        enough data to be sure — requires more investigation.
        """
        if self.probability >= 0.08 and self.confidence < 0.50:
            return "⚠️  HIGH-PROB / LOW-CONF"
        return ""


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class DiagnosisScorer:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def score(
        self,
        symptoms_text: str,
        anamnesis_text: str,
        top_k: int = 10,
    ) -> List[CandidateDiagnosis]:
        reported_symptoms = _tokenise(symptoms_text)
        reported_risk     = _tokenise(anamnesis_text)

        candidates: List[CandidateDiagnosis] = []

        for _, row in self.dataset.iterrows():
            disease  = row["disease"]
            sym_list = [_normalise(s) for s in row["symptoms"]]
            rf_list  = [_normalise(r) for r in row["risk_factors"]]
            prior    = float(row["base_prevalence"])

            s_exact, s_partial, s_miss, s_obs = _overlap_score(reported_symptoms, sym_list)
            rf_exact, rf_partial, rf_miss, rf_obs = _overlap_score(reported_risk, rf_list)

            cand = CandidateDiagnosis(
                disease=disease,
                symptom_match_exact=s_exact,
                symptom_match_partial=s_partial,
                symptom_total=max(len(sym_list), 1),
                symptom_observed=s_obs,
                risk_factor_hits=rf_exact + rf_partial,
                risk_factor_total=max(len(rf_list), 1),
                risk_factor_observed=rf_obs,
                prior=prior,
            )
            self._compute_probability(cand)
            self._compute_confidence(cand)
            candidates.append(cand)

        # Normalise probabilities across all diseases (true posterior)
        total_raw = sum(c.probability_raw for c in candidates) or 1.0
        for c in candidates:
            c.probability = c.probability_raw / total_raw
            # Scale CI proportionally
            c.prob_ci_lo = c.prob_ci_lo / total_raw
            c.prob_ci_hi = c.prob_ci_hi / total_raw
            c.prob_std   = c.prob_std   / total_raw

        # Sort: descending probability, then descending confidence as tiebreak
        candidates.sort(key=lambda c: (-c.probability, -c.confidence))
        return candidates[:top_k]

    @staticmethod
    def _compute_probability(cand: CandidateDiagnosis):
        """
        Probability = posterior mean of Beta likelihood × prior.
        Fully determined by *what* the evidence says (match quality).
        """
        s_eff_match = cand.symptom_match_exact + 0.5 * cand.symptom_match_partial
        s_eff_miss  = max(cand.symptom_total - s_eff_match, 0)

        rf_hits = cand.risk_factor_hits
        rf_miss = max(cand.risk_factor_total - rf_hits, 0)

        sym_alpha, sym_beta = s_eff_match + 1.0, s_eff_miss + 1.0
        rf_alpha,  rf_beta  = rf_hits + 1.0,     rf_miss + 1.0

        sym_samples = RNG.beta(sym_alpha, sym_beta, size=MC_SAMPLES)
        rf_samples  = RNG.beta(rf_alpha,  rf_beta,  size=MC_SAMPLES)
        prior_noise = np.clip(
            RNG.normal(cand.prior, cand.prior * 0.1, size=MC_SAMPLES), 1e-6, 1
        )

        combined = (
            W_SYMPTOMS * sym_samples +
            W_RISK     * rf_samples  +
            W_PRIOR    * prior_noise
        )

        cand.samples       = combined
        cand.probability_raw = float(np.mean(combined))
        cand.prob_std      = float(np.std(combined))
        cand.prob_ci_lo    = float(np.percentile(combined, 5))
        cand.prob_ci_hi    = float(np.percentile(combined, 95))

    @staticmethod
    def _compute_confidence(cand: CandidateDiagnosis):
        """
        Confidence = how much evidence did we actually observe?
        Completely independent of whether that evidence matched or not.

        Uses a saturating function: conf = n / (n + k)
        where n = observed evidence tokens, k = half-saturation constant.
        Full confidence requires observing ~10+ symptoms/RF items.
        """
        k = 6.0  # half-saturation: conf=0.5 when we've observed 6 tokens
        n = cand.symptom_observed + cand.risk_factor_observed
        cand.evidence_tokens = n
        cand.confidence = n / (n + k)
        