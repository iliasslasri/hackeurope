"""
mappers.py
----------
Utility functions for converting internal scorer types to backend schema types.

The scorer (agents/scorer.py) operates on CandidateDiagnosis dataclasses.
The pipeline and frontend expect DDxEntry Pydantic models (backend/schemas.py).
This module bridges the two without coupling them directly.
"""

from __future__ import annotations

from typing import List

from agents.scorer import CandidateDiagnosis
from backend.schemas import DDxEntry, SuspicionLevel


# ---------------------------------------------------------------------------
# SuspicionLevel mapping
# ---------------------------------------------------------------------------

_PROBABILITY_LABEL_TO_SUSPICION: dict[str, SuspicionLevel] = {
    "Very High": SuspicionLevel.HIGH,
    "High":      SuspicionLevel.HIGH,
    "Moderate":  SuspicionLevel.MEDIUM,
    "Low":       SuspicionLevel.LOW,
    "Very Low":  SuspicionLevel.LOW,
}


def suspicion_from_candidate(candidate: CandidateDiagnosis) -> SuspicionLevel:
    """
    Map a CandidateDiagnosis probability label to a SuspicionLevel enum value.

    CandidateDiagnosis.probability_label returns one of:
        "Very High" | "High" | "Moderate" | "Low" | "Very Low"

    These are collapsed into the three-tier SuspicionLevel:
        High / Medium / Low
    """
    return _PROBABILITY_LABEL_TO_SUSPICION.get(
        candidate.probability_label, SuspicionLevel.LOW
    )


# ---------------------------------------------------------------------------
# CandidateDiagnosis → DDxEntry
# ---------------------------------------------------------------------------

def candidate_to_ddx_entry(candidate: CandidateDiagnosis, rank: int) -> DDxEntry:
    """
    Convert a single CandidateDiagnosis to a DDxEntry.

    Args:
        candidate:  Scored diagnosis produced by DiagnosisScorer.
        rank:       1-based position in the ranked DDx list.

    Returns:
        A DDxEntry Pydantic model ready for inclusion in AuraUIPayload.ddx.

    Notes:
        - key_supporting is populated from the top semantic/string matches
          collected by DiagnosisScorer (candidate.top_matches).
        - key_against is left empty here; the Strategist LLM layer can
          enrich this field with its own reasoning later.
    """
    return DDxEntry(
        rank=rank,
        disease=candidate.disease,
        suspicion=suspicion_from_candidate(candidate),
        key_supporting=[label for label, _score in candidate.top_matches],
        key_against=[],
    )


def candidates_to_ddx(
    candidates: List[CandidateDiagnosis],
) -> List[DDxEntry]:
    """
    Convert an ordered list of CandidateDiagnosis objects to a ranked DDx list.

    Args:
        candidates: Already-sorted list returned by DiagnosisScorer.score().

    Returns:
        List of DDxEntry with rank starting at 1.
    """
    return [
        candidate_to_ddx_entry(candidate, rank=i + 1)
        for i, candidate in enumerate(candidates)
    ]
