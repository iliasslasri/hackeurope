"""
medicalDiagnosisAgent.py
------------------------
Bridge between the AuraPipeline and the agents/ probabilistic scorer.

Wraps agents.agent.MedicalDiagnosisAgent so it works in a fully automatic,
non-interactive mode — no Q&A session, no manual input.

The PatientHistory object produced by triageGenie is used to construct the
symptom and anamnesis strings that the scorer needs, then the top-K ranked
diseases are returned as a plain list of strings for the QuestionGenie.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agents.agent import MedicalDiagnosisAgent as _BaseAgent
from backend.schemas import PatientHistory

logger = logging.getLogger(__name__)


class MedicalDiagnosisAgent:
    """
    Wraps the probabilistic agents.MedicalDiagnosisAgent for automatic use
    inside the AuraPipeline.

    The scorer is initialised once (dataset load + semantic index build) and
    reused across every transcript chunk — keeping latency low per turn.
    """

    def __init__(self, top_k: int = 5) -> None:
        """
        Parameters
        ----------
        top_k : int
            Number of top candidate diseases to return per run.
        """
        self.top_k = top_k
        logger.info("Initialising MedicalDiagnosisAgent scorer (this may take a moment)…")
        self._agent = _BaseAgent(verbose=False)
        logger.info("MedicalDiagnosisAgent scorer ready.")

    # ------------------------------------------------------------------
    # Public API consumed by AuraPipeline
    # ------------------------------------------------------------------

    def generate_candidate_diseases(
        self,
        patient_history: PatientHistory,
        top_k: int | None = None,
    ) -> list[str]:
        """
        Score the patient against the disease dataset and return the top-K
        disease names ranked by posterior probability.

        Parameters
        ----------
        patient_history : PatientHistory
            The structured patient data extracted by triageGenie.
        top_k : int | None
            Override the instance-level top_k if provided.

        Returns
        -------
        list[str]
            Ordered list of disease names (most likely first).
            Returns [] if no symptoms are available yet.
        """
        k = top_k or self.top_k

        # Build the symptom string from extracted structured fields
        symptoms_str = ", ".join(patient_history.symptoms) if patient_history.symptoms else ""

        # Build the anamnesis string: relevant_history + risk_factors
        anamnesis_parts: list[str] = []
        if patient_history.relevant_history:
            anamnesis_parts.append(patient_history.relevant_history)
        if patient_history.risk_factors:
            anamnesis_parts.append(", ".join(patient_history.risk_factors))
        anamnesis_str = " ".join(anamnesis_parts)

        if not symptoms_str:
            logger.warning(
                "MedicalDiagnosisAgent: no symptoms in PatientHistory yet — skipping scoring."
            )
            return []

        try:
            candidates = self._agent.diagnose(symptoms_str, anamnesis_str, top_k=k)
            disease_names = [c.disease for c in candidates]
            logger.info(
                "Scorer ranked top-%d: %s",
                len(disease_names),
                ", ".join(disease_names),
            )
            return disease_names

        except Exception as exc:  # noqa: BLE001
            logger.error("MedicalDiagnosisAgent scoring failed: %s", exc)
            return []
