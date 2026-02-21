"""
agents.py
---------
AuraPipeline — orchestrator for the full Aura backend.

Flow per transcript chunk
--------------------------
1. triageGenie       → extract / update PatientHistory from the transcript
2. MedicalDiagnosisAgent → fast local Bayesian scorer → ranked candidate diseases
3. QuestionGenie     → generate targeted clinical questions for each disease
4. Assemble AuraUIPayload and return to the frontend

Step 2 runs automatically from any PatientHistory — no user interaction needed.
The scorer's ranked disease names feed step 3 so question generation is
grounded in evidence rather than relying on the LLM alone.
"""

from __future__ import annotations

import asyncio
import logging

from backend.schemas import PatientHistory, AuraUIPayload
from backend.triageGenie import update_patient_async
from backend.questionGenie import generate_questions_async
from backend.medicalDiagnosisAgent import MedicalDiagnosisAgent

logger = logging.getLogger(__name__)

class AuraPipeline:
    """
    Stateful orchestrator — holds the patient history across transcript chunks.

    Usage
    -----
    pipeline = AuraPipeline()                         # once, at startup
    payload  = pipeline.run(transcript_chunk)         # each time new text arrives
    """

    def __init__(self) -> None:
        self.patient_history = PatientHistory()
        # Initialise once — the scorer loads datasets and builds a semantic index.
        # This is the expensive step (~2-5 s); afterwards every run() call is fast.
        self._medical_ai = MedicalDiagnosisAgent(top_k=5)

    # ------------------------------------------------------------------
    # Async entrypoint (preferred)
    # ------------------------------------------------------------------

    async def run_async(self, full_transcript: str) -> AuraUIPayload:
        """
        Process a new transcript chunk and return an updated UI payload.

        Parameters
        ----------
        full_transcript : str
            The latest doctor-patient conversation text.

        Returns
        -------
        AuraUIPayload with updateUi=False if nothing changed, else full payload.
        """
        # Step 1 — Extract / update structured patient data
        updated_history = await update_patient_async(self.patient_history, full_transcript)

        # If nothing changed, skip expensive downstream calls
        if updated_history == self.patient_history:
            logger.info("AuraPipeline: no new clinical information detected — skipping update.")
            return AuraUIPayload(
                patient_history=self.patient_history,
                updateUi=False,
            )

        self.patient_history = updated_history

        # Step 2 — Run the local Bayesian scorer automatically
        candidate_diseases = self._medical_ai.generate_candidate_diseases(self.patient_history)
        ### if we have enough symptoms, we can generate questions
        if len(self.patient_history.symptoms) < 3:
            return AuraUIPayload(
                patient_history=self.patient_history,
                updateUi=False,
            )

        if not candidate_diseases:
            # Not enough symptoms yet to rank diseases — return partial payload
            return AuraUIPayload(
                patient_history=self.patient_history,
                updateUi=True,
            )

        # Step 3 — Generate targeted clinical questions for each candidate disease
        questions_result = await generate_questions_async(
            patient_history=self.patient_history.model_dump(),
            candidate_diseases=candidate_diseases,
        )

        # Assemble the final payload consumed by the Streamlit frontend
        return AuraUIPayload(
            transcript_chunk=full_transcript,
            patient_history=self.patient_history,
            questions_by_disease=questions_result.get("results", []),
            updateUi=True,
        )

    # ------------------------------------------------------------------
    # Sync convenience wrapper (used by Streamlit callbacks)
    # ------------------------------------------------------------------

    def run(self, full_transcript: str) -> AuraUIPayload:
        """Blocking wrapper around run_async for non-async callers."""
        return asyncio.run(self.run_async(full_transcript))
