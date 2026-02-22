"""
main_agent.py
-------------
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

from backend.schemas import PatientHistory, AuraUIPayload, DDxEntry, SuspicionLevel
from backend.triageGenie import update_patient_async
from backend.questionGenie import generate_questions_async

# FIX Bug 1: correct import path — agents/MedicalDiagnosisAgent.py, not backend/
import agents  # noqa: F401  — ensures agents/ is on sys.path via agents/__init__.py
from agents.MedicalDiagnosisAgent import MedicalDiagnosisAgent

logger = logging.getLogger(__name__)

# Maximum number of diseases to generate questions for (keeps parallel calls small)
_QUESTION_TOP_K = 3


def _patient_to_scorer_inputs(patient: PatientHistory) -> tuple[str, str]:
    """
    Bridge a structured PatientHistory into the two free-text strings
    expected by DiagnosisScorer.score(symptoms_text, anamnesis_text).

    The scorer tokenises both strings, so a comma-separated list works well.
    """
    symptoms_text  = ", ".join(patient.symptoms)

    anamnesis_parts = list(patient.risk_factors)
    if patient.relevant_history:
        anamnesis_parts.append(patient.relevant_history)
    anamnesis_text = ". ".join(anamnesis_parts)

    return symptoms_text, anamnesis_text


class AuraPipeline:
    """
    Stateful orchestrator — holds the patient history across transcript chunks.

    Usage
    -----
    pipeline = AuraPipeline()                         # once, at startup
    payload  = pipeline.run(transcript_chunk)         # each time new text arrives
    """

    def __init__(self, top_k: int = 5) -> None:
        self.patient_history = PatientHistory()
        self._top_k = top_k
        # FIX Bug 2: constructor takes (verbose, hf_token), NOT top_k.
        # Initialise once — the scorer loads datasets and builds a semantic index.
        # This is the expensive step (~2-5 s); afterwards every run() call is fast.
        self._medical_ai = MedicalDiagnosisAgent(verbose=False)

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

        # Guard: require at least 3 symptoms before scoring
        if len(self.patient_history.symptoms) < 2:
            return AuraUIPayload(
                patient_history=self.patient_history,
                updateUi=False,
            )

        # Step 2 — FIX Bug 3: bridge PatientHistory → scorer strings, then call .diagnose()
        symptoms_text, anamnesis_text = _patient_to_scorer_inputs(self.patient_history)
        raw_candidates = self._medical_ai.diagnose(
            symptoms=symptoms_text,
            anamnesis=anamnesis_text,
            top_k=self._top_k,
        )

        ddx_entries = []
        for i, c in enumerate(raw_candidates, start=1):
            if c.probability_label in ["Very High", "High"]:
                susp = SuspicionLevel.HIGH
            elif c.probability_label == "Moderate":
                susp = SuspicionLevel.MEDIUM
            else:
                susp = SuspicionLevel.LOW
                
            ddx_entries.append(DDxEntry(
                rank=i,
                disease=c.disease,
                suspicion=susp,
                key_supporting=[m[0] for m in c.top_matches]
            ))

        # Extract plain disease names for the downstream questionGenie
        # Only generate questions for the top-K diseases to keep latency low
        candidate_diseases: list[str] = [c.disease for c in raw_candidates[:_QUESTION_TOP_K]]

        if not candidate_diseases:
            # Not enough symptom signal to rank — return partial payload
            return AuraUIPayload(
                patient_history=self.patient_history,
                ddx=ddx_entries,
                updateUi=True,
            )

        # Step 3 — Generate targeted clinical questions (parallel LLM calls)
        questions_result = await generate_questions_async(
            patient_history=self.patient_history.model_dump(),
            candidate_diseases=candidate_diseases,
        )

        # Assemble the final payload consumed by the Streamlit frontend
        return AuraUIPayload(
            transcript_chunk=full_transcript,
            patient_history=self.patient_history,
            ddx=ddx_entries,
            questions_by_disease=questions_result.get("results", []),
            updateUi=True,
        )

    # ------------------------------------------------------------------
    # Sync convenience wrapper (used by Streamlit callbacks)
    # ------------------------------------------------------------------

    def run(self, full_transcript: str) -> AuraUIPayload:
        """Blocking wrapper around run_async for non-async callers."""
        return asyncio.run(self.run_async(full_transcript))
