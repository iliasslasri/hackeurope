from __future__ import annotations

import asyncio
import logging

from backend.schemas import PatientHistory, AuraUIPayload, DDxEntry, SuspicionLevel
from backend.triageGenie import update_patient_async
from backend.questionGenie import generate_questions_async
from backend.qaGenie import extract_qa_async

# FIX Bug 1: correct import path — agents/MedicalDiagnosisAgent.py, not backend/
import agents  # noqa: F401  — ensures agents/ is on sys.path via agents/__init__.py
from agents.MedicalDiagnosisAgent import MedicalDiagnosisAgent
from agents.question_strategy import Question

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

    def __init__(self, initial_history: PatientHistory = None, top_k: int = 5) -> None:
        self.patient_history = initial_history or PatientHistory()
        self._top_k = top_k
        self._has_ddx = False   # True once diagnose() has run at least once
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
        # Step 1 — Run triageGenie and qaGenie concurrently (no dependency between them)
        updated_history, qa_pairs = await asyncio.gather(
            update_patient_async(self.patient_history, full_transcript),
            extract_qa_async(full_transcript),
        )

        history_changed = updated_history != self.patient_history
        self.patient_history = updated_history

        # Skip if: nothing new
        if not history_changed and not qa_pairs:
            logger.info("AuraPipeline: no new info — skipping.")
            return AuraUIPayload(
                patient_history=self.patient_history,
                updateUi=False,
            )

        # Step 2 — Initial scoring (re-run whenever history changes or first time)
        if history_changed or not self._has_ddx:
            symptoms_text, anamnesis_text = _patient_to_scorer_inputs(self.patient_history)
            self._medical_ai.diagnose(
                symptoms=symptoms_text,
                anamnesis=anamnesis_text,
                top_k=self._top_k,
            )
            self._has_ddx = True

        # Step 2b — Refine scores using actual Q&A pairs from the conversation
        if qa_pairs:
            logger.info("qaGenie: feeding %d Q&A pair(s) into update_scores.", len(qa_pairs))
            for pair in qa_pairs:
                try:
                    q = Question(
                        prompt=pair["question"],
                        target_symptom=None,    # extracted automatically via semantic similarity
                        question_type="symptom_probe",
                    )
                    self._medical_ai.update_scores(q, pair["answer"])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("update_scores failed for pair %r: %s", pair, exc)

        # Read back the (possibly updated) candidate list
        raw_candidates = self._medical_ai.get_candidates() or []

        ddx_entries = []
        print("raw_candidates ", raw_candidates)
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
                probability_pct=round(c.probability_raw * 100, 1),
                confidence=round(c.confidence * 100, 1),
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

        # Derive a single follow-up question from the generated questions.
        # Priority: first "rule_in" question from the top-ranked disease,
        # falling back to the very first question if none is rule_in.
        follow_up = ""
        results_list = questions_result.get("results", [])
        if results_list:
            for dq in results_list:
                qs = getattr(dq, "questions", None) or (dq.get("questions", []) if isinstance(dq, dict) else [])
                for q in qs:
                    q_target = getattr(q, "target", None) or (q.get("target") if isinstance(q, dict) else None)
                    q_text = getattr(q, "question", None) or (q.get("question", "") if isinstance(q, dict) else "")
                    target_val = q_target.value if hasattr(q_target, "value") else str(q_target)
                    if target_val == "rule_in" and q_text:
                        follow_up = q_text
                        break
                if follow_up:
                    break
            # Fallback: first question of the first disease
            if not follow_up and results_list:
                first_dq = results_list[0]
                first_qs = getattr(first_dq, "questions", None) or (first_dq.get("questions", []) if isinstance(first_dq, dict) else [])
                if first_qs:
                    first_q = first_qs[0]
                    follow_up = getattr(first_q, "question", None) or (first_q.get("question", "") if isinstance(first_q, dict) else "")

        # Assemble the final payload consumed by the Streamlit frontend
        return AuraUIPayload(
            transcript_chunk=full_transcript,
            patient_history=self.patient_history,
            ddx=ddx_entries,
            follow_up_question=follow_up,
            questions_by_disease=results_list,
            updateUi=True,
        )

    # ------------------------------------------------------------------
    # Sync convenience wrapper (used by Streamlit callbacks)
    # ------------------------------------------------------------------

    def run(self, full_transcript: str) -> AuraUIPayload:
        """Blocking wrapper around run_async for non-async callers."""
        return asyncio.run(self.run_async(full_transcript))


