"""
agents.py
---------
Orchestrator that wires the three Aura agents into a single pipeline:

    [Transcript chunk]
         │
         ▼
    InformationExtractorAgent   →  PatientHistory (JSON)
         │
         ▼
    StrategistAgent             →  DDx + follow-up question (JSON)
         │
         ▼
    SafetyReviewerAgent         →  Approved or revised DDx (JSON)
         │
         ▼
    QuestionGenieAgent          →  Per-disease question banks (JSON)
         │
         ▼
    [AuraUIPayload] →  Streamlit frontend

Each agent class wraps exactly one LLM call and validates its output
against the Pydantic schemas from schemas.py.

For the hack, the QuestionGenieAgent delegates entirely to questionGenie.py.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .llm_client import call_llm_sync
from .prompts import (
    INFORMATION_EXTRACTOR_SYSTEM,
    INFORMATION_EXTRACTOR_USER_TEMPLATE,
    STRATEGIST_SYSTEM,
    STRATEGIST_USER_TEMPLATE,
    SAFETY_REVIEWER_SYSTEM,
    SAFETY_REVIEWER_USER_TEMPLATE,
)
from .schemas import (
    AuraUIPayload,
    DDxEntry,
    DiseaseQuestions,
    PatientHistory,
    SafetyReviewerOutput,
    StrategistOutput,
    SuspicionLevel,
    QuestionTarget,
    ClinicalQuestion,
)
from .questionGenie import generate_questions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_parse(raw: dict[str, Any], model):
    """
    Attempt to parse *raw* into *model*.
    On validation error, log and return None.
    """
    try:
        return model.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        logger.error("Schema validation failed for %s: %s", model.__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Agent 1 — Information Extractor
# ---------------------------------------------------------------------------


class InformationExtractorAgent:
    """
    Parses a raw transcript chunk and returns a validated PatientHistory.

    If the LLM call fails or returns malformed JSON, a fallback
    empty PatientHistory is returned so the pipeline can continue.
    """

    def run(self, transcript: str) -> PatientHistory:
        user_prompt = INFORMATION_EXTRACTOR_USER_TEMPLATE.format(transcript=transcript)
        try:
            raw = call_llm_sync(
                INFORMATION_EXTRACTOR_SYSTEM,
                user_prompt,
                temperature=0.1,
                max_tokens=512,
            )
            result = _safe_parse(raw, PatientHistory)
            return result or PatientHistory()
        except Exception as exc:  # noqa: BLE001
            logger.error("InformationExtractorAgent failed: %s", exc)
            return PatientHistory()


# ---------------------------------------------------------------------------
# Agent 2 — Strategist
# ---------------------------------------------------------------------------


class StrategistAgent:
    """
    Takes structured patient data + RAG context and produces:
    - A ranked DDx (3–5 entries)
    - A diagnostic gap description
    - A targeted follow-up question
    """

    def run(
        self,
        patient_history: PatientHistory,
        rag_context: str = "",
    ) -> StrategistOutput | None:
        patient_data_json = json.dumps(patient_history.model_dump(), indent=2)
        user_prompt = STRATEGIST_USER_TEMPLATE.format(
            patient_data_json=patient_data_json,
            rag_context=rag_context or "No additional context available.",
        )
        try:
            raw = call_llm_sync(
                STRATEGIST_SYSTEM,
                user_prompt,
                temperature=0.2,
                max_tokens=1024,
            )
            result = _safe_parse(raw, StrategistOutput)

            # --- MOCK fallback: use pre-built output if validation fails -------
            if result is None:
                result = _mock_strategist_output(patient_history)

            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("StrategistAgent failed: %s", exc)
            return _mock_strategist_output(patient_history)


def _mock_strategist_output(patient_history: PatientHistory) -> StrategistOutput:
    """
    Deterministic fallback so the UI always has something to render
    even when the LLM is unavailable (e.g. during local development).
    """
    symptoms = set(patient_history.symptoms)
    has_fatigue = "fatigue" in symptoms
    has_weight_gain = "weight gain" in symptoms
    has_sadness = any(s in symptoms for s in ["persistent sadness", "depression"])

    ddx = []
    if has_fatigue and has_weight_gain:
        ddx.append(DDxEntry(rank=1, disease="Hypothyroidism",
                             suspicion=SuspicionLevel.HIGH,
                             key_supporting=["fatigue", "weight gain"],
                             key_against=[]))
        ddx.append(DDxEntry(rank=2, disease="Depression",
                             suspicion=SuspicionLevel.HIGH,
                             key_supporting=["fatigue", "persistent sadness"],
                             key_against=["weight gain (atypical)"]))
        ddx.append(DDxEntry(rank=3, disease="Anemia",
                             suspicion=SuspicionLevel.MEDIUM,
                             key_supporting=["fatigue"],
                             key_against=["no pallor reported"]))
    else:
        ddx.append(DDxEntry(rank=1, disease="Unspecified",
                             suspicion=SuspicionLevel.LOW,
                             key_supporting=list(symptoms)[:2],
                             key_against=[]))

    return StrategistOutput(
        ddx=ddx,
        diagnostic_gap="Thyroid function tests not yet discussed",
        follow_up_question=(
            "Has the patient experienced cold intolerance, hair loss, or constipation "
            "recently? These would strongly support a Hypothyroidism diagnosis over Depression."
        ),
        reasoning=(
            "The overlap of fatigue, weight gain, and low mood is shared by Hypothyroidism "
            "and Depression. A TSH level and cold intolerance history are the key "
            "discriminators. Anemia is less likely without reported pallor."
        ),
    )


# ---------------------------------------------------------------------------
# Agent 3 — Safety Reviewer
# ---------------------------------------------------------------------------


class SafetyReviewerAgent:
    """
    Final-pass agent.  Verifies the Strategist output against three safety gates:
    1. Common-first rule
    2. Red-flag check (life-threatening conditions)
    3. Hallucination guard
    """

    def run(
        self,
        patient_history: PatientHistory,
        strategist_output: StrategistOutput,
    ) -> SafetyReviewerOutput:
        patient_data_json = json.dumps(patient_history.model_dump(), indent=2)
        proposed_output_json = json.dumps(strategist_output.model_dump(), indent=2)
        user_prompt = SAFETY_REVIEWER_USER_TEMPLATE.format(
            patient_data_json=patient_data_json,
            proposed_output_json=proposed_output_json,
        )
        try:
            raw = call_llm_sync(
                SAFETY_REVIEWER_SYSTEM,
                user_prompt,
                temperature=0.1,
                max_tokens=768,
            )
            result = _safe_parse(raw, SafetyReviewerOutput)
            return result or SafetyReviewerOutput(approved=True)
        except Exception as exc:  # noqa: BLE001
            logger.error("SafetyReviewerAgent failed: %s", exc)
            return SafetyReviewerOutput(approved=True)  # fail-open: don't block UI


# ---------------------------------------------------------------------------
# Agent 4 — Question Genie (delegates to questionGenie.py)
# ---------------------------------------------------------------------------


class QuestionGenieAgent:
    """
    Thin wrapper that exposes the questionGenie module as a pipeline agent.
    Generates ≥3 targeted clinical questions per candidate disease.
    """

    def run(
        self,
        patient_history: PatientHistory,
        candidate_diseases: list[str],
    ) -> list[DiseaseQuestions]:
        raw_output = generate_questions(
            patient_history=patient_history.model_dump(),
            candidate_diseases=candidate_diseases,
        )
        results: list[DiseaseQuestions] = []
        for item in raw_output.get("results", []):
            questions = [
                ClinicalQuestion(
                    question=q.get("question", ""),
                    clinical_rationale=q.get("clinical_rationale", ""),
                    target=QuestionTarget(q.get("target", "differentiate")),
                )
                for q in item.get("questions", [])
            ]
            results.append(
                DiseaseQuestions(
                    disease=item.get("disease", "Unknown"),
                    questions=questions,
                    error=item.get("error"),
                )
            )
        return results


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class AuraPipeline:
    """
    Runs the full Aura agent pipeline for a single transcript chunk and
    returns an AuraUIPayload ready for the Streamlit frontend to render.

    Usage
    -----
    >>> pipeline = AuraPipeline()
    >>> payload = pipeline.run(transcript_chunk, rag_context)
    >>> # payload is an AuraUIPayload — render it in the frontend
    """

    def __init__(self):
        self.extractor     = InformationExtractorAgent()
        self.strategist    = StrategistAgent()
        self.reviewer      = SafetyReviewerAgent()
        self.genie         = QuestionGenieAgent()

    def run(
        self,
        transcript_chunk: str,
        rag_context: str = "",
    ) -> AuraUIPayload:
        # --- Step 1: extract structured patient data ---
        patient_history = self.extractor.run(transcript_chunk)

        # --- Step 2: generate DDx and primary follow-up question ---
        strategist_result = self.strategist.run(patient_history, rag_context)
        if strategist_result is None:
            strategist_result = _mock_strategist_output(patient_history)

        # --- Step 3: safety review ---
        safety_result = self.reviewer.run(patient_history, strategist_result)

        # Choose approved or revised output
        if safety_result.approved:
            final_ddx = strategist_result.ddx
            final_question = strategist_result.follow_up_question
        else:
            final_ddx = safety_result.revised_ddx or strategist_result.ddx
            final_question = (
                safety_result.revised_follow_up_question
                or strategist_result.follow_up_question
            )

        # --- Step 4: generate per-disease question banks ---
        disease_names = [entry.disease for entry in final_ddx]
        questions_by_disease = self.genie.run(patient_history, disease_names)

        return AuraUIPayload(
            transcript_chunk=transcript_chunk,
            patient_history=patient_history,
            ddx=final_ddx,
            follow_up_question=final_question,
            questions_by_disease=questions_by_disease,
            safety_issues=safety_result.issues,
            approved_by_safety_reviewer=safety_result.approved,
        )
