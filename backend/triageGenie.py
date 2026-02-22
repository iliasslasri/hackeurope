"""
triageGenie.py
--------------
Analyses a new doctor-patient interaction transcript and merges any newly
discovered clinical information into an existing PatientHistory object.

Public API
----------
    update_patient(patient: PatientHistory, transcript: str) -> PatientHistory
    update_patient_async(patient: PatientHistory, transcript: str) -> PatientHistory

Both functions share the same interface:

Parameters
----------
patient : PatientHistory
    The current structured patient data (see schemas.PatientHistory).
    Fields that are already known will be preserved.  New information from
    the transcript is merged in (additive for lists, overwrite for scalars
    only when a better value is found, append for relevant_history).

transcript : str
    Raw text of the latest doctor-patient interaction.  Can be a full
    transcript, a brief exchange, or clinical notes — anything free-text.

Returns
-------
PatientHistory — updated object with newly extracted clinical data merged in.

Example
-------
>>> from backend.schemas import PatientHistory
>>> from backend.triageGenie import update_patient
>>>
>>> patient = PatientHistory(
...     symptoms=["fatigue", "weight gain"],
...     duration="3 months",
...     severity="moderate",
...     negated_symptoms=["fever", "chest pain"],
...     risk_factors=["female sex", "work stress"],
...     medications=[],
...     relevant_history="38-year-old female with gradual fatigue and low mood.",
... )
>>>
>>> transcript = '''
... Doctor: Any new symptoms since last time?
... Patient: Yes, I've been feeling really cold all the time, even indoors.
... Doctor: Cold intolerance noted. Are you constipated?
... Patient: Actually yes, that started last week.
... Doctor: We will check TSH levels. I'll also prescribe levothyroxine to start.
... '''
>>>
>>> updated = update_patient(patient, transcript)
>>> print(updated.symptoms)
['fatigue', 'weight gain', 'cold intolerance', 'constipation']
>>> print(updated.medications)
['levothyroxine']
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .llm_client import call_llm
from .prompts import TRIAGE_GENIE_SYSTEM, TRIAGE_GENIE_USER_TEMPLATE
from .schemas import PatientHistory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_user_prompt(patient: PatientHistory, transcript: str) -> str:
    """Render TRIAGE_GENIE_USER_TEMPLATE with the current patient data and transcript."""
    current_patient_json = json.dumps(patient.model_dump(), indent=2, ensure_ascii=False)
    return TRIAGE_GENIE_USER_TEMPLATE.format(
        current_patient_json=current_patient_json,
        transcript=transcript,
    )


def _merge_patient_data(original: PatientHistory, llm_output: dict[str, Any]) -> PatientHistory:
    """
    Merge the LLM-returned dict into the original PatientHistory.

    This is a belt-and-suspenders step: even if the LLM mistakenly drops
    some fields, we ensure the original values are never lost by falling
    back to the original when the LLM returns an empty value.
    """
    def _merge_list(original_list: list[str], new_list: list | None) -> list[str]:
        """Union of both lists, preserving order and deduplicating (case-insensitive)."""
        seen: set[str] = set()
        result: list[str] = []
        for item in (original_list or []) + (new_list or []):
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                result.append(item.strip())
        return result

    def _prefer_non_null(original_val: str | None, new_val: str | None) -> str | None:
        """Keep the new value if it is non-null/non-empty, otherwise keep original."""
        if new_val and new_val.strip():
            return new_val.strip()
        return original_val

    return PatientHistory(
        patient_name=_prefer_non_null(original.patient_name, llm_output.get("patient_name")),
        symptoms=_merge_list(original.symptoms, llm_output.get("symptoms")),
        negated_symptoms=_merge_list(original.negated_symptoms, llm_output.get("negated_symptoms")),
        risk_factors=_merge_list(original.risk_factors, llm_output.get("risk_factors")),
        medications=_merge_list(original.medications, llm_output.get("medications")),
        duration=_prefer_non_null(original.duration, llm_output.get("duration")),
        severity=_prefer_non_null(original.severity, llm_output.get("severity")),
        relevant_history=_prefer_non_null(
            original.relevant_history, llm_output.get("relevant_history")
        ),
    )


# ---------------------------------------------------------------------------
# Public async entry point
# ---------------------------------------------------------------------------


async def update_patient_async(
    patient: PatientHistory,
    transcript: str,
) -> PatientHistory:
    """
    Analyse ``transcript`` and return an updated :class:`PatientHistory`.

    The LLM is instructed to merge new information only — it must not drop
    fields that already exist in the patient object.  A local
    :func:`_merge_patient_data` step acts as a safety net in case the LLM
    accidentally omits something.

    Parameters
    ----------
    patient    : Current patient data.
    transcript : Raw text of the latest doctor-patient interaction.

    Returns
    -------
    PatientHistory with newly discovered information integrated.
    """
    if not transcript or not transcript.strip():
        logger.warning("triageGenie received an empty transcript — returning patient unchanged.")
        return patient

    user_prompt = _build_user_prompt(patient, transcript)

    try:
        raw_response = await call_llm(
            system_prompt=TRIAGE_GENIE_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.1,        # low: we want faithful extraction, not creativity
            max_tokens=1024,
            response_format="json_object",
        )

        updated = _merge_patient_data(patient, raw_response)
        logger.info(
            "triageGenie updated patient: +%d symptom(s), +%d risk factor(s).",
            len(updated.symptoms) - len(patient.symptoms),
            len(updated.risk_factors) - len(patient.risk_factors),
        )
        return updated

    except Exception as exc:  # noqa: BLE001
        logger.error("triageGenie LLM call failed: %s — returning patient unchanged.", exc)
        return patient


# ---------------------------------------------------------------------------
# Public sync entry point (convenience wrapper for Streamlit / scripts)
# ---------------------------------------------------------------------------


def update_patient(
    patient: PatientHistory,
    transcript: str,
) -> PatientHistory:
    """
    Blocking wrapper around :func:`update_patient_async`.

    Use this when you cannot use ``await`` (e.g. inside a Streamlit callback
    or a plain script).

    Parameters
    ----------
    patient    : Current patient data.
    transcript : Raw text of the latest doctor-patient interaction.

    Returns
    -------
    PatientHistory with newly discovered information integrated.
    """
    return asyncio.run(update_patient_async(patient, transcript))


# ---------------------------------------------------------------------------
# CLI / quick smoke-test (run with: python -m backend.triageGenie)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    _SAMPLE_PATIENT = PatientHistory(
        symptoms=["fatigue", "weight gain", "persistent sadness", "sluggishness"],
        duration="3 months",
        severity="moderate",
        negated_symptoms=["fever", "chest pain"],
        risk_factors=["female sex", "work stress"],
        medications=[],
        relevant_history=(
            "38-year-old female presenting with gradual onset of fatigue and low mood "
            "over the past 3 months. Denies fever or acute pain. Doctor suspects "
            "burnout or mild depression, but weight gain and sluggishness may point "
            "to Hypothyroidism."
        ),
    )

    _SAMPLE_TRANSCRIPT = """
Doctor: Good morning. How have you been feeling since our last appointment?
Patient: A bit worse, honestly. I've been feeling extremely cold even when everyone
    else is warm, and my skin has become really dry and flaky.
Doctor: Cold intolerance and dry skin — noted. Any changes in your bowel habits?
Patient: Yes, I've been quite constipated this past week.
Doctor: Given these new symptoms, I'm now more strongly suspecting Hypothyroidism.
    Your TSH was mildly elevated last time. I'm going to start you on levothyroxine
    25 mcg daily and we'll recheck labs in 6 weeks.
Patient: Okay, I also forgot to mention I have a family history of thyroid problems —
    my mother was on thyroid medication her whole life.
Doctor: That's very relevant. Family history of thyroid disease is a definite risk
    factor. I'll add that to your notes.
"""

    print("=" * 60)
    print("Triage Genie — smoke test")
    print("=" * 60)
    print("\n--- Original patient data ---")
    pprint.pprint(_SAMPLE_PATIENT.model_dump(), width=100, sort_dicts=False)

    updated_patient = update_patient(_SAMPLE_PATIENT, _SAMPLE_TRANSCRIPT)

    print("\n--- Updated patient data ---")
    pprint.pprint(updated_patient.model_dump(), width=100, sort_dicts=False)
