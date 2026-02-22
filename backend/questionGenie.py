"""
questionGenie.py
----------------
Generates clinically targeted follow-up questions for a given set of
candidate diseases based on a patient's history.

Public API
----------
    generate_questions(patient_history: dict, candidate_diseases: list[str]) -> dict
    generate_questions_async(patient_history: dict, candidate_diseases: list[str]) -> dict

Both functions share the same interface:

Parameters
----------
patient_history : dict
    Structured patient data.  Expected shape (all fields optional):
    {
        "symptoms":          ["fatigue", "weight gain", ...],
        "duration":          "3 months",
        "severity":          "moderate",
        "negated_symptoms":  ["fever"],
        "risk_factors":      ["female sex", "family history of thyroid disease"],
        "medications":       ["levothyroxine"],
        "relevant_history":  "Patient reports recent work stress ..."
    }

candidate_diseases : list[str]
    Ordered list of disease names to generate questions for.
    E.g. ["Hypothyroidism", "Depression", "Anemia"]

Returns
-------
dict — shape:
{
    "patient_history_snapshot": { ... },   # echo of the input (for traceability)
    "results": [
        {
            "disease": "Hypothyroidism",
            "questions": [
                {
                    "question": "Has the patient noticed cold intolerance ...",
                    "clinical_rationale": "Cold intolerance is a pathognomonic ...",
                    "target": "rule_in"
                },
                ...                       # at least 3 questions per disease
            ],
            "error": null                 # populated only on LLM failure
        },
        ...
    ]
}
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .llm_client import call_llm
from .prompts import QUESTION_GENIE_SYSTEM, QUESTION_GENIE_USER_TEMPLATE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_user_prompt(patient_history: dict[str, Any], disease_name: str) -> str:
    """Render the QUESTION_GENIE_USER_TEMPLATE with concrete values."""
    patient_history_json = json.dumps(patient_history, indent=2, ensure_ascii=False)
    return QUESTION_GENIE_USER_TEMPLATE.format(
        patient_history_json=patient_history_json,
        disease_name=disease_name,
    )


_VALID_TARGETS = {"rule_in", "rule_out", "differentiate"}


def _sanitise_questions(raw: list[dict]) -> list[dict]:
    """
    Drop any question dict that the LLM returned with garbled / non-English
    field names (e.g. '家喻户晓' instead of 'question').

    A valid question dict must have:
      - 'question'  : non-empty string
      - 'target'    : one of the accepted QuestionTarget values
    Missing 'clinical_rationale' is tolerated (schema default = "").
    """
    clean = []
    for item in raw:
        q_text = item.get("question", "")
        target  = item.get("target", "")
        if isinstance(q_text, str) and q_text.strip() and target in _VALID_TARGETS:
            clean.append({
                "question":           q_text.strip(),
                "clinical_rationale": item.get("clinical_rationale", ""),
                "target":             target,
            })
        else:
            logger.warning("Dropping malformed question from LLM output: %s", item)
    return clean


async def _generate_for_disease(
    patient_history: dict[str, Any],
    disease_name: str,
) -> dict[str, Any]:
    """
    Call the LLM once for a single disease.

    Returns a dict:
    {
        "disease": str,
        "questions": [...],   # list of question objects
        "error": str | None   # None on success
    }
    """
    user_prompt = _build_user_prompt(patient_history, disease_name)

    try:
        raw_response = await call_llm(
            system_prompt=QUESTION_GENIE_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=512,          # 3 questions fit easily in 512 tokens
            response_format="json_object",
        )

        # Sanitise before any Pydantic validation — drop garbled LLM keys
        questions: list[dict] = _saniltise_questions(
            raw_response.get("questions", [])
        )[:3]   # cap at 3 per disease
        if len(questions) < 3:
            logger.warning(
                "LLM returned only %d valid question(s) for '%s' — expected ≥3.",
                len(questions),
                disease_name,
            )

        return {
            "disease": disease_name,
            "questions": questions,
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to generate questions for '%s': %s", disease_name, exc)
        return {
            "disease": disease_name,
            "questions": [],
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Public async entry point
# ---------------------------------------------------------------------------


async def generate_questions_async(
    patient_history: dict[str, Any],
    candidate_diseases: list[str],
    max_concurrent: int = 5,   # parallel LLM calls; increase if rate limits allow
) -> dict[str, Any]:
    """
    Generate >=3 clinical questions for each candidate disease.

    Calls are made **in parallel** (up to *max_concurrent* at once) using
    asyncio.gather, which cuts wall-clock time from O(N) to O(1) relative
    to the number of diseases.

    Parameters
    ----------
    patient_history  : Structured patient data dict.
    candidate_diseases : List of disease names (strings).
    max_concurrent   : Max simultaneous LLM calls (soft rate-limit guard).

    Returns
    -------
    Structured result dict (see module docstring for schema).
    """
    if not candidate_diseases:
        return {
            "patient_history_snapshot": patient_history,
            "results": [],
        }

    # Semaphore prevents bursting more than max_concurrent calls at once
    sem = asyncio.Semaphore(max_concurrent)

    async def _guarded(disease: str) -> dict[str, Any]:
        async with sem:
            return await _generate_for_disease(patient_history, disease)

    results: list[dict[str, Any]] = await asyncio.gather(
        *[_guarded(d) for d in candidate_diseases]
    )

    return {
        "patient_history_snapshot": patient_history,
        "results": list(results),
    }


# ---------------------------------------------------------------------------
# Public sync entry point (convenience wrapper for Streamlit / scripts)
# ---------------------------------------------------------------------------


def generate_questions(
    patient_history: dict[str, Any],
    candidate_diseases: list[str],
) -> dict[str, Any]:
    """
    Blocking wrapper around `generate_questions_async`.

    Use this when you cannot use `await` (e.g. inside a Streamlit callback).

    Parameters
    ----------
    patient_history     : Structured patient data dict.
    candidate_diseases  : List of disease names (strings).

    Returns
    -------
    Structured result dict (see module docstring for schema).
    """
    return asyncio.run(generate_questions_async(patient_history, candidate_diseases))


# ---------------------------------------------------------------------------
# CLI / quick smoke-test (run with: python -m backend.questionGenie)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    _SAMPLE_HISTORY = {
        "symptoms": ["fatigue", "weight gain", "persistent sadness", "sluggishness"],
        "duration": "3 months",
        "severity": "moderate",
        "negated_symptoms": ["fever", "chest pain"],
        "risk_factors": ["female sex", "work stress"],
        "medications": [],
        "relevant_history": (
            "Patient is a 38-year-old female presenting with gradual onset of "
            "fatigue and low mood over the past 3 months. Denies fever or pain."
        ),
    }

    _CANDIDATE_DISEASES = ["Hypothyroidism", "Depression", "Anemia"]

    print("=" * 60)
    print("Question Genie — smoke test")
    print("=" * 60)

    output = generate_questions(_SAMPLE_HISTORY, _CANDIDATE_DISEASES)
    pprint.pprint(output, width=100, sort_dicts=False)
