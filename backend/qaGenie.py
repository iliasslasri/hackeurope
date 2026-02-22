"""
qaGenie.py
----------
Extracts medical question-answer pairs from a doctor-patient transcript.

If the transcript contains explicit clinical questions with direct patient answers,
those pairs are returned. If no such pairs exist, an empty list is returned.

Public API
----------
    extract_qa(transcript: str) -> list[dict]
    extract_qa_async(transcript: str) -> list[dict]

Returns
-------
list of dicts, each with keys:
    {
        "question": "<the medical question>",
        "answer":   "<the patient's answer>"
    }
Returns [] if no Q&A pairs are found.
"""

from __future__ import annotations

import asyncio
import logging

from .llm_client import call_llm
from .prompts import QA_GENIE_SYSTEM, QA_GENIE_USER_TEMPLATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public async entry point
# ---------------------------------------------------------------------------


async def extract_qa_async(transcript: str) -> list[dict]:
    """
    Analyse *transcript* and return all medical Q&A pairs found within it.

    Parameters
    ----------
    transcript : str
        Raw doctor-patient conversation text.

    Returns
    -------
    list[dict]  — each dict has ``"question"`` and ``"answer"`` keys.
                  Empty list if no pairs are found or the transcript is empty.
    """
    if not transcript or not transcript.strip():
        return []

    user_prompt = QA_GENIE_USER_TEMPLATE.format(transcript=transcript)

    try:
        raw = await call_llm(
            system_prompt=QA_GENIE_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.1,        # faithful extraction, no creativity
            max_tokens=1024,
            response_format="json_object",
        )
        pairs: list[dict] = raw.get("qa_pairs", [])

        # Sanitise: keep only dicts with non-empty question AND answer strings
        clean = [
            {"question": p["question"].strip(), "answer": p["answer"].strip()}
            for p in pairs
            if isinstance(p.get("question"), str) and p["question"].strip()
            and isinstance(p.get("answer"), str)  and p["answer"].strip()
        ]
        logger.info("qaGenie extracted %d Q&A pair(s) from transcript.", len(clean))
        return clean

    except Exception as exc:  # noqa: BLE001
        logger.error("qaGenie LLM call failed: %s — returning empty list.", exc)
        return []


# ---------------------------------------------------------------------------
# Public sync entry point (convenience wrapper)
# ---------------------------------------------------------------------------


def extract_qa(transcript: str) -> list[dict]:
    """
    Blocking wrapper around :func:`extract_qa_async`.

    Parameters
    ----------
    transcript : str
        Raw doctor-patient conversation text.

    Returns
    -------
    list[dict]  — each dict has ``"question"`` and ``"answer"`` keys.
    """
    return asyncio.run(extract_qa_async(transcript))


# ---------------------------------------------------------------------------
# CLI / quick smoke-test (run with: python -m backend.qaGenie)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    _SAMPLE = """
Doctor: How long have you been experiencing these headaches?
Patient: About two weeks now, mostly in the mornings.

Doctor: Have you had any nausea or vomiting with them?
Patient: Yes, I've felt nauseous a few times, but no vomiting.

Doctor: Any recent changes to your sleep or stress levels?
Patient: Work has been very stressful lately and I haven't been sleeping well.

Doctor: Great, let's get you checked in.
Patient: Thank you.
"""

    print("=" * 60)
    print("QA Genie — smoke test")
    print("=" * 60)
    result = extract_qa(_SAMPLE)
    pprint.pprint(result, width=100)
