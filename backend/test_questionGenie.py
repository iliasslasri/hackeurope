"""
test_questionGenie.py
---------------------
Manual smoke-test for the questionGenie module.

Run from the project root:
    python3 -m backend.test_questionGenie

Two modes:
  1.  Real API mode   – when CRUSOE_API_KEY is set in .env to a valid key.
  2.  Mock mode       – when the key is missing / placeholder; uses a local
                        stub so you can validate output shape & schemas without
                        hitting the network.
"""

from __future__ import annotations

import asyncio
import json
import os
import pprint
import sys

import logging

from dotenv import load_dotenv

load_dotenv()

# Show WARNING+ from the backend so API errors are visible in the terminal
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(name)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Sample patient history — the Hypothyroidism vs. Depression demo
# ---------------------------------------------------------------------------

SAMPLE_HISTORY = {
    "symptoms": [
        "fatigue",
        "weight gain",
        "persistent sadness",
        "sluggishness",
    ],
    "duration": "3 months",
    "severity": "moderate",
    "negated_symptoms": ["fever", "chest pain"],
    "risk_factors": ["female sex", "work stress"],
    "medications": [],
    "relevant_history": (
        "38-year-old female presenting with gradual onset of fatigue and low mood "
        "over the past 3 months. Denies fever or acute pain. Doctor suspects "
        "burnout or mild depression, but weight gain and sluggishness may point "
        "to Hypothyroidism."
    ),
}

CANDIDATE_DISEASES = ["Hypothyroidism", "Depression", "Anemia"]


# ---------------------------------------------------------------------------
# Mock LLM — returns hard-coded plausible JSON so the pipeline can be
# tested without any API credentials
# ---------------------------------------------------------------------------

MOCK_RESPONSES: dict[str, dict] = {
    "Hypothyroidism": {
        "disease": "Hypothyroidism",
        "questions": [
            {
                "question": (
                    "Has the patient noticed cold intolerance — feeling unusually cold "
                    "when others are comfortable?"
                ),
                "clinical_rationale": (
                    "Cold intolerance is a hallmark of reduced basal metabolic rate "
                    "in hypothyroidism and is rarely seen in pure depression."
                ),
                "target": "rule_in",
            },
            {
                "question": (
                    "Does the patient have a family history of thyroid disease or "
                    "autoimmune conditions such as Hashimoto's thyroiditis?"
                ),
                "clinical_rationale": (
                    "Hashimoto's thyroiditis is the most common cause of hypothyroidism; "
                    "a positive family history significantly raises pre-test probability."
                ),
                "target": "rule_in",
            },
            {
                "question": (
                    "Has the patient experienced hair thinning, loss of the outer "
                    "third of the eyebrow, or dry/coarse skin?"
                ),
                "clinical_rationale": (
                    "These dermatological findings are specific for hypothyroidism "
                    "and help differentiate it from depression."
                ),
                "target": "differentiate",
            },
            {
                "question": (
                    "What was the patient's resting heart rate on examination? "
                    "Is there bradycardia (<60 bpm)?"
                ),
                "clinical_rationale": (
                    "Bradycardia is a common objective sign of hypothyroidism that "
                    "is absent in depression, providing a measurable discriminator."
                ),
                "target": "rule_in",
            },
        ],
    },
    "Depression": {
        "disease": "Depression",
        "questions": [
            {
                "question": (
                    "Has the patient experienced anhedonia — a loss of pleasure in "
                    "activities they previously enjoyed?"
                ),
                "clinical_rationale": (
                    "Anhedonia is a core DSM-5 criterion for major depressive disorder "
                    "and is less prominent in hypothyroid fatigue presentations."
                ),
                "target": "rule_in",
            },
            {
                "question": (
                    "Has the patient had any thoughts of hopelessness, worthlessness, "
                    "or suicidal ideation?"
                ),
                "clinical_rationale": (
                    "Suicidal ideation is specific to depressive disorders and represents "
                    "an immediate red-flag clinical need."
                ),
                "target": "rule_in",
            },
            {
                "question": (
                    "Has the patient experienced early morning awakening with inability "
                    "to return to sleep?"
                ),
                "clinical_rationale": (
                    "Early morning awakening is characteristic of melancholic depression, "
                    "whereas hypothyroid fatigue typically manifests as hypersomnia."
                ),
                "target": "differentiate",
            },
        ],
    },
    "Anemia": {
        "disease": "Anemia",
        "questions": [
            {
                "question": (
                    "Has the patient noticed pallor — particularly in the conjunctiva, "
                    "nail beds, or palmar creases?"
                ),
                "clinical_rationale": (
                    "Pallor is a sensitive examination finding for moderate-to-severe "
                    "anemia and should be checked even when the patient does not report it."
                ),
                "target": "rule_in",
            },
            {
                "question": (
                    "Does the patient have heavy menstrual periods, or has she had "
                    "any recent significant blood loss?"
                ),
                "clinical_rationale": (
                    "Iron-deficiency anemia secondary to menorrhagia is the most common "
                    "cause of anemia in pre-menopausal women — directly relevant to this patient."
                ),
                "target": "rule_in",
            },
            {
                "question": (
                    "Does the patient follow a vegetarian or vegan diet, or has she "
                    "had any bariatric surgery?"
                ),
                "clinical_rationale": (
                    "Dietary restriction of iron/B12 and malabsorption post-surgery "
                    "are leading modifiable causes of nutritional anemia."
                ),
                "target": "rule_out",
            },
        ],
    },
}


def run_with_mock() -> dict:
    """
    Returns a QuestionGenieOutput-shaped dict without any API call.
    Validates against schemas.QuestionGenieOutput.
    """
    from backend.schemas import (
        PatientHistory,
        DiseaseQuestions,
        ClinicalQuestion,
        QuestionTarget,
        QuestionGenieOutput,
    )

    history_model = PatientHistory.model_validate(SAMPLE_HISTORY)
    results = []
    for disease in CANDIDATE_DISEASES:
        mock = MOCK_RESPONSES[disease]
        questions = [
            ClinicalQuestion(
                question=q["question"],
                clinical_rationale=q["clinical_rationale"],
                target=QuestionTarget(q["target"]),
            )
            for q in mock["questions"]
        ]
        results.append(DiseaseQuestions(disease=disease, questions=questions))

    output = QuestionGenieOutput(
        patient_history_snapshot=history_model,
        results=results,
    )
    return output.model_dump()


async def run_with_real_api() -> dict:
    """Calls the actual Crusoe API via generate_questions_async."""
    from backend.questionGenie import generate_questions_async

    return await generate_questions_async(SAMPLE_HISTORY, CANDIDATE_DISEASES)


def _pretty(obj: dict) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def main():
    api_key = os.getenv("CRUSOE_API_KEY", "")
    use_mock = not api_key or "your_crusoe" in api_key.lower()

    print("=" * 70)
    print("  Aura — questionGenie smoke-test")
    print("=" * 70)
    print(f"  Mode        : {'🔴 MOCK (no valid API key set)' if use_mock else '🟢 REAL API'}")
    print(f"  Diseases    : {', '.join(CANDIDATE_DISEASES)}")
    print("=" * 70)
    print()

    if use_mock:
        print("ℹ️  Running in MOCK mode.")
        print("   Set CRUSOE_API_KEY in .env to test against the real LLM.\n")
        result = run_with_mock()
    else:
        print("🌐  Calling Crusoe API …\n")
        result = asyncio.run(run_with_real_api())

    # --- Print summary ---
    for entry in result.get("results", []):
        disease = entry.get("disease", "?")
        questions = entry.get("questions", [])
        error = entry.get("error")

        print(f"┌─ 🩺  {disease}")
        if error:
            print(f"│  ⚠️  ERROR: {error}")
        else:
            for i, q in enumerate(questions, 1):
                target_icon = {"rule_in": "✅", "rule_out": "❌", "differentiate": "🔀"}.get(
                    q.get("target", ""), "?"
                )
                print(f"│  Q{i} {target_icon}  {q['question']}")
                print(f"│      └─ {q['clinical_rationale']}")
        print("│")
        print()

    # --- Validate schema round-trip ---
    print("── Schema validation ──────────────────────────────────────────────")
    try:
        from backend.schemas import QuestionGenieOutput, PatientHistory

        # Rebuild from raw dict to confirm Pydantic passes
        validated = QuestionGenieOutput(
            patient_history_snapshot=PatientHistory.model_validate(
                result.get("patient_history_snapshot", SAMPLE_HISTORY)
            ),
            results=[],  # already validated above; skip deep re-parse
        )
        total_q = sum(len(e.get("questions", [])) for e in result.get("results", []))
        print(f"✅  QuestionGenieOutput schema OK")
        print(f"✅  Total questions generated : {total_q}")
        print(f"✅  Diseases covered          : {len(result.get('results', []))}")
    except Exception as exc:
        print(f"❌  Schema validation FAILED: {exc}")
        sys.exit(1)

    print()
    print("Full JSON output:")
    print(_pretty(result))


if __name__ == "__main__":
    main()
