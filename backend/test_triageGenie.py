"""
test_triageGenie.py
-------------------
Smoke-test and unit tests for the triageGenie module.

Run from the project root:
    python3 -m backend.test_triageGenie

Two modes:
  1.  Real API mode  – when CRUSOE_API_KEY is set in .env to a valid key.
  2.  Mock mode      – when the key is missing / placeholder; uses a local
                       stub to validate merge logic & schema without any
                       network call.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import logging
from unittest.mock import AsyncMock, patch

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(name)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INITIAL_PATIENT = {
    "symptoms": ["fatigue", "weight gain", "persistent sadness", "sluggishness"],
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

NEW_TRANSCRIPT = """
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

# What the LLM should ideally return for the above transcript
MOCK_LLM_RESPONSE = {
    "symptoms": [
        "fatigue",
        "weight gain",
        "persistent sadness",
        "sluggishness",
        "cold intolerance",
        "dry skin",
        "constipation",
    ],
    "duration": "3 months",
    "severity": "moderate",
    "negated_symptoms": ["fever", "chest pain"],
    "risk_factors": ["female sex", "work stress", "family history of thyroid disease"],
    "medications": ["levothyroxine 25 mcg daily"],
    "relevant_history": (
        "38-year-old female presenting with gradual onset of fatigue and low mood "
        "over the past 3 months. Denies fever or acute pain. Doctor suspects "
        "burnout or mild depression, but weight gain and sluggishness may point "
        "to Hypothyroidism. "
        "Follow-up: patient reports cold intolerance, dry skin, and constipation. "
        "Family history of thyroid disease confirmed. TSH mildly elevated. "
        "Levothyroxine 25 mcg/day started."
    ),
}


# ---------------------------------------------------------------------------
# Unit tests for the internal merge helper
# ---------------------------------------------------------------------------

def _run_unit_tests() -> bool:
    """
    Test _merge_patient_data directly — no LLM, no network.
    Returns True if all assertions pass.
    """
    from backend.schemas import PatientHistory
    from backend.triageGenie import _merge_patient_data   # type: ignore[attr-defined]

    print("── Unit tests: _merge_patient_data ────────────────────────────────")

    original = PatientHistory.model_validate(INITIAL_PATIENT)
    all_passed = True

    # --- Test 1: new symptoms are added, originals preserved ---
    llm_out_1 = {**MOCK_LLM_RESPONSE}
    result_1 = _merge_patient_data(original, llm_out_1)
    for sym in INITIAL_PATIENT["symptoms"]:
        if sym not in result_1.symptoms:
            print(f"  ❌ FAIL T1: original symptom '{sym}' was lost")
            all_passed = False
    for sym in ["cold intolerance", "dry skin", "constipation"]:
        if sym not in result_1.symptoms:
            print(f"  ❌ FAIL T1: new symptom '{sym}' not added")
            all_passed = False
    if all_passed:
        print(f"  ✅  T1 PASS — symptoms merged correctly ({len(result_1.symptoms)} total)")

    # --- Test 2: medications are updated ---
    passed_t2 = "levothyroxine 25 mcg daily" in result_1.medications
    if passed_t2:
        print(f"  ✅  T2 PASS — medication added: {result_1.medications}")
    else:
        print(f"  ❌ FAIL T2 — expected levothyroxine in medications, got: {result_1.medications}")
        all_passed = False

    # --- Test 3: risk_factors extended ---
    passed_t3 = "family history of thyroid disease" in result_1.risk_factors
    if passed_t3:
        print(f"  ✅  T3 PASS — risk factor added: {result_1.risk_factors}")
    else:
        print(f"  ❌ FAIL T3 — expected family history in risk_factors, got: {result_1.risk_factors}")
        all_passed = False

    # --- Test 4: negated_symptoms preserved ---
    for ns in INITIAL_PATIENT["negated_symptoms"]:
        if ns not in result_1.negated_symptoms:
            print(f"  ❌ FAIL T4: negated symptom '{ns}' was lost")
            all_passed = False
    if all(ns in result_1.negated_symptoms for ns in INITIAL_PATIENT["negated_symptoms"]):
        print(f"  ✅  T4 PASS — negated_symptoms preserved: {result_1.negated_symptoms}")

    # --- Test 5: LLM returning empty lists doesn't wipe original data ---
    llm_out_empty = {
        "symptoms": [],
        "negated_symptoms": [],
        "risk_factors": [],
        "medications": [],
        "duration": None,
        "severity": None,
        "relevant_history": None,
    }
    result_empty = _merge_patient_data(original, llm_out_empty)
    if result_empty.symptoms == original.symptoms and result_empty.duration == original.duration:
        print("  ✅  T5 PASS — empty LLM output does not wipe original data")
    else:
        print("  ❌ FAIL T5 — original data was lost on empty LLM response")
        all_passed = False

    # --- Test 6: no duplicates even if LLM repeats existing data ---
    llm_out_dup = {**MOCK_LLM_RESPONSE, "symptoms": ["fatigue", "fatigue", "cold intolerance"]}
    result_dup = _merge_patient_data(original, llm_out_dup)
    symptom_counts = {s: result_dup.symptoms.count(s) for s in result_dup.symptoms}
    duplicates = {s: c for s, c in symptom_counts.items() if c > 1}
    if not duplicates:
        print("  ✅  T6 PASS — no duplicate symptoms in merged list")
    else:
        print(f"  ❌ FAIL T6 — duplicates found: {duplicates}")
        all_passed = False

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Integration test: mock the LLM call, run full async flow
# ---------------------------------------------------------------------------

async def _run_mock_integration() -> bool:
    """
    Patches call_llm so no real HTTP request is made.
    Verifies the full update_patient_async pipeline end-to-end.
    """
    from backend.schemas import PatientHistory
    from backend.triageGenie import update_patient_async

    print("── Integration test: update_patient_async (mocked LLM) ────────────")

    original = PatientHistory.model_validate(INITIAL_PATIENT)
    all_passed = True

    with patch("backend.triageGenie.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = MOCK_LLM_RESPONSE
        updated = await update_patient_async(original, NEW_TRANSCRIPT)

    # Verify call_llm was called exactly once
    if mock_llm.call_count == 1:
        print("  ✅  LLM called exactly once")
    else:
        print(f"  ❌ LLM called {mock_llm.call_count} times (expected 1)")
        all_passed = False

    # Check schema is valid PatientHistory
    if isinstance(updated, PatientHistory):
        print("  ✅  Return type is PatientHistory")
    else:
        print(f"  ❌ Wrong return type: {type(updated)}")
        all_passed = False

    # New symptoms present
    for sym in ["cold intolerance", "dry skin", "constipation"]:
        if sym in updated.symptoms:
            print(f"  ✅  Symptom '{sym}' present after update")
        else:
            print(f"  ❌ Symptom '{sym}' missing after update")
            all_passed = False

    # Medication added
    if any("levothyroxine" in m for m in updated.medications):
        print(f"  ✅  Medication updated: {updated.medications}")
    else:
        print(f"  ❌ Medication not updated: {updated.medications}")
        all_passed = False

    # relevant_history is non-empty and longer than the original
    orig_len = len(INITIAL_PATIENT["relevant_history"])
    if updated.relevant_history and len(updated.relevant_history) >= orig_len:
        print(f"  ✅  relevant_history updated ({len(updated.relevant_history)} chars)")
    else:
        print(f"  ❌ relevant_history not properly updated: {updated.relevant_history!r}")
        all_passed = False

    # Empty transcript → patient returned unchanged
    unchanged = await update_patient_async(original, "   ")
    if unchanged == original:
        print("  ✅  Empty transcript returns patient unchanged")
    else:
        print("  ❌ Empty transcript changed the patient unexpectedly")
        all_passed = False

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Real-API integration test
# ---------------------------------------------------------------------------

async def _run_real_api() -> bool:
    """Calls the actual Crusoe LLM and prints the diff."""
    from backend.schemas import PatientHistory
    from backend.triageGenie import update_patient_async

    original = PatientHistory.model_validate(INITIAL_PATIENT)
    print("🌐  Calling Crusoe API …\n")
    updated = await update_patient_async(original, NEW_TRANSCRIPT)

    print("  ── Original patient:")
    _print_patient(original)
    print("\n  ── Updated patient:")
    _print_patient(updated)

    # Simple diff
    new_symptoms = [s for s in updated.symptoms if s not in original.symptoms]
    new_risks = [r for r in updated.risk_factors if r not in original.risk_factors]
    new_meds = [m for m in updated.medications if m not in original.medications]
    print("\n  ── Delta:")
    print(f"    ➕ New symptoms      : {new_symptoms or '(none)'}")
    print(f"    ➕ New risk factors  : {new_risks or '(none)'}")
    print(f"    ➕ New medications   : {new_meds or '(none)'}")
    print()
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_patient(p) -> None:
    data = p.model_dump()
    for key, val in data.items():
        if isinstance(val, list):
            print(f"    {key}: {val}")
        else:
            preview = str(val)[:120] + ("…" if val and len(str(val)) > 120 else "")
            print(f"    {key}: {preview}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.getenv("CRUSOE_API_KEY", "")
    use_mock = not api_key or "your_crusoe" in api_key.lower()

    print("=" * 70)
    print("  Aura — triageGenie test suite")
    print("=" * 70)
    print(f"  Mode : {'🔴 MOCK (no valid API key set)' if use_mock else '🟢 REAL API'}")
    print("=" * 70)
    print()

    overall_pass = True

    # Unit tests (always run — no network needed)
    overall_pass &= _run_unit_tests()

    # Integration / real-API
    if use_mock:
        print("ℹ️  Running integration test in MOCK mode (LLM patched).")
        print("   Set CRUSOE_API_KEY in .env to test against the real LLM.\n")
        overall_pass &= asyncio.run(_run_mock_integration())
    else:
        overall_pass &= asyncio.run(_run_real_api())

    # Final verdict
    print("=" * 70)
    if overall_pass:
        print("✅  All tests PASSED")
    else:
        print("❌  Some tests FAILED — see output above")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
