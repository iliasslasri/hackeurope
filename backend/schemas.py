"""
schemas.py
----------
Pydantic models that define the shared JSON contract between the
Aura backend agents and the Streamlit frontend.

These are the single source of truth — the frontend renders whatever
shape these models produce; agents must always return data that
validates against them.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------


class SuspicionLevel(str, Enum):
    HIGH   = "High"
    MEDIUM = "Medium"
    LOW    = "Low"


class QuestionTarget(str, Enum):
    RULE_IN       = "rule_in"
    RULE_OUT      = "rule_out"
    DIFFERENTIATE = "differentiate"


# ---------------------------------------------------------------------------
# Information Extractor output
# ---------------------------------------------------------------------------


class PatientHistory(BaseModel):
    """Structured clinical data extracted from a transcript chunk."""

    symptoms: list[str] = Field(default_factory=list)
    duration: Optional[str] = None
    severity: Optional[str] = None
    negated_symptoms: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    relevant_history: Optional[str] = None

    # TODO update method

# ---------------------------------------------------------------------------
# Strategist output
# ---------------------------------------------------------------------------


class DDxEntry(BaseModel):
    """A single entry in the Differential Diagnosis list."""

    rank: int
    disease: str
    suspicion: SuspicionLevel
    key_supporting: list[str] = Field(default_factory=list)
    key_against: list[str] = Field(default_factory=list)


class StrategistOutput(BaseModel):
    """Full output from the Strategist agent."""

    ddx: list[DDxEntry]
    diagnostic_gap: str
    follow_up_question: str
    reasoning: str


# ---------------------------------------------------------------------------
# Safety Reviewer output
# ---------------------------------------------------------------------------


class SafetyReviewerOutput(BaseModel):
    """Output from the Safety Reviewer agent."""

    approved: bool
    issues: list[str] = Field(default_factory=list)
    revised_ddx: Optional[list[DDxEntry]] = None
    revised_follow_up_question: Optional[str] = None


# ---------------------------------------------------------------------------
# Question Genie output
# ---------------------------------------------------------------------------


class ClinicalQuestion(BaseModel):
    """One targeted question generated for a specific disease."""

    question: str
    clinical_rationale: str
    target: QuestionTarget


class DiseaseQuestions(BaseModel):
    """Questions generated for a single candidate disease."""

    disease: str
    questions: list[ClinicalQuestion] = Field(default_factory=list)
    error: Optional[str] = None   # populated only on LLM failure


class QuestionGenieOutput(BaseModel):
    """Full output from the Question Genie function."""

    patient_history_snapshot: PatientHistory
    results: list[DiseaseQuestions] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Complete UI payload (assembled by the orchestrator, consumed by the frontend)
# ---------------------------------------------------------------------------


class AuraUIPayload(BaseModel):
    """
    The single object sent to the Streamlit frontend on each transcript update.

    Frontend zones:
    - transcript_chunk  → Live Transcript panel
    - ddx               → DDx Tracker leaderboard
    - follow_up_question → Co-Pilot Prompter alert box
    - questions_by_disease → expandable per-disease Q cards
    """

    transcript_chunk: str = ""
    patient_history: PatientHistory = Field(default_factory=PatientHistory)
    ddx: list[DDxEntry] = Field(default_factory=list)
    follow_up_question: str = ""
    questions_by_disease: list[DiseaseQuestions] = Field(default_factory=list)
    safety_issues: list[str] = Field(default_factory=list)
    approved_by_safety_reviewer: bool = True
