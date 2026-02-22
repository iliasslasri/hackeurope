from __future__ import annotations
from typing import List, Optional

from agents.medical_dataset import load_datasets
from agents.scorer          import DiagnosisScorer, CandidateDiagnosis, _normalise
from agents.updater         import SequentialUpdater
from agents.question_strategy import Question

class MedicalDiagnosisAgent:

    def __init__(self, verbose: bool = True, hf_token: str = ""):
        self.df, self.dataset_source = load_datasets()
        self.candidates: Optional[List[CandidateDiagnosis]] = None

        from agents.semantic_index import SemanticIndex
        all_phrases: List[str] = []
        for _, row in self.df.iterrows():
            all_phrases.extend(row["symptoms"])
            all_phrases.extend(row["risk_factors"])
        self.sem_index = SemanticIndex(list(set(all_phrases)),
                                       hf_token=hf_token, verbose=verbose)

        self.scorer = DiagnosisScorer(self.df, self.sem_index)

        self.profiles: dict = {
            row["disease"]: {
                "symptoms":     [_normalise(s) for s in row["symptoms"]],
                "risk_factors": [_normalise(r) for r in row["risk_factors"]],
            }
            for _, row in self.df.iterrows()
        }

        self.updater: Optional[SequentialUpdater] = None

        if verbose:
            print(f"  [Agent] {len(self.df)} diseases · {self.dataset_source}")

    # ------------------------------------------------------------------

    def diagnose(self, symptoms: str, anamnesis: str,
                 top_k: int = 10) -> List[CandidateDiagnosis]:
        """Initial scoring + initialise updater for Q&A session."""
        self.candidates = self.scorer.score(symptoms, anamnesis, top_k=top_k)
        self.updater = SequentialUpdater(
            self.candidates,
            self.profiles,
            sem_index=self.sem_index,
        )
        return self.candidates

    def get_candidates(self) -> Optional[List[CandidateDiagnosis]]:
        return self.candidates

    def update_scores(self, question: Question,
                      answer: str) -> List[CandidateDiagnosis]:
        """
        Update the differential given a Q&A pair.

        Target symptoms are extracted automatically from question.prompt
        via semantic similarity — no manual tagging required.

        Any question text works:
          "Do you have chest pain?"      → targets: [chest pain]
          "Any fever or chills?"         → targets: [fever, chills]
          "Tell me about your breathing" → targets: [shortness of breath, cough, ...]
          "When did this start?"         → targets: [] → answer scanned for mentions
        """
        if self.candidates is None:
            raise ValueError("No candidates — call diagnose() first.")
        if self.updater is None:
            raise ValueError("Updater not initialised — call diagnose() first.")

        _parsed, updates = self.updater.apply_answer(question, answer)

        if updates:
            ml_targets = updates[0].get("ml_targets", [])
            n_updated  = max(u["n_updated"] for u in updates)
            if ml_targets:
                print(f"  → targets extracted: {ml_targets}")
                print(f"  → {n_updated} disease(s) updated")

        self.candidates = self.updater.current_candidates(len(self.candidates))
        return self.candidates

    def next_question(self) -> Optional[Question]:
        """Return the next most informative question from the auto-strategy."""
        if self.updater is None:
            raise ValueError("Updater not initialised — call diagnose() first.")
        return self.updater.next_question()

