
from __future__ import annotations
from typing import List, Optional

from medical_dataset import load_datasets
from scorer       import DiagnosisScorer, CandidateDiagnosis, _normalise
from reporter     import render_console, render_markdown
from updater      import SequentialUpdater


# ---------------------------------------------------------------------------
# MedicalDiagnosisAgent
# ---------------------------------------------------------------------------

class MedicalDiagnosisAgent:

    def __init__(self, verbose: bool = True, hf_token: str = ""):
        self.df, self.dataset_source = load_datasets()
        self.candidates= None

        # Build semantic index over all symptom + RF phrases
        from semantic_index import SemanticIndex
        all_phrases: List[str] = []
        for _, row in self.df.iterrows():
            all_phrases.extend(row["symptoms"])
            all_phrases.extend(row["risk_factors"])
        self.sem_index = SemanticIndex(list(set(all_phrases)),
                                       hf_token=hf_token, verbose=verbose)

        self.scorer = DiagnosisScorer(self.df, self.sem_index)

        # Build disease profile dict for the updater
        self.profiles: dict = {
            row["disease"]: {
                "symptoms":     [_normalise(s) for s in row["symptoms"]],
                "risk_factors": [_normalise(r) for r in row["risk_factors"]],
            }
            for _, row in self.df.iterrows()
        }

        if verbose:
            print(f"  [Agent] {len(self.df)} diseases · {self.dataset_source}")

    # ------------------------------------------------------------------

    def diagnose(self, symptoms: str, anamnesis: str,
                 top_k: int = 10) -> List[CandidateDiagnosis]:
        self.candidates = self.scorer.score(symptoms, anamnesis, top_k=top_k)
        return self.candidates
    def get_candidates(self) -> Optional[List[CandidateDiagnosis]]:
        return self.candidates

    def print_report(self, results: List[CandidateDiagnosis],
                     symptoms: str, anamnesis: str):
        print(render_console(results, symptoms, anamnesis, self.dataset_source))

    def save_markdown(self, results: List[CandidateDiagnosis],
                      symptoms: str, anamnesis: str, path: str):
        md = render_markdown(results, symptoms, anamnesis, self.dataset_source)
        with open(path, "w") as f:
            f.write(md)
        print(f"  [Agent] Saved → {path}")
    
    def update_scores(self, question, answer) -> List[CandidateDiagnosis]:
        """Update candidate scores based on a new Q&A pair."""
        if self.candidates is None:
            raise ValueError("No initial candidates. Run diagnose() first.")
        updater = SequentialUpdater(self.candidates, self.profiles)
        parsed, updates = updater.apply_answer(question.target_symptom, answer)
        self.candidates = updater.current_candidates(len(self.candidates))
        return self.candidates

    # ------------------------------------------------------------------
    # Q&A session
    # ------------------------------------------------------------------

    def qa_session(self,
                   initial_candidates: List[CandidateDiagnosis],
                   max_questions: int = 10,
                   auto_answers: Optional[dict] = None) -> List[CandidateDiagnosis]:
        """
        Run an interactive Q&A session with open-ended clinical questions.
        Answers can be full free text; the parser extracts polarity, modifiers,
        and any extra symptoms mentioned.

        auto_answers : {symptom_key: answer_text} for demo/testing
        """
        updater = SequentialUpdater(initial_candidates, self.profiles)

        print("\n" + "─"*72)
        print("  🩺  CLINICAL INTERVIEW  —  Refining differential diagnosis")
        print("  Answer freely in natural language.  Type 'done' to finish early.")
        print("─"*72)

        for q_num in range(1, max_questions + 1):
            question = updater.next_question()
            if question is None:
                print("\n  [No more informative questions]")
                break

            # ── Print the question ──
            tag = f"[{question.question_type}]"
            print(f"\n  Q{q_num} {tag}")
            print(f"  ❓ {question.prompt}")

            # ── Get answer ──
            if auto_answers is not None:
                key = question.target_symptom or question.question_type
                raw = auto_answers.get(key, auto_answers.get(question.question_type, "skip"))
                print(f"  → [auto] \"{raw}\"")
            else:
                raw = input("  ▶ ").strip()
                if raw.lower() in ("done", "exit", "quit", "stop"):
                    break

            # ── Parse & update ──
            parsed, updates = updater.apply_answer(question.target_symptom, raw)

            # ── Show parse summary ──
            print(f"  📋 {parsed.explanation}")
            if parsed.extra_symptoms:
                print(f"  💡 Extra symptoms detected: {', '.join(parsed.extra_symptoms)}")

            # ── Show delta table for top candidates ──
            top5_diseases = {c.disease for c in updater.current_candidates(5)}
            moved = [u for u in updates
                     if u["disease"] in top5_diseases and (
                         abs(u["delta_prob_raw"]) > 0.0005 or u["delta_conf"] > 0.001)]
            moved.sort(key=lambda u: -abs(u["delta_prob_raw"]))

            if moved:
                print(f"\n  {'Disease':<30}  {'ΔProb':>9}  {'ΔConf':>7}  Profile  Extras")
                print("  " + "─"*66)
                for u in moved[:6]:
                    dp = u["delta_prob_raw"]; dc = u["delta_conf"]
                    ap = "↑" if dp > 0.0005 else "↓" if dp < -0.0005 else "·"
                    ac = "↑" if dc > 0.001 else "·"
                    pf = "✓" if u["in_profile"] else "—"
                    ex = (",".join(u["extras"])[:18]) if u["extras"] else ""
                    print(f"  {u['disease']:<30}  {ap}{dp:>+8.4f}  {ac}{dc:>+6.4f}"
                          f"  [{pf:^5}]  {ex}")

            # ── Mini ranking ──
            top5 = updater.current_candidates(5)
            print(f"\n  ── Top-5 after Q{q_num} ──")
            for i, c in enumerate(top5, 1):
                bp = "█" * min(round(c.probability*200), 16) + \
                     "░" * max(16 - round(c.probability*200), 0)
                bc = "█" * min(round(c.confidence*10), 10) + \
                     "░" * max(10 - round(c.confidence*10), 0)
                fl = " ⚠️" if c.risk_flag else ""
                print(f"  {i}. {c.disease:<32} "
                      f"P:[{bp}]{c.probability*100:5.1f}%  "
                      f"C:[{bc}]{c.confidence*100:5.1f}%{fl}")

        return updater.current_candidates(10)