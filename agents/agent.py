"""
agent.py
--------
Medical Diagnosis Agent — main entry point.

Usage
-----
  python agent.py --demo                    # three auto-answered demo cases
  python agent.py --symptoms "..." --anamnesis "..." [--qa] [--output report.md]
  python agent.py                           # interactive mode

Environment
-----------
  HF_TOKEN=<your_token>   enables PubMedBERT embeddings via HuggingFace API
                          (free account at huggingface.co, no GPU needed)
"""

from __future__ import annotations
import argparse
from typing import List, Optional

from agents.medical_dataset import load_datasets
from agents.scorer       import DiagnosisScorer, CandidateDiagnosis, _normalise
from agents.reporter     import render_console, render_markdown
from agents.updater      import SequentialUpdater


# ---------------------------------------------------------------------------
# MedicalDiagnosisAgent
# ---------------------------------------------------------------------------

class MedicalDiagnosisAgent:

    def __init__(self, verbose: bool = True, hf_token: str = ""):
        self.df, self.dataset_source = load_datasets()

        # Build semantic index over all symptom + RF phrases
        from agents.semantic_index import SemanticIndex
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
        return self.scorer.score(symptoms, anamnesis, top_k=top_k)

    def print_report(self, results: List[CandidateDiagnosis],
                     symptoms: str, anamnesis: str):
        print(render_console(results, symptoms, anamnesis, self.dataset_source))

    def save_markdown(self, results: List[CandidateDiagnosis],
                      symptoms: str, anamnesis: str, path: str):
        md = render_markdown(results, symptoms, anamnesis, self.dataset_source)
        with open(path, "w") as f:
            f.write(md)
        print(f"  [Agent] Saved → {path}")

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


# ---------------------------------------------------------------------------
# Demo cases
# ---------------------------------------------------------------------------

DEMO_CASES = [
    {
        "name": "Case 1 – Respiratory / COVID-like",
        "symptoms":  ("high fever, persistent dry cough, shortness of breath, "
                      "loss of taste, loss of smell, severe fatigue, headache, muscle aches"),
        "anamnesis": ("45-year-old male, obese BMI 32, hypertension, "
                      "no COVID-19 vaccine, non-smoker, office worker"),
        "qa_answers": {
            "runny nose":      "no, definitely no runny nose",
            "wheezing":        "not really, no wheeze",
            "rapid breathing": "yes constantly, it's quite severe",
            "chest pain":      "kind of, a mild tightness, and I've also been sweating a lot",
            "chills":          "yes absolutely, with the fever",
            "nausea":          "I had some nausea last week but it went away",
            "timeline":        "started about 5 days ago, came on quite suddenly",
            "lifestyle":       "I don't smoke, drink occasionally, mostly sedentary desk job",
            "family_history":  "no particular family history of lung disease",
        },
    },
    {
        "name": "Case 2 – Cardiac / Metabolic",
        "symptoms":  ("chest pain radiating to left arm, shortness of breath on exertion, "
                      "sweating, fatigue, occasional dizziness, palpitations"),
        "anamnesis": ("60-year-old male, heavy smoker 30 pack-years, type 2 diabetes, "
                      "high cholesterol, sedentary, family history of heart disease"),
        "qa_answers": {
            "nausea":           "sort of, mild nausea especially after meals",
            "rapid heart rate": "yes very much so, my heart races and I also get chest tightness",
            "radiating arm pain": "definitely, severe pain down my left arm",
            "cold hands and feet": "no, hands feel fine",
            "timeline":         "been building for 3 months, getting worse with exertion",
            "lifestyle":        "smoke a pack a day for 30 years, barely exercise, bad diet",
            "family_history":   "father had a heart attack at 58, brother has angina",
            "medication":       "taking metformin and a statin, no recent changes",
        },
    },
    {
        "name": "Case 3 – Sparse symptoms → watch ⚠️ flags resolve",
        "symptoms":  "fatigue, chest pain",
        "anamnesis": "50-year-old male, smoker",
        "qa_answers": {
            "sweating":         "yes, night sweats mostly, comes and goes, also shortness of breath on exertion",
            "palpitations":     "yes quite a lot, and sometimes dizziness as well",
            "nausea":           "no nausea at all",
            "radiating arm pain": "kind of, not strong but sometimes feel it in my shoulder",
            "rapid heart rate": "definitely, measured it, over 100 sometimes",
            "timeline":         "fatigue for 2 months, chest pain started 3 weeks ago",
            "lifestyle":        "smoke 15 per day, minimal exercise, work night shifts",
        },
    },
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(agent: MedicalDiagnosisAgent, symptoms: str, anamnesis: str,
        top_k: int, output: Optional[str], qa: bool = False,
        auto_answers: Optional[dict] = None):
    print("\n  Running initial diagnosis...")
    results = agent.diagnose(symptoms, anamnesis, top_k=top_k)
    agent.print_report(results, symptoms, anamnesis)
    if qa:
        results = agent.qa_session(results, auto_answers=auto_answers)
        print("\n  ══ Final ranking after interview ══")
        agent.print_report(results, symptoms, anamnesis)
    if output:
        agent.save_markdown(results, symptoms, anamnesis, output)


def main():
    parser = argparse.ArgumentParser(description="Medical Diagnosis Agent")
    parser.add_argument("--symptoms",  type=str)
    parser.add_argument("--anamnesis", type=str)
    parser.add_argument("--top_k",     type=int, default=10)
    parser.add_argument("--output",    type=str)
    parser.add_argument("--demo",      action="store_true")
    parser.add_argument("--qa",        action="store_true")
    parser.add_argument("--hf_token",  type=str, default="",
                        help="HuggingFace token for PubMedBERT embeddings")
    args = parser.parse_args()

    print("\n" + "="*72)
    print("  🏥  MEDICAL DIAGNOSIS AGENT")
    print("="*72)

    agent = MedicalDiagnosisAgent(verbose=True, hf_token=args.hf_token)

    if args.demo:
        for case in DEMO_CASES:
            print(f"\n\n{'#'*72}\n  {case['name']}\n{'#'*72}")
            run(agent, case["symptoms"], case["anamnesis"], args.top_k,
                args.output, qa=True, auto_answers=case["qa_answers"])
        return

    if args.symptoms and args.anamnesis:
        run(agent, args.symptoms, args.anamnesis, args.top_k, args.output, qa=args.qa)
        return

    # Interactive mode
    print("\n  Interactive mode — enter patient information below.")
    print("  (Press Enter twice to finish each field)\n")

    def read_block(prompt: str) -> str:
        print(prompt)
        lines = []
        while True:
            line = input()
            if line == "" and lines:
                break
            lines.append(line)
        return " ".join(l for l in lines if l)

    syms = read_block("Symptoms (free text):")
    anmn = read_block("Medical history / anamnesis:")
    run(agent, syms, anmn, args.top_k, args.output, qa=True)


if __name__ == "__main__":
    main()
