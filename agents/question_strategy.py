"""
question_strategy.py
--------------------
Generates open-ended clinical questions, not just "do you have X?".

QUESTION TYPES
==============
  symptom_probe    -- ask about a new symptom not yet explored
  characterize     -- "describe the pain: sharp, dull, burning, pressure?"
  severity         -- "on a scale of 1-10 how bad is the [symptom]?"
  location         -- "where exactly? does it spread anywhere?"
  timeline         -- "when did this start? sudden or gradual?"
  triggers         -- "what makes it better or worse?"
  associated       -- "anything else accompanying the [symptom]?"
  family_history   -- "any family history of [relevant condition]?"
  lifestyle        -- "do you smoke? alcohol? exercise?"
  medication       -- "any medications or recent changes?"
  context          -- "travel recently? contact with sick people?"
  systems_review   -- "any changes in weight, sleep, appetite, bowel?"

SELECTION LOGIC
===============
Each turn the strategy scores all candidate questions and returns the
highest-value one, avoiding repetition of already-asked topics.

Priority:
  1. High-prob/low-conf disease  -> questions that build confidence fast
  2. Most discriminating symptom -> separates top-3 candidates
  3. Confirmed symptom drill     -> characterize / severity / location
  4. Missing timeline            -> ask once early
  5. Key risk factors            -> family history / lifestyle / context
  6. Systems review              -> catch anything broad
"""

from __future__ import annotations
import random, numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from updater import DiseaseState

W_SYMPTOMS = 0.65
W_RISK     = 0.20
W_PRIOR    = 0.15

# ---------------------------------------------------------------------------
# Question dataclass
# ---------------------------------------------------------------------------

@dataclass
class Question:
    prompt:         str            # the text shown to the clinician/patient
    target_symptom: Optional[str]  # symptom to update in Bayesian model (None = context Q)
    question_type:  str
    info_score:     float = 0.0

    def __str__(self) -> str:
        return self.prompt


# ---------------------------------------------------------------------------
# Question templates (randomised per call to avoid robotic repetition)
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, List[str]] = {
    "symptom_probe": [
        "Have you been experiencing {s}?",
        "Have you noticed any {s}?",
        "Any {s} recently?",
        "Do you have or have you had {s}?",
    ],
    "characterize": [
        "Can you describe the {s} in more detail? For example — is it sharp, dull, burning, a pressure, or something else?",
        "How would you describe the {s} — constant or comes and goes? Sharp or dull?",
        "Tell me more about the {s}: what does it feel like exactly?",
        "What quality does the {s} have — throbbing, tight, cramping, aching, burning?",
    ],
    "severity": [
        "On a scale of 1 to 10, how severe is the {s}?",
        "How bad is the {s} — mild, moderate, or severe?",
        "Does the {s} stop you from doing normal activities?",
    ],
    "location": [
        "Where exactly do you feel the {s}? Does it spread or radiate anywhere?",
        "Can you point to where the {s} is worst? Does it move?",
        "Is the {s} in one specific area, or does it cover a wider region?",
    ],
    "timeline": [
        "When did these symptoms first start? Did they come on suddenly or gradually?",
        "How long have you been feeling this way, and has it been getting better, worse, or staying the same?",
        "Did anything happen — an illness, stress, injury, or diet change — around the time the symptoms began?",
    ],
    "triggers": [
        "Does anything make the {s} better or worse — such as exercise, eating, rest, lying down, or stress?",
        "Have you noticed any pattern or triggers for the {s}?",
        "Does the {s} change with activity, food, position, or time of day?",
    ],
    "associated": [
        "Along with the {s}, have you noticed anything else — like fever, sweating, nausea, or shortness of breath?",
        "Does the {s} come with any other symptoms you haven't mentioned yet?",
        "Any other symptoms that seem to accompany the {s}?",
    ],
    "family_history": [
        "Has anyone in your immediate family — parents, siblings — been diagnosed with {c}?",
        "Is there a family history of {c} on either side?",
        "Do you know of any relatives who've had {c} or similar conditions?",
    ],
    "lifestyle": [
        "Can you tell me about your lifestyle — do you smoke or drink alcohol, and how active are you?",
        "How would you describe your diet and exercise habits?",
        "Do you smoke? How much alcohol do you drink per week? Do you exercise regularly?",
    ],
    "medication": [
        "Are you currently taking any medications, supplements, or over-the-counter drugs?",
        "Have there been any recent changes to your medications or treatments?",
        "Any regular prescriptions, or anything you've started or stopped recently?",
    ],
    "context": [
        "Have you been in contact with anyone who was unwell recently, or travelled anywhere?",
        "Any recent travel, unusual food, or exposure to sick people?",
        "Have you been in any new environments — hospitals, crowded places, or abroad recently?",
    ],
    "systems_review": [
        "Have you noticed any unintentional weight changes, changes in sleep, or changes in appetite?",
        "Any changes in your bowel habits, urination, or vision lately?",
        "Anything else going on generally — fatigue, night sweats, or mood changes?",
    ],
}

def _t(qtype: str, **kwargs) -> str:
    tmpl = random.choice(TEMPLATES.get(qtype, ["{s}"]))
    try:
        return tmpl.format(**kwargs)
    except KeyError:
        return tmpl


# ---------------------------------------------------------------------------
# QuestionStrategy
# ---------------------------------------------------------------------------

class QuestionStrategy:
    """
    Selects the next best clinical question given the current diagnostic state.

    Parameters
    ----------
    states          : disease -> DiseaseState  (from SequentialUpdater)
    idf             : symptom specificity weights
    asked_symptoms  : set of symptom phrases already queried
    confirmed_syms  : symptoms the patient confirmed (for drill-down Qs)
    """

    def __init__(self,
                 states: Dict[str, "DiseaseState"],
                 idf: Dict[str, float],
                 asked_symptoms: Set[str],
                 confirmed_syms: Set[str]):
        self._states    = states
        self._idf       = idf
        self._asked     = asked_symptoms
        self._confirmed = confirmed_syms
        self._type_count: Dict[str, int] = {}
        self._turn = 0

    def next_question(self) -> Optional[Question]:
        self._turn += 1
        tops = self._top_states(5)
        if not tops:
            return None

        sym_q     = self._best_symptom_q(tops)
        sym_score = sym_q.info_score if sym_q else 1.0

        # Determine what non-symptom question is due this turn
        context_q: Optional[tuple[float, Question]] = None

        # Timeline: ask on turn 2 (after first symptom probe)
        if self._type_count.get("timeline", 0) == 0 and self._turn == 2:
            context_q = (sym_score + 10, Question(
                prompt=_t("timeline"), target_symptom=None,
                question_type="timeline"))

        # Drill confirmed symptom on even turns (characterize → severity → triggers → location)
        elif self._confirmed and self._turn >= 3 and self._turn % 3 == 0:
            sym = next(iter(self._confirmed))
            for qtype in ["characterize", "severity", "triggers", "location"]:
                if self._type_count.get(qtype, 0) < 1:
                    context_q = (sym_score + 10, Question(
                        prompt=_t(qtype, s=sym), target_symptom=sym,
                        question_type=qtype))
                    break

        # Risk factor / family history on turn 4
        elif self._turn == 4 and tops:
            q = self._rf_question(tops[0])
            if q:
                context_q = (sym_score + 10, q)

        # Lifestyle on turn 6
        elif self._turn == 6 and self._type_count.get("lifestyle", 0) == 0:
            context_q = (sym_score + 10, Question(
                prompt=_t("lifestyle"), target_symptom=None,
                question_type="lifestyle"))

        # Associated on turn 7
        elif self._turn == 7 and self._confirmed and self._type_count.get("associated", 0) == 0:
            sym = next(iter(self._confirmed))
            context_q = (sym_score + 10, Question(
                prompt=_t("associated", s=sym), target_symptom=None,
                question_type="associated"))

        # Systems review on turn 9
        elif self._turn == 9 and self._type_count.get("systems_review", 0) == 0:
            context_q = (sym_score + 10, Question(
                prompt=_t("systems_review"), target_symptom=None,
                question_type="systems_review"))

        # HIGH-PROB/LOW-CONF override: only override context questions if one disease
        # is very dominant AND very low confidence (strong clinical urgency)
        for st in tops[:1]:  # only check the TOP candidate
            p = self._prob(st); c = st.evidence_n / (st.evidence_n + 6.0)
            if p >= 0.25 and c < 0.35:   # very dominant + very low confidence
                context_q = None          # force symptom probe to build confidence
                break

        scored: List[tuple[float, Question]] = []
        if context_q:
            scored.append(context_q)
        if sym_q:
            scored.append((sym_score, sym_q))
        # Also add rf question as fallback
        if tops and not context_q:
            q = self._rf_question(tops[0])
            if q:
                scored.append((sym_score * 0.9, q))

        if not scored:
            return None

        scored.sort(key=lambda x: -x[0])
        _, best = scored[0]
        self._type_count[best.question_type] = self._type_count.get(best.question_type, 0) + 1
        if best.target_symptom:
            self._asked.add(best.target_symptom)
        return best

    # ------------------------------------------------------------------

    def _top_states(self, k: int) -> List["DiseaseState"]:
        return sorted(self._states.values(), key=self._prob, reverse=True)[:k]

    def _prob(self, st: "DiseaseState") -> float:
        sm = st.sym_alpha / (st.sym_alpha + st.sym_beta)
        rm = st.rf_alpha  / (st.rf_alpha  + st.rf_beta)
        return W_SYMPTOMS * sm + W_RISK * rm + W_PRIOR * st.prior

    def _best_symptom_q(self, tops: List["DiseaseState"],
                         boost: float = 1.0) -> Optional[Question]:
        all_syms: Set[str] = set()
        for st in tops: all_syms.update(st.sym_profile)

        best_score, best_sym = -1.0, None
        for sym in all_syms:
            if sym in self._asked: continue
            idf  = self._idf.get(sym, 1.0)
            frac = sum(sym in st.sym_profile for st in tops) / len(tops)
            if frac == 0: continue
            disc = -(frac * np.log(frac+1e-9) + (1-frac) * np.log(1-frac+1e-9))
            sc   = idf * disc * boost
            if sc > best_score: best_score, best_sym = sc, sym

        if best_sym is None: return None

        if best_sym in self._confirmed:
            qtype = random.choice(["characterize", "severity", "triggers"])
        else:
            qtype = "symptom_probe"

        return Question(prompt=_t(qtype, s=best_sym), target_symptom=best_sym,
                        question_type=qtype, info_score=best_score)

    def _rf_question(self, st: "DiseaseState") -> Optional[Question]:
        for rf in st.rf_profile:
            if rf in self._asked: continue
            rfl = rf.lower()
            if "family history" in rfl:
                cond = rfl.replace("family history of", "").strip()
                if self._type_count.get("family_history", 0) < 2:
                    return Question(prompt=_t("family_history", c=cond),
                                    target_symptom=rf, question_type="family_history",
                                    info_score=1.1)
            elif any(w in rfl for w in ("smok","alcohol","sedentary","exercise","diet","obese","bmi")):
                if self._type_count.get("lifestyle", 0) == 0:
                    return Question(prompt=_t("lifestyle"), target_symptom=rf,
                                    question_type="lifestyle", info_score=1.0)
            elif any(w in rfl for w in ("medic","drug","pill","therapy","treatment")):
                if self._type_count.get("medication", 0) == 0:
                    return Question(prompt=_t("medication"), target_symptom=rf,
                                    question_type="medication", info_score=0.9)
            elif any(w in rfl for w in ("contact","travel","food","water","exposure")):
                if self._type_count.get("context", 0) == 0:
                    return Question(prompt=_t("context"), target_symptom=rf,
                                    question_type="context", info_score=0.9)
            else:
                return Question(prompt=_t("symptom_probe", s=rf), target_symptom=rf,
                                question_type="symptom_probe", info_score=0.85)
        return None
