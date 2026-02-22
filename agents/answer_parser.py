"""
answer_parser.py
----------------
Parses free-text patient answers into structured signals for Bayesian updating.

A patient doesn't say "yes" — they say things like:
  "yes but only when I exercise"
  "kind of, it started 3 weeks ago and is getting worse"
  "no, not really, maybe a tiny bit when I lie down"
  "I had it last month but it went away"
  "my chest hurts sometimes, and I've also been sweating a lot at night"

This module handles all of that and outputs a ParsedAnswer with:

  polarity       : the core yes/no/mild/unsure signal (maps to ANSWER_WEIGHTS)
  strength       : float [0, 1] — how strongly to weight the Bayesian update
  temporal       : "current" | "recent" | "past" | "intermittent" | "unknown"
  severity       : "severe" | "moderate" | "mild" | "trace" | "unknown"
  extra_symptoms : list of symptom phrases mentioned incidentally in the answer
  raw            : the original text
  explanation    : human-readable summary of what was parsed


DESIGN PRINCIPLES
=================
1. Rules-first, no ML required. Pure regex + pattern matching so it works
   with no external dependencies and is fully explainable.

2. Modifiers attenuate or amplify the base polarity weight:
   - temporal:  past events are weighted lower (symptom may have resolved)
   - severity:  mild/trace reduce the alpha weight; severe amplifies it
   - frequency: intermittent reduces alpha weight

3. Extra symptoms mentioned in the answer are extracted and fed back into
   the updater as additional observations — this is the big win from free text.
   "yes, and I've also been having night sweats and joint pain" extracts two
   new symptoms that weren't explicitly asked about.

4. Negation is handled carefully — "no, not really" is still "no", but
   "I don't NOT have it" should map to uncertain/mild. We use a simple
   negation stack rather than trying to parse full syntax.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Literal

from agents.scorer import _normalise, _tokenise

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

PolarityType  = Literal["yes", "no", "mild", "unsure", "skip"]
TemporalType  = Literal["current", "recent", "past", "intermittent", "unknown"]
SeverityType  = Literal["severe", "moderate", "mild", "trace", "unknown"]


# ---------------------------------------------------------------------------
# Pattern libraries
# ---------------------------------------------------------------------------

# ---- POLARITY ----
_YES_STRONG  = re.compile(
    r"\b(yes|yeah|yep|yup|definitely|absolutely|certainly|clearly|"
    r"confirm|confirmed|positive|indeed|correct|that'?s right|"
    r"i do\b|i have\b|i'?ve (had|been|got)|i (feel|felt)|"
    r"i (notice[d]?|experience[d]?)|present|affirmative)\b",
    re.I
)
# Negated forms of 'i have / i do / i feel' — used to cancel false yes hits
_YES_NEGATED = re.compile(
    r"\b(i do(n'?t| not)|i have(n'?t| not)|i (don'?t|did ?not|does ?not) (have|feel|notice|experience))\b",
    re.I
)
_YES_MILD    = re.compile(
    r"\b(kind of|sort of|somewhat|a (little|bit)|slightly|partially|"
    r"mildly|moderate(ly)?|sometimes|occasional(ly)?|on and off|"
    r"now and then|here and there|not (too|very) (bad|severe|strong)|"
    r"a tad|faintly|vaguely|weakly|borderline)\b",
    re.I
)
_NO_STRONG   = re.compile(
    r"\b(no\b|nope|nah|not (at all|really)|never|negative|absent|"
    r"do(es)? not (have|feel|notice|experience)|"
    r"don'?t (have|feel|notice|experience)|haven'?t|didn'?t|"
    r"i do(n'?t| not)|not present|without|denies|denied)\b",
    re.I
)
_UNSURE      = re.compile(
    r"\b(not sure|unsure|i('?m)? not (sure|certain)|hard to say|"
    r"difficult to tell|maybe|perhaps|possibly|might|could be|"
    r"i think( so)?|i guess|probably|i don'?t know|unclear|"
    r"can'?t (tell|say))\b",
    re.I
)
_SKIP        = re.compile(
    r"\b(skip|pass|next|n/?a|not (applicable|relevant)|"
    r"i don'?t understand|what do you mean)\b",
    re.I
)

# ---- TEMPORAL ----
_TEMPORAL = [
    ("past",         re.compile(r"\b(used to|used to have|had (it|this|that)|"
                                r"(last|past|previous) (week|month|year|months|years|time|episode)|"
                                r"(weeks?|months?|years?) ago|before|previously|formerly|"
                                r"went away|resolved|no (longer|more)|not anymore|stopped)\b", re.I)),
    ("recent",       re.compile(r"\b(recently|just (started|began)|a few days? ago|"
                                r"this week|since (yesterday|monday|tuesday|wednesday|thursday|friday)|"
                                r"new(ly)?|onset|started (recently|lately))\b", re.I)),
    ("intermittent", re.compile(r"\b(sometimes|occasionally|on and off|comes? and goes?|"
                                r"intermittent(ly)?|sporadic(ally)?|now and then|episodic(ally)?|"
                                r"not (always|constant(ly)?)|only (sometimes|when|if)|"
                                r"flares? up|recurring)\b", re.I)),
    ("current",      re.compile(r"\b(right now|currently|at the moment|still|ongoing|"
                                r"all (day|the time)|constant(ly)?|persistent(ly)?|"
                                r"every day|daily|always|continuously?)\b", re.I)),
]

# ---- SEVERITY ----
_SEVERITY = [
    ("severe",   re.compile(r"\b(severe(ly)?|extreme(ly)?|very (bad|strong|intense|painful)|"
                            r"terrible|awful|unbearable|excruciating|intense(ly)?|"
                            r"a lot|significantly|heavily|quite( a lot)?|"
                            r"high(ly)?|strong(ly)?|major|significant)\b", re.I)),
    ("moderate", re.compile(r"\b(moderate(ly)?|medium|fair(ly)?|quite|"
                            r"reasonably|noticeably|considerably|"
                            r"(pretty|rather) (bad|strong|bad))\b", re.I)),
    ("mild",     re.compile(r"\b(mild(ly)?|slight(ly)?|a (little|bit)|minor|"
                            r"not (too|very) (bad|strong|severe)|"
                            r"low[- ]grade|faint(ly)?|subtle|barely)\b", re.I)),
    ("trace",    re.compile(r"\b(barely|hardly|trace|almost (not|none)|"
                            r"very (slight(ly)?|faint(ly)?|mild(ly)?)|"
                            r"just a hint|minimal(ly)?|negligible)\b", re.I)),
]

# ---- EXTRA SYMPTOM TRIGGERS ----
# Phrases that introduce additional symptoms mentioned in the answer
_EXTRA_SYM_INTRO = re.compile(
    r"\b(also|and (also|i (have|feel|notice|experience|had|been))|"
    r"plus|as well(?: as)?|in addition|additionally|along with|"
    r"together with|on top of (that|it)|not to mention|"
    r"i'?ve (also|been) (having|feeling|experiencing|noticing))\b",
    re.I
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParsedAnswer:
    # Core signal
    polarity:       PolarityType
    strength:       float           # 0.0 – 1.0, attenuated/amplified polarity

    # Modifiers
    temporal:       TemporalType    = "unknown"
    severity:       SeverityType    = "unknown"
    is_intermittent: bool           = False

    # Bonus: new symptoms mentioned in the answer text
    extra_symptoms: List[str]       = field(default_factory=list)

    # Audit trail
    raw:            str             = ""
    explanation:    str             = ""

    @property
    def answer_type(self) -> str:
        """Maps back to the AnswerType expected by updater.apply_answer."""
        if self.polarity == "yes" and self.strength < 0.35:
            return "mild"
        return self.polarity

    @property
    def effective_strength(self) -> float:
        """
        Final weight scalar applied to the Beta update.
        Base strength × temporal modifier × severity modifier.
        """
        t_mod = {"current": 1.0, "recent": 0.85, "intermittent": 0.65,
                 "past": 0.35, "unknown": 0.90}[self.temporal]
        s_mod = {"severe": 1.3, "moderate": 1.0, "mild": 0.65,
                 "trace": 0.30, "unknown": 1.0}[self.severity]
        return min(self.strength * t_mod * s_mod, 2.0)  # cap at 2× boost


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class FreeTextAnswerParser:
    """
    Converts a raw patient answer string into a ParsedAnswer.

    Usage
    -----
    parser = FreeTextAnswerParser(all_known_symptoms)
    parsed = parser.parse("yes but only at night, and I've also had joint pain")
    # parsed.polarity      → "yes"
    # parsed.temporal      → "intermittent"
    # parsed.extra_symptoms → ["joint pain"]
    # parsed.effective_strength → 0.65 × 1.0 × ...
    """

    def __init__(self, known_symptoms: List[str]):
        """
        known_symptoms: all symptom phrases across all disease profiles,
                        used for fuzzy-matching of extra symptoms mentioned.
        """
        self._known = [_normalise(s) for s in known_symptoms]

    def parse(self, raw_text: str) -> ParsedAnswer:
        text = raw_text.strip()
        norm = _normalise(text)

        # --- 1. Polarity ---
        polarity, base_strength, notes = self._parse_polarity(norm)

        # --- 2. Temporal ---
        temporal = self._parse_temporal(norm)

        # --- 3. Severity ---
        severity = self._parse_severity(norm)

        # --- 4. Extra symptoms ---
        extra = self._extract_extra_symptoms(norm, polarity)

        # --- 5. Intermittency ---
        intermittent = temporal == "intermittent"

        # --- 6. Final strength ---
        parsed = ParsedAnswer(
            polarity=polarity,
            strength=base_strength,
            temporal=temporal,
            severity=severity,
            is_intermittent=intermittent,
            extra_symptoms=extra,
            raw=text,
            explanation=self._explain(polarity, base_strength, temporal,
                                      severity, intermittent, extra, notes),
        )
        return parsed

    # ------------------------------------------------------------------

    def _parse_polarity(self, norm: str) -> tuple[PolarityType, float, str]:
        """
        Returns (polarity, base_strength, notes).
        Priority order: skip > explicit_no > double_negation_yes > unsure > mild_yes > strong_yes > fallback
        """
        skip_hit      = bool(_SKIP.search(norm))
        unsure_hit    = bool(_UNSURE.search(norm))
        no_hit        = bool(_NO_STRONG.search(norm))
        yes_mild_hit  = bool(_YES_MILD.search(norm))
        # A yes-strong hit is only real if it's not negated
        yes_raw       = bool(_YES_STRONG.search(norm))
        yes_negated   = bool(_YES_NEGATED.search(norm))
        yes_hit       = yes_raw and not yes_negated
        notes = []

        if skip_hit:
            return "skip", 0.0, ["skip/pass"]

        # Strong no with no competing yes → clearly negative
        if no_hit and not yes_hit:
            strength = 1.0
            if yes_mild_hit:
                # "not really, maybe a tiny bit" → still a weak no / trace positive
                strength = 0.15
                notes.append("hedged no → trace positive")
            return "no", strength, notes

        # Strong no AND strong yes in same sentence → unsure / double-negation
        if no_hit and yes_hit:
            # Check for real double-negation pattern: "not without", "don't not have"
            double_neg = bool(re.search(
                r"\b(not without|do(n'?t)? not|never not|without not)\b", norm, re.I
            ))
            if double_neg:
                notes.append("double-negation → positive")
                return "yes", 0.6, notes
            # Otherwise "no ... definitely" is still a no with emphasis
            return "no", 1.0, ["strong no with emphasis"]

        # Unsure wins if there's genuine hedging and no strong yes
        if unsure_hit and not yes_hit:
            return "unsure", 0.5, ["explicit uncertainty"]

        # Mild positive qualifier
        if yes_mild_hit and not yes_hit:
            notes.append("mild qualifier")
            return "yes", 0.45, notes

        # Clear yes
        if yes_hit:
            notes.append("strong yes")
            return "yes", 1.0, notes

        # Weak unsure fallback
        if unsure_hit:
            return "unsure", 0.4, ["uncertain with some positive signal"]

        if len(norm.split()) >= 3:
            return "unsure", 0.35, ["no clear polarity detected"]

        return "skip", 0.0, ["empty or unrecognised"]

    # ------------------------------------------------------------------

    def _parse_temporal(self, norm: str) -> TemporalType:
        # Order matters: past beats current if "used to … no longer"
        for label, pattern in _TEMPORAL:
            if pattern.search(norm):
                return label  # type: ignore
        return "unknown"

    def _parse_severity(self, norm: str) -> SeverityType:
        for label, pattern in _SEVERITY:
            if pattern.search(norm):
                return label  # type: ignore
        return "unknown"

    # ------------------------------------------------------------------

    def _extract_extra_symptoms(
        self, norm: str, polarity: PolarityType
    ) -> List[str]:
        """
        Find symptom phrases mentioned alongside the main answer.

        Strategy:
          1. Split on "also", "and I've been having", etc. to find bonus clauses
          2. Try to match each chunk against the known symptom list
          3. Return matched known symptoms (not raw text, to keep downstream clean)
        """
        found: List[str] = []

        # Find everything after an intro trigger
        parts: List[str] = []
        for m in _EXTRA_SYM_INTRO.finditer(norm):
            tail = norm[m.end():].strip()
            if tail:
                parts.append(tail)

        # Also scan the whole text using the tokeniser
        tokens = _tokenise(norm)

        # Match against known symptoms (exact or partial)
        for sym in self._known:
            sym_words = set(sym.split())
            # Exact match in tokens
            if sym in tokens:
                found.append(sym)
                continue
            # Fuzzy: all words of symptom appear in the full text
            if len(sym_words) >= 2 and sym_words.issubset(tokens):
                found.append(sym)
                continue
            # Tail-fragment match
            for part in parts:
                if sym in part or any(w in part for w in sym_words if len(w) > 4):
                    found.append(sym)
                    break

        # Deduplicate, preserve order
        seen = set()
        unique = []
        for s in found:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    # ------------------------------------------------------------------

    def extract_all_mentions(
        self, norm_text: str
    ) -> List[tuple[str, str, float]]:
        """
        Scan normalised answer text for ALL known symptom/RF mentions.

        Returns a list of (symptom_phrase, polarity, effective_strength)
        tuples — one per detected symptom.  The polarity and strength
        are derived from the surrounding text context using the same
        parse() pipeline.
        """
        parsed = self.parse(norm_text)
        pol = parsed.answer_type
        eff = parsed.effective_strength

        tokens = _tokenise(norm_text)
        found: List[tuple[str, str, float]] = []
        seen: set = set()

        for sym in self._known:
            if sym in seen:
                continue
            sym_words = set(sym.split())
            matched = False

            # Exact token match
            if sym in tokens:
                matched = True
            # Fuzzy: all words of multi-word symptom appear in text
            elif len(sym_words) >= 2 and sym_words.issubset(tokens):
                matched = True

            if matched:
                seen.add(sym)
                # Check if this specific symptom is negated in context
                local_pol = pol
                local_eff = eff
                # Simple negation check: "no <symptom>" or "denies <symptom>"
                neg_pattern = re.compile(
                    r'\b(no|not|denies?|without|absent|negative)\b.{0,15}'
                    + re.escape(sym), re.I
                )
                if neg_pattern.search(norm_text):
                    local_pol = "no"
                    local_eff = 1.0
                found.append((sym, local_pol, local_eff))

        return found

    # ------------------------------------------------------------------

    @staticmethod
    def _explain(polarity, strength, temporal, severity,
                 intermittent, extra, notes) -> str:
        parts = [f"Polarity: {polarity} (strength={strength:.2f})"]
        if temporal != "unknown":
            parts.append(f"Temporal: {temporal}")
        if severity != "unknown":
            parts.append(f"Severity: {severity}")
        if intermittent:
            parts.append("intermittent presentation")
        if notes:
            parts.append("Notes: " + "; ".join(notes))
        if extra:
            parts.append(f"Extra symptoms detected: {extra}")
        return " | ".join(parts)
