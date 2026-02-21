"""
reporter.py
-----------
Renders a ranked diagnosis report to the console (ASCII) and/or a markdown file.

Two distinct axes are shown for every candidate:

  PROBABILITY  — what the evidence says about this disease
                 (normalised posterior, sums to 100% across all diseases)

  CONFIDENCE   — how much evidence we actually observed
                 (independent of match quality; based on evidence quantity only)

The ⚠️  flag marks the clinically dangerous quadrant:
  High probability AND low confidence → worrying signal with insufficient data.
"""

from __future__ import annotations
from typing import List
from agents.scorer import CandidateDiagnosis

PROB_ICONS = {
    "Very High": "🔴",
    "High":      "🟠",
    "Moderate":  "🟡",
    "Low":       "🔵",
    "Very Low":  "⚪",
}
CONF_ICONS = {
    "High":     "✅",
    "Moderate": "🟨",
    "Low":      "🟧",
    "Very Low": "❓",
}


def _bar(value: float, width: int = 16) -> str:
    filled = round(min(max(value, 0.0), 1.0) * width)
    return "█" * filled + "░" * (width - filled)


def render_console(
    candidates: List[CandidateDiagnosis],
    symptoms_text: str,
    anamnesis_text: str,
    dataset_source: str,
) -> str:
    lines = []
    SEP  = "=" * 80
    SEP2 = "-" * 80

    lines += [
        SEP,
        "  🏥  MEDICAL DIAGNOSIS AGENT  —  Differential Ranking",
        SEP,
        f"  Dataset   : {dataset_source}",
        f"  Symptoms  : {symptoms_text[:75]}{'…' if len(symptoms_text)>75 else ''}",
        f"  Anamnesis : {anamnesis_text[:75]}{'…' if len(anamnesis_text)>75 else ''}",
        SEP,
        "",
        "  ┌─────────────────────────────────────────────────────────────────────────┐",
        "  │  PROBABILITY = what the evidence says  │  CONFIDENCE = how much we have │",
        "  └─────────────────────────────────────────────────────────────────────────┘",
        "",
        f"  {'#':<3}  {'Disease':<30}  {'Prob%':>6}  {'CI 90%':>13}  {'Conf%':>6}  {'Evidence':>9}  Flag",
        "  " + SEP2,
    ]

    for rank, c in enumerate(candidates, 1):
        prob_pct   = c.probability * 100
        conf_pct   = c.confidence  * 100
        ci_str     = f"[{c.prob_ci_lo*100:.1f},{c.prob_ci_hi*100:.1f}]"
        prob_icon  = PROB_ICONS.get(c.probability_label, "⚪")
        conf_icon  = CONF_ICONS.get(c.confidence_label, "❓")
        flag       = c.risk_flag

        lines.append(
            f"  {rank:<3}  {c.disease:<30}  "
            f"{prob_icon} {prob_pct:>5.1f}%  {ci_str:>13}  "
            f"{conf_icon} {conf_pct:>5.1f}%  "
            f"[{c.symptom_observed}+{c.risk_factor_observed} tok]  {flag}"
        )

        # Two mini-bars side by side — one per axis
        prob_bar = _bar(c.probability * 5)   # scale: max ~20% → full bar
        conf_bar = _bar(c.confidence)
        lines.append(
            f"       Prob [{prob_bar}] {prob_pct:.1f}%   "
            f"Conf [{conf_bar}] {conf_pct:.1f}%"
        )

    lines += [
        "",
        SEP,
        "  LEGEND",
        "  ──────",
        "  Prob%     : normalised posterior probability — all candidates sum to 100%",
        "  CI 90%    : 5th–95th percentile credible interval on probability",
        "  Conf%     : evidence-quantity confidence, independent of match quality",
        "              conf = n/(n+6) where n = observed symptom+RF tokens",
        "  Evidence  : [symptom tokens observed] + [risk-factor tokens observed]",
        "  ⚠️  flag  : HIGH-PROB / LOW-CONF — worrying signal, not enough data to be sure",
        "",
        "  Probability icons: 🔴 Very High  🟠 High  🟡 Moderate  🔵 Low  ⚪ Very Low",
        "  Confidence  icons: ✅ High  🟨 Moderate  🟧 Low  ❓ Very Low",
        SEP,
    ]

    return "\n".join(lines)


def render_markdown(
    candidates: List[CandidateDiagnosis],
    symptoms_text: str,
    anamnesis_text: str,
    dataset_source: str,
) -> str:
    lines = [
        "# 🏥 Medical Diagnosis Agent — Differential Ranking",
        "",
        f"**Dataset:** {dataset_source}  ",
        f"**Symptoms:** {symptoms_text}  ",
        f"**Anamnesis:** {anamnesis_text}  ",
        "",
        "---",
        "",
        "> **Two independent axes:**  ",
        "> **Probability** = what the evidence *says* (match quality, normalised posterior)  ",
        "> **Confidence** = how much evidence we *have* (evidence quantity, independent of match)  ",
        "> A ⚠️ flag marks the dangerous quadrant: high probability + low confidence.",
        "",
        "## Ranked Candidates",
        "",
        "| # | Disease | Prob% | CI 90% | Conf% | Ev. tokens | Flag |",
        "|---|---------|------:|:------:|------:|:----------:|------|",
    ]

    for rank, c in enumerate(candidates, 1):
        p_icon = PROB_ICONS.get(c.probability_label, "⚪")
        c_icon = CONF_ICONS.get(c.confidence_label,  "❓")
        ci     = f"[{c.prob_ci_lo*100:.1f}, {c.prob_ci_hi*100:.1f}]"
        ev     = f"{c.symptom_observed}+{c.risk_factor_observed}"
        lines.append(
            f"| {rank} | **{c.disease}** "
            f"| {p_icon} {c.probability*100:.1f}% "
            f"| {ci} "
            f"| {c_icon} {c.confidence*100:.1f}% "
            f"| {ev} "
            f"| {c.risk_flag} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Probability vs Confidence (top candidates)",
        "",
        "```",
        f"{'Disease':<32}  Probability                Confidence",
        "-" * 70,
    ]
    for c in candidates[:8]:
        prob_bar = _bar(c.probability * 5, 14)
        conf_bar = _bar(c.confidence,      14)
        lines.append(
            f"{c.disease:<32}  [{prob_bar}] {c.probability*100:5.1f}%  "
            f"[{conf_bar}] {c.confidence*100:5.1f}%"
            + ("  ⚠️" if c.risk_flag else "")
        )
    lines += ["```", ""]

    lines += [
        "---",
        "## Methodology",
        "",
        "### Probability (evidence quality)",
        "- Symptom matching modelled as **Beta(exact + 0.5·partial + 1, misses + 1)**",
        "- Risk factor matching modelled as **Beta(hits + 1, misses + 1)**",
        "- Prior prevalence perturbed with Gaussian noise (σ = 10%)",
        "- Combined score: `0.65 × symptoms + 0.20 × risk_factors + 0.15 × prior`",
        "- Monte Carlo N=5000 → posterior mean and 90% CI",
        "- **Normalised** across all diseases so values form a true probability distribution",
        "",
        "### Confidence (evidence quantity)",
        "- `confidence = n / (n + 6)` where `n` = observed symptom + RF tokens",
        "- Saturates toward 1.0 as more evidence is gathered",
        "- **Completely independent** of whether evidence matched — a disease with",
        "  many observed-but-unmatched symptoms still gets high confidence (we're",
        "  confident it's unlikely). A disease with few observations gets low confidence",
        "  even if every symptom matched (we can't rule other things out).",
        "",
        "> ⚠️ **Disclaimer:** Research/educational tool only. Not medical advice.",
        "> Always consult a qualified physician.",
    ]

    return "\n".join(lines)

