"""Explanations: template feedback text and the human-review flag (no LLM involved)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .models.concept_grader import ConceptResult, GraderConfig


def _join(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def build_feedback(rows: list["ConceptResult"], empty: bool) -> str:
    if empty:
        return "No answer was given. Attempt the question to earn credit."
    covered = [r.name for r in rows if r.status == "covered"]
    partial = [r.name for r in rows if r.status == "partial"]
    negated = [r.name for r in rows if r.negated]
    missing = [r.name for r in rows if r.status == "missing" and not r.negated]
    parts = []
    if covered:
        parts.append(f"You correctly explained {_join(covered)}.")
    if partial:
        parts.append(f"You partly addressed {_join(partial)}; state it more precisely.")
    if negated:
        parts.append(f"Check your statement about {_join(negated)}: it appears to be negated or contradicted.")
    if missing:
        parts.append(f"Missing: {_join(missing)}.")
    if not missing and not partial and not negated:
        parts.append("All key concepts are present.")
    return " ".join(parts)


def review_flag(rows: list["ConceptResult"], empty: bool, cfg: "GraderConfig") -> tuple[bool, list[str]]:
    """Flag for human review when the grade is fragile.

    - empty answer
    - negation detected
    - a semantic similarity within ``cfg.margin`` of t_high / t_low, where that concept's
      credit depends on the semantic channel (no clean lexical match)
    - lexical/semantic disagreement: a synonym matched but the semantic cosine is below
      ``cfg.disagree_floor`` (possible keyword-stuffing / out-of-context mention)
    """
    reasons: list[str] = []
    if empty:
        reasons.append("empty answer")
    for r in rows:
        if r.negated:
            reasons.append(f"negation near '{r.name}'")
    if cfg.use_semantic:
        for r in rows:
            if r.semantic is None:
                continue
            if r.method in ("semantic", "—", "negated"):
                for t, label in ((cfg.t_high, "t_high"), (cfg.t_low, "t_low")):
                    if abs(r.semantic - t) < cfg.margin:
                        reasons.append(f"'{r.name}' similarity {r.semantic:.2f} is near {label}={t:.2f}")
                        break
            if r.method in ("lexical", "fuzzy") and r.semantic < cfg.disagree_floor:
                reasons.append(f"'{r.name}': keyword matched but meaning similarity is low ({r.semantic:.2f})")
    return bool(reasons), reasons
