"""Model C: concept-graph partial-credit grader.

For each rubric concept i with weight w_i:
    lexical  p = 1 if a synonym matches (word-boundary regex; '*' = up to 3 words;
               an optional plural suffix on the last word; misspelling-tolerant fallback)
    semantic (toggleable) max cosine(description, response sentence/clause):
               >= t_high -> 1.0, >= t_low -> 0.5, else 0
    p_i = max(lexical, semantic) unless negated (negation word within 3 tokens before the match).
Score = min(M, max(0, sum w_i * p_i)). Graph edges are for explanation only.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Iterable

from ..explain import build_feedback, review_flag
from ..preprocess import (
    SCOPE_BREAK,
    Token,
    Unit,
    clean,
    content_words,
    is_negation,
    lower_aligned,
    semantic_units,
    tokenize,
)
from ..rubric import Concept, Rubric


@dataclass
class GraderConfig:
    use_semantic: bool = True
    t_high: float = 0.70
    t_low: float = 0.50
    check_negation: bool = True
    uniform_weights: bool = False
    fuzzy: bool = True
    neg_window: int = 3
    margin: float = 0.05
    disagree_floor: float = 0.30  # keyword matched but cosine below this -> review


@dataclass
class ConceptResult:
    concept_id: str
    name: str
    weight: float
    coverage: float
    status: str  # covered | partial | missing
    method: str  # lexical | fuzzy | semantic | negated | —
    evidence: str
    start: int | None
    end: int | None
    negated: bool
    lexical: float
    semantic: float | None  # raw cosine
    semantic_credit: float
    credit: float


@dataclass
class GradeResult:
    question_id: str
    response: str  # cleaned display text; offsets refer to this
    score: float
    max_marks: float
    concepts: list[ConceptResult]
    missing: list[str]
    review: bool
    review_reasons: list[str]
    feedback: str
    config: dict = field(default_factory=dict)

    @property
    def review_reason(self) -> str:
        return "; ".join(self.review_reasons)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["review_reason"] = self.review_reason
        return d

    def concept_rows(self) -> list[dict]:
        return [asdict(c) for c in self.concepts]


# ---------------------------------------------------------------- lexical matching
_WORD = r"[\w']+"


@lru_cache(maxsize=4096)
def synonym_pattern(synonym: str) -> re.Pattern | None:
    words = lower_aligned(clean(synonym)).split()
    if not words:
        return None
    parts: list[str] = []
    gap = False
    for w in words:
        if w == "*":
            gap = True
            continue
        if parts:
            parts.append(rf"(?:\W+{_WORD}){{0,3}}\W+" if gap else r"\W+")
        parts.append(re.escape(w))
        gap = False
    if not parts:
        return None
    if words[-1].isalpha() and len(words[-1]) >= 3:
        parts.append(r"(?:s|es)?")
    return re.compile(r"(?<!\w)" + "".join(parts) + r"(?!\w)")


def _osa_distance(a: str, b: str) -> int:
    """Optimal-string-alignment (Damerau) edit distance."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def _fuzzy_eq(word: str, tok: str) -> bool:
    if word == tok:
        return True
    n = len(word)
    if n < 6 or tok[:1] != word[:1] or abs(len(tok) - n) > 2 or is_negation(tok):
        return False
    return _osa_distance(word, tok) <= (2 if n >= 9 else 1)


def _negated_before(tokens: list[Token], idx: int, window: int) -> bool:
    """Is there a negation within ``window`` tokens before token ``idx`` (not crossing clause punctuation)?"""
    seen = 0
    j = idx - 1
    while j >= 0 and seen < window:
        t = tokens[j].text
        if t in SCOPE_BREAK:
            return False
        if is_negation(t):
            return True
        if t.isalnum() or "'" in t:
            seen += 1
        j -= 1
    return False


def _token_index(tokens: list[Token], char: int) -> int:
    for i, t in enumerate(tokens):
        if t.end > char:
            return i
    return len(tokens)


@dataclass
class _Match:
    start: int
    end: int
    negated: bool
    fuzzy: bool


def lexical_matches(concept: Concept, low: str, tokens: list[Token], cfg: GraderConfig) -> list[_Match]:
    found: list[_Match] = []
    for syn in concept.synonyms:
        pat = synonym_pattern(syn)
        if pat is None:
            continue
        for m in pat.finditer(low):
            i0 = _token_index(tokens, m.start())
            i1 = _token_index(tokens, m.end() - 1)
            own = {w for w in lower_aligned(clean(syn)).split()}
            neg = _negated_before(tokens, i0, cfg.neg_window) or any(
                is_negation(tokens[k].text) and tokens[k].text not in own for k in range(i0, min(i1 + 1, len(tokens)))
            )
            found.append(_Match(m.start(), m.end(), neg, False))
    if found or not cfg.fuzzy:
        return found
    words = [(i, t) for i, t in enumerate(tokens) if t.text[:1].isalnum()]
    for syn in concept.synonyms:
        sw = lower_aligned(clean(syn)).split()
        if not sw or sw[0] == "*":
            continue
        for k in range(len(words)):
            hit = _fuzzy_seq(sw, words, k)
            if hit is not None:
                a, b, gap_neg = hit
                neg = gap_neg or _negated_before(tokens, words[a][0], cfg.neg_window)
                found.append(_Match(words[a][1].start, words[b][1].end, neg, True))
    return found


def _fuzzy_seq(sw: list[str], words: list[tuple[int, Token]], k: int):
    """Match synonym words (with '*' gaps of 0-3 words) starting at word k; at least one word must be fuzzy."""

    def rec(si: int, wi: int, fuzzy_used: bool, gap_neg: bool):
        if si == len(sw):
            return (wi - 1, gap_neg) if fuzzy_used else None
        if sw[si] == "*":
            for g in range(0, 4):
                if wi + g > len(words):
                    break
                gn = gap_neg or any(is_negation(words[x][1].text) for x in range(wi, wi + g))
                r = rec(si + 1, wi + g, fuzzy_used, gn)
                if r is not None:
                    return r
            return None
        if wi >= len(words) or not _fuzzy_eq(sw[si], words[wi][1].text):
            return None
        return rec(si + 1, wi + 1, fuzzy_used or sw[si] != words[wi][1].text, gap_neg)

    r = rec(0, k, False, False)
    return None if r is None else (k, r[0], r[1])


# ---------------------------------------------------------------- grader
class ConceptGrader:
    name = "C_concept"

    def __init__(self, embedder=None, config: GraderConfig | None = None) -> None:
        self.embedder = embedder
        self.config = config or GraderConfig()

    @staticmethod
    def concept_text(c: Concept) -> str:
        return clean(c.description or c.name)

    def prefetch(self, pairs: Iterable[tuple[Rubric, str]]) -> None:
        """Batch-encode every concept description and response unit up front (benchmark speed-up)."""
        if self.embedder is None:
            return
        texts: list[str] = []
        for rubric, response in pairs:
            texts.extend(self.concept_text(c) for c in rubric.concepts)
            texts.extend(u.text for u in semantic_units(clean(response)))
        self.embedder.prefetch(texts)

    def _semantic_negated(self, unit: Unit, low: str, tokens: list[Token], concept: Concept, cfg: GraderConfig) -> bool:
        concept_text = concept.description + " " + concept.name + " " + " ".join(concept.synonyms)
        if any(is_negation(t.text) for t in tokenize(lower_aligned(clean(concept_text)))):
            return False  # the concept is itself phrased negatively ("cannot be null")
        cue = content_words(concept_text)
        unit_toks = [i for i, t in enumerate(tokens) if t.start >= unit.start and t.end <= unit.end]
        return any(_negated_before(tokens, i, cfg.neg_window) for i in unit_toks if tokens[i].text in cue)

    def grade(self, rubric: Rubric, response: str, config: GraderConfig | None = None) -> GradeResult:
        cfg = config or self.config
        text = clean(response)
        low = lower_aligned(text)
        tokens = tokenize(low)
        empty = not any(t.text[:1].isalnum() for t in tokens)
        M = float(rubric.max_marks)
        n = len(rubric.concepts)

        units: list[Unit] = []
        sims = None
        if cfg.use_semantic and not empty:
            if self.embedder is None:
                raise RuntimeError("semantic matching is enabled but no embedder was provided")
            units = semantic_units(text)
            if units:
                sims = self.embedder.cosine([self.concept_text(c) for c in rubric.concepts], [u.text for u in units])

        rows: list[ConceptResult] = []
        for ci, c in enumerate(rubric.concepts):
            w = M / n if cfg.uniform_weights else c.weight
            matches = [] if empty else lexical_matches(c, low, tokens, cfg)
            clean_m = [m for m in matches if not m.negated] if cfg.check_negation else matches
            exact = [m for m in clean_m if not m.fuzzy]
            best_lex = (exact or clean_m or [None])[0]
            lex_neg_only = cfg.check_negation and bool(matches) and not clean_m

            sem_cos, sem_credit, sem_unit, sem_neg = None, 0.0, None, False
            if sims is not None:
                j = int(sims[ci].argmax())
                sem_cos = float(sims[ci, j])
                sem_unit = units[j]
                sem_credit = 1.0 if sem_cos >= cfg.t_high else 0.5 if sem_cos >= cfg.t_low else 0.0
                if cfg.check_negation and sem_credit > 0:
                    sem_neg = self._semantic_negated(sem_unit, low, tokens, c, cfg)

            negated = False
            if best_lex is not None:
                p, method, span = 1.0, "fuzzy" if best_lex.fuzzy else "lexical", (best_lex.start, best_lex.end)
            elif sem_credit > 0 and not sem_neg and not (
                lex_neg_only and any(m.start < sem_unit.end and m.end > sem_unit.start for m in matches)
            ):
                p, method, span = sem_credit, "semantic", (sem_unit.start, sem_unit.end)
            elif lex_neg_only or sem_neg:
                negated = True
                m0 = matches[0] if lex_neg_only else None
                span = (m0.start, m0.end) if m0 else (sem_unit.start, sem_unit.end)
                p, method = 0.0, "negated"
            else:
                p, method, span = 0.0, "—", None

            status = "covered" if p >= 1.0 else "partial" if p > 0 else "missing"
            rows.append(
                ConceptResult(
                    concept_id=c.id,
                    name=c.name,
                    weight=round(w, 6),
                    coverage=p,
                    status=status,
                    method=method,
                    evidence=text[span[0] : span[1]] if span else "",
                    start=span[0] if span else None,
                    end=span[1] if span else None,
                    negated=negated,
                    lexical=1.0 if matches and (best_lex is not None or not cfg.check_negation) else 0.0,
                    semantic=None if sem_cos is None else round(sem_cos, 4),
                    semantic_credit=sem_credit,
                    credit=round(w * p, 6),
                )
            )

        score = min(M, max(0.0, sum(r.credit for r in rows)))
        flag, reasons = review_flag(rows, empty, cfg)
        return GradeResult(
            question_id=rubric.question_id,
            response=text,
            score=round(score, 4),
            max_marks=M,
            concepts=rows,
            missing=[r.name for r in rows if r.status == "missing"],
            review=flag,
            review_reasons=reasons,
            feedback=build_feedback(rows, empty),
            config=asdict(cfg),
        )


__all__ = ["ConceptGrader", "GraderConfig", "GradeResult", "ConceptResult"]
