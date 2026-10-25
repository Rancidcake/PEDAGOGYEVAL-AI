"""Text preprocessing.

Rules (deliberately conservative so evidence spans stay faithful):
- NFKC normalisation and whitespace cleanup; the cleaned original is kept for display.
- A lowercase copy with identical character offsets is used for matching.
- No stemming. Stopwords, negations, numbers and units are all kept.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

NEGATIONS = frozenset({"not", "never", "no", "without", "cannot", "nor", "none", "neither"})
# Clause punctuation stops a negation's scope.
SCOPE_BREAK = frozenset({",", ";", ".", ":", "!", "?"})

STOPWORDS = frozenset(
    """a an the and or but if then so of to in on at by for from with as is are was were be been being
    it its this that these those there their they them he she his her we our you your i me my
    do does did doing has have had having can could will would should may might must shall
    which who whom whose what when where why how than too very also just only into onto over under
    about after before because while such each any all some more most other same both own up down out
    off again further once here""".split()
)

TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")
SENT_RE = re.compile(r"[^.!?;\n]+[.!?;]*")
CLAUSE_SPLIT_RE = re.compile(
    r",|;|:|\s(?:but|because|so|since|whereas|while|which|therefore|thus|hence)\s", re.IGNORECASE
)


@dataclass(frozen=True)
class Token:
    text: str  # lowercase form
    start: int
    end: int


@dataclass(frozen=True)
class Unit:
    """A sentence or clause of a response, with offsets into the cleaned text."""

    text: str
    start: int
    end: int


def clean(text: str | None) -> str:
    """NFKC-normalise and collapse whitespace. Returns the display text."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def lower_aligned(text: str) -> str:
    """Lowercase copy whose character offsets match ``text`` exactly."""
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)


def tokenize(lower_text: str) -> list[Token]:
    return [Token(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(lower_text)]


def is_negation(tok: str) -> bool:
    return tok in NEGATIONS or tok.endswith("n't")


def content_words(text: str) -> set[str]:
    """Lowercase non-stopword word tokens of length > 2 (used for overlap checks only)."""
    return {
        t.text
        for t in tokenize(lower_aligned(clean(text)))
        if t.text.isalnum() and t.text not in STOPWORDS and len(t.text) > 2
    }


def sentences(text: str) -> list[Unit]:
    units = []
    for m in SENT_RE.finditer(text):
        raw = m.group(0)
        lead = len(raw) - len(raw.lstrip())
        body = raw.strip()
        if body and any(ch.isalnum() for ch in body):
            units.append(Unit(body, m.start() + lead, m.start() + lead + len(body)))
    return units


def semantic_units(text: str) -> list[Unit]:
    """Sentences plus their clauses, de-duplicated, for sentence-level semantic matching."""
    out: dict[tuple[int, int], Unit] = {}
    for sent in sentences(text):
        out[(sent.start, sent.end)] = sent
        pos = 0
        pieces = []
        for m in CLAUSE_SPLIT_RE.finditer(sent.text):
            pieces.append((pos, m.start()))
            pos = m.end()
        pieces.append((pos, len(sent.text)))
        if len(pieces) < 2:
            continue
        for a, b in pieces:
            frag = sent.text[a:b]
            lead = len(frag) - len(frag.lstrip())
            body = frag.strip().rstrip(".!?;")
            if len(body.split()) >= 2:
                s = sent.start + a + lead
                out[(s, s + len(body))] = Unit(body, s, s + len(body))
    return sorted(out.values(), key=lambda u: (u.start, u.end))
