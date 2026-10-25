"""Model A: TF-IDF cosine baseline.

mark = round(cosine(reference, response) * max_marks). Unigram+bigram TF-IDF.
The vectoriser is fitted only on the texts passed to ``fit`` (train split for the
benchmark; reference answers only for the pilot and the app), never on evaluation responses.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..preprocess import clean


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


class TfidfBaseline:
    name = "A_tfidf"

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), lowercase=True, token_pattern=r"(?u)\b\w+\b", sublinear_tf=True
        )
        self.fitted = False

    def fit(self, texts: Iterable[str]) -> "TfidfBaseline":
        docs = [clean(t) for t in texts if clean(t)]
        self.vectorizer.fit(docs)
        self.fitted = True
        return self

    def similarity(self, references: list[str], responses: list[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TfidfBaseline.fit() must be called before scoring")
        r = self.vectorizer.transform([clean(t) for t in references])
        s = self.vectorizer.transform([clean(t) for t in responses])
        # rows are L2-normalised by TfidfVectorizer, so the row-wise dot product is the cosine
        return np.asarray(r.multiply(s).sum(axis=1)).ravel()

    def grade(self, reference: str, response: str, max_marks: float) -> dict:
        if not clean(response):
            return {"model": self.name, "similarity": 0.0, "score": 0.0, "max_marks": max_marks}
        sim = float(self.similarity([reference], [response])[0])
        return {"model": self.name, "similarity": sim, "score": float(round_half_up(sim * max_marks)), "max_marks": max_marks}
