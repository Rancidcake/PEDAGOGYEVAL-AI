"""Evaluation metrics: MAE, RMSE, exact match (marks) and accuracy, macro/weighted F1, confusion matrix (labels)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .models.tfidf_baseline import round_half_up


def regression_metrics(gold: Sequence[float], pred: Sequence[float]) -> dict[str, float]:
    """MAE/RMSE on raw predicted marks; exact match after rounding predictions half-up."""
    g = np.asarray(gold, dtype=float)
    p = np.asarray(pred, dtype=float)
    err = p - g
    return {
        "n": int(len(g)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "exact_match": float(np.mean([round_half_up(x) == round_half_up(y) for x, y in zip(p, g)])),
    }


def classification_metrics(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> dict:
    return {
        "n": int(len(gold)),
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, labels=list(labels), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(gold, pred, labels=list(labels), average="weighted", zero_division=0)),
    }


def confusion_long(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> pd.DataFrame:
    """Confusion matrix as a long table (gold, pred, count) for CSV storage."""
    cm = confusion_matrix(gold, pred, labels=list(labels))
    return pd.DataFrame(
        [{"gold": g, "pred": p, "count": int(cm[i, j])} for i, g in enumerate(labels) for j, p in enumerate(labels)]
    )


def scores_to_labels(scores: Sequence[float], threshold: float, labels: Sequence[str], negated: Sequence[bool] | None = None) -> list[str]:
    """Map normalised scores to labels: >= threshold -> 'correct'.

    Below the threshold, 3-way tasks use 'contradictory' when a negation flag is set, else 'incorrect'.
    """
    out = []
    for i, s in enumerate(scores):
        if s >= threshold:
            out.append("correct")
        elif "contradictory" in labels and negated is not None and negated[i]:
            out.append("contradictory")
        else:
            out.append("incorrect")
    return out


def tune_threshold(scores: Sequence[float], gold: Sequence[str], labels: Sequence[str], negated=None) -> tuple[float, float]:
    """Grid-search the decision threshold that maximises macro-F1 (call on dev data only)."""
    best_t, best_f = 0.5, -1.0
    for t in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        f = f1_score(gold, scores_to_labels(scores, t, labels, negated), labels=list(labels), average="macro", zero_division=0)
        if f > best_f + 1e-12:
            best_t, best_f = float(t), float(f)
    return best_t, best_f
