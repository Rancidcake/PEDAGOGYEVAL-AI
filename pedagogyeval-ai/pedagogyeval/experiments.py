"""Experiment runners shared by the notebook and the app. All numbers are computed here, never hard-coded.

Pilot protocol (24 authored responses, 6 questions):
  A  TF-IDF fitted on the rubric reference answers only (no student responses).
  B  MiniLM cosine, isotonic calibration fitted leave-one-question-out (never sees the graded question).
  C  concept-graph grader (full + ablations).
SciEntsBank protocol: thresholds tuned on dev (carved from train) to maximise macro-F1, then applied
unchanged to the official test splits (unseen-answers / unseen-questions / unseen-domains).
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .evaluate import classification_metrics, confusion_long, regression_metrics, scores_to_labels, tune_threshold
from .models.concept_grader import ConceptGrader, GraderConfig
from .models.embedding_baseline import Embedder, EmbeddingBaseline
from .models.tfidf_baseline import TfidfBaseline, round_half_up
from .rubric import Rubric

ABLATIONS: dict[str, GraderConfig] = {
    "C_concept": GraderConfig(),
    "C_no_semantic": GraderConfig(use_semantic=False),
    "C_uniform_weights": GraderConfig(uniform_weights=True),
    "C_no_negation": GraderConfig(check_negation=False),
}
MODEL_LABELS = {
    "A_tfidf": "A · TF-IDF",
    "B_embedding": "B · MiniLM",
    "C_concept": "C · Concept graph",
    "C_no_semantic": "C − semantic",
    "C_uniform_weights": "C − weights (uniform)",
    "C_no_negation": "C − negation check",
}


# ------------------------------------------------------------------ pilot
def fit_tfidf_on_rubrics(rubrics: dict[str, Rubric]) -> TfidfBaseline:
    return TfidfBaseline().fit(r.reference_answer for r in rubrics.values())


def b_calibrator(embedder: Embedder, rubrics: dict[str, Rubric], pilot: pd.DataFrame, exclude_qid: str | None) -> EmbeddingBaseline:
    """Model B with isotonic calibration fitted on pilot questions other than ``exclude_qid``."""
    train = pilot[pilot.question_id != exclude_qid]
    b = EmbeddingBaseline(embedder)
    sims = b.similarity([rubrics[q].reference_answer for q in train.question_id], list(train.response))
    gold = np.array([g / rubrics[q].max_marks for g, q in zip(train.gold_marks, train.question_id)])
    return b.fit_isotonic(sims, gold)


def run_pilot(embedder: Embedder, rubrics: dict[str, Rubric], pilot: pd.DataFrame, configs: dict[str, GraderConfig] | None = None):
    configs = configs or ABLATIONS
    tfidf = fit_tfidf_on_rubrics(rubrics)
    grader = ConceptGrader(embedder)
    grader.prefetch((rubrics[q], t) for q, t in zip(pilot.question_id, pilot.response))
    calibrators = {q: b_calibrator(embedder, rubrics, pilot, q) for q in pilot.question_id.unique()}
    rows = []
    for _, r in pilot.iterrows():
        rub = rubrics[r.question_id]
        base = {
            "response_id": r.response_id, "question_id": r.question_id, "response_type": r.response_type,
            "response": r.response, "gold": float(r.gold_marks), "max_marks": rub.max_marks,
        }
        a = tfidf.grade(rub.reference_answer, r.response, rub.max_marks)
        rows.append({**base, "model": "A_tfidf", "pred": a["score"], "similarity": a["similarity"]})
        b = calibrators[r.question_id].grade(rub.reference_answer, r.response, rub.max_marks)
        rows.append({**base, "model": "B_embedding", "pred": b["score"], "similarity": b["similarity"]})
        for name, cfg in configs.items():
            c = grader.grade(rub, r.response, cfg)
            rows.append({
                **base, "model": name, "pred": c.score, "review": c.review, "review_reason": c.review_reason,
                "missing": "; ".join(c.missing), "feedback": c.feedback,
                "concepts": " | ".join(f"{x.name}:{x.status}:{x.method}" for x in c.concepts),
            })
    preds = pd.DataFrame(rows)
    preds["pred_rounded"] = preds.pred.map(round_half_up)
    preds["abs_error"] = (preds.pred - preds.gold).abs()
    preds["norm_abs_error"] = preds.abs_error / preds.max_marks
    metrics = []
    for name, g in preds.groupby("model", sort=False):
        m = regression_metrics(g.gold, g.pred)
        m["norm_mae"] = float(g.norm_abs_error.mean())
        if "review" in g and g.review.notna().any():
            m["review_rate"] = float(g.review.astype(bool).mean())
        metrics.append({"model": name, "label": MODEL_LABELS.get(name, name), **m})
    metrics = pd.DataFrame(metrics)
    by_type = (
        preds.groupby(["model", "response_type"]).agg(n=("gold", "size"), mae=("abs_error", "mean"), mean_gold=("gold", "mean"), mean_pred=("pred", "mean")).reset_index()
    )
    return preds, metrics, by_type


def pilot_sample_table(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per response: gold vs A/B/C marks plus C's feedback and review flag."""
    wide = preds[preds.model.isin(["A_tfidf", "B_embedding", "C_concept"])].pivot_table(
        index=["response_id", "question_id", "response_type", "response", "gold", "max_marks"], columns="model", values="pred"
    ).reset_index()
    c = preds[preds.model == "C_concept"].set_index("response_id")[["review", "review_reason", "missing", "feedback"]]
    wide = wide.join(c, on="response_id")
    wide.columns.name = None
    return wide.rename(columns={"A_tfidf": "A", "B_embedding": "B", "C_concept": "C"})


# ------------------------------------------------------------------ SciEntsBank
def score_scientsbank(frames: dict[str, pd.DataFrame], embedder: Embedder, configs: dict[str, GraderConfig] | None = None, log=print) -> pd.DataFrame:
    """Normalised [0,1] score per answer per model for dev + test splits."""
    from .data import derive_rubric

    configs = configs or ABLATIONS
    train = frames["train"]
    tfidf = TfidfBaseline().fit(list(train.reference_answer.unique()) + list(train.student_answer))
    b = EmbeddingBaseline(embedder)
    grader = ConceptGrader(embedder)
    eval_splits = [k for k in frames if k != "train"]
    rubric_cache: dict[tuple, Rubric] = {}
    out = []
    for split in eval_splits:
        df = frames[split]
        log(f"  scoring {split} (n={len(df)})")
        rubs = []
        for q, qt, ref in zip(df.question_id, df.question, df.reference_answer):
            key = (q, ref)
            if key not in rubric_cache:
                rubric_cache[key] = derive_rubric(str(q), qt, ref)
            rubs.append(rubric_cache[key])
        grader.prefetch(zip(rubs, df.student_answer))
        base = df[["answer_id", "question_id", "student_answer", "reference_answer", "label_2way", "label_3way"]].assign(split=split)
        a_sim = tfidf.similarity(list(df.reference_answer), list(df.student_answer))
        out.append(base.assign(model="A_tfidf", score=np.clip(a_sim, 0, 1), negated=False))
        b_sim = b.similarity(list(df.reference_answer), list(df.student_answer))
        out.append(base.assign(model="B_embedding", score=np.clip(b_sim, 0, 1), negated=False))
        for name, cfg in configs.items():
            res = [grader.grade(r, t, cfg) for r, t in zip(rubs, df.student_answer)]
            out.append(base.assign(
                model=name,
                score=[x.score / x.max_marks for x in res],
                negated=[any(c.negated for c in x.concepts) for x in res],
                review=[x.review for x in res],
            ))
    scores = pd.concat(out, ignore_index=True)
    scores["review"] = scores["review"].astype("boolean")
    return scores


def evaluate_scientsbank(scores: pd.DataFrame, tasks=("2way", "3way")):
    """Tune one threshold per model/task on dev, apply to every split. Returns (metrics, confusion, predictions)."""
    metrics, conf, preds = [], [], []
    for task in tasks:
        gold_col = f"label_{task}"
        labels = sorted(scores[gold_col].dropna().unique(), key=lambda x: ["correct", "contradictory", "incorrect"].index(x))
        for model, g in scores.groupby("model", sort=False):
            g = g[g[gold_col].notna()]
            dev = g[g.split == "dev"]
            neg_dev = dev.negated.values if task == "3way" else None
            t, dev_f1 = tune_threshold(dev.score.values, dev[gold_col].values, labels, neg_dev)
            for split, s in g.groupby("split", sort=False):
                pred = scores_to_labels(s.score.values, t, labels, s.negated.values if task == "3way" else None)
                m = classification_metrics(s[gold_col].values, pred, labels)
                metrics.append({"model": model, "label": MODEL_LABELS.get(model, model), "task": task, "split": split, "threshold": t, "tuned_on": "dev", **m})
                conf.append(confusion_long(s[gold_col].values, pred, labels).assign(model=model, task=task, split=split))
                preds.append(s[["answer_id", "split", "model", "score", "negated", gold_col]].rename(columns={gold_col: "gold"}).assign(task=task, pred=pred))
    return pd.DataFrame(metrics), pd.concat(conf, ignore_index=True), pd.concat(preds, ignore_index=True)


def worst_errors_pilot(preds: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    keep = ["model", "response_id", "response_type", "response", "gold", "pred", "max_marks", "abs_error", "review_reason", "missing"]
    return (
        preds[preds.model.isin(["A_tfidf", "B_embedding", "C_concept"])]
        .sort_values(["model", "abs_error", "response_id"], ascending=[True, False, True])
        .groupby("model").head(k)[[c for c in keep if c in preds.columns]]
    )


def worst_errors_scientsbank(preds: pd.DataFrame, scores: pd.DataFrame, k: int = 10, task: str = "3way") -> pd.DataFrame:
    """Most confident misclassifications on the test splits: wrong label with score farthest from the gold side."""
    p = preds[(preds.task == task) & (preds.split != "dev") & (preds.gold != preds.pred) & preds.model.isin(["A_tfidf", "B_embedding", "C_concept"])].copy()
    p["target"] = (p.gold == "correct").astype(float)
    p["confidence_gap"] = (p.score - p.target).abs()
    text = scores.drop_duplicates(["answer_id", "split"]).set_index(["answer_id", "split"])[["question_id", "student_answer", "reference_answer"]]
    p = p.join(text, on=["answer_id", "split"])
    return (
        p.sort_values(["model", "confidence_gap", "answer_id"], ascending=[True, False, True]).groupby("model").head(k)
        [["model", "split", "answer_id", "question_id", "reference_answer", "student_answer", "gold", "pred", "score", "negated", "confidence_gap"]]
    )


def config_table() -> pd.DataFrame:
    return pd.DataFrame([{"config": k, **asdict(v)} for k, v in ABLATIONS.items()])
