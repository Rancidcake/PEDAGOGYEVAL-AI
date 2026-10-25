"""Shared app state: cached model/rubrics, session rubric store, and A/B/C grading."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
for p in (str(ROOT), str(APP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from pedagogyeval import ConceptGrader, Embedder, GraderConfig, ModelLoadError, Rubric  # noqa: E402
from pedagogyeval.data import load_pilot, load_pilot_rubrics  # noqa: E402
from pedagogyeval.experiments import MODEL_LABELS, b_calibrator  # noqa: E402
from pedagogyeval.models.tfidf_baseline import TfidfBaseline  # noqa: E402

RESULTS = ROOT / "results"
MODELS = {"C_concept": "C · CONCEPT GRAPH", "B_embedding": "B · MINILM", "A_tfidf": "A · TF-IDF"}


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading MiniLM from models_cache/ …")
def get_embedder() -> Embedder:
    return Embedder()


def embedder_or_error() -> tuple[Embedder | None, str | None]:
    try:
        return get_embedder(), None
    except ModelLoadError as exc:
        return None, str(exc)
    except Exception as exc:  # never crash the demo on model load
        return None, f"embedding model failed to load: {exc}"


@st.cache_data(show_spinner=False)
def file_rubrics() -> dict[str, dict]:
    return {k: v.to_dict() for k, v in load_pilot_rubrics().items()}


@st.cache_data(show_spinner=False)
def pilot_frame() -> pd.DataFrame:
    return load_pilot()


@st.cache_data(show_spinner=False)
def demo_examples() -> list[dict]:
    return json.loads((ROOT / "data" / "demo_examples.json").read_text(encoding="utf-8"))["examples"]


@st.cache_data(show_spinner=False)
def team() -> dict:
    p = ROOT / "data" / "team.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"members": []}


@st.cache_resource(show_spinner=False)
def tfidf_for(reference_answers: tuple[str, ...]) -> TfidfBaseline:
    return TfidfBaseline().fit(reference_answers)


@st.cache_resource(show_spinner="Calibrating Model B (leave-one-question-out) …")
def calibrator_for(exclude_qid: str | None, _embedder: Embedder):
    return b_calibrator(_embedder, load_pilot_rubrics(), load_pilot(), exclude_qid)


# ------------------------------------------------------------------ session rubric store
def rubrics() -> dict[str, dict]:
    """Session-scoped rubric dicts (edited on the Rubric page, used immediately by Grade)."""
    if "rubrics" not in st.session_state:
        st.session_state["rubrics"] = {k: json.loads(json.dumps(v)) for k, v in file_rubrics().items()}
    return st.session_state["rubrics"]


def rubric_obj(qid: str) -> Rubric:
    return Rubric.from_dict(rubrics()[qid], validate=False)


# ------------------------------------------------------------------ grading
def grade_all(qid: str, response: str, cfg: GraderConfig) -> dict:
    """Grade one response with A, B and C. Returns a JSON-able dict; never raises for model problems."""
    rub = rubric_obj(qid)
    problems = rub.problems()
    refs = tuple(sorted({r["reference_answer"] for r in rubrics().values()}))
    out: dict = {"question_id": qid, "response": response, "config": cfg.__dict__.copy(), "rubric_problems": problems, "errors": {}}
    out["A_tfidf"] = tfidf_for(refs).grade(rub.reference_answer, response, rub.max_marks)
    emb, err = embedder_or_error()
    if emb is None:
        out["errors"]["model"] = err
        out["B_embedding"] = None
        cfg = GraderConfig(**{**cfg.__dict__, "use_semantic": False})
        out["config"]["use_semantic"] = False
    else:
        pilot_qids = set(pilot_frame().question_id)
        cal = calibrator_for(qid if qid in pilot_qids else None, emb)
        out["B_embedding"] = cal.grade(rub.reference_answer, response, rub.max_marks)
        out["B_embedding"]["calibration"] = "isotonic, fitted on pilot questions other than this one" if qid in pilot_qids else "isotonic, fitted on all pilot questions"
    res = ConceptGrader(emb, cfg).grade(rub, response)
    out["C_concept"] = res.to_dict()
    return out


def result_csv(result: dict) -> str:
    c = result["C_concept"]
    rows = pd.DataFrame(c["concepts"])
    rows.insert(0, "question_id", result["question_id"])
    rows["total_score"] = c["score"]
    rows["max_marks"] = c["max_marks"]
    rows["review"] = c["review"]
    rows["review_reason"] = c["review_reason"]
    for m in ("A_tfidf", "B_embedding"):
        rows[f"{m}_score"] = None if result.get(m) is None else result[m]["score"]
    return rows.to_csv(index=False)


def results_available(*names: str) -> list[str]:
    return [n for n in names if not (RESULTS / n).exists()]


__all__ = ["MODEL_LABELS", "MODELS", "RESULTS", "ROOT", "grade_all", "rubrics", "rubric_obj"]
