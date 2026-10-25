"""Unit tests for the grading package (run offline: HF_HUB_OFFLINE=1 pytest)."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pedagogyeval import ConceptGrader, Embedder, GraderConfig, Rubric, RubricError  # noqa: E402
from pedagogyeval.data import derive_rubric, load_pilot, load_pilot_rubrics  # noqa: E402
from pedagogyeval.evaluate import classification_metrics, regression_metrics, tune_threshold  # noqa: E402
from pedagogyeval.models.tfidf_baseline import TfidfBaseline  # noqa: E402
from pedagogyeval.preprocess import clean, lower_aligned, semantic_units  # noqa: E402


@pytest.fixture(scope="session")
def rubrics():
    return load_pilot_rubrics()


@pytest.fixture(scope="session")
def grader():
    return ConceptGrader(Embedder())


@pytest.fixture(scope="session")
def lexical_grader():
    return ConceptGrader(None, GraderConfig(use_semantic=False))


def row(result, cid):
    return next(r for r in result.concepts if r.concept_id == cid)


# ------------------------------------------------------------------ required behaviours
@pytest.mark.parametrize("text", ["", "   ", "\n\t", "?!..."])
def test_empty_answer_scores_zero_and_flags(grader, rubrics, text):
    res = grader.grade(rubrics["Q1_photosynthesis"], text)
    assert res.score == 0
    assert res.review
    assert "empty answer" in res.review_reason


def test_negation_gets_no_credit_and_flag(grader, rubrics):
    res = grader.grade(rubrics["Q1_photosynthesis"], "Plants do not use sunlight")
    light = row(res, "c1")
    assert light.negated
    assert light.coverage == 0
    assert light.method == "negated"
    assert res.score == 0
    assert res.review and "negation" in res.review_reason


def test_negation_only_hits_the_negated_concept(grader, rubrics):
    res = grader.grade(rubrics["Q1_photosynthesis"], "Plants do not need sunlight; they make glucose and release oxygen.")
    assert row(res, "c1").negated and row(res, "c1").coverage == 0
    assert row(res, "c3").status == "covered"
    assert row(res, "c4").status == "covered"


def test_contraction_negation(grader, rubrics):
    res = grader.grade(rubrics["Q1_photosynthesis"], "Plants don't need sunlight at all.")
    assert row(res, "c1").negated


def test_negatively_phrased_synonym_is_not_self_negating(grader, rubrics):
    res = grader.grade(rubrics["Q3_primary_key"], "A primary key cannot be null.")
    c2 = row(res, "c2")
    assert c2.status == "covered" and not c2.negated


@pytest.mark.parametrize(
    "qid,text,cid",
    [
        ("Q2_osmosis", "Water crosses a selectively permeable barrier.", "c2"),
        ("Q1_photosynthesis", "Green leaves capture the energy of sunshine.", "c1"),
    ],
)
def test_paraphrase_lexical_misses_semantic_catches(grader, lexical_grader, rubrics, qid, text, cid):
    lex_only = row(lexical_grader.grade(rubrics[qid], text), cid)
    assert lex_only.coverage == 0, "paraphrase must be outside the synonym list"
    full = row(grader.grade(rubrics[qid], text), cid)
    assert full.method == "semantic"
    assert full.coverage > 0
    assert full.lexical == 0


def test_score_never_exceeds_max(grader, rubrics):
    pilot = load_pilot()
    for _, r in pilot.iterrows():
        rub = rubrics[r.question_id]
        res = grader.grade(rub, r.response)
        assert 0 <= res.score <= rub.max_marks
    # keyword stuffing: every synonym repeated
    rub = rubrics["Q5_dns"]
    stuffed = " ".join(s for c in rub.concepts for s in c.synonyms if "*" not in s) * 3
    res = grader.grade(rub, stuffed)
    assert res.score <= rub.max_marks


def test_score_clamped_when_weights_exceed_max(lexical_grader, rubrics):
    rub = copy.deepcopy(rubrics["Q5_dns"])
    for c in rub.concepts:
        c.weight = 5  # deliberately invalid; grading must still clamp
    res = lexical_grader.grade(rub, "DNS translates a website's name to an IP address.")
    assert res.score == rub.max_marks


def test_punctuation_noise(grader, rubrics):
    clean_res = grader.grade(rubrics["Q1_photosynthesis"], "Plants use sunlight and carbon dioxide to make glucose and oxygen.")
    noisy = grader.grade(
        rubrics["Q1_photosynthesis"], "Plants!!!   use... SUNLIGHT,,, and carbon   dioxide??? to make glucose & oxygen!!"
    )
    assert noisy.score == clean_res.score == 4
    assert noisy.response == "Plants!!! use... SUNLIGHT,,, and carbon dioxide??? to make glucose & oxygen!!"


def test_misspelling(grader, rubrics):
    res = grader.grade(rubrics["Q1_photosynthesis"], "Plants use sunlihgt and carbon dioxde to make glucose and oxygen.")
    assert row(res, "c1").method == "fuzzy" and row(res, "c1").status == "covered"
    assert row(res, "c2").status == "covered"
    assert res.score == 4


def test_fuzzy_does_not_overmatch_short_words(lexical_grader, rubrics):
    res = lexical_grader.grade(rubrics["Q1_photosynthesis"], "They might fight at night.")
    assert res.score == 0


def test_evidence_offsets_point_into_display_text(grader, rubrics):
    res = grader.grade(rubrics["Q5_dns"], "DNS translates a website's name to an IP address.")
    assert res.score == 3
    for r in res.concepts:
        assert res.response[r.start : r.end] == r.evidence


def test_result_has_required_fields(grader, rubrics):
    d = grader.grade(rubrics["Q5_dns"], "DNS is like a phonebook that stores the names of websites.").to_dict()
    for k in ("score", "max_marks", "concepts", "missing", "review", "review_reason", "feedback"):
        assert k in d
    for k in ("name", "weight", "coverage", "status", "method", "evidence", "start", "end", "negated"):
        assert k in d["concepts"][0]
    assert d["feedback"].startswith(("You", "Missing", "Check", "No"))


def test_ablation_switches(grader, rubrics):
    rub = rubrics["Q1_photosynthesis"]
    no_neg = grader.grade(rub, "Plants do not use sunlight", GraderConfig(check_negation=False))
    assert row(no_neg, "c1").coverage == 1 and not row(no_neg, "c1").negated
    no_sem = grader.grade(rub, "Green leaves capture the energy of sunshine.", GraderConfig(use_semantic=False))
    assert no_sem.score == 0


def test_spec_examples(grader, rubrics):
    assert grader.grade(rubrics["Q5_dns"], "DNS translates a website's name to an IP address.").score == 3
    assert grader.grade(rubrics["Q5_dns"], "DNS encrypts all internet traffic.").score == 0
    assert grader.grade(rubrics["Q1_photosynthesis"], "Plants use sunlight and carbon dioxide to make glucose and oxygen.").score == 4


# ------------------------------------------------------------------ rubric validation
def _base():
    return {
        "question_id": "t",
        "question": "q?",
        "reference_answer": "ref",
        "max_marks": 2,
        "concepts": [
            {"id": "a", "name": "A", "description": "A", "weight": 1, "synonyms": ["alpha"]},
            {"id": "b", "name": "B", "description": "B", "weight": 1, "synonyms": ["beta"]},
        ],
        "edges": [{"source": "a", "target": "b", "relation": "causes"}],
    }


def test_valid_rubric_loads():
    assert Rubric.from_dict(_base()).max_marks == 2


def test_weights_must_sum_to_max():
    d = _base()
    d["concepts"][0]["weight"] = 3
    with pytest.raises(RubricError, match="weights sum to 4 but max_marks is 2"):
        Rubric.from_dict(d)


def test_edge_must_reference_existing_concepts():
    d = _base()
    d["edges"].append({"source": "a", "target": "zzz", "relation": "causes"})
    with pytest.raises(RubricError, match="unknown concept 'zzz'"):
        Rubric.from_dict(d)


def test_edge_relation_must_be_known():
    d = _base()
    d["edges"][0]["relation"] = "is_friends_with"
    with pytest.raises(RubricError, match="expected one of causes, requires, part_of, leads_to"):
        Rubric.from_dict(d)


def test_missing_fields_and_duplicates():
    d = _base()
    del d["reference_answer"]
    with pytest.raises(RubricError, match="missing field"):
        Rubric.from_dict(d)
    d = _base()
    d["concepts"][1]["id"] = "a"
    d["edges"] = []
    with pytest.raises(RubricError, match="duplicate concept id"):
        Rubric.from_dict(d)


def test_all_errors_reported_together():
    d = _base()
    d["concepts"][0]["weight"] = 5
    d["edges"][0]["target"] = "nope"
    with pytest.raises(RubricError) as exc:
        Rubric.from_dict(d)
    assert "weights sum" in str(exc.value) and "unknown concept" in str(exc.value)


def test_pilot_rubrics_are_valid(rubrics):
    assert len(rubrics) == 6
    for r in rubrics.values():
        assert 3 <= len(r.concepts) <= 5
        assert len({c.weight for c in r.concepts}) == 1


def test_derived_rubric_is_valid():
    r = derive_rubric(
        "x",
        "Can you use aluminum foil to connect the circuit? Why?",
        "Yes, aluminum foil is a metal and all metals conduct. The aluminum foil would complete the circuit.",
    )
    assert 1 <= len(r.concepts) <= 4
    assert abs(sum(c.weight for c in r.concepts) - r.max_marks) < 1e-6


# ------------------------------------------------------------------ preprocessing / metrics / baselines
def test_preprocess_keeps_offsets_and_tokens():
    t = clean("  Ｐlants do  NOT use 5 kg  ")
    assert t == "Plants do NOT use 5 kg"
    assert len(lower_aligned(t)) == len(t) and "not" in lower_aligned(t) and "5 kg" in t
    units = semantic_units("Water moves in. It goes through a membrane, because of osmosis.")
    assert any(u.text.startswith("It goes through") for u in units)


def test_metrics():
    m = regression_metrics([3, 0, 2], [3, 1, 1.5])
    assert m["mae"] == pytest.approx(0.5) and m["exact_match"] == pytest.approx(2 / 3)
    c = classification_metrics(["correct", "incorrect"], ["correct", "correct"], ["correct", "incorrect"])
    assert c["accuracy"] == 0.5
    t, f = tune_threshold([0.9, 0.1, 0.8, 0.2], ["correct", "incorrect", "correct", "incorrect"], ["correct", "incorrect"])
    assert f == 1.0 and 0.2 < t <= 0.8


def test_tfidf_baseline(rubrics):
    m = TfidfBaseline().fit(r.reference_answer for r in rubrics.values())
    rub = rubrics["Q5_dns"]
    full = m.grade(rub.reference_answer, rub.reference_answer, rub.max_marks)
    assert full["score"] == rub.max_marks
    assert m.grade(rub.reference_answer, "", rub.max_marks)["score"] == 0
