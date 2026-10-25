"""Streamlit AppTest smoke tests: every page loads; every quick-load + GRADE runs without exceptions."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
TIMEOUT = 180
EXAMPLES = json.loads((ROOT / "data" / "demo_examples.json").read_text())["examples"]


def run(path: Path) -> AppTest:
    at = AppTest.from_file(str(path), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def html_of(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


@pytest.mark.parametrize("page", ["Home.py", "pages/1_Grade.py", "pages/2_Rubric.py", "pages/3_Results.py"])
def test_page_loads(page):
    at = run(APP / page)
    assert "pe-" in html_of(at) or at.markdown, "theme helpers should render"


def test_home_has_title_and_pipeline():
    at = run(APP / "Home.py")
    assert "PEDAGOGYEVAL/AI" in html_of(at)
    assert "HUMAN EXAMINER" in html_of(at)


@pytest.mark.parametrize("ex", EXAMPLES, ids=[e["key"] for e in EXAMPLES])
def test_quick_load_then_grade(ex):
    at = run(APP / "pages" / "1_Grade.py")
    at.button(key=f"ql_{ex['key']}").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.text_area(key="grade_response").value == ex["response"]
    assert at.selectbox(key="grade_qid").value == ex["question_id"]
    at.button(key="grade_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    page = html_of(at)
    assert "pe-score" in page and "pe-ledger" in page and "pe-strip" in page
    assert at.session_state["graded"]["response"] == ex["response"]
    if ex["key"] == "negation":
        assert "NEEDS HUMAN REVIEW" in page and "<del" in page
    if ex["key"] == "wrong":
        assert '<div class="big">0<span class="den">/3</span>' in page


def test_all_models_and_toggles_render():
    at = run(APP / "pages" / "1_Grade.py")
    at.button(key="ql_paraphrase").click().run()
    at.button(key="grade_btn").click().run()
    for m in ("A_tfidf", "B_embedding", "C_concept"):
        at.radio(key="grade_model").set_value(m).run()
        assert not at.exception, [e.value for e in at.exception]
    at.toggle(key="grade_sem").set_value(False).run()
    assert not at.exception
    at.toggle(key="grade_neg").set_value(False).run()
    assert not at.exception


def test_empty_response_grades_cleanly():
    at = run(APP / "pages" / "1_Grade.py")
    at.button(key="grade_btn").click().run()
    assert not at.exception
    assert "empty answer" in html_of(at)


def test_rubric_page_new_question_and_save():
    at = run(APP / "pages" / "2_Rubric.py")
    assert "rubric is valid" in html_of(at)
    at.button(key="rub_new").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="rub_qid").value.startswith("Q_new_")
    at.button(key="rub_save").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(k.startswith("Q_new_") for k in at.session_state["rubrics"])


def test_rubric_validation_flags_bad_max_marks():
    at = run(APP / "pages" / "2_Rubric.py")
    qid = at.selectbox(key="rub_qid").value
    num = next(n for n in at.number_input if n.key.startswith(f"rub_max_{qid}"))
    num.set_value(10.0).run()
    assert not at.exception
    page = html_of(at)
    total = sum(c["weight"] for c in at.session_state["rubrics"][qid]["concepts"])
    assert "pe-valid bad" in page and f"weights sum to {total:g} but max_marks is 10" in page
    assert at.button(key="rub_save").disabled


def test_results_page_tabs_render():
    at = run(APP / "pages" / "3_Results.py")
    assert len(at.tabs) == 3


def test_results_page_shows_ledgers_when_csvs_exist():
    if not (ROOT / "results" / "pilot_metrics.csv").exists():
        pytest.skip("results/ not generated yet (run the notebook)")
    at = run(APP / "pages" / "3_Results.py")
    page = html_of(at)
    assert "RESULTS NOT FOUND" not in page
    assert page.count("pe-ledger") >= 4 and 'class="best"' in page
    at.radio(key="res_task").set_value("2way").run()
    assert not at.exception
