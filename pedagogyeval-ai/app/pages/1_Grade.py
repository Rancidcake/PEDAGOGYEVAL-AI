"""Grade: the main demo screen."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import theme  # noqa: E402

theme.apply("Grade")

import streamlit as st  # noqa: E402

import common  # noqa: E402
from pedagogyeval import GraderConfig  # noqa: E402

ss = st.session_state
rubs = common.rubrics()
qids = list(rubs)
ss.setdefault("grade_qid", qids[0])
ss.setdefault("grade_response", "")
if ss["grade_qid"] not in rubs:
    ss["grade_qid"] = qids[0]

# ------------------------------------------------------------------ sidebar controls
with st.sidebar:
    theme.render(theme.section_label("S", "Marking scheme"))
    model = st.radio("Model", list(common.MODELS), format_func=common.MODELS.get, key="grade_model")
    use_sem = st.toggle("Semantic matching", value=True, key="grade_sem")
    neg = st.toggle("Negation check", value=True, key="grade_neg")
    t_high = st.slider("t_high (full credit)", 0.30, 0.95, 0.70, 0.01, key="grade_thigh", disabled=not use_sem)
    t_low = st.slider("t_low (half credit)", 0.20, 0.90, 0.50, 0.01, key="grade_tlow", disabled=not use_sem)
    if t_low > t_high:
        theme.render(theme.notice("t_low > t_high: values swapped for grading.", error=True))
        t_low, t_high = t_high, t_low
    theme.render('<div class="pe-legend">Thresholds and toggles drive Model C; A and B are holistic.</div>')

cfg = GraderConfig(use_semantic=use_sem, t_high=t_high, t_low=t_low, check_negation=neg)


# ------------------------------------------------------------------ quick-load
def _load(ex: dict) -> None:
    ss["grade_qid"] = ex["question_id"]
    ss["grade_response"] = ex["response"]
    ss["graded"] = None
    ss["loaded_from"] = f"{ex['label']} ({ex['source']})"


theme.render(theme.section_label("0", "Quick load", "prepared examples, no typing needed"))
cols = st.columns(4)
for col, ex in zip(cols, common.demo_examples()):
    col.button(ex["label"], key=f"ql_{ex['key']}", on_click=_load, args=(ex,), width="stretch")

left, right = st.columns([1, 1.25], gap="large")

# ------------------------------------------------------------------ left: the form
with left:
    theme.render(theme.section_label("1", "Question"))
    st.selectbox("Question", qids, key="grade_qid", format_func=lambda q: f"{q} · {rubs[q]['question']}", label_visibility="collapsed")
    r = rubs[ss["grade_qid"]]
    theme.render(theme.field("Question", r["question"]) + theme.field("Reference answer", r["reference_answer"]) + theme.field("Max marks", theme.fmt_mark(r["max_marks"])))
    theme.render(theme.section_label("2", "Response", ss.get("loaded_from") or "type or quick-load"))
    st.text_area("Student response", key="grade_response", height=130, label_visibility="collapsed", placeholder="Student answer…")
    if st.button("GRADE", type="primary", key="grade_btn", width="stretch"):
        ss["graded"] = {"qid": ss["grade_qid"], "response": ss["grade_response"]}

# ------------------------------------------------------------------ right: the verdict
with right:
    theme.render(theme.section_label("3", "Verdict", common.MODELS[model]))
    graded = ss.get("graded")
    if not graded:
        theme.render(theme.notice("No script marked yet.\nQuick-load an example or type a response, then press GRADE."))
    elif graded["qid"] not in rubs:
        theme.render(theme.notice("The graded question no longer exists in the session rubrics.", error=True))
    else:
        result = common.grade_all(graded["qid"], graded["response"], cfg)
        c = result["C_concept"]
        if result["errors"].get("model"):
            theme.render(theme.notice("MODEL B / SEMANTIC UNAVAILABLE: " + result["errors"]["model"] + "\nModel C is running lexical-only.", error=True))
        if result["rubric_problems"]:
            theme.render(theme.notice("Rubric has problems (fix on the RUBRIC page):\n" + "\n".join(result["rubric_problems"]), error=True))

        chosen = result.get(model)
        if chosen is None:
            theme.render(theme.notice(f"{common.MODELS[model]} is unavailable (embedding model not loaded).", error=True))
        elif model == "C_concept":
            theme.render(theme.score_block(c["score"], c["max_marks"], "concept-graph partial credit"))
        else:
            theme.render(theme.score_block(chosen["score"], chosen["max_marks"], f"holistic · cosine={chosen['similarity']:.2f}"))

        theme.render(
            theme.model_strip(
                {
                    k: (common.MODELS[k], None if result.get(k) is None else result[k]["score"], c["max_marks"])
                    for k in ("A_tfidf", "B_embedding", "C_concept")
                },
                model,
            )
        )
        if model != "C_concept":
            theme.render('<div class="pe-legend">Holistic models give no concept breakdown; the analysis below is Model C\'s.</div>')

        theme.render(theme.section_label("3.1", "Marked script"))
        theme.render(theme.highlighted_response(c["response"], c["concepts"]))
        theme.render(theme.section_label("3.2", "Concept ledger"))
        theme.render(theme.concept_table(c["concepts"]))
        theme.render(theme.feedback_note(c["feedback"]))
        theme.render(theme.review_box(c["review"], c["review_reasons"]))
        theme.render(theme.section_label("3.3", "Concept graph", "edges explain; they never award marks"))
        rub = rubs[graded["qid"]]
        st.graphviz_chart(
            theme.concept_graph_dot(rub["concepts"], rub["edges"], {x["concept_id"]: x["status"] for x in c["concepts"]}),
            width="stretch",
        )
        d1, d2 = st.columns(2)
        d1.download_button("DOWNLOAD JSON", json.dumps(result, indent=2, default=str), file_name=f"grade_{graded['qid']}.json", mime="application/json", width="stretch", key="dl_json")
        d2.download_button("DOWNLOAD CSV", common.result_csv(result), file_name=f"grade_{graded['qid']}.csv", mime="text/csv", width="stretch", key="dl_csv")

theme.render('<div class="pe-foot">DECISION SUPPORT ONLY. HUMAN REVIEW REQUIRED FOR HIGH-STAKES MARKS.</div>')
