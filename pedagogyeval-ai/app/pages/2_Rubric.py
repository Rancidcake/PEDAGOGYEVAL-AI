"""Rubric: edit concepts, weights, synonyms and relations; validated live; used immediately by Grade."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import theme  # noqa: E402

theme.apply("Rubric")

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import common  # noqa: E402
from pedagogyeval.rubric import RELATIONS, Rubric, RubricError, rubrics_to_json  # noqa: E402

ss = st.session_state
rubs = common.rubrics()
ss.setdefault("rub_version", 0)
qids = list(rubs)
if "rub_qid_next" in ss:  # set by save/upload; applied before the selectbox is created
    ss["rub_qid"] = ss.pop("rub_qid_next")
if ss.get("rub_qid") not in rubs:
    ss["rub_qid"] = ss.get("grade_qid") if ss.get("grade_qid") in rubs else qids[0]


def _bump() -> None:
    ss["rub_version"] += 1


def _new_question() -> None:
    n = 1
    while f"Q_new_{n}" in rubs:
        n += 1
    qid = f"Q_new_{n}"
    rubs[qid] = {
        "question_id": qid, "question": "New question?", "reference_answer": "Reference answer.", "max_marks": 2,
        "concepts": [
            {"id": "c1", "name": "concept one", "description": "First key idea", "weight": 1, "synonyms": []},
            {"id": "c2", "name": "concept two", "description": "Second key idea", "weight": 1, "synonyms": []},
        ],
        "edges": [],
    }
    ss["rub_qid"] = qid
    _bump()


def _reset() -> None:
    ss["rubrics"] = {k: json.loads(json.dumps(v)) for k, v in common.file_rubrics().items()}
    ss["rub_qid"] = list(ss["rubrics"])[0]
    _bump()


theme.render(theme.section_label("R", "Rubric editor", "edits apply to this session"))
top = st.columns([3, 1, 1])
top[0].selectbox("Question", qids, key="rub_qid", on_change=_bump, format_func=lambda q: f"{q} · {rubs[q]['question']}")
top[1].button("+ NEW QUESTION", on_click=_new_question, width="stretch", key="rub_new")
top[2].button("RESET TO FILE", on_click=_reset, width="stretch", key="rub_reset")

qid = ss["rub_qid"]
cur = rubs[qid]
v = ss["rub_version"]

left, right = st.columns([1.35, 1], gap="large")
with left:
    theme.render(theme.section_label("1", "Question metadata"))
    new_id = st.text_input("question_id", cur["question_id"], key=f"rub_id_{qid}_{v}")
    question = st.text_area("question", cur["question"], height=70, key=f"rub_q_{qid}_{v}")
    reference = st.text_area("reference_answer", cur["reference_answer"], height=90, key=f"rub_ref_{qid}_{v}")
    max_marks = st.number_input("max_marks", min_value=0.5, max_value=100.0, value=float(cur["max_marks"]), step=0.5, key=f"rub_max_{qid}_{v}")

    theme.render(theme.section_label("2", "Concepts", "synonyms separated by ;  ·  * = up to 3 words"))
    cdf = pd.DataFrame(
        [{"id": c["id"], "name": c["name"], "description": c["description"], "weight": float(c["weight"]), "synonyms": "; ".join(c["synonyms"])} for c in cur["concepts"]],
        columns=["id", "name", "description", "weight", "synonyms"],
    )
    cdf_ed = st.data_editor(
        cdf, num_rows="dynamic", width="stretch", key=f"rub_concepts_{qid}_{v}", hide_index=True,
        column_config={
            "id": st.column_config.TextColumn("id", width="small", required=True),
            "name": st.column_config.TextColumn("name", required=True),
            "description": st.column_config.TextColumn("description", width="large"),
            "weight": st.column_config.NumberColumn("weight", min_value=0.0, step=0.25, format="%.2f", required=True),
            "synonyms": st.column_config.TextColumn("synonyms", width="large"),
        },
    )
    concept_ids = [str(x) for x in cdf_ed["id"].dropna() if str(x).strip()]

    theme.render(theme.section_label("3", "Edges", "relations explain; they never award marks"))
    edf = pd.DataFrame(cur["edges"], columns=["source", "target", "relation"])
    edf_ed = st.data_editor(
        edf, num_rows="dynamic", width="stretch", key=f"rub_edges_{qid}_{v}", hide_index=True,
        column_config={
            "source": st.column_config.SelectboxColumn("source", options=concept_ids or [""], required=True),
            "target": st.column_config.SelectboxColumn("target", options=concept_ids or [""], required=True),
            "relation": st.column_config.SelectboxColumn("relation", options=list(RELATIONS), required=True),
        },
    )


def _s(x) -> str:
    return "" if x is None or (isinstance(x, float) and x != x) else str(x).strip()


draft = {
    "question_id": _s(new_id),
    "question": question,
    "reference_answer": reference,
    "max_marks": float(max_marks),
    "concepts": [
        {
            "id": _s(r["id"]), "name": _s(r["name"]), "description": _s(r["description"]) or _s(r["name"]),
            "weight": float(r["weight"]) if _s(r["weight"]) else 0.0,
            "synonyms": [s.strip() for s in _s(r["synonyms"]).split(";") if s.strip()],
        }
        for _, r in cdf_ed.iterrows()
        if _s(r["id"]) or _s(r["name"])
    ],
    "edges": [
        {"source": _s(r["source"]), "target": _s(r["target"]), "relation": _s(r["relation"])}
        for _, r in edf_ed.iterrows()
        if _s(r["source"]) or _s(r["target"])
    ],
}

problems: list[str]
try:
    draft_rub = Rubric.from_dict(draft, validate=False)
    problems = draft_rub.problems()
except RubricError as exc:
    draft_rub, problems = None, [str(exc)]
if draft["question_id"] != qid and draft["question_id"] in rubs:
    problems.append(f"question_id '{draft['question_id']}' already exists")

with right:
    theme.render(theme.section_label("4", "Validation", "live"))
    total = sum(c["weight"] for c in draft["concepts"])
    ok = []
    if not any("weights sum" in p for p in problems):
        ok.append(f"weights sum to {theme.fmt_mark(total)}/{theme.fmt_mark(draft['max_marks'])}")
    if not any("edge" in p for p in problems):
        ok.append(f"{len(draft['edges'])} edge(s) reference existing concepts")
    if not problems:
        ok.append("rubric is valid")
    theme.render(theme.validation_lines([p.split("] ", 1)[-1] for p in problems], ok))

    theme.render(theme.section_label("5", "Graph preview"))
    st.graphviz_chart(theme.concept_graph_dot(draft["concepts"], [e for e in draft["edges"] if e["source"] and e["target"]]), width="stretch")

    theme.render(theme.section_label("6", "Save / transfer"))
    if st.button("SAVE TO SESSION", type="primary", width="stretch", key="rub_save", disabled=bool(problems)):
        if draft["question_id"] != qid:
            rubs.pop(qid, None)
            if ss.get("grade_qid") == qid:
                ss["grade_qid"] = draft["question_id"]
        rubs[draft["question_id"]] = draft
        ss["rub_qid_next"] = draft["question_id"]
        ss["rub_saved"] = draft["question_id"]
        _bump()
        st.rerun()
    if ss.get("rub_saved"):
        theme.render(f'<div class="pe-valid ok">✓ saved {theme.esc(ss["rub_saved"])}: the GRADE page uses it now</div>')
    if problems:
        theme.render('<div class="pe-legend">Fix the ✗ lines to enable saving.</div>')

    all_json = rubrics_to_json(Rubric.from_dict(r, validate=False) for r in rubs.values())
    st.download_button("DOWNLOAD RUBRICS JSON", all_json, file_name="rubrics.json", mime="application/json", width="stretch", key="rub_dl")
    up = st.file_uploader("Upload rubrics JSON", type=["json"], key="rub_upload")
    if up is not None and ss.get("rub_upload_done") != up.file_id:
        try:
            data = json.loads(up.getvalue().decode("utf-8"))
            items = data["rubrics"] if isinstance(data, dict) and "rubrics" in data else data
            items = [items] if isinstance(items, dict) else items
            loaded = [Rubric.from_dict(d) for d in items]
            for rb in loaded:
                rubs[rb.question_id] = rb.to_dict()
            ss["rub_upload_done"] = up.file_id
            ss["rub_qid_next"] = loaded[0].question_id
            _bump()
            st.rerun()
        except (RubricError, ValueError, KeyError, TypeError) as exc:
            theme.render(theme.notice(f"Upload rejected:\n{exc}", error=True))
