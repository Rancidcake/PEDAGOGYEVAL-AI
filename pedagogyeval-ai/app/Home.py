"""PedagogyEval AI: home page."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theme  # noqa: E402

theme.apply("Home")

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import common  # noqa: E402

t = common.team()
theme.render(theme.ascii_title("PEDAGOGYEVAL/AI  v1.0"))
theme.render(
    '<div class="pe-sub">FINE-GRAINED CONCEPT GRAPH MAPPING AND EXPLAINABLE<br>PARTIAL-CREDIT GRADING FOR SHORT ANSWERS</div>'
)
members = " &nbsp;·&nbsp; ".join(f"{theme.esc(m['name'])} <span style='color:var(--ink-soft)'>[{theme.esc(m['prn'])}]</span>" for m in t.get("members", []))
theme.render(f'<div style="margin:.8rem 0 0">{theme.esc(t.get("course", ""))} &nbsp;—&nbsp; {members}</div>')

theme.render(theme.section_label("0", "Pipeline", "concept graph is explanation only; scores come from evidence"))
st.graphviz_chart(theme.pipeline_dot(), width="stretch")

theme.render(theme.section_label("1", "How it marks"))
theme.render(
    theme.ledger_table(
        pd.DataFrame(
            [
                ["A", "TF-IDF", "cosine(reference, response) × max marks"],
                ["B", "MiniLM embeddings", "cosine, isotonic-calibrated on held-out pilot questions"],
                ["C", "Concept graph", "Σ weight × coverage per concept (lexical ∨ semantic, negation-aware)"],
            ],
            columns=["Model", "Kind", "Mark"],
        )
    )
)

metrics = common.RESULTS / "pilot_metrics.csv"
if metrics.exists():
    m = pd.read_csv(metrics).set_index("model")
    items = [(f"{common.MODEL_LABELS[k]} MAE", f"{m.loc[k, 'mae']:.2f}") for k in ("A_tfidf", "B_embedding", "C_concept") if k in m.index]
    theme.render(theme.section_label("2", "Pilot at a glance", "from results/pilot_metrics.csv"))
    theme.render(theme.stats_row(items))

theme.render(theme.section_label("3", "Pages"))
theme.render(
    theme.field("GRADE", "Quick-load an example, press GRADE, read the verdict, evidence and concept graph.")
    + theme.field("RUBRIC", "Edit concepts, weights, synonyms and relations; validated live and used immediately.")
    + theme.field("RESULTS", "Pilot, SciEntsBank and ablation tables read from results/*.csv (no recomputation).")
)
theme.render(
    '<div class="pe-foot">DECISION-SUPPORT TOOL. A HUMAN EXAMINER MUST REVIEW ANY HIGH-STAKES MARK.</div>'
)
