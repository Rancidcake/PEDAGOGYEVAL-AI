"""Results: reads results/*.csv only (no recomputation)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import theme  # noqa: E402

theme.apply("Results")

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import common  # noqa: E402

R = common.RESULTS
P = theme.PALETTE
MODEL_ORDER = ["A_tfidf", "B_embedding", "C_concept"]
ABL_ORDER = ["C_concept", "C_no_semantic", "C_uniform_weights", "C_no_negation"]
MODEL_SCALE = alt.Scale(domain=[common.MODEL_LABELS[m] for m in MODEL_ORDER], range=[theme.MODEL_COLORS[m] for m in MODEL_ORDER])
ABL_SCALE = alt.Scale(domain=[common.MODEL_LABELS[m] for m in ABL_ORDER], range=[P["redpen"], P["ink-soft"], P["ochre"], P["moss"]])
RUN_HINT = "Run notebooks/PedagogyEval_Experiments.ipynb (all cells) to create it, then reload this page."


def read(name: str) -> pd.DataFrame | None:
    p = R / name
    return pd.read_csv(p) if p.exists() else None


def missing(*names: str) -> bool:
    gone = [n for n in names if not (R / n).exists()]
    if gone:
        theme.render(theme.notice("RESULTS NOT FOUND\n  " + "\n  ".join(f"results/{n}" for n in gone) + "\n\n" + RUN_HINT))
    return bool(gone)


def bars(df: pd.DataFrame, x: str, y: str, color: str, scale, title: str, fmt: str = ".2f", offset: bool = False, legend: bool = True):
    extra = {"xOffset": alt.XOffset(f"{color}:N", sort=None)} if offset else {}
    base = alt.Chart(df).encode(
        **extra,
        x=alt.X(f"{x}:N", title=None, sort=None, axis=alt.Axis(labelAngle=0, labelLimit=160)),
        y=alt.Y(f"{y}:Q", title=y.upper().replace("_", " "), scale=alt.Scale(domain=[0, float(df[y].max()) * 1.18 or 1], nice=False)),
        color=alt.Color(f"{color}:N", scale=scale, legend=alt.Legend(orient="bottom", title=None, columns=4, labelLimit=220) if legend else None),
        tooltip=[x, color, alt.Tooltip(f"{y}:Q", format=".3f")],
    )
    chart = base.mark_bar(stroke=P["ink"], strokeWidth=1) + base.mark_text(dy=-7, fontSize=10, color=P["ink"]).encode(text=alt.Text(f"{y}:Q", format=fmt))
    return chart.properties(height=240, title=title)


def confusion_chart(cm: pd.DataFrame, title: str, best: bool):
    order = ["correct", "contradictory", "incorrect"]
    labels = [l for l in order if l in set(cm.gold)]
    cm = cm.copy()
    cm["share"] = cm["count"] / cm.groupby("gold")["count"].transform("sum").clip(lower=1)
    base = alt.Chart(cm).encode(
        x=alt.X("pred:N", sort=labels, title="PREDICTED", axis=alt.Axis(labelAngle=0, labelFontSize=10)),
        y=alt.Y("gold:N", sort=labels, title="GOLD", axis=alt.Axis(labelFontSize=10)),
    )
    heat = base.mark_rect(stroke=P["paper"]).encode(
        color=alt.Color("share:Q", scale=alt.Scale(domain=[0, 1], range=theme.SEQ_RANGE), legend=None),
        tooltip=["gold", "pred", "count", alt.Tooltip("share:Q", format=".0%")],
    )
    txt = base.mark_text(fontSize=12, fontWeight="bold").encode(
        text="count:Q", color=alt.condition("datum.share > 0.5", alt.value(P["paper"]), alt.value(P["ink"]))
    )
    view = {"stroke": P["redpen"], "strokeWidth": 3} if best else {"stroke": P["ink"], "strokeWidth": 1}
    return (heat + txt).properties(width=alt.Step(84), height=alt.Step(46), title=("▶ " if best else "") + title).configure_view(**view)


theme.render(theme.section_label("R", "Results ledger", "read-only · results/*.csv"))
tab_pilot, tab_sb, tab_abl = st.tabs(["PILOT", "SCIENTSBANK", "ABLATIONS"])

# ------------------------------------------------------------------ PILOT
with tab_pilot:
    if not missing("pilot_metrics.csv", "pilot_by_type.csv"):
        m = read("pilot_metrics.csv")
        m = m[m.model.isin(MODEL_ORDER)].reset_index(drop=True)
        best = m.mae.idxmin()
        show = m[["label", "n", "mae", "rmse", "exact_match", "norm_mae"]].rename(columns={"label": "model", "exact_match": "exact", "norm_mae": "mae/max"})
        theme.render(theme.section_label("1", "Pilot metrics", "24 authored responses · lower MAE is better"))
        theme.render(theme.ledger_table(show, highlight_index=best, num_cols=["mae", "rmse", "exact", "mae/max"]))
        theme.render('<div class="pe-legend">Exact = rounded mark equals gold. Pilot data is authored test data, not a benchmark.</div>')
        bt = read("pilot_by_type.csv")
        bt = bt[bt.model.isin(MODEL_ORDER)].assign(model=lambda d: d.model.map(common.MODEL_LABELS))
        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.altair_chart(bars(m.assign(model=m.label.str.split(" · ").str[0]), "model", "mae", "model",
                                 alt.Scale(domain=["A", "B", "C"], range=MODEL_SCALE.range), "MAE BY MODEL", legend=False), width="stretch")
        with c2:
            st.altair_chart(bars(bt, "response_type", "mae", "model", MODEL_SCALE, "MAE BY RESPONSE TYPE", offset=True), width="stretch")
        s = read("pilot_sample_outputs.csv")
        if s is not None:
            theme.render(theme.section_label("2", "Sample outputs", "gold vs A | B | C"))
            cols = ["response_id", "response", "gold", "A", "B", "C", "review", "missing"]
            st.dataframe(s[[c for c in cols if c in s]].round(2), hide_index=True, width="stretch", height=320)
        w = read("worst_errors_pilot.csv")
        if w is not None:
            with st.expander("Worst errors per model (pilot)"):
                st.dataframe(w.round(2), hide_index=True, width="stretch")

# ------------------------------------------------------------------ SCIENTSBANK
with tab_sb:
    if not missing("scientsbank_metrics.csv", "scientsbank_confusion.csv"):
        sm = read("scientsbank_metrics.csv")
        cm = read("scientsbank_confusion.csv")
        theme.render(
            theme.notice(
                "SciEntsBank (SemEval-2013 Task 7). Thresholds tuned on dev (carved from train), applied unchanged to the official "
                "test splits. Model C's concepts are AUTO-DERIVED from reference answers (approximation, not expert rubrics)."
            )
        )
        c1, c2 = st.columns(2)
        task = c1.radio("Task", sorted(sm.task.unique()), horizontal=True, key="res_task", format_func=lambda t: t.upper())
        splits = [s for s in ["dev", "unseen-answers", "unseen-questions", "unseen-domains"] if s in set(sm.split)]
        split = c2.selectbox("Split for confusion matrices", splits, index=min(1, len(splits) - 1), key="res_split")
        sub = sm[(sm.task == task) & sm.model.isin(MODEL_ORDER)]
        piv = sub.pivot_table(index="model", columns="split", values="macro_f1").reindex(MODEL_ORDER)[splits]
        piv.insert(0, "model", [common.MODEL_LABELS[m] for m in piv.index])
        piv = piv.reset_index(drop=True)
        best_model = sub[sub.split != "dev"].groupby("model").macro_f1.mean().idxmax()
        theme.render(theme.section_label("1", f"Macro-F1 · {task}", "best model by mean test macro-F1 in red"))
        theme.render(theme.ledger_table(piv, highlight_index=MODEL_ORDER.index(best_model), num_cols=splits))
        detail = sub[sub.split == split][["label", "accuracy", "macro_f1", "weighted_f1", "threshold", "n"]].rename(columns={"label": "model"}).reset_index(drop=True)
        theme.render(theme.section_label("2", f"All metrics · {split}"))
        theme.render(theme.ledger_table(detail, highlight_index=int(detail.macro_f1.idxmax()), num_cols=["accuracy", "macro_f1", "weighted_f1", "threshold"]))
        chart_df = sub[sub.split != "dev"].assign(model=lambda d: d.model.map(common.MODEL_LABELS))
        st.altair_chart(bars(chart_df, "split", "macro_f1", "model", MODEL_SCALE, f"MACRO-F1 BY TEST SPLIT · {task.upper()}", offset=True), width="stretch")
        theme.render(theme.section_label("3", f"Confusion matrices · {split}", "paper→ink = share of gold row; ▶ = best model"))
        cc = st.columns(3)
        for col, mname in zip(cc, MODEL_ORDER):
            part = cm[(cm.model == mname) & (cm.task == task) & (cm.split == split)]
            if len(part):
                col.altair_chart(confusion_chart(part, common.MODEL_LABELS[mname], mname == best_model), width="content")
        bal = read("scientsbank_class_balance.csv")
        if bal is not None:
            with st.expander("Class balance (computed from the dataset)"):
                st.dataframe(bal, hide_index=True, width="stretch")
        w = read("worst_errors_scientsbank.csv")
        if w is not None:
            with st.expander("Worst errors per model (test splits, 3-way)"):
                st.dataframe(w.round(3), hide_index=True, width="stretch")

# ------------------------------------------------------------------ ABLATIONS
with tab_abl:
    if not missing("ablations_pilot.csv"):
        ap = read("ablations_pilot.csv").reset_index(drop=True)
        theme.render(theme.section_label("1", "Pilot ablations", "Model C variants · lower MAE is better"))
        show = ap[["label", "mae", "rmse", "exact_match", "delta_mae_vs_full"]].rename(columns={"label": "variant", "exact_match": "exact", "delta_mae_vs_full": "Δmae vs full"})
        theme.render(theme.ledger_table(show, highlight_index=int(ap.mae.idxmin()), num_cols=["mae", "rmse", "exact", "Δmae vs full"]))
        if "note" in ap and ap.note.notna().any():
            theme.render('<div class="pe-legend">' + "<br>".join(theme.esc(n) for n in ap.note.dropna().unique()) + "</div>")
        st.altair_chart(bars(ap.assign(variant=ap.label), "variant", "mae", "variant", ABL_SCALE, "PILOT MAE BY VARIANT"), width="stretch")
    asb = read("ablations_scientsbank.csv")
    if asb is not None:
        theme.render(theme.section_label("2", "SciEntsBank ablations", "macro-F1 · mean of 3 test splits · red = best 3-way"))
        piv = asb.pivot_table(index="label", columns="task", values="macro_f1_test_mean").reindex([common.MODEL_LABELS[m] for m in ABL_ORDER]).reset_index().rename(columns={"label": "variant"})
        cols = [c for c in piv.columns if c != "variant"]
        theme.render(theme.ledger_table(piv, highlight_index=int(piv[cols[-1]].idxmax()), num_cols=cols))
        long = asb.assign(variant=asb.label)
        st.altair_chart(bars(long, "task", "macro_f1_test_mean", "variant", ABL_SCALE, "MEAN TEST MACRO-F1 BY VARIANT", fmt=".3f", offset=True), width="stretch")
    else:
        missing("ablations_scientsbank.csv")
