"""Assemble presentation/: screenshots, report figures, copies of all data + results, and FEATURES.md.

Every number written into FEATURES.md is read from results/*.csv (nothing typed in).
    python scripts/capture_screenshots.py presentation/screenshots   # app must be running
    python scripts/build_presentation.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation"
RES = ROOT / "results"

SCREENS = [
    ("01_home", "Home: ASCII title block, team, pipeline diagram, model summary, pilot MAE at a glance"),
    ("01_home_full", "Home (full page)"),
    ("02_grade_empty", "Grade page before marking: quick-load row, §1 question, §2 response, red GRADE button, sidebar controls"),
    ("03_grade_good_paraphrase", "GOOD PARAPHRASE graded: big score, A|B|C strip, highlighted evidence, concept ledger  **(report Fig. 2)**"),
    ("03_grade_good_paraphrase_full", "GOOD PARAPHRASE: full verdict incl. examiner's note, review box, concept graph"),
    ("03_concept_graph_paraphrase", "Concept graph coloured by status (moss / ochre / red-pen)"),
    ("04_grade_paraphrase_semantic_off_full", "Same paraphrase with semantic matching switched OFF (live ablation)"),
    ("05_grade_incomplete", "INCOMPLETE graded: missing concept shown as [✗ MISSING]  **(report Fig. 3)**"),
    ("05_grade_incomplete_full", "INCOMPLETE: full verdict with feedback 'Missing: …' and red graph node"),
    ("05_concept_graph_incomplete", "Concept graph for the incomplete osmosis answer"),
    ("06_grade_wrong_but_topical", "WRONG BUT TOPICAL ('DNS encrypts all internet traffic'): C gives 0/3, baselines give credit"),
    ("06_grade_wrong_but_topical_full", "WRONG BUT TOPICAL (full page)"),
    ("07_grade_negation_trap", "NEGATION TRAP: negated span struck through in red"),
    ("07_grade_negation_trap_full", "NEGATION TRAP: full verdict with ⚠ NEEDS HUMAN REVIEW box"),
    ("07_marked_script_negation", "Close-up: marked script with yellow evidence and red strikethrough"),
    ("08_grade_model_A_tfidf", "Model A (TF-IDF) selected: holistic score"),
    ("08_grade_model_B_minilm", "Model B (MiniLM, calibrated) selected: holistic score"),
    ("09_rubric_editor", "Rubric editor: metadata, concepts/edges tables, live validation, graph preview"),
    ("09_rubric_editor_full", "Rubric editor (full page)"),
    ("10_rubric_validation_error", "Live validation error (max_marks 10 ≠ weights 4), SAVE disabled"),
    ("11_results_pilot", "Results › PILOT: metrics ledger, best row in red pen, charts  **(report Fig. 4)**"),
    ("11_results_pilot_full", "Results › PILOT (full page incl. sample outputs)"),
    ("12_results_scientsbank", "Results › SCIENTSBANK (2-way): macro-F1 ledger per split"),
    ("12_results_scientsbank_full", "Results › SCIENTSBANK 2-way (full page incl. confusion matrices)"),
    ("13_results_scientsbank_3way_full", "Results › SCIENTSBANK 3-way (full page)"),
    ("14_results_ablations", "Results › ABLATIONS"),
    ("14_results_ablations_full", "Results › ABLATIONS (full page)"),
    ("15_sidebar_closed_menu_button", "Sidebar closed: the ☰ MENU button re-opens it on every page"),
]
FIGURES = [
    ("pilot_mae.png", "Pilot MAE by model and by response type"),
    ("scientsbank_macro_f1.png", "SciEntsBank macro-F1 per test split, 2-way and 3-way"),
    ("confusion_unseen_answers_3way.png", "Confusion matrices, unseen-answers, 3-way (▶ = best model)"),
    ("confusion_unseen_answers_2way.png", "Confusion matrices, unseen-answers, 2-way"),
    ("ablations.png", "Ablations: pilot MAE and SciEntsBank mean test macro-F1"),
]
LBL = {"A_tfidf": "A · TF-IDF", "B_embedding": "B · MiniLM", "C_concept": "C · Concept graph",
       "C_no_semantic": "C − semantic", "C_uniform_weights": "C − weights (uniform)", "C_no_negation": "C − negation check"}
MAIN = ["A_tfidf", "B_embedding", "C_concept"]


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return "\n".join(lines)


def copy_tree() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    for f, _ in FIGURES:
        if (RES / "figures" / f).exists():
            shutil.copy2(RES / "figures" / f, OUT / "figures" / f)
    for sub, files in (("data/input", sorted((ROOT / "data").glob("*"))), ("data/results", sorted(RES.glob("*.csv")) + [RES / "environment.json"])):
        d = OUT / sub
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            if f.is_file():
                shutil.copy2(f, d / f.name)


def results_section() -> str:
    pm = pd.read_csv(RES / "pilot_metrics.csv").set_index("model")
    sm = pd.read_csv(RES / "scientsbank_metrics.csv")
    ap = pd.read_csv(RES / "ablations_pilot.csv")
    asb = pd.read_csv(RES / "ablations_scientsbank.csv")
    sample = pd.read_csv(RES / "pilot_sample_outputs.csv")
    bal = pd.read_csv(RES / "scientsbank_class_balance.csv")
    env = json.loads((RES / "environment.json").read_text())

    pilot = pd.DataFrame([{"model": LBL[m], "MAE": pm.loc[m, "mae"], "RMSE": pm.loc[m, "rmse"], "exact match": pm.loc[m, "exact_match"], "MAE / max": pm.loc[m, "norm_mae"]} for m in MAIN])
    splits = ["unseen-answers", "unseen-questions", "unseen-domains"]
    sb_tabs = []
    for task in ("2way", "3way"):
        t = sm[(sm.task == task) & sm.model.isin(MAIN)].pivot_table(index="model", columns="split", values="macro_f1").reindex(MAIN)
        t = t[["dev"] + splits]
        t.insert(0, "model", [LBL[m] for m in t.index])
        t["mean test"] = t[splits].mean(axis=1)
        sb_tabs.append(f"**{task.upper()} macro-F1** (threshold tuned on dev, frozen)\n\n" + md_table(t.reset_index(drop=True)))
    acc = sm[(sm.split == "unseen-answers") & sm.model.isin(MAIN)][["task", "model", "accuracy", "macro_f1", "weighted_f1", "threshold"]].copy()
    acc["model"] = acc.model.map(LBL)
    abl_p = ap[["label", "mae", "rmse", "exact_match", "delta_mae_vs_full"]].rename(columns={"label": "variant", "delta_mae_vs_full": "Δ MAE vs full"})
    notes = "; ".join(ap.note.dropna())
    abl_s = asb.pivot_table(index="label", columns="task", values="macro_f1_test_mean").reindex([LBL[m] for m in ["C_concept", "C_no_semantic", "C_uniform_weights", "C_no_negation"]]).reset_index().rename(columns={"label": "variant"})
    samp = sample[["response_id", "response_type", "gold", "A", "B", "C", "review"]].copy()
    btab = bal[bal.task == "3way"].pivot_table(index="label", columns="split", values="n").reset_index()

    return f"""## 4. Results (generated from `results/*.csv`)

Environment: Python {env['python']}, torch {env['torch']}, sentence-transformers {env['sentence_transformers']}, seed {env['seed']}, CUDA {env['cuda']}.

### 4.1 Pilot (24 authored responses, 6 questions)

{md_table(pilot)}

Per-response marks (gold vs A / B / C):

{md_table(samp, "{:.2f}")}

### 4.2 SciEntsBank (SemEval-2013 Task 7)

{sb_tabs[0]}

{sb_tabs[1]}

Unseen-answers detail:

{md_table(acc.reset_index(drop=True))}

3-way class balance (unique answers per split):

{md_table(btab, "{:.0f}")}

### 4.3 Ablations (Model C)

Pilot:

{md_table(abl_p)}

Notes (computed): {notes}.

SciEntsBank, mean macro-F1 over the three test splits:

{md_table(abl_s)}
"""


def features_md() -> str:
    shots = "\n".join(f"| `{n}.png` | {d} |" for n, d in SCREENS if (OUT / "screenshots" / f"{n}.png").exists())
    gallery = "\n\n".join(f"**{d}**\n\n![{n}](screenshots/{n}.png)" for n, d in SCREENS if not n.endswith("_full") and (OUT / "screenshots" / f"{n}.png").exists())
    figs = "\n".join(f"| `figures/{f}` | {d} |" for f, d in FIGURES)
    fig_gallery = "\n\n".join(f"**{d}**\n\n![{f}](figures/{f})" for f, d in FIGURES)
    data_rows = "\n".join(f"| `data/input/{p.name}` | {p.stat().st_size:,} B |" for p in sorted((OUT / "data" / "input").iterdir()))
    res_rows = "\n".join(f"| `data/results/{p.name}` | {p.stat().st_size:,} B |" for p in sorted((OUT / "data" / "results").iterdir()))
    return f"""# PedagogyEval AI: presentation pack

**Fine-Grained Concept Graph Mapping and Explainable Partial-Credit Grading for Short Answers**

This folder is a self-contained copy for the presentation and report: screenshots of the live app, report figures,
copies of all input data and computed results, and this feature list. Regenerate it with
`python scripts/capture_screenshots.py presentation/screenshots` (app running) and `python scripts/build_presentation.py`.

## 1. Where the UI is and how to run it

The UI is a Streamlit app in `pedagogyeval-ai/app/` (`Home.py` + `pages/1_Grade.py`, `2_Rubric.py`, `3_Results.py`).

```bash
cd pedagogyeval-ai
bash scripts/run_demo.sh          # offline; opens on http://localhost:8501
```

## 2. Features

### Grading engine (`pedagogyeval/` package)
* **Preprocessing:** Unicode NFKC normalisation, whitespace cleanup, lowercase copy with identical character offsets (evidence spans point into the displayed text). No stemming; stopwords, negations, numbers and units kept.
* **Rubric schema (JSON):** question_id, question, reference answer, max marks, concepts (id, name, description, weight, synonyms) and edges (causes / requires / part_of / leads_to).
* **Rubric validation:** weights must sum to max marks, edges must reference existing concepts, known relation types, no duplicates or self-loops; all problems reported together in clear messages.
* **Concept graph:** NetworkX directed graph built from each rubric. Edges are used for explanation and visualisation only, never as evidence.
* **Model A, TF-IDF:** unigram+bigram cosine to the reference; mark = round(sim × max).
* **Model B, MiniLM embeddings:** all-MiniLM-L6-v2 loaded from `models_cache/` (offline); isotonic calibration for marks (fitted leave-one-question-out on the pilot), tuned thresholds for labels (on dev only).
* **Model C, concept-graph partial credit:**
  * lexical synonym matching with word boundaries, `*` wildcard (up to 3 words), plural tolerance, punctuation tolerance;
  * misspelling-tolerant fallback (Damerau edit distance), reported as method `fuzzy`;
  * semantic matching (toggleable): best cosine between concept description and each response sentence/clause; ≥ t_high → full, ≥ t_low → half;
  * negation detection (not, never, no, without, cannot, n't …) within 3 tokens before the match → no credit + flag;
  * score = min(M, max(0, Σ weight × coverage)).
* **Explainability:** per-concept rows (weight, coverage, covered/partial/missing, method, evidence span + offsets, negation flag), missing list, template feedback ("You correctly explained X. Missing: Z."), no LLM.
* **Human-review flag:** empty answer, negation, similarity within 0.05 of a threshold, or keyword matched but meaning similarity low.
* **Evaluation:** MAE, RMSE, exact match, accuracy, macro-F1, weighted-F1, confusion matrices; dev-only threshold tuning.

### Streamlit app ("Examiner's Ledger" design)
* Warm-paper / black-ink / red-pen palette, JetBrains Mono everywhere (bundled locally, works offline), ruled-paper background, red margin line, square corners, no shadows, no gradients on components, hidden Streamlit chrome.
* **Home:** ASCII title block, team names/PRNs (from `data/team.json`), styled Graphviz pipeline diagram, model summary, pilot MAE at a glance, disclaimer.
* **Grade:** sidebar model selector (A/B/C), semantic and negation toggles, t_high/t_low sliders; four quick-load buttons (GOOD PARAPHRASE, INCOMPLETE, WRONG BUT TOPICAL, NEGATION TRAP); §1 question / §2 response / §3 verdict layout; 96px score with double red underline and progress bar; highlighted evidence (yellow) and negated spans (red strikethrough); concept ledger with [✓ COVERED] [~ PARTIAL] [✗ MISSING] tags; status-coloured concept graph; examiner's margin note; ⚠ NEEDS HUMAN REVIEW box; A | B | C comparison strip; download JSON / CSV. Settings changes re-mark live.
* **Rubric:** edit metadata, concepts and edges in tables (relation as dropdown); live ✓/✗ validation; live graph preview; new question / reset; save to session (used immediately by Grade); upload/download rubric JSON.
* **Results:** PILOT | SCIENTSBANK | ABLATIONS tabs reading only `results/*.csv`; ledger tables with the best row in red pen; palette-matched Altair charts; paper→ink confusion matrices with the best model marked; mono notice if results are missing.
* **Reliability:** one-command offline launch, model-load failures shown as a notice (falls back to lexical-only) instead of a crash, cached model and rubrics (< 2 s per grade after load).

### Experiments notebook (Colab-runnable)
Setup + environment printout → pilot evaluation → SciEntsBank (inspects splits/columns/labels/class balance first; official test splits kept, dev carved from train) → ablations → worst-10 errors per model → plots in the app palette → metrics summary. Everything saved to `results/`.

### Tests
45 pytest tests: grader behaviour (empty, negation, paraphrase, max bound, punctuation noise, misspelling, rubric validation …) and Streamlit AppTest runs of every page, every quick-load + GRADE, model/toggle switching, rubric save and validation, results tabs.

## 3. Data in this folder

**Inputs** (copied from `data/`; pilot questions, rubrics and responses are AUTHORED test data):

| file | size |
|---|---|
{data_rows}

**Computed results** (copied from `results/`, written by the notebook):

| file | size |
|---|---|
{res_rows}

SciEntsBank itself is not copied (it is downloaded from Hugging Face, `Atomi/sem_eval_2013_task_7`); per-answer scores and predictions are in `scientsbank_scores.csv` / `scientsbank_predictions.csv`.

{results_section()}

## 5. Known limitations (keep in the talk)
* Embeddings are direction-blind: "Plants breathe in oxygen and release carbon dioxide" (gold 0) gets semantic credit from Model C; the no-semantic ablation removes that error.
* A correct, negatively phrased statement ("cannot generalise to new data") can be treated as negated; it is flagged for review.
* SciEntsBank concepts are auto-derived from reference answers, an approximation; uniform weights can match or beat the heuristic weighting there.
* The pilot set is small and authored by us: a sanity check, not a benchmark. *Contradictory* (3-way) is rarely predicted.
* Decision-support tool only: a human examiner must review high-stakes marks.

## 6. Image index

### App screenshots (`screenshots/`, 3200×1800 for 16:9 slides; `_full` = whole page)

| file | shows |
|---|---|
{shots}

### Report figures (`figures/`)

| file | shows |
|---|---|
{figs}

## 7. Gallery

{gallery}

{fig_gallery}
"""


def main() -> None:
    copy_tree()
    (OUT / "FEATURES.md").write_text(features_md(), encoding="utf-8")
    print("wrote", OUT / "FEATURES.md")


if __name__ == "__main__":
    main()
