# PedagogyEval AI

**Fine-Grained Concept Graph Mapping and Explainable Partial-Credit Grading for Short Answers** (NLP lab report)

```
┌────────────────────────────────┐
│ PEDAGOGYEVAL/AI  v1.0          │
└────────────────────────────────┘
```

Three graders, one shared package, one Colab notebook, one Streamlit app ("Examiner's Ledger"):

| Model | Idea | Output |
|---|---|---|
| **A** TF-IDF | cosine(reference, response), unigram+bigram | `round(sim × M)` |
| **B** MiniLM | all-MiniLM-L6-v2 cosine, isotonic calibration (marks) / tuned threshold (labels) | calibrated mark |
| **C** Concept graph | per concept: lexical synonym match ∨ semantic (t_low / t_high) unless negated; `min(M, Σ wᵢ·pᵢ)` | mark + evidence, feedback, review flag |

Python 3.11, seed 42 everywhere. **No result is hard-coded**: every number is computed by the notebook and saved to `results/*.csv`; the app's Results page only reads those files.

---

## Project layout

```
pedagogyeval/            shared package (all grading logic)
  preprocess.py          NFKC, whitespace, offset-preserving lowercase copy, sentence/clause units
  rubric.py              schema, JSON load/save, validation (clear errors), NetworkX graph
  models/                tfidf_baseline.py (A) · embedding_baseline.py (B + shared Embedder) · concept_grader.py (C)
  explain.py             template feedback + review flag
  evaluate.py            MAE, RMSE, exact match, accuracy, macro/weighted-F1, confusion, dev threshold tuning
  data.py                pilot set + SciEntsBank loader (inspects first) + approximate concept derivation
  experiments.py         pilot / SciEntsBank / ablation / worst-error runners (used by notebook and app)
data/pilot.csv           24 AUTHORED responses (6 questions × full / partial / wrong-but-topical / paraphrase)
data/rubrics.json        6 AUTHORED rubrics (3–4 equal-weight concepts each)
data/demo_examples.json  the four quick-load examples
data/team.json           names + PRNs shown on the Home page (edit this, no code changes)
models_cache/            MiniLM saved locally for the offline demo
notebooks/PedagogyEval_Experiments.ipynb
app/Home.py · app/pages/1_Grade.py · 2_Rubric.py · 3_Results.py
app/theme.py             ALL CSS + HTML helpers + palette (also used for Altair and matplotlib)
app/common.py            cached model/rubrics, session rubric store, A/B/C grading for the app
app/static/fonts/        JetBrains Mono (served locally so the design survives with Wi-Fi off)
scripts/download_model.py · scripts/run_demo.sh
results/                 CSVs + figures written by the notebook
tests/test_grader.py · tests/test_app.py
```

---

## 1. Setup (once, while online)

```bash
cd pedagogyeval-ai
python3.11 -m venv .venv && source .venv/bin/activate     # or: uv venv --python 3.11 .venv
pip install -r requirements.txt                          # or: uv pip install -r requirements.txt
python scripts/download_model.py                         # saves MiniLM into models_cache/ (~87 MB)
```

## 2. Run the tests

```bash
HF_HUB_OFFLINE=1 python -m pytest tests/ -q
```

`tests/test_grader.py` covers: empty answer → 0 + flag; `"Plants do not use sunlight"` → negation flag, no credit;
a paraphrase outside the synonym list (lexical misses, semantic catches); score ≤ max (including keyword stuffing and
over-weighted rubrics); punctuation noise; misspelling; rubric validation errors.
`tests/test_app.py` uses `streamlit.testing.v1.AppTest` to load every page, click each quick-load button then GRADE,
switch models/toggles, edit and save a rubric, and render the Results tabs, asserting no exceptions.

## 3. Run the experiments notebook

**Locally:** open `notebooks/PedagogyEval_Experiments.ipynb` from the `notebooks/` folder and *Run All* (~2 min on a laptop CPU).

**On Google Colab:**
1. Zip the project folder as `pedagogyeval-ai.zip` (including `models_cache/` is optional; without it MiniLM is downloaded).
2. In Colab: *File → Upload notebook* → `notebooks/PedagogyEval_Experiments.ipynb`.
3. Upload `pedagogyeval-ai.zip` to `/content` (Files pane), or set `REPO_URL` in the first code cell to a git URL.
4. *Runtime → Run all.* Download `/content/pedagogyeval-ai/results/` and copy it into your local `results/`.

Sections: 0 setup + environment printout → 1 pilot (MAE, RMSE, exact match, sample outputs) → 2 SciEntsBank
(prints splits/columns/labels/class balance first; accuracy, macro-F1, weighted-F1, confusion per split) →
3 ablations (no semantic / uniform weights / no negation check) → 4 worst 10 errors per model → 5 plots from the saved
CSVs in the app palette + JetBrains Mono → 6 metrics summary (`results/metrics_summary.csv`).

**Protocols (so the numbers are honest):**
* Pilot: Model A's TF-IDF is fitted on reference answers only; Model B's isotonic calibrator is fitted
  **leave-one-question-out** (never on the question being graded). The app uses the same LOQO calibrator.
* SciEntsBank (`Atomi/sem_eval_2013_task_7`, SciEntsBank rows only, 2-way and 3-way labels): official test splits kept
  (unseen-answers / unseen-questions / unseen-domains); dev = 15 % of train, stratified, seed 42. One threshold per
  model × task is tuned on dev for macro-F1 and frozen. In 3-way, Model C predicts *contradictory* when it is below
  threshold and a negation flag fired.
* SciEntsBank has no concept rubrics, so Model C's concepts are **auto-derived** from each reference answer
  (clauses + stopword-delimited chunks, weights ∝ content words). This is an approximation and is labelled as such everywhere.

## 4. Launch the demo (one command, offline)

```bash
bash scripts/run_demo.sh            # → http://localhost:8501   (PORT=8502 bash scripts/run_demo.sh)
```

The script activates `.venv` if present, sets `HF_HUB_OFFLINE=1` (plus the transformers/datasets offline flags) and runs
`streamlit run app/Home.py`. With Wi-Fi off the app starts, grades and renders in its own fonts (served from
`app/static/fonts`). If the model folder is missing, the app does **not** crash: it shows a red notice and Model C falls back
to lexical-only grading. **Do one full rehearsal with Wi-Fi off before the viva.**

---

## 5. Demo script (≈4 minutes)

Open **GRADE**. Sidebar: model **C · CONCEPT GRAPH**, semantic **on**, negation **on**, t_high 0.70, t_low 0.50.

| # | Click | Then | What to say |
|---|---|---|---|
| 1 | **GOOD PARAPHRASE** | GRADE | "No exact keywords for *light* or *oxygen release*: 'sunshine' and 'giving off oxygen' were caught by the **semantic** channel (half credit each, cos shown in the ledger). Yellow marks show the evidence and superscripts say which concept. Compare the A / B / C strip: holistic scores say nothing about *what* is missing." |
| 2 | Sidebar: toggle **Semantic matching off** | (verdict updates) | "Lexical-only C loses the paraphrased concepts; this is the *no semantic* ablation on the Results page." Toggle it back on. |
| 3 | **INCOMPLETE** | GRADE | "2/3. The ledger shows exactly what is missing: **[✗ MISSING] dilute to concentrated**, and the margin note tells the student. The concept graph colours that node red-pen." |
| 4 | **WRONG BUT TOPICAL** | GRADE | "'DNS encrypts all internet traffic': on topic, uses the key term, gets **0/3** from C, while the similarity baselines still give it credit (see the strip)." |
| 5 | **NEGATION TRAP** | GRADE | "'Plants do **not** need sunlight…': the sunlight span is struck through in red, the concept gets no credit, and the boxed **⚠ NEEDS HUMAN REVIEW** explains why. It's a decision-support tool: fragile cases are flagged, not hidden." |
| 6 | **RUBRIC** page | add a synonym (e.g. `sunshine` to c1), SAVE TO SESSION, back to GRADE, GOOD PARAPHRASE → GRADE | "Rubrics are editable live, validated (✓/✗ lines), and used immediately. *Light energy* is now matched lexically, so the paraphrase goes from 3 to 3.5/4." |
| 7 | **RESULTS** page | PILOT → SCIENTSBANK → ABLATIONS | "Everything here is read from `results/*.csv` written by the notebook; nothing is recomputed or typed in. On SciEntsBank, C's concepts are auto-derived, so treat those numbers as a lower bound." |

Optional: download the verdict as **JSON/CSV** from the bottom of the verdict column.

### Screenshots for the report

| Figure | Page / action | Capture |
|---|---|---|
| **Fig. 2**, good paraphrase | GRADE → GOOD PARAPHRASE → GRADE | score block, A/B/C strip, marked script, concept ledger (semantic rows visible) |
| **Fig. 3**, incomplete answer with the missing concept | GRADE → INCOMPLETE → GRADE | ledger row **[✗ MISSING] dilute to concentrated**, examiner's note, concept graph with the red-pen node |
| **Fig. 4**, results dashboard | RESULTS → PILOT (metrics ledger + charts), optionally SCIENTSBANK confusion matrices | |

Static figures for the report body are also in `results/figures/` (`pilot_mae.png`, `scientsbank_macro_f1.png`,
`confusion_unseen_answers_{2way,3way}.png`, `ablations.png`).

---

## Grading details

* **Preprocessing:** NFKC, whitespace collapse, lowercase copy with identical offsets (evidence spans index the displayed
  text). No stemming; stopwords, negations, numbers and units kept.
* **Lexical:** synonym regex with word boundaries; `*` in a synonym = up to 3 intervening words (e.g. `use * carbon dioxide`);
  optional plural `s/es` on the last word; punctuation between words tolerated. If no exact match, a misspelling-tolerant
  fallback (Damerau distance ≤ 1 for words ≥ 6 letters, ≤ 2 for ≥ 9, same first letter) is used and reported as `fuzzy`.
* **Semantic:** max cosine between the concept description and each response sentence *or clause*: ≥ t_high → 1.0, ≥ t_low → 0.5.
* **Negation:** `not, never, no, without, cannot, nor, none, neither` and `…n't` within 3 tokens before the match (scope stops at
  `, ; . : ! ?`) → p = 0 and a flag. Negations that are part of the synonym itself (`cannot be null`) don't count.
* **Review flag:** empty answer; negation; a semantic score within 0.05 of t_low/t_high where that concept's credit depends on
  the semantic channel; or lexical/semantic disagreement (keyword matched but cosine < 0.30, i.e. a possible out-of-context keyword).
* **Edges** (`causes | requires | part_of | leads_to`) are drawn for explanation only and never add marks.

## Known limitations (found by our own tests, keep them in the report)

* **Direction blindness of embeddings.** "Plants breathe in oxygen and release carbon dioxide" (gold 0) gets semantic credit for
  *carbon dioxide intake*, because MiniLM scores "release CO₂" close to "takes in CO₂". The *no semantic* ablation removes this error.
* **Negation false positives.** A correct, negatively phrased statement ("it cannot generalise to new data") is treated as negated.
  It is flagged for review rather than silently scored.
* **Auto-derived SciEntsBank rubrics** are crude; uniform weights can match or beat the content-word weighting there.
* The pilot set is small and authored by us; treat it as a sanity check, not a benchmark.
* *Contradictory* (3-way) is rarely predicted by any model.

## Disclaimer

Decision-support tool: a human examiner must review any high-stakes mark.
