# PedagogyEval AI: presentation pack

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
| `data/input/demo_examples.json` | 1,038 B |
| `data/input/pilot.csv` | 3,238 B |
| `data/input/rubrics.json` | 8,581 B |
| `data/input/team.json` | 244 B |

**Computed results** (copied from `results/`, written by the notebook):

| file | size |
|---|---|
| `data/results/ablations_pilot.csv` | 728 B |
| `data/results/ablations_scientsbank.csv` | 1,202 B |
| `data/results/environment.json` | 407 B |
| `data/results/grader_configs.csv` | 320 B |
| `data/results/metrics_summary.csv` | 1,183 B |
| `data/results/pilot_by_type.csv` | 1,651 B |
| `data/results/pilot_metrics.csv` | 708 B |
| `data/results/pilot_predictions.csv` | 52,494 B |
| `data/results/pilot_sample_outputs.csv` | 7,852 B |
| `data/results/scientsbank_class_balance.csv` | 963 B |
| `data/results/scientsbank_confusion.csv` | 16,925 B |
| `data/results/scientsbank_metrics.csv` | 5,713 B |
| `data/results/scientsbank_predictions.csv` | 6,191,791 B |
| `data/results/scientsbank_scores.csv` | 3,342,343 B |
| `data/results/worst_errors_pilot.csv` | 5,163 B |
| `data/results/worst_errors_scientsbank.csv` | 7,005 B |

SciEntsBank itself is not copied (it is downloaded from Hugging Face, `Atomi/sem_eval_2013_task_7`); per-answer scores and predictions are in `scientsbank_scores.csv` / `scientsbank_predictions.csv`.

## 4. Results (generated from `results/*.csv`)

Environment: Python 3.11.9, torch 2.14.1, sentence-transformers 6.2.0, seed 42, CUDA False.

### 4.1 Pilot (24 authored responses, 6 questions)

| model | MAE | RMSE | exact match | MAE / max |
|---|---|---|---|---|
| A · TF-IDF | 1.125 | 1.339 | 0.208 | 0.354 |
| B · MiniLM | 1.107 | 1.294 | 0.250 | 0.348 |
| C · Concept graph | 0.500 | 0.946 | 0.708 | 0.156 |

Per-response marks (gold vs A / B / C):

| response_id | response_type | gold | A | B | C | review |
|---|---|---|---|---|---|---|
| Q1_full | full | 4.00 | 2.00 | 2.04 | 4.00 | False |
| Q1_para | paraphrase | 4.00 | 2.00 | 2.04 | 3.00 | False |
| Q1_partial | partial | 2.00 | 1.00 | 2.04 | 2.00 | False |
| Q1_wrong | wrong_topical | 0.00 | 1.00 | 2.04 | 2.00 | True |
| Q2_full | full | 3.00 | 3.00 | 3.00 | 3.00 | False |
| Q2_para | paraphrase | 3.00 | 1.00 | 1.61 | 2.00 | True |
| Q2_partial | partial | 2.00 | 1.00 | 2.22 | 2.00 | False |
| Q2_wrong | wrong_topical | 0.00 | 1.00 | 1.61 | 0.00 | False |
| Q3_full | full | 3.00 | 2.00 | 3.00 | 3.00 | False |
| Q3_para | paraphrase | 3.00 | 1.00 | 1.61 | 1.00 | True |
| Q3_partial | partial | 1.00 | 1.00 | 2.00 | 1.00 | False |
| Q3_wrong | wrong_topical | 0.00 | 1.00 | 1.61 | 0.00 | False |
| Q4_full | full | 3.00 | 2.00 | 3.00 | 3.00 | False |
| Q4_para | paraphrase | 3.00 | 1.00 | 1.50 | 0.50 | True |
| Q4_partial | partial | 2.00 | 2.00 | 1.59 | 2.00 | False |
| Q4_wrong | wrong_topical | 0.00 | 1.00 | 1.60 | 0.00 | False |
| Q5_full | full | 3.00 | 1.00 | 1.54 | 3.00 | False |
| Q5_para | paraphrase | 3.00 | 1.00 | 1.54 | 1.00 | True |
| Q5_partial | partial | 1.00 | 1.00 | 1.54 | 0.50 | False |
| Q5_wrong | wrong_topical | 0.00 | 1.00 | 1.54 | 0.00 | False |
| Q6_full | full | 3.00 | 2.00 | 2.30 | 3.00 | False |
| Q6_para | paraphrase | 3.00 | 1.00 | 1.50 | 2.00 | False |
| Q6_partial | partial | 1.00 | 1.00 | 2.00 | 1.00 | False |
| Q6_wrong | wrong_topical | 0.00 | 1.00 | 1.62 | 0.00 | False |

### 4.2 SciEntsBank (SemEval-2013 Task 7)

**2WAY macro-F1** (threshold tuned on dev, frozen)

| model | dev | unseen-answers | unseen-questions | unseen-domains | mean test |
|---|---|---|---|---|---|
| A · TF-IDF | 0.598 | 0.609 | 0.603 | 0.616 | 0.610 |
| B · MiniLM | 0.638 | 0.650 | 0.639 | 0.630 | 0.640 |
| C · Concept graph | 0.663 | 0.654 | 0.600 | 0.650 | 0.635 |

**3WAY macro-F1** (threshold tuned on dev, frozen)

| model | dev | unseen-answers | unseen-questions | unseen-domains | mean test |
|---|---|---|---|---|---|
| A · TF-IDF | 0.381 | 0.387 | 0.397 | 0.404 | 0.396 |
| B · MiniLM | 0.408 | 0.417 | 0.419 | 0.410 | 0.416 |
| C · Concept graph | 0.435 | 0.423 | 0.400 | 0.417 | 0.413 |

Unseen-answers detail:

| task | model | accuracy | macro_f1 | weighted_f1 | threshold |
|---|---|---|---|---|---|
| 2way | A · TF-IDF | 0.609 | 0.609 | 0.608 | 0.080 |
| 2way | B · MiniLM | 0.650 | 0.650 | 0.651 | 0.510 |
| 2way | C · Concept graph | 0.659 | 0.654 | 0.660 | 0.460 |
| 3way | A · TF-IDF | 0.552 | 0.387 | 0.517 | 0.080 |
| 3way | B · MiniLM | 0.593 | 0.417 | 0.558 | 0.510 |
| 3way | C · Concept graph | 0.583 | 0.423 | 0.556 | 0.370 |

3-way class balance (unique answers per split):

| label | dev | train | unseen-answers | unseen-domains | unseen-questions |
|---|---|---|---|---|---|
| contradictory | 75 | 424 | 58 | 417 | 64 |
| correct | 301 | 1707 | 233 | 1917 | 301 |
| incorrect | 370 | 2092 | 249 | 2228 | 368 |

### 4.3 Ablations (Model C)

Pilot:

| variant | mae | rmse | exact_match | Δ MAE vs full |
|---|---|---|---|---|
| C · Concept graph | 0.500 | 0.946 | 0.708 | 0.000 |
| C − semantic | 0.542 | 1.099 | 0.708 | 0.042 |
| C − weights (uniform) | 0.500 | 0.946 | 0.708 | 0.000 |
| C − negation check | 0.500 | 0.946 | 0.708 | 0.000 |

Notes (computed): pilot rubrics use equal weights: True; pilot responses with a negation word: 2; negation flag fired on: 0.

SciEntsBank, mean macro-F1 over the three test splits:

| variant | 2way | 3way |
|---|---|---|
| C · Concept graph | 0.635 | 0.413 |
| C − semantic | 0.612 | 0.388 |
| C − weights (uniform) | 0.645 | 0.412 |
| C − negation check | 0.636 | 0.405 |


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
| `01_home.png` | Home: ASCII title block, team, pipeline diagram, model summary, pilot MAE at a glance |
| `01_home_full.png` | Home (full page) |
| `02_grade_empty.png` | Grade page before marking: quick-load row, §1 question, §2 response, red GRADE button, sidebar controls |
| `03_grade_good_paraphrase.png` | GOOD PARAPHRASE graded: big score, A|B|C strip, highlighted evidence, concept ledger  **(report Fig. 2)** |
| `03_grade_good_paraphrase_full.png` | GOOD PARAPHRASE: full verdict incl. examiner's note, review box, concept graph |
| `03_concept_graph_paraphrase.png` | Concept graph coloured by status (moss / ochre / red-pen) |
| `04_grade_paraphrase_semantic_off_full.png` | Same paraphrase with semantic matching switched OFF (live ablation) |
| `05_grade_incomplete.png` | INCOMPLETE graded: missing concept shown as [✗ MISSING]  **(report Fig. 3)** |
| `05_grade_incomplete_full.png` | INCOMPLETE: full verdict with feedback 'Missing: …' and red graph node |
| `05_concept_graph_incomplete.png` | Concept graph for the incomplete osmosis answer |
| `06_grade_wrong_but_topical.png` | WRONG BUT TOPICAL ('DNS encrypts all internet traffic'): C gives 0/3, baselines give credit |
| `06_grade_wrong_but_topical_full.png` | WRONG BUT TOPICAL (full page) |
| `07_grade_negation_trap.png` | NEGATION TRAP: negated span struck through in red |
| `07_grade_negation_trap_full.png` | NEGATION TRAP: full verdict with ⚠ NEEDS HUMAN REVIEW box |
| `07_marked_script_negation.png` | Close-up: marked script with yellow evidence and red strikethrough |
| `08_grade_model_A_tfidf.png` | Model A (TF-IDF) selected: holistic score |
| `08_grade_model_B_minilm.png` | Model B (MiniLM, calibrated) selected: holistic score |
| `09_rubric_editor.png` | Rubric editor: metadata, concepts/edges tables, live validation, graph preview |
| `09_rubric_editor_full.png` | Rubric editor (full page) |
| `10_rubric_validation_error.png` | Live validation error (max_marks 10 ≠ weights 4), SAVE disabled |
| `11_results_pilot.png` | Results › PILOT: metrics ledger, best row in red pen, charts  **(report Fig. 4)** |
| `11_results_pilot_full.png` | Results › PILOT (full page incl. sample outputs) |
| `12_results_scientsbank.png` | Results › SCIENTSBANK (2-way): macro-F1 ledger per split |
| `12_results_scientsbank_full.png` | Results › SCIENTSBANK 2-way (full page incl. confusion matrices) |
| `13_results_scientsbank_3way_full.png` | Results › SCIENTSBANK 3-way (full page) |
| `14_results_ablations.png` | Results › ABLATIONS |
| `14_results_ablations_full.png` | Results › ABLATIONS (full page) |
| `15_sidebar_closed_menu_button.png` | Sidebar closed: the ☰ MENU button re-opens it on every page |

### Report figures (`figures/`)

| file | shows |
|---|---|
| `figures/pilot_mae.png` | Pilot MAE by model and by response type |
| `figures/scientsbank_macro_f1.png` | SciEntsBank macro-F1 per test split, 2-way and 3-way |
| `figures/confusion_unseen_answers_3way.png` | Confusion matrices, unseen-answers, 3-way (▶ = best model) |
| `figures/confusion_unseen_answers_2way.png` | Confusion matrices, unseen-answers, 2-way |
| `figures/ablations.png` | Ablations: pilot MAE and SciEntsBank mean test macro-F1 |

## 7. Gallery

**Home: ASCII title block, team, pipeline diagram, model summary, pilot MAE at a glance**

![01_home](screenshots/01_home.png)

**Grade page before marking: quick-load row, §1 question, §2 response, red GRADE button, sidebar controls**

![02_grade_empty](screenshots/02_grade_empty.png)

**GOOD PARAPHRASE graded: big score, A|B|C strip, highlighted evidence, concept ledger  **(report Fig. 2)****

![03_grade_good_paraphrase](screenshots/03_grade_good_paraphrase.png)

**Concept graph coloured by status (moss / ochre / red-pen)**

![03_concept_graph_paraphrase](screenshots/03_concept_graph_paraphrase.png)

**INCOMPLETE graded: missing concept shown as [✗ MISSING]  **(report Fig. 3)****

![05_grade_incomplete](screenshots/05_grade_incomplete.png)

**Concept graph for the incomplete osmosis answer**

![05_concept_graph_incomplete](screenshots/05_concept_graph_incomplete.png)

**WRONG BUT TOPICAL ('DNS encrypts all internet traffic'): C gives 0/3, baselines give credit**

![06_grade_wrong_but_topical](screenshots/06_grade_wrong_but_topical.png)

**NEGATION TRAP: negated span struck through in red**

![07_grade_negation_trap](screenshots/07_grade_negation_trap.png)

**Close-up: marked script with yellow evidence and red strikethrough**

![07_marked_script_negation](screenshots/07_marked_script_negation.png)

**Model A (TF-IDF) selected: holistic score**

![08_grade_model_A_tfidf](screenshots/08_grade_model_A_tfidf.png)

**Model B (MiniLM, calibrated) selected: holistic score**

![08_grade_model_B_minilm](screenshots/08_grade_model_B_minilm.png)

**Rubric editor: metadata, concepts/edges tables, live validation, graph preview**

![09_rubric_editor](screenshots/09_rubric_editor.png)

**Live validation error (max_marks 10 ≠ weights 4), SAVE disabled**

![10_rubric_validation_error](screenshots/10_rubric_validation_error.png)

**Results › PILOT: metrics ledger, best row in red pen, charts  **(report Fig. 4)****

![11_results_pilot](screenshots/11_results_pilot.png)

**Results › SCIENTSBANK (2-way): macro-F1 ledger per split**

![12_results_scientsbank](screenshots/12_results_scientsbank.png)

**Results › ABLATIONS**

![14_results_ablations](screenshots/14_results_ablations.png)

**Sidebar closed: the ☰ MENU button re-opens it on every page**

![15_sidebar_closed_menu_button](screenshots/15_sidebar_closed_menu_button.png)

**Pilot MAE by model and by response type**

![pilot_mae.png](figures/pilot_mae.png)

**SciEntsBank macro-F1 per test split, 2-way and 3-way**

![scientsbank_macro_f1.png](figures/scientsbank_macro_f1.png)

**Confusion matrices, unseen-answers, 3-way (▶ = best model)**

![confusion_unseen_answers_3way.png](figures/confusion_unseen_answers_3way.png)

**Confusion matrices, unseen-answers, 2-way**

![confusion_unseen_answers_2way.png](figures/confusion_unseen_answers_2way.png)

**Ablations: pilot MAE and SciEntsBank mean test macro-F1**

![ablations.png](figures/ablations.png)
