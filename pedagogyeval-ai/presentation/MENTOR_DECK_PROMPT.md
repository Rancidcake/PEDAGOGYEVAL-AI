You are a presentation designer and technical storyteller. Build a 12–14 slide deck presenting my NLP project to an INDUSTRY MENTOR (a senior engineer/product person, not an examiner). The deck must look like it belongs to the project's own web app: a design called "EXAMINER'S LEDGER": an examiner's marking sheet crossed with a terminal. Warm paper, black ink, red-pen corrections.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
§1 PROJECT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Title: PEDAGOGYEVAL/AI v1.0
Subtitle: Fine-Grained Concept Graph Mapping and Explainable Partial-Credit Grading for Short Answers
Team: [NAME ONE · PRN] · [NAME TWO · PRN]   (leave as placeholders)

One-line pitch: An explainable short-answer grader that marks a response concept by concept, shows the exact evidence for every mark, awards partial credit, catches negations, and flags fragile cases for a human instead of hiding them.

Problem (for the mentor): Teachers spend hours marking short answers. Similarity-based auto-graders give one opaque number, reward on-topic but wrong answers, and can't say what is missing. Real classrooms need partial credit, feedback the student can act on, and a reason to trust (or override) the machine.

How it works:
- Rubric = weighted concepts (name, description, weight, synonyms) + relation edges (causes / requires / part_of / leads_to). Edges explain; they never award marks.
- Pipeline: Input → Preprocess (NFKC, offset-preserving lowercase, no stemming, negations kept) → Concept Graph → Lexical match (synonym regex, `*` wildcards, plural & misspelling tolerance) + Semantic match (MiniLM sentence/clause cosine: ≥0.70 full, ≥0.50 half) → Negation check (not/never/no/without/cannot within 3 tokens → no credit + flag) → Weighted aggregation min(M, Σ w·p) → Explanation (evidence spans, template feedback, review flag) → Evaluation.
- Three models compared: A = TF-IDF cosine baseline · B = MiniLM embeddings, isotonic-calibrated (leave-one-question-out) · C = our concept-graph grader.
- Human-review flag fires on: empty answer, negation, similarity within 0.05 of a threshold, or keyword present but meaning dissimilar.
- Product: 3-page Streamlit app (Grade / Rubric editor / Results dashboard), runs fully offline with one command, 45 automated tests, Colab notebook, fixed seed 42, every number computed and saved to CSV (none hand-typed).

RESULTS (use exactly these numbers; do not round differently or invent any others):
Pilot: 24 authored responses, 6 questions (photosynthesis, osmosis, primary key, overfitting, DNS, phishing), full / partial / wrong-but-topical / paraphrase.
| Model | MAE (marks) | RMSE | Exact match |
| A · TF-IDF | 1.125 | 1.339 | 20.8% |
| B · MiniLM | 1.107 | 1.294 | 25.0% |
| C · Concept graph | 0.500 | 0.946 | 70.8% |
→ C cuts mark error by ~55% vs both baselines on the pilot.

SciEntsBank (SemEval-2013 Task 7, public benchmark; thresholds tuned on dev, frozen for the official test splits; C's concepts AUTO-DERIVED from reference answers = an approximation):
Mean macro-F1 over the 3 test splits: 2-way: A 0.610 · B 0.640 · C 0.635 | 3-way: A 0.396 · B 0.416 · C 0.413
Unseen-answers split: C is best (2-way macro-F1 0.654, accuracy 65.9%; 3-way macro-F1 0.423).
Honest read: on the benchmark C is on par with the embedding baseline (B), not clearly better, because its rubrics are auto-generated, while C additionally explains every mark.
Ablations (C): removing semantic matching hurts everywhere (pilot MAE 0.500 → 0.542; SciEntsBank 2-way 0.635 → 0.612, 3-way 0.413 → 0.388). Uniform weights match or beat our auto-weighting heuristic on SciEntsBank (2-way 0.645).

Known limitations (show them, mentors respect this):
- Embeddings are direction-blind: "Plants breathe in oxygen and release CO₂" (gold 0) still gets semantic credit.
- Correct negatively phrased answers ("it cannot generalise to new data") can be flagged as negated → sent to review.
- Auto-derived benchmark rubrics are crude; the "contradictory" class is rarely predicted; pilot set is small and self-authored.
- Decision-support tool: a human examiner reviews high-stakes marks.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
§2 SLIDE PLAN (mentor-oriented: problem → value → how → evidence → risks → ask)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
01 Title: ASCII box title block, subtitle, team, one-line pitch.
02 The problem: why opaque similarity scores fail teachers (3 short bracketed points).
03 What we built: one sentence + the 3 product surfaces (GRADE · RUBRIC · RESULTS) + "runs offline, one command".
04 How it marks: pipeline diagram as ink-outlined boxes with arrows; the aggregation box outlined in red pen.
05 Live example (good paraphrase): screenshot 03_grade_good_paraphrase.png, callouts on score, evidence highlights, concept ledger.
06 Partial credit and feedback: screenshot 05_grade_incomplete.png; callout the [✗ MISSING] row and the margin-note feedback.
07 Trust and safety: screenshot 07_grade_negation_trap.png; negation strikethrough + "⚠ NEEDS HUMAN REVIEW" box; list the 4 review triggers.
08 Pilot results: ledger table (A/B/C), C row highlighted in red pen; chart figures/pilot_mae.png.
09 Benchmark results: SciEntsBank table + figures/scientsbank_macro_f1.png; state the honest read in one line.
10 Ablations: what each component contributes (figures/ablations.png).
11 Limitations and risks: bracketed list, each with a mitigation.
12 Roadmap: expert rubrics for real courses, direction-aware semantic matching (NLI), teacher-in-the-loop review queue, LMS integration, pilot with a real class.
13 The ask: what we want from the mentor (feedback on productisation, evaluation design, a real dataset or classroom pilot, deployment advice).
14 (Optional) Appendix: rubric editor screenshot 09_rubric_editor.png, test and reproducibility facts.

Speaker notes for every slide: 2–4 sentences, plain spoken English, mentor-facing (impact, trade-offs, what we'd do next).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
§3 DESIGN SYSTEM: "EXAMINER'S LEDGER" (follow strictly)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Palette (use only these):
  paper     #F3EFE6  slide background
  paper-2   #EAE4D7  panels, table header rows
  ink       #1B1A17  text, borders, arrows
  ink-soft  #5E5A52  secondary text, captions, baseline model A
  rule      #C9C1B0  hairlines, gridlines, faint ruled lines
  redpen    #C8372D  the ONE accent: key numbers, best row, missing items, our model C
  ochre     #B7861B  partial / warnings
  moss      #3F6B34  covered / success
  highlight #F2D64B at ~45% opacity: highlighter-pen marks on key words only
Chart colour mapping: A = ink-soft, B = ink, C = redpen. Heatmaps: single hue paper → ink.

Typography: JetBrains Mono for everything (fallback IBM Plex Mono, then monospace). Weights 400 / 500 / 700.
  Headings: UPPERCASE, letter-spacing 0.08em, bold, ink.
  Body: 15–18 pt equivalent, line-height 1.6.
  Hero numbers (e.g. "0.50 MAE", "−55%"): JetBrains Mono 700, very large (~96 px), with a DOUBLE red underline (3px double #C8372D).

Layout and components:
  - Background: paper with faint horizontal ruled lines (#C9C1B0 at ~35% opacity, ~32 px apart) and a thin vertical red margin line (#C8372D at ~30%) near the left edge, like an exercise book.
  - Square corners (0–2 px radius), 1 px solid ink borders, NO shadows, NO gradients on any element.
  - Slide titles styled as form-field section labels: red "§N" followed by the uppercase title, with a 1 px ink rule underneath; an optional small uppercase hint right-aligned in ink-soft.
  - Status tags as bracketed mono text, not pills: [✓ COVERED] moss · [~ PARTIAL] ochre · [✗ MISSING] redpen.
  - Tables as ledgers: paper-2 header row, uppercase 0.72 em headers, ink outer border, rule-coloured row lines, right-aligned tabular numbers; the best row tinted faint red with a 3 px red-pen left border and bold text.
  - Callouts / speaker-style notes as "margin notes": 3 px red-pen left border, italic mono text, small "EXAMINER'S NOTE" label.
  - Warnings as an ochre-outlined box starting with "⚠".
  - Diagrams: rectangular ink-outlined boxes on paper fill, ink arrows, mono labels; emphasise one box with a red-pen outline.
  - Screenshots: framed with a 1 px ink border, no rounded corners, no drop shadow; annotate with red-pen arrows/circles like a teacher's corrections.
  - Generous whitespace, left-aligned text, at most ~6 lines of text per slide; must read well on a projector and at 1366×768.

STRICTLY AVOID: purple/violet/indigo, blue gradients or any gradients, glassmorphism, neon, sparkle/robot/brain emojis or AI clip-art, rounded pill cards with drop shadows, stock photos, default template blue/orange charts, chat-bubble visuals.

Assets I will provide (folder presentation/): screenshots/01_home.png, 03_grade_good_paraphrase.png, 05_grade_incomplete.png, 07_grade_negation_trap.png, 07_marked_script_negation.png, 09_rubric_editor.png, 11_results_pilot.png, 12_results_scientsbank.png, 14_results_ablations.png, 15_sidebar_closed_menu_button.png; figures/pilot_mae.png, scientsbank_macro_f1.png, confusion_unseen_answers_3way.png, ablations.png. Leave clearly labelled image placeholders where these go if you can't embed them.

Tone: confident, precise, honest; industry vocabulary (impact, reliability, human-in-the-loop, offline-first, reproducible). No hype, no invented metrics, no claims beyond the numbers above.

OUTPUT: the full deck (slide-by-slide content, layout, and visuals in the style above) plus speaker notes. If you produce code (HTML/reveal.js or python-pptx), embed the palette as variables and load JetBrains Mono.
