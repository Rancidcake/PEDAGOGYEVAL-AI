"""Data loading: the authored pilot set and SciEntsBank (SemEval-2013 Task 7).

SciEntsBank is loaded from the Hugging Face mirror "Atomi/sem_eval_2013_task_7". The loader
first prints splits, columns, labels and class balance, and resolves column names from
candidates instead of assuming them. Official test splits are kept; dev is carved from train.

Concepts for SciEntsBank questions are AUTO-DERIVED from the reference answer (clauses +
stopword-delimited noun-phrase-like chunks). This is an approximation, not an expert rubric.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocess import STOPWORDS, clean, lower_aligned, tokenize
from .rubric import Concept, Edge, Rubric, load_rubrics

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SEED = 42
HF_DATASET = "Atomi/sem_eval_2013_task_7"


# ------------------------------------------------------------------ pilot
def load_pilot(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "pilot.csv")
    need = {"response_id", "question_id", "response_type", "response", "gold_marks"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"pilot.csv is missing columns: {sorted(missing)}")
    return df


def load_pilot_rubrics(path: str | Path | None = None) -> dict[str, Rubric]:
    return load_rubrics(path or DATA_DIR / "rubrics.json")


# ------------------------------------------------------------------ SciEntsBank
COLUMN_CANDIDATES = {
    "corpus": ["corpus", "dataset"],
    "task": ["classification_type", "task", "label_type"],
    "test_set": ["test_set", "subset", "split_name"],
    "question_id": ["question_id", "qid"],
    "question": ["question", "question_text"],
    "reference_answer": ["reference_answer", "reference", "ref_answer"],
    "answer_id": ["student_answer_id", "answer_id", "id"],
    "student_answer": ["student_answer", "answer", "response"],
    "label": ["student_answer_label", "label", "accuracy"],
}


def resolve_columns(columns: list[str]) -> dict[str, str]:
    out, missing = {}, []
    for key, cands in COLUMN_CANDIDATES.items():
        hit = next((c for c in cands if c in columns), None)
        if hit is None:
            missing.append(f"{key} (tried {cands})")
        else:
            out[key] = hit
    if missing:
        raise KeyError("could not resolve dataset columns: " + "; ".join(missing) + f". Available: {columns}")
    return out


def inspect_dataset(ds) -> dict[str, str]:
    """Print splits, columns, label sets and class balance. Returns the resolved column map."""
    print(f"Dataset: {HF_DATASET}")
    for split in ds:
        print(f"  split '{split}': {len(ds[split])} rows")
    first = next(iter(ds))
    cols = ds[first].column_names
    print("  columns:", cols)
    cmap = resolve_columns(cols)
    print("  resolved:", cmap)
    for split in ds:
        df = ds[split].to_pandas()
        print(f"\n  [{split}] rows by corpus x task x test_set:")
        print(df.groupby([cmap["corpus"], cmap["task"], cmap["test_set"]]).size().to_string().replace("\n", "\n    "))
        for (corpus, task), g in df.groupby([cmap["corpus"], cmap["task"]]):
            bal = Counter(g[cmap["label"]])
            tot = sum(bal.values())
            pretty = ", ".join(f"{k}={v} ({v / tot:.1%})" for k, v in bal.most_common())
            print(f"    {corpus}/{task} labels: {pretty}")
    return cmap


def load_scientsbank(dev_frac: float = 0.15, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Return one frame per evaluation split with both 2-way and 3-way labels attached.

    Keys: train, dev, unseen-answers, unseen-questions, unseen-domains.
    Columns: answer_id, question_id, question, reference_answer, student_answer, label_2way, label_3way.
    """
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET)
    cmap = inspect_dataset(ds) if verbose else resolve_columns(ds[next(iter(ds))].column_names)
    frames = []
    for split in ds:
        df = ds[split].to_pandas()
        df = df[df[cmap["corpus"]].astype(str).str.lower().str.contains("scient")]
        frames.append(df.assign(_hf_split=split))
    df = pd.concat(frames, ignore_index=True)
    base_cols = ["_hf_split", cmap["test_set"], cmap["question_id"], cmap["question"], cmap["reference_answer"], cmap["answer_id"], cmap["student_answer"]]
    wide = None
    for task in sorted(df[cmap["task"]].unique()):
        sub = df[df[cmap["task"]] == task][base_cols + [cmap["label"]]].drop_duplicates(subset=[cmap["answer_id"], "_hf_split"])
        sub = sub.rename(columns={cmap["label"]: f"label_{task}"})
        wide = sub if wide is None else wide.merge(sub, on=base_cols, how="outer")
    wide = wide.rename(
        columns={
            cmap["test_set"]: "test_set",
            cmap["question_id"]: "question_id",
            cmap["question"]: "question",
            cmap["reference_answer"]: "reference_answer",
            cmap["answer_id"]: "answer_id",
            cmap["student_answer"]: "student_answer",
        }
    )
    wide["student_answer"] = wide["student_answer"].fillna("").astype(str)
    train_all = wide[wide["_hf_split"] == "train"].reset_index(drop=True)
    strat = train_all["label_3way"] if "label_3way" in train_all else None
    train, dev = train_test_split(train_all, test_size=dev_frac, random_state=SEED, stratify=strat)
    out = {"train": train.reset_index(drop=True), "dev": dev.reset_index(drop=True)}
    test = wide[wide["_hf_split"] != "train"]
    for ts in sorted(test["test_set"].unique()):
        out[str(ts)] = test[test["test_set"] == ts].reset_index(drop=True)
    if verbose:
        print("\n  SciEntsBank evaluation frames (unique answers):")
        for k, v in out.items():
            lab = ", ".join(f"{a}={b}" for a, b in Counter(v.get("label_3way", [])).most_common())
            print(f"    {k:<17} n={len(v):>5}  3-way: {lab}")
    return out


# ------------------------------------------------------------------ concept derivation (approximation)
CLAUSE_RE = re.compile(
    r"[.;:!?]|,|\s(?:because|so|since|therefore|thus|which|when|if|but|while|whereas|then|and then)\s", re.IGNORECASE
)
CONNECTOR_REL = {"because": "causes", "since": "causes", "so": "leads_to", "therefore": "leads_to", "thus": "leads_to", "then": "leads_to"}


def _chunks(text: str) -> list[str]:
    """Maximal runs of non-stopword word tokens (a dependency-free stand-in for noun chunks)."""
    runs, cur = [], []
    for t in tokenize(lower_aligned(clean(text))):
        w = t.text
        if w[:1].isalnum() and w not in STOPWORDS and not w.endswith("n't"):
            cur.append(w)
        else:
            if cur:
                runs.append(" ".join(cur))
            cur = []
    if cur:
        runs.append(" ".join(cur))
    return runs


def derive_rubric(question_id: str, question: str, reference_answer: str, max_concepts: int = 4) -> Rubric:
    """Auto-derive an approximate rubric from a reference answer (labelled approximation)."""
    ref = clean(reference_answer)
    pieces, connectors, pos = [], [], 0
    for m in CLAUSE_RE.finditer(ref):
        pieces.append(ref[pos : m.start()])
        connectors.append(m.group(0).strip().lower())
        pos = m.end()
    pieces.append(ref[pos:])
    clauses: list[list] = []  # [text, n_content, connector_before]
    pending_conn = ""
    for i, p in enumerate(pieces):
        p = p.strip()
        n = sum(len(c.split()) for c in _chunks(p))
        conn = connectors[i - 1] if i > 0 else ""
        if n == 0:
            pending_conn = conn or pending_conn
            continue
        if clauses and n < 2:
            clauses[-1][0] += " " + p
            clauses[-1][1] += n
        else:
            clauses.append([p, n, conn or pending_conn])
        pending_conn = ""
    if not clauses:
        clauses = [[ref or question, max(1, len(_chunks(ref))), ""]]
    while len(clauses) > max_concepts:
        k = min(range(len(clauses) - 1), key=lambda i: clauses[i][1] + clauses[i + 1][1])
        a, b = clauses[k], clauses.pop(k + 1)
        a[0] = f"{a[0]} {b[0]}"
        a[1] += b[1]
    q_words = {t.text for t in tokenize(lower_aligned(clean(question)))}
    total = sum(c[1] for c in clauses)
    M = float(len(clauses))
    concepts, edges = [], []
    for i, (txt, n, conn) in enumerate(clauses):
        chunks = _chunks(txt)
        syns = [c for c in chunks if not set(c.split()) <= q_words and (len(c.split()) > 1 or len(c) >= 4)]
        cid = f"c{i + 1}"
        concepts.append(Concept(cid, (chunks[0] if chunks else txt)[:40], txt, round(M * n / total, 6), syns))
        if i > 0:
            rel = CONNECTOR_REL.get(conn, "leads_to")
            src, dst = (cid, f"c{i}") if rel == "causes" else (f"c{i}", cid)
            edges.append(Edge(src, dst, rel))
    drift = M - sum(c.weight for c in concepts)
    concepts[-1].weight = round(concepts[-1].weight + drift, 6)
    return Rubric(question_id, clean(question), ref, M, concepts, edges).validate()
