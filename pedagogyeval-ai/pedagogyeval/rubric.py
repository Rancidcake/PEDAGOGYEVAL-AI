"""Rubric schema, JSON load/save, validation and concept-graph construction."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

RELATIONS = ("causes", "requires", "part_of", "leads_to")
WEIGHT_TOL = 1e-6


class RubricError(ValueError):
    """Raised when a rubric is malformed. The message lists every problem found."""


@dataclass
class Concept:
    id: str
    name: str
    description: str
    weight: float
    synonyms: list[str] = field(default_factory=list)


@dataclass
class Edge:
    source: str
    target: str
    relation: str


@dataclass
class Rubric:
    question_id: str
    question: str
    reference_answer: str
    max_marks: float
    concepts: list[Concept]
    edges: list[Edge] = field(default_factory=list)

    # ---------- serialisation ----------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any], validate: bool = True) -> "Rubric":
        missing = [k for k in ("question_id", "question", "reference_answer", "max_marks", "concepts") if k not in d]
        if missing:
            raise RubricError(f"rubric is missing field(s): {', '.join(missing)}")
        try:
            concepts = [
                Concept(
                    id=str(c["id"]),
                    name=str(c["name"]),
                    description=str(c.get("description") or c["name"]),
                    weight=float(c["weight"]),
                    synonyms=[str(s) for s in (c.get("synonyms") or []) if str(s).strip()],
                )
                for c in d["concepts"]
            ]
            edges = [Edge(str(e["source"]), str(e["target"]), str(e["relation"])) for e in d.get("edges") or []]
            rub = cls(
                question_id=str(d["question_id"]),
                question=str(d["question"]),
                reference_answer=str(d["reference_answer"]),
                max_marks=float(d["max_marks"]),
                concepts=concepts,
                edges=edges,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RubricError(f"rubric '{d.get('question_id', '?')}' has a malformed field: {exc!r}") from exc
        if validate:
            rub.validate()
        return rub

    # ---------- validation ----------
    def problems(self) -> list[str]:
        """Return a list of human-readable problems (empty means valid)."""
        errs: list[str] = []
        qid = self.question_id or "?"
        if not self.question_id.strip():
            errs.append("question_id is empty")
        if not self.reference_answer.strip():
            errs.append(f"[{qid}] reference_answer is empty")
        if self.max_marks <= 0:
            errs.append(f"[{qid}] max_marks must be > 0 (got {self.max_marks:g})")
        if not self.concepts:
            errs.append(f"[{qid}] rubric has no concepts")
        ids = [c.id for c in self.concepts]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            errs.append(f"[{qid}] duplicate concept id(s): {', '.join(dupes)}")
        for c in self.concepts:
            if not c.id.strip() or not c.name.strip():
                errs.append(f"[{qid}] every concept needs an id and a name")
            if c.weight < 0:
                errs.append(f"[{qid}] concept '{c.id}' has negative weight {c.weight:g}")
        total = sum(c.weight for c in self.concepts)
        if self.concepts and abs(total - self.max_marks) > WEIGHT_TOL:
            errs.append(f"[{qid}] concept weights sum to {total:g} but max_marks is {self.max_marks:g}")
        known = set(ids)
        for e in self.edges:
            for end in (e.source, e.target):
                if end not in known:
                    errs.append(f"[{qid}] edge {e.source} -> {e.target} references unknown concept '{end}'")
            if e.relation not in RELATIONS:
                errs.append(
                    f"[{qid}] edge {e.source} -> {e.target} has relation '{e.relation}'; "
                    f"expected one of {', '.join(RELATIONS)}"
                )
            if e.source == e.target:
                errs.append(f"[{qid}] edge {e.source} -> {e.target} is a self-loop")
        return errs

    def validate(self) -> "Rubric":
        errs = self.problems()
        if errs:
            raise RubricError("invalid rubric:\n  - " + "\n  - ".join(errs))
        return self

    def concept(self, cid: str) -> Concept:
        for c in self.concepts:
            if c.id == cid:
                return c
        raise KeyError(cid)


def build_graph(rubric: Rubric) -> nx.DiGraph:
    """Directed concept graph. Edges carry the relation; they are never used as grading evidence."""
    g = nx.DiGraph(question_id=rubric.question_id)
    for c in rubric.concepts:
        g.add_node(c.id, name=c.name, weight=c.weight, description=c.description)
    for e in rubric.edges:
        g.add_edge(e.source, e.target, relation=e.relation)
    return g


def load_rubrics(path: str | Path, validate: bool = True) -> dict[str, Rubric]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["rubrics"] if isinstance(data, dict) and "rubrics" in data else data
    if isinstance(items, dict):
        items = [items]
    rubrics = [Rubric.from_dict(d, validate=validate) for d in items]
    return {r.question_id: r for r in rubrics}


def rubrics_to_json(rubrics: Iterable[Rubric]) -> str:
    return json.dumps({"rubrics": [r.to_dict() for r in rubrics]}, indent=2, ensure_ascii=False)


def save_rubrics(rubrics: Iterable[Rubric], path: str | Path) -> None:
    Path(path).write_text(rubrics_to_json(rubrics) + "\n", encoding="utf-8")
