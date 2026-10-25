"""PedagogyEval AI: concept-graph mapping and explainable partial-credit grading for short answers."""
import os
import random

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "1.0.0"

from .rubric import Concept, Edge, Rubric, RubricError, build_graph, load_rubrics, save_rubrics  # noqa: E402
from .models import (  # noqa: E402
    ConceptGrader, Embedder, EmbeddingBaseline, GradeResult, GraderConfig, ModelLoadError, TfidfBaseline,
)

__all__ = [
    "SEED", "Concept", "Edge", "Rubric", "RubricError", "build_graph", "load_rubrics", "save_rubrics",
    "ConceptGrader", "Embedder", "EmbeddingBaseline", "GradeResult", "GraderConfig", "ModelLoadError", "TfidfBaseline",
]
