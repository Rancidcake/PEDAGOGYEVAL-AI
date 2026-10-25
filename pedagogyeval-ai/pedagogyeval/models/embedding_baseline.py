"""Model B: sentence-embedding baseline, plus the shared MiniLM ``Embedder``.

The model is loaded from ``models_cache/all-MiniLM-L6-v2`` (saved by
``scripts/download_model.py``). It is downloaded only if the local copy is missing.
Calibration (isotonic regression for marks, tuned thresholds for labels) is fitted
on train/dev data only.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..preprocess import clean

ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_DIR = ROOT / "models_cache" / "all-MiniLM-L6-v2"
HUB_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42


class ModelLoadError(RuntimeError):
    pass


class Embedder:
    """MiniLM wrapper with an in-memory text -> unit-vector cache."""

    def __init__(self, model_dir: str | Path | None = None, batch_size: int = 128) -> None:
        self.batch_size = batch_size
        self.source, self.model = self._load(Path(model_dir) if model_dir else LOCAL_MODEL_DIR)
        self._cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _load(model_dir: Path):
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.manual_seed(SEED)
        except Exception as exc:  # pragma: no cover - environment problem
            raise ModelLoadError(f"sentence-transformers / torch could not be imported: {exc}") from exc
        if (model_dir / "config.json").exists() or (model_dir / "modules.json").exists():
            try:
                return str(model_dir), SentenceTransformer(str(model_dir), device="cpu")
            except Exception as exc:
                raise ModelLoadError(f"local model at {model_dir} failed to load: {exc}") from exc
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise ModelLoadError(
                f"MiniLM not found at {model_dir} and HF_HUB_OFFLINE=1. "
                "Run `python scripts/download_model.py` once while online."
            )
        try:
            return HUB_MODEL_ID, SentenceTransformer(HUB_MODEL_ID, device="cpu")
        except Exception as exc:
            raise ModelLoadError(
                f"MiniLM not found at {model_dir} and download failed ({exc}). "
                "Run `python scripts/download_model.py` while online."
            ) from exc

    def prefetch(self, texts: Iterable[str]) -> None:
        todo = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if not todo:
            return
        vecs = self.model.encode(
            todo, batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        for t, v in zip(todo, vecs):
            self._cache[t] = v.astype(np.float32)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.prefetch(texts)
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return np.stack([self._cache[t] for t in texts])

    def cosine(self, a: Sequence[str], b: Sequence[str]) -> np.ndarray:
        """Matrix of cosine similarities, shape (len(a), len(b))."""
        return self.encode(a) @ self.encode(b).T


class EmbeddingBaseline:
    """cosine(reference, response) -> calibrated mark (isotonic) or label (thresholds)."""

    name = "B_embedding"

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.iso: IsotonicRegression | None = None

    def similarity(self, references: list[str], responses: list[str]) -> np.ndarray:
        refs = [clean(t) for t in references]
        resp = [clean(t) for t in responses]
        self.embedder.prefetch(refs + resp)
        sims = np.array([float(self.embedder.cosine([r], [s])[0, 0]) if s else 0.0 for r, s in zip(refs, resp)])
        return sims

    def fit_isotonic(self, sims: np.ndarray, normalised_gold: np.ndarray) -> "EmbeddingBaseline":
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(sims, normalised_gold)
        return self

    def calibrated(self, sims: np.ndarray) -> np.ndarray:
        if self.iso is None:
            return np.clip(sims, 0.0, 1.0)
        return self.iso.predict(np.asarray(sims, dtype=float))

    def grade(self, reference: str, response: str, max_marks: float) -> dict:
        if not clean(response):
            return {"model": self.name, "similarity": 0.0, "score": 0.0, "max_marks": max_marks, "calibrated": self.iso is not None}
        sim = float(self.similarity([reference], [response])[0])
        score = float(self.calibrated(np.array([sim]))[0]) * max_marks
        return {"model": self.name, "similarity": sim, "score": score, "max_marks": max_marks, "calibrated": self.iso is not None}
