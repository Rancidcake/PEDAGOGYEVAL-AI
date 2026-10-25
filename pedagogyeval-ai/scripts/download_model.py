"""Download all-MiniLM-L6-v2 into models_cache/ so the demo runs offline.

Run once while online:
    python scripts/download_model.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "models_cache" / "all-MiniLM-L6-v2"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> int:
    os.environ.pop("HF_HUB_OFFLINE", None)
    from sentence_transformers import SentenceTransformer

    if (TARGET / "config.json").exists():
        print(f"[ok] model already present at {TARGET}")
        return 0
    print(f"[..] downloading {MODEL_ID}")
    model = SentenceTransformer(MODEL_ID, device="cpu")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(TARGET))
    vec = model.encode(["sanity check"])
    print(f"[ok] saved to {TARGET}  (dim={vec.shape[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
