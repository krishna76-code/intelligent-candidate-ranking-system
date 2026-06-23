"""
embedder.py — wraps the local sentence-transformer model.

The model weights are CACHED INTO THE REPO (artifacts/model/) so rank.py can run
with the NETWORK OFF. We never hit huggingface.co at runtime.

Usage:
    # one-time, online, to cache the model into the repo:
    python -m src.data.embedder --cache

    # in code (offline-safe):
    from src.data.embedder import load_model, embed_texts
    model = load_model()
    vecs = embed_texts(model, ["some text", ...])
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Small, fast on CPU, 384-dim. The plan standardized on this.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "artifacts" / "model"


def cache_model() -> None:
    """Download the model ONCE (online) and save it into artifacts/model/."""
    from sentence_transformers import SentenceTransformer

    print(f"[embedder] downloading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_DIR))
    print(f"[embedder] cached to {MODEL_DIR}")


def load_model():
    """Load from the local cache (offline-safe); fall back to hub only if missing."""
    from sentence_transformers import SentenceTransformer

    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return SentenceTransformer(str(MODEL_DIR), device="cpu")

    print(f"[embedder] WARNING: local cache {MODEL_DIR} empty; loading from hub.")
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_texts(model, texts, batch_size: int = 256):
    """Encode strings to a (N, 384) float32 numpy array, L2-normalized (cosine == dot)."""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache the embedding model into the repo.")
    parser.add_argument("--cache", action="store_true", help="Download & cache the model.")
    args = parser.parse_args()
    if args.cache:
        cache_model()
    else:
        m = load_model()
        print(f"[embedder] loaded OK. dim = {m.get_sentence_embedding_dimension()}")
