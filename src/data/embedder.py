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
    """
    Load model from local cache.

    Offline-safe during rank.py execution.
    Falls back to hub only if cache is missing.
    """

    from sentence_transformers import SentenceTransformer

    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        return SentenceTransformer(
            str(MODEL_DIR),
            device="cpu"
        )

    print(
        f"[embedder] WARNING: local cache {MODEL_DIR} missing."
    )

    return SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )


def embed_texts(
    model,
    texts,
    batch_size: int = 256
):
    """
    Encode strings into L2-normalized float32 vectors.

    Returns
    -------
    np.ndarray
        Shape (N,384), dtype float32.
    """

    import numpy as np
    import torch

    if len(texts) == 0:

        dim = model.get_sentence_embedding_dimension()

        return np.empty(
            (0, dim),
            dtype=np.float32
        )

    with torch.inference_mode():

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    return embeddings.astype(
        np.float32,
        copy=False
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache",
        action="store_true"
    )

    args = parser.parse_args()

    if args.cache:

        cache_model()

    else:

        model = load_model()

        vec = embed_texts(
            model,
            ["hello world"]
        )

        print(
            f"[embedder] loaded OK."
        )

        print(
            f"dim = {vec.shape[1]}"
        )

        print(
            f"dtype = {vec.dtype}"
        )if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache",
        action="store_true"
    )

    args = parser.parse_args()

    if args.cache:

        cache_model()

    else:

        model = load_model()

        vec = embed_texts(
            model,
            ["hello world"]
        )

        print(
            f"[embedder] loaded OK."
        )

        print(
            f"dim = {vec.shape[1]}"
        )

        print(
            f"dtype = {vec.dtype}"
        )