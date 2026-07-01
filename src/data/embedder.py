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

# BGE-base: 768-dim, top retrieval quality (the JD names "BGE, E5"). The model
# only runs OFFLINE during precompute; rank.py loads the saved .npy vectors only.
# Note: on a 4GB GPU use a small batch (~32-64) to avoid CUDA OOM.
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# BGE retrieval works best when the QUERY (the JD) is prefixed; documents are not.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

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


def _resolve_device(device: str | None) -> str:
    """
    Pick the compute device.
      - None / "auto" -> use GPU if available, else CPU (good for OFFLINE precompute)
      - "cpu"         -> force CPU (rank.py uses this — the graded step is CPU-only)
      - "cuda"        -> force GPU
    """
    if device in ("cpu", "cuda"):
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_model(device: str | None = "cpu"):
    """
    Load model from local cache.

    device: "cpu" (default, used by rank.py — graded step is CPU-only),
            "cuda" to force GPU, or "auto"/None to use GPU if available
            (precompute.py passes "auto" to speed up the offline embedding).

    Offline-safe. Falls back to hub only if the local cache is missing.
    """

    from sentence_transformers import SentenceTransformer

    dev = _resolve_device(device)

    # A cached model must contain actual weights — not just a .gitkeep placeholder.
    # (On hosted deploys the repo may ship an empty artifacts/model/ with only
    # .gitkeep; treating that as "cached" would force OFFLINE mode and crash. In
    # that case we fall through and download from the hub instead.)
    has_weights = MODEL_DIR.exists() and (
        (MODEL_DIR / "model.safetensors").exists()
        or (MODEL_DIR / "pytorch_model.bin").exists()
        or (MODEL_DIR / "config.json").exists()
    )

    if has_weights:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        print(f"[embedder] loading cached model on device={dev}")
        return SentenceTransformer(
            str(MODEL_DIR),
            device=dev
        )

    # No local weights (e.g. fresh cloud clone) -> download from the hub.
    print(f"[embedder] no cached weights in {MODEL_DIR}; downloading {MODEL_NAME} ...")
    return SentenceTransformer(
        MODEL_NAME,
        device=dev
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
        )