from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.loader import (
    load_candidates,
    build_text_blob
)

from src.data.embedder import (
    cache_model,
    load_model,
    embed_texts,
    BGE_QUERY_PREFIX,
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Precompute candidate + JD embeddings."
    )

    parser.add_argument(
        "--candidates",
        required=True
    )

    parser.add_argument(
        "--jd",
        default="data/job_description.txt"
    )

    parser.add_argument(
        "--out-dir",
        default="artifacts"
    )

    parser.add_argument(
        "--cache-model",
        action="store_true"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,  # safe for bge-base on a 4GB GPU; CPU handles any size
        help="Embedding batch size. Lower if you hit CUDA out-of-memory."
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    out_dir = Path(args.out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.cache_model:

        print("[precompute] caching model...")
        cache_model()

    # OFFLINE step: use the GPU if available (allowed — only rank.py must be CPU).
    print("[precompute] loading model (GPU if available)...")
    model = load_model(device="auto")

    candidate_ids = []
    texts = []

    print("[precompute] reading candidates...")

    for candidate in tqdm(
            load_candidates(args.candidates)):

        candidate_ids.append(
            candidate["candidate_id"]
        )

        texts.append(
            build_text_blob(candidate)
        )

    print(
        f"[precompute] embedding {len(texts):,} candidates..."
    )

    candidate_embeddings = embed_texts(
        model=model,
        texts=texts,
        batch_size=args.batch_size  # default 64: fits bge-base in 4GB VRAM; raise on bigger GPUs
    )

    candidate_embeddings = candidate_embeddings.astype(
        np.float32,
        copy=False
    )

    np.save(
        out_dir / "candidate_embeddings.npy",
        candidate_embeddings
    )

    np.save(
        out_dir / "candidate_ids.npy",
        np.array(candidate_ids)
    )

    print("[precompute] embedding JD...")

    with open(
            args.jd,
            "r",
            encoding="utf-8"
    ) as f:

        jd_text = f.read()

    # The JD is the QUERY -> prefix it so BGE retrieval scores against documents
    # correctly. Candidate blobs are documents and are embedded without a prefix.
    jd_embedding = embed_texts(
        model=model,
        texts=[BGE_QUERY_PREFIX + jd_text]
    )

    jd_embedding = jd_embedding.astype(
        np.float32,
        copy=False
    )

    np.save(
        out_dir / "jd_embedding.npy",
        jd_embedding
    )

    print("[precompute] done.")

    print(
        f"[precompute] saved to {out_dir}"
    )


if __name__ == "__main__":
    main()