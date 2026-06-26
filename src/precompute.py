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
    embed_texts
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

    print("[precompute] loading model...")
    model = load_model()

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
        batch_size=256
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

    jd_embedding = embed_texts(
        model=model,
        texts=[jd_text]
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