"""
precompute.py — OFFLINE step (no time limit, runs on your PC).

Builds the artifacts that rank.py loads at runtime so the graded step needs
NO network and stays under 5 minutes.

Run:
    python src/precompute.py --candidates ./data/candidates.jsonl
    # add --cache-model the first time, to download & cache the model weights:
    python src/precompute.py --candidates ./data/candidates.jsonl --cache-model

Produces (in artifacts/):
    candidate_embeddings.npy   (N x 384) float32, L2-normalized
    candidate_ids.npy          (N,) the candidate_id for each row (keeps order aligned)
    jd_embedding.npy           (1 x 384) the JD vector

Anti-stuffer rule: the embedded text is profile.summary + headline + career
descriptions (see loader.build_text_blob) — never the skills list.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.loader import load_candidates, build_text_blob
from src.data.embedder import cache_model, load_model, embed_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute candidate + JD embeddings (offline).")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--jd", default="data/job_description.txt")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--cache-model", action="store_true",
                        help="Download & cache the model weights into artifacts/model/ first.")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_model:
        print("[precompute] caching model ...")
        cache_model()

    print("[precompute] loading model ...")
    model = load_model()

    print("[precompute] reading candidates ...")
    candidate_ids = []
    texts = []
    for candidate in tqdm(load_candidates(args.candidates)):
        candidate_ids.append(candidate.get("candidate_id"))
        texts.append(build_text_blob(candidate))

    print(f"[precompute] embedding {len(texts):,} candidates ...")
    candidate_embeddings = embed_texts(model=model, texts=texts, batch_size=args.batch_size)
    candidate_embeddings = candidate_embeddings.astype(np.float32, copy=False)

    np.save(out_dir / "candidate_embeddings.npy", candidate_embeddings)
    np.save(out_dir / "candidate_ids.npy", np.array(candidate_ids))

    print("[precompute] embedding JD ...")
    jd_text = Path(args.jd).read_text(encoding="utf-8")
    jd_embedding = embed_texts(model=model, texts=[jd_text]).astype(np.float32, copy=False)
    np.save(out_dir / "jd_embedding.npy", jd_embedding)

    print(f"[precompute] done. saved {len(candidate_ids):,} candidates to {out_dir}/")


if __name__ == "__main__":
    main()
