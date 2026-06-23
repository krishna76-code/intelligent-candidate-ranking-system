"""
precompute.py — OFFLINE step (no time limit, runs on your PC).

Builds the artifacts that rank.py loads at runtime so the graded step needs
NO network and stays under 5 minutes.

Run:
    python src/precompute.py --candidates ./data/candidates.jsonl

Produces:
    artifacts/candidate_embeddings.npy   (~150 MB, the 100K vectors, float32, L2-normalized)
    artifacts/jd_embedding.npy           (the JD vector)
    artifacts/model/                     (cached all-MiniLM-L6-v2 weights, see data/embedder.py)

Important anti-stuffer rule (see implementation_plan.md / claude.md):
    The text blob per candidate = summary + headline + career-history DESCRIPTIONS.
    Do NOT embed the skills list — that is how keyword stuffers game the system.

TODO: implement. This is a placeholder skeleton.
"""

from __future__ import annotations

import argparse
# import numpy as np
# from pathlib import Path
# from src.data.loader import load_candidates, build_text_blob
# from src.data.embedder import cache_model, load_model, embed_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute candidate + JD embeddings (offline).")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--jd", default="data/job_description.txt")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--cache-model", action="store_true", help="Also download & cache the model weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "precompute.py skeleton — drop the embedding-generation code here. "
        f"Would read {args.candidates} and write embeddings to {args.out_dir}/."
    )


if __name__ == "__main__":
    main()
