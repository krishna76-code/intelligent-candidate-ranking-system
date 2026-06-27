"""
embedding_separation.py — measure how well the embeddings discriminate.

Computes the mean JD-similarity for clearly-STRONG candidates (real ML/AI title
+ retrieval/ranking evidence) vs clearly-WEAK ones, and reports the gap. A bigger
gap means the embedding model actually tells good candidates from bad — which is
the real test of whether an embedding upgrade was worth it.

This is how we measured MiniLM (~0.007 separation = near noise) vs bge-base
(~0.048 = ~7x better) on this pool.

Run from the repo root:
    python -m evaluation.embedding_separation --candidates ./data/candidates.jsonl
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import numpy as np

from evaluation._common import iter_candidates, DEFAULT_CANDIDATES, DEFAULT_EMB, DEFAULT_IDS, DEFAULT_JD
from src.rank import Constants, sanitize_candidate
from src.scoring.hard_filters import is_disqualified
from src.scoring.features import title_role_match, career_evidence


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure embedding strong-vs-weak separation.")
    ap.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    ap.add_argument("--embeddings", default=str(DEFAULT_EMB))
    ap.add_argument("--candidate-ids", default=str(DEFAULT_IDS))
    ap.add_argument("--jd-embedding", default=str(DEFAULT_JD))
    args = ap.parse_args()

    embs = np.load(args.embeddings)
    ids = np.load(args.candidate_ids, allow_pickle=True)
    jd = np.load(args.jd_embedding)
    jd_vec = jd[0] if jd.ndim == 2 else jd
    id_to_row = {str(c): i for i, c in enumerate(ids)}
    C = Constants()

    strong, weak = [], []
    for cand in iter_candidates(args.candidates):
        cand = sanitize_candidate(cand)
        if is_disqualified(cand)[0]:
            continue
        row = id_to_row.get(str(cand.get("candidate_id")))
        if row is None:
            continue
        sim = float(np.dot(embs[row], jd_vec))
        t = title_role_match(cand)
        ce = career_evidence(cand, C.CORE_AI_SKILLS)
        if t >= 0.9 and ce >= 0.66:
            strong.append(sim)
        elif t <= 0.6 and ce <= 0.34:
            weak.append(sim)

    s_mean = st.mean(strong) if strong else 0.0
    w_mean = st.mean(weak) if weak else 0.0
    print("=== EMBEDDING SEPARATION (strong vs weak candidates) ===")
    print(f"  embeddings : {args.embeddings}")
    print(f"  dimension  : {embs.shape[1]}")
    print(f"  strong (n={len(strong)}): mean JD-sim = {s_mean:.4f}")
    print(f"  weak   (n={len(weak)}): mean JD-sim = {w_mean:.4f}")
    print(f"  SEPARATION : {s_mean - w_mean:.4f}")
    print()
    print("  Interpretation: bigger separation => embeddings better distinguish")
    print("  good from bad. (MiniLM ~0.007 = noise; bge-base ~0.048 = real signal.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
