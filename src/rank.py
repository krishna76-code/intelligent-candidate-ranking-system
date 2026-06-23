"""
rank.py — MAIN ENTRY POINT (the graded ≤5-minute step).

Reproduce command (goes in README + submission_metadata.yaml):
    python src/rank.py --candidates ./data/candidates.jsonl --out ./output/<participant_id>.csv

Constraints this file MUST honor:
    - Runtime  : <= 5 minutes wall-clock
    - Memory   : <= 16 GB RAM
    - Compute  : CPU only (no GPU)
    - Network  : OFF (no API calls) — embeddings are precomputed offline
    - Disk     : <= 5 GB

Pipeline (see implementation_plan.md):
    1. Load candidates.jsonl + artifacts/candidate_embeddings.npy + artifacts/jd_embedding.npy
    2. HARD FILTER   -> score 0   (scoring/hard_filters.py)
    3. FEATURE SCORE -> 0..1      (scoring/features.py)
    4. JD PENALTIES  -> reduce    (scoring/penalties.py)
    5. BEHAVIORAL    -> x multiplier (scoring/behavioral.py)
    6. SORT desc; ties by candidate_id ascending; take top 100
    7. REASONING                  (scoring/reasoning.py)
    8. WRITE <participant_id>.csv ; then run scripts/validate_submission.py

TODO: implement. This is a placeholder skeleton.
"""

from __future__ import annotations

import argparse
# import csv
# import json
# import numpy as np
# import yaml

# from src.data.loader import load_candidates
# from src.data.embedder import load_model            # only needed if embedding the JD at runtime
# from src.scoring.hard_filters import is_disqualified
# from src.scoring.features import feature_score
# from src.scoring.penalties import penalty_multiplier
# from src.scoring.behavioral import behavioral_multiplier
# from src.scoring.reasoning import build_reasoning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank top-100 candidates for the Senior AI Engineer JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl (or .jsonl.gz).")
    parser.add_argument("--out", required=True, help="Path to write the submission CSV.")
    parser.add_argument("--config", default="config/weights.yaml", help="Feature weights & thresholds.")
    parser.add_argument("--embeddings", default="artifacts/candidate_embeddings.npy")
    parser.add_argument("--jd-embedding", default="artifacts/jd_embedding.npy")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "rank.py skeleton — drop the ranking pipeline here. "
        f"Would read {args.candidates} and write {args.out}."
    )


if __name__ == "__main__":
    main()
