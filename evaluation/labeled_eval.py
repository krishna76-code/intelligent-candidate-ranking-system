"""
labeled_eval.py — check the ranking against a small HAND-LABELED set.

We (humans) labeled ~25 real candidates from the pool as clearly-good
(relevance 2-3) or clearly-bad (relevance 0) in eval_labels.json. This script
runs the full ranking and checks:
  - BAD candidates are kept OUT of the top 100 (and how deep they sink)
  - GOOD candidates land high (we report their rank among all survivors)

This is NOT training data — the scorer never sees these labels; they are only an
independent check that the ranking agrees with human judgment. (No leakage: the
weights/features are fixed by reasoning, not fitted to these labels.)

Run from the repo root:
    python -m evaluation.labeled_eval --candidates ./data/candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation._common import rank_all, DEFAULT_CANDIDATES

LABELS_PATH = Path(__file__).resolve().parent / "eval_labels.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate against the hand-labeled set.")
    ap.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    ap.add_argument("--labels", default=str(LABELS_PATH))
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    labels = json.load(open(args.labels, "r", encoding="utf-8"))["labels"]
    good = [l for l in labels if l["relevance"] >= 2]
    bad = [l for l in labels if l["relevance"] == 0]

    ranked, pos, dq_ids, _n = rank_all(candidates_path=args.candidates)
    total = len(ranked)

    def where(cid):
        if cid in dq_ids:
            return "hard-filtered"
        return pos.get(cid)

    print(f"(survivors scored: {total})\n")

    print("=== GOOD candidates — want them ranked HIGH ===")
    good_in_topk = 0
    for l in sorted(good, key=lambda x: -x["relevance"]):
        cid = l["candidate_id"]
        r = where(cid)
        if isinstance(r, int):
            pct = f"top {100 * r / total:.1f}%"
            if r <= args.top_k:
                good_in_topk += 1
        else:
            pct = r
        print(f"  rel{l['relevance']}  {cid}: rank {str(r):<14} ({pct})   [{l['why'][:42]}]")

    print("\n=== BAD candidates — want them OUT of the top 100 ===")
    bad_out = 0
    for l in bad:
        cid = l["candidate_id"]
        r = where(cid)
        if r == "hard-filtered":
            out = True
            shown = "hard-filtered (removed up front)"
        else:
            out = (r is None) or (r > args.top_k)
            shown = f"rank {r}" + ("" if out else "  <-- LEAKED INTO TOP 100!")
        if out:
            bad_out += 1
        print(f"  {cid}: {shown:<40} [{l['why'][:38]}]")

    print(f"\n=== SCORE ===")
    print(f"  GOOD in top {args.top_k}: {good_in_topk}/{len(good)} "
          f"(rest still expected high among {total}; see ranks above)")
    print(f"  BAD kept out of top {args.top_k}: {bad_out}/{len(bad)}")
    ok = bad_out == len(bad)
    print(f"  RESULT: {'PASS — no bad candidate leaked into the top 100' if ok else 'FAIL — a bad candidate leaked in'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
