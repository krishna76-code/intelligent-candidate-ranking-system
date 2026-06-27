"""
eyeball_test.py — sanity-check the TOP-100 ranking by inspection.

Asks the questions a human recruiter would: are the top picks real AI/ML
engineers? Any keyword-stuffers or honeypots that slipped in? It prints the top-N
with titles/companies/experience and reports aggregate stats over the top 100.

Run from the repo root:
    python -m evaluation.eyeball_test --candidates ./data/candidates.jsonl
    python -m evaluation.eyeball_test --candidates ./data/candidates.jsonl --show 20

Exit code is 0 if the top 100 is clean (0 stuffers, 0 honeypots), else 1 — so it
can double as a CI guardrail.
"""

from __future__ import annotations

import argparse

from evaluation._common import rank_all, load_profiles, DEFAULT_CANDIDATES
from src.rank import sanitize_candidate
from src.scoring.hard_filters import (
    is_honeypot_company_founding,
    is_honeypot_zero_duration_expert,
    is_honeypot_experience_inflation,
)
from src.utils.constants import COMPANY_FOUNDED

NON_ENG_MARKERS = (
    "marketing", "hr manager", "sales executive", "accountant",
    "graphic designer", "content writer", "recruiter", "operations manager",
    "customer support",
)
ENG_MARKERS = (
    "engineer", "scientist", "ml", "ai", "machine learning", "nlp",
    "data scien", "recommendation", "search", "applied", "research",
)


def is_honeypot(cand: dict) -> bool:
    return (
        is_honeypot_company_founding(cand, COMPANY_FOUNDED)
        or is_honeypot_zero_duration_expert(cand)
        or is_honeypot_experience_inflation(cand)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Eyeball-test the top-100 ranking.")
    ap.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--show", type=int, default=15, help="How many top rows to print.")
    args = ap.parse_args()

    ranked, pos, _dq, _n = rank_all(candidates_path=args.candidates)
    top_ids = [cid for cid, _s in ranked[: args.top_k]]
    profiles = load_profiles(args.candidates, top_ids)

    print(f"=== TOP {args.show} ===")
    print(f"{'rank':>4}  {'score':>7}  {'title':<32}  {'yrs':>4}  company")
    score_by_id = dict(ranked[: args.top_k])
    for i, cid in enumerate(top_ids[: args.show], start=1):
        p = profiles.get(cid, {}).get("profile", {})
        print(f"{i:>4}  {score_by_id[cid]:>7.4f}  "
              f"{(p.get('current_title') or '')[:32]:<32}  "
              f"{str(p.get('years_of_experience') or ''):>4}  "
              f"{(p.get('current_company') or '')[:18]}")

    # Aggregate stats over the whole top-k
    titles = [(profiles.get(cid, {}).get("profile", {}).get("current_title") or "").lower()
              for cid in top_ids]
    eng = sum(1 for t in titles if any(m in t for m in ENG_MARKERS))
    stuffers = sum(1 for t in titles if any(m in t for m in NON_ENG_MARKERS))
    honeypots = sum(1 for cid in top_ids
                    if is_honeypot(sanitize_candidate(profiles.get(cid, {}))))

    print(f"\n=== TOP {args.top_k} STATS ===")
    print(f"  engineering / AI titles : {eng}/{args.top_k}")
    print(f"  keyword-stuffer titles  : {stuffers}/{args.top_k}   (want 0)")
    print(f"  honeypots               : {honeypots}/{args.top_k}   (want 0; DQ if >10%)")

    clean = stuffers == 0 and honeypots == 0
    print(f"\n  RESULT: {'PASS — top 100 is clean' if clean else 'FAIL — review flagged rows'}")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
