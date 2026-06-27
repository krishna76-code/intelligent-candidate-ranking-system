"""
rank.py — MAIN ENTRY POINT (the graded <=5-minute step).

Reproduce command:
    python src/rank.py --candidates ./data/candidates.jsonl --out ./output/<participant_id>.csv

Pipeline:
    1. Load candidates + precomputed embeddings (+ candidate_ids for alignment) + JD vector
    2. HARD FILTER   -> skip (score 0)
    3. FEATURE SCORE -> 0..1
    4. JD PENALTIES  -> x multiplier
    5. BEHAVIORAL    -> x multiplier
    6. SORT desc; ties by candidate_id ascending; take top 100
    7. REASONING
    8. WRITE candidate_id,rank,score,reasoning  (exact spec order)

Runtime: CPU only, no network (embeddings are precomputed). Embeddings are
L2-normalized at precompute time, so cosine similarity == dot product.
"""

from __future__ import annotations

import argparse
import csv
import numpy as np
import yaml
from pathlib import Path

from src.data.loader import load_candidates
from src.scoring.hard_filters import is_disqualified
from src.scoring.features import feature_score
from src.scoring.penalties import penalty_multiplier
from src.scoring.behavioral import behavioral_multiplier
from src.scoring.reasoning import build_reasoning


class Constants:
    """Namespace passed to scoring functions (they read attributes via getattr)."""
    from src.utils.constants import (  # noqa
        CORE_AI_SKILLS, CONSULTING_FIRMS, PRODUCT_COMPANIES,
    )
    RELEVANT_SKILLS = CORE_AI_SKILLS


def sanitize_candidate(candidate: dict) -> dict:
    """
    Coerce a candidate into a shape the scoring functions can safely read.

    The real 100K pool is clean, but the sandbox accepts uploaded samples — a
    field can be ABSENT or present-but-None. The scoring modules use
    `.get(k, default)` (covers absent) but a null value still slips through and
    crashes on `.lower()` / `.get()`. We fix all of that once, here, so every
    downstream module gets safe types.
    """
    c = candidate if isinstance(candidate, dict) else {}

    profile = c.get("profile")
    c["profile"] = profile if isinstance(profile, dict) else {}
    # null string fields -> ""
    for k in ("current_title", "headline", "summary", "country", "current_company"):
        if c["profile"].get(k) is None:
            c["profile"][k] = ""
    if c["profile"].get("years_of_experience") is None:
        c["profile"]["years_of_experience"] = 0

    history = c.get("career_history")
    history = history if isinstance(history, list) else []
    clean_history = []
    for job in history:
        job = job if isinstance(job, dict) else {}
        for k in ("title", "description", "company", "industry", "start_date"):
            if job.get(k) is None:
                job[k] = ""
        if job.get("duration_months") is None:
            job["duration_months"] = 0
        clean_history.append(job)
    c["career_history"] = clean_history

    skills = c.get("skills")
    skills = skills if isinstance(skills, list) else []
    clean_skills = []
    for sk in skills:
        sk = sk if isinstance(sk, dict) else {}
        if sk.get("name") is None:
            sk["name"] = ""
        for k in ("duration_months", "endorsements"):
            if sk.get(k) is None:
                sk[k] = 0
        clean_skills.append(sk)
    c["skills"] = clean_skills

    signals = c.get("redrob_signals")
    signals = signals if isinstance(signals, dict) else {}
    # drop null numeric/string signals so the scorers' float()/strptime don't choke
    for k in ("recruiter_response_rate", "notice_period_days", "github_activity_score",
              "last_active_date", "open_to_work_flag"):
        if signals.get(k) is None:
            signals.pop(k, None)
    c["redrob_signals"] = signals

    return c


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank top-100 candidates for the Senior AI Engineer JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl (or .jsonl.gz).")
    parser.add_argument("--out", required=True, help="Path to write the submission CSV.")
    parser.add_argument("--config", default="config/weights.yaml")
    parser.add_argument("--embeddings", default="artifacts/candidate_embeddings.npy")
    parser.add_argument("--candidate-ids", default="artifacts/candidate_ids.npy")
    parser.add_argument("--jd-embedding", default="artifacts/jd_embedding.npy")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load config; weights are NESTED under "feature_weights" ---
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    weights = config.get("feature_weights", config)  # fall back to flat for safety

    # --- Load embeddings + their candidate_ids (alignment is by ID, not row order) ---
    cand_embs = np.load(args.embeddings)
    cand_ids = np.load(args.candidate_ids, allow_pickle=True)
    jd_emb = np.load(args.jd_embedding)
    jd_vec = jd_emb[0] if jd_emb.ndim == 2 else jd_emb  # saved as (1, D) -> take the row

    # Map candidate_id -> its embedding row, so we never rely on file line order.
    id_to_row = {str(cid): i for i, cid in enumerate(cand_ids)}

    constants = Constants()
    ranked = []

    for candidate in load_candidates(args.candidates):
        candidate = sanitize_candidate(candidate)
        cid = candidate.get("candidate_id")

        # 2. Hard filter -> skip disqualified (is_disqualified returns (bool, reason))
        dq, _reason = is_disqualified(candidate)
        if dq:
            continue

        # Embedding similarity (cosine == dot, vectors are L2-normalized)
        row = id_to_row.get(str(cid))
        sim = float(np.dot(cand_embs[row], jd_vec)) if row is not None else 0.0

        # 3. Feature score
        feat, breakdown = feature_score(candidate, sim, weights, constants)

        # 4. JD penalties  5. Behavioral
        pen_mult, pen_reasons = penalty_multiplier(candidate, constants)
        beh_mult, beh_breakdown = behavioral_multiplier(candidate)

        final_score = feat * pen_mult * beh_mult
        ranked.append({
            "candidate": candidate,
            "raw_score": final_score,
            "breakdown": breakdown,
            "penalties": pen_reasons,
            "behavioral": beh_breakdown,
        })

    # Normalize scores into a clean [0, 1] range. The behavioral multiplier can
    # reach 1.1, so a raw final score can slightly exceed 1.0; dividing by the max
    # raw score rescales everything to <=1.0 WITHOUT changing the ranking order
    # (it's a monotonic transform). Then round to the precision we write, so the
    # sort tie-break agrees with the validator.
    max_raw = max((e["raw_score"] for e in ranked), default=1.0) or 1.0
    for e in ranked:
        e["score"] = round(e["raw_score"] / max_raw, 4)

    # 6. Sort: score DESC, ties by candidate_id ASC (matches the validator)
    ranked.sort(key=lambda x: (-x["score"], str(x["candidate"].get("candidate_id", ""))))
    top = ranked[: args.top_k]

    # 8. Write CSV — EXACT columns/order: candidate_id,rank,score,reasoning
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, entry in enumerate(top):
            rank = i + 1
            c = entry["candidate"]
            reasoning = build_reasoning(c, rank, entry["breakdown"], entry["penalties"], entry["behavioral"])
            writer.writerow([
                c.get("candidate_id"),
                rank,
                f'{entry["score"]:.4f}',
                reasoning,
            ])

    print(f"[rank] wrote {len(top)} rows to {out_path}")


if __name__ == "__main__":
    main()
