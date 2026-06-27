"""
_common.py — shared helpers for the evaluation scripts.

These evaluation utilities are DEV TOOLS, not part of the graded rank.py path.
They reproduce the exact same scoring pipeline rank.py uses (hard filters ->
features -> penalties -> behavioral) so the numbers they report match the real
submission. They read the full candidate pool + precomputed embeddings.

Run from the repo root, e.g.:
    python -m evaluation.eyeball_test --candidates ./data/candidates.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from src.rank import Constants, sanitize_candidate
from src.scoring.hard_filters import is_disqualified
from src.scoring.features import feature_score
from src.scoring.penalties import penalty_multiplier
from src.scoring.behavioral import behavioral_multiplier

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = REPO_ROOT / "data" / "candidates.jsonl"
DEFAULT_EMB = REPO_ROOT / "artifacts" / "candidate_embeddings.npy"
DEFAULT_IDS = REPO_ROOT / "artifacts" / "candidate_ids.npy"
DEFAULT_JD = REPO_ROOT / "artifacts" / "jd_embedding.npy"
DEFAULT_CONFIG = REPO_ROOT / "config" / "weights.yaml"


def load_weights(config_path=DEFAULT_CONFIG) -> dict:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8")) or {}
    return cfg.get("feature_weights", cfg)


def iter_candidates(path):
    """Yield candidate dicts from .jsonl, tolerant of malformed lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line, strict=False)
            except json.JSONDecodeError:
                continue


def rank_all(candidates_path=DEFAULT_CANDIDATES,
             emb_path=DEFAULT_EMB, ids_path=DEFAULT_IDS, jd_path=DEFAULT_JD,
             config_path=DEFAULT_CONFIG):
    """
    Score every candidate exactly as rank.py does (filter -> features -> penalties
    -> behavioral). Returns:
        ranked   : list of (candidate_id, final_score) sorted best-first (survivors only)
        pos      : dict candidate_id -> 1-based rank among survivors
        dq_ids   : set of candidate_ids removed by the hard filters
        n_total  : total candidates read
    """
    weights = load_weights(config_path)
    cand_embs = np.load(emb_path)
    cand_ids = np.load(ids_path, allow_pickle=True)
    jd_emb = np.load(jd_path)
    jd_vec = jd_emb[0] if jd_emb.ndim == 2 else jd_emb
    id_to_row = {str(c): i for i, c in enumerate(cand_ids)}

    constants = Constants()
    ranked, dq_ids, n_total = [], set(), 0

    for cand in iter_candidates(candidates_path):
        n_total += 1
        cand = sanitize_candidate(cand)
        cid = cand.get("candidate_id")
        dq, _ = is_disqualified(cand)
        if dq:
            dq_ids.add(cid)
            continue
        row = id_to_row.get(str(cid))
        sim = float(np.dot(cand_embs[row], jd_vec)) if row is not None else 0.0
        feat, _ = feature_score(cand, sim, weights, constants)
        pen, _ = penalty_multiplier(cand, constants)
        beh, _ = behavioral_multiplier(cand)
        ranked.append((cid, round(feat * pen * beh, 4)))

    ranked.sort(key=lambda x: (-x[1], str(x[0])))
    pos = {cid: i + 1 for i, (cid, _s) in enumerate(ranked)}
    return ranked, pos, dq_ids, n_total


def load_profiles(candidates_path, wanted_ids):
    """Return {candidate_id: full candidate dict} for the requested ids."""
    wanted = set(wanted_ids)
    out = {}
    for cand in iter_candidates(candidates_path):
        cid = cand.get("candidate_id")
        if cid in wanted:
            out[cid] = cand
    return out
