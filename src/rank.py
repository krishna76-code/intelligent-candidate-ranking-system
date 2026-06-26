"""
rank.py — MAIN ENTRY POINT.
"""

import argparse
import json
import csv
import numpy as np
import yaml
from pathlib import Path

# Importing the logic modules
from src.scoring.hard_filters import is_disqualified
from src.scoring.features import feature_score
from src.scoring.penalties import penalty_multiplier
from src.scoring.behavioral import behavioral_multiplier
from src.scoring.reasoning import build_reasoning

class Constants:
    """Namespace to pass configuration constants to scoring functions."""
    CORE_AI_SKILLS = {"nlp", "retrieval", "ranking", "search", "recommendation", "llm", "ai", "ml"}
    CONSULTING_FIRMS = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}
    PRODUCT_COMPANIES = {"google", "meta", "amazon", "microsoft", "apple", "netflix"}
    PRODUCTION_TERMS = {"production", "deployed", "shipped", "scale", "infrastructure", "docker", "kubernetes"}

def main() -> None:
    args = parse_args()
    
    # Load Configuration & Embeddings
    with open(args.config, 'r') as f:
        weights = yaml.safe_load(f)
    
    cand_embs = np.load(args.embeddings)
    jd_emb = np.load(args.jd_embedding)
    
    constants = Constants()
    ranked_results = []

    # Process Candidates
    with open(args.candidates, 'r') as f:
        for idx, line in enumerate(f):
            candidate = json.loads(line)
            
            # 1. Hard Filter
            if is_disqualified(candidate):
                continue
            
            # 2. Embedding Similarity (using precomputed)
            sim = float(np.dot(cand_embs[idx], jd_emb) / (np.linalg.norm(cand_embs[idx]) * np.linalg.norm(jd_emb)))
            
            # 3. Feature Scoring
            feat_score, breakdown = feature_score(candidate, sim, weights, constants)
            
            # 4. JD Penalties
            penalty_mult, penalty_reasons = penalty_multiplier(candidate, constants)
            
            # 5. Behavioral Multiplier
            beh_mult, beh_breakdown = behavioral_multiplier(candidate)
            
            # Aggregate Final Score
            final_score = feat_score * penalty_mult * beh_mult
            
            ranked_results.append({
                "candidate": candidate,
                "score": final_score,
                "breakdown": breakdown,
                "penalties": penalty_reasons,
                "behavioral": beh_breakdown
            })

    # Sort: Score DESC, then candidate_id ASC
    ranked_results.sort(key=lambda x: (-x["score"], x["candidate"].get("candidate_id", "")))
    top_candidates = ranked_results[:args.top_k]

    # Write Output
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "score", "reasoning"])
        writer.writeheader()
        
        for i, entry in enumerate(top_candidates):
            c = entry["candidate"]
            reasoning = build_reasoning(
                c, i + 1, entry["breakdown"], entry["penalties"], entry["behavioral"]
            )
            writer.writerow({
                "candidate_id": c.get("candidate_id"),
                "score": round(entry["score"], 4),
                "reasoning": reasoning
            })

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank top-100 candidates.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="config/weights.yaml")
    parser.add_argument("--embeddings", default="artifacts/candidate_embeddings.npy")
    parser.add_argument("--jd-embedding", default="artifacts/jd_embedding.npy")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()

if __name__ == "__main__":
    main()