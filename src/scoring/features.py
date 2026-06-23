"""
features.py — the 8 weighted features (each returns 0..1).

Embedding similarity is ONE feature (~0.18), NOT the backbone. The decisive
signals are title-role match and career-history evidence.

Default weights (live in config/weights.yaml, not hard-coded):
    title_role_match      0.22   current/recent title is really ML/AI/DS/eng
    career_evidence       0.20   descriptions mention retrieval/ranking/search/recommendation/NLP
    embedding_similarity  0.18   cosine(JD, career text)
    experience_fit        0.12   5-9 yrs ideal, soft edges
    product_vs_services   0.10   product-company bonus, services penalty
    skill_trust           0.08   duration_months * endorsements, capped
    assessment_bonus      0.05   relevant Redrob assessment present -> bonus (never penalize absence)
    writes_code_recently  0.05   penalize architecture/tech-lead-only with no recent code

API to implement:
    def feature_score(candidate, embedding_sim, weights, constants) -> tuple[float, dict]:
        '''Return (weighted_total_0_1, per_feature_breakdown) for explainability.'''

TODO: implement. Placeholder skeleton.
"""

from __future__ import annotations


def title_role_match(candidate: dict) -> float:
    raise NotImplementedError


def career_evidence(candidate: dict, retrieval_terms: set) -> float:
    raise NotImplementedError


def experience_fit(candidate: dict, ideal_min: float = 5, ideal_max: float = 9) -> float:
    raise NotImplementedError


def product_vs_services(candidate: dict, consulting_firms: set, product_companies: set) -> float:
    raise NotImplementedError


def skill_trust(candidate: dict, cap: float = 1.0) -> float:
    """duration_months * endorsements, capped; discount 0-endorsement, <6-month skills."""
    raise NotImplementedError


def assessment_bonus(candidate: dict, relevant_skills: set) -> float:
    """Bonus only when a relevant assessment score is present. NEVER penalize absence."""
    raise NotImplementedError


def writes_code_recently(candidate: dict) -> float:
    raise NotImplementedError


def feature_score(candidate: dict, embedding_sim: float, weights: dict, constants=None) -> "tuple[float, dict]":
    """Combine all features into a weighted 0..1 score + per-feature breakdown."""
    raise NotImplementedError
