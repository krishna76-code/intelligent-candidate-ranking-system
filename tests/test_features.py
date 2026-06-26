"""
Tests for src/scoring/features.py.

Verify each feature returns a value in [0, 1] and that the weighted total
behaves sensibly (strong ML candidate > weak/keyword candidate).
"""

import pytest

from src.scoring.features import (
    feature_score, 
    experience_fit, 
    assessment_bonus, 
    title_role_match
)

class MockConstants:
    """Mock constants namespace to simulate src.utils.constants during testing."""
    CORE_AI_SKILLS = {"nlp", "retrieval", "ranking", "search", "recommendation"}
    CONSULTING_FIRMS = {"tcs", "infosys", "wipro"}
    PRODUCT_COMPANIES = {"google", "meta", "amazon"}
    RELEVANT_SKILLS = CORE_AI_SKILLS

MOCK_WEIGHTS = {
    "title_role_match": 0.22,
    "career_evidence": 0.20,
    "embedding_similarity": 0.18,
    "experience_fit": 0.12,
    "product_vs_services": 0.10,
    "skill_trust": 0.08,
    "assessment_bonus": 0.05,
    "writes_code_recently": 0.05
}

def test_experience_fit_peaks_in_band():
    """7 yrs => high; 1 yr => low; 20 yrs => reduced"""
    # Peak sweet spot (5-9 years)
    assert experience_fit({"profile": {"years_of_experience": 7.0}}, ideal_min=5, ideal_max=9) == 1.0
    
    # Just outside lower edge (e.g., 4 years) gets a partial score, not zero
    assert experience_fit({"profile": {"years_of_experience": 4.0}}, ideal_min=5, ideal_max=9) == 0.7
    
    # Far outside (1 year => severe penalty)
    assert experience_fit({"profile": {"years_of_experience": 1.0}}, ideal_min=5, ideal_max=9) == 0.2
    
    # Too experienced (20 years => out of optimal target scope)
    assert experience_fit({"profile": {"years_of_experience": 20.0}}, ideal_min=5, ideal_max=9) == 0.2


def test_assessment_bonus_never_penalizes_absence():
    """Candidate with no skill_assessment_scores must NOT be penalized."""
    candidate_no_scores = {
        "redrob_signals": {}
    }
    
    # Must return exactly 0.0 (neutral baseline), never a negative penalty
    score = assessment_bonus(candidate_no_scores, {"nlp", "python"})
    assert score == 0.0

    # Ensure a candidate with valid scores gets a proper fraction
    candidate_with_scores = {
        "redrob_signals": {
            "skill_assessment_scores": {"nlp": 85.0}
        }
    }
    score_valid = assessment_bonus(candidate_with_scores, {"nlp"})
    assert score_valid == 0.85


def test_feature_score_in_range():
    """Weighted total stays within [0, 1]."""
    candidate = {
        "profile": {
            "current_title": "Senior AI Engineer",
            "years_of_experience": 6.5
        },
        "career_history": [
            {
                "title": "Machine Learning Engineer",
                "company": "Google",
                "description": "Built scalable retrieval pipelines in python and pytorch.",
                "industry": "Internet"
            }
        ],
        "skills": [
            {"name": "nlp", "duration_months": 24, "endorsements": 12},
            {"name": "retrieval", "duration_months": 36, "endorsements": 18}
        ],
        "redrob_signals": {
            "skill_assessment_scores": {"nlp": 92.0},
            "github_activity_score": 7.5
        }
    }
    
    sim = 0.88 # Mock cosine similarity
    total_score, breakdown = feature_score(candidate, sim, MOCK_WEIGHTS, MockConstants())
    
    assert 0.0 <= total_score <= 1.0, f"Total score {total_score} is out of bounds!"
    
    # Ensure all sub-features are also cleanly bounded 0..1
    for feat_name, feat_val in breakdown.items():
        assert 0.0 <= feat_val <= 1.0, f"{feat_name} returned out-of-bounds score: {feat_val}"


def test_strong_ml_vs_keyword_stuffer():
    """Ensure the weighted total behaves sensibly: strong ML candidate > keyword candidate."""
    strong_candidate = {
        "profile": {
            "current_title": "AI Engineer",
            "years_of_experience": 6
        },
        "career_history": [
            {
                "title": "AI Engineer",
                "company": "Meta",
                "description": "Shipped search and ranking models. Programmed in python."
            }
        ],
        "skills": [{"name": "nlp", "duration_months": 36, "endorsements": 15}]
    }
    
    stuffer_candidate = {
        "profile": {
            "current_title": "Marketing Manager",
            "years_of_experience": 6
        },
        "career_history": [
            {
                "title": "Marketing Manager",
                "company": "TCS",
                "description": "Marketing and sales operations."
            }
        ],
        # Stuffed skills array but no career backing
        "skills": [{"name": "nlp", "duration_months": 36, "endorsements": 15}],
        "redrob_signals": {}
    }
    
    strong_score, _ = feature_score(strong_candidate, 0.80, MOCK_WEIGHTS, MockConstants())
    stuffer_score, _ = feature_score(stuffer_candidate, 0.80, MOCK_WEIGHTS, MockConstants())
    
    # The scoring matrix should decisively favor the real engineer, 
    # even if embedding similarity is identical and they both claim the same skills.
    assert strong_score > stuffer_score + 0.3