import pytest
from src.scoring.features import (
    feature_score, experience_fit, skill_trust, assessment_bonus
)

# Mock constants for testing
class MockConstants:
    CORE_AI_SKILLS = {"nlp", "retrieval", "ranking", "search", "llm"}
    CONSULTING_FIRMS = {"tcs", "infosys"}
    PRODUCT_COMPANIES = {"google", "meta"}

MOCK_WEIGHTS = {
    "title_role_match": 0.22, "career_evidence": 0.20, "embedding_similarity": 0.18,
    "experience_fit": 0.12, "product_vs_services": 0.10, "skill_trust": 0.08,
    "assessment_bonus": 0.05, "writes_code_recently": 0.05
}

def test_experience_fit_peaks_in_band():
    """Verify that experience fit score follows the target window logic."""
    ideal = {"ideal_min": 5, "ideal_max": 9}
    # Test peak
    assert experience_fit({"profile": {"years_of_experience": 7}}, 5, 9) == 1.0
    # Test low
    assert experience_fit({"profile": {"years_of_experience": 1}}, 5, 9) < 0.5
    # Test high
    assert experience_fit({"profile": {"years_of_experience": 20}}, 5, 9) < 0.5

def test_assessment_bonus_never_penalizes_absence():
    """Verify that no signals results in exactly 0.0, not a negative penalty."""
    candidate_no_signals = {"redrob_signals": {}}
    assert assessment_bonus(candidate_no_signals, {"nlp"}) == 0.0

def test_skill_trust_discounting():
    """Verify that skills with 0 endorsements or <6 months duration are ignored."""
    candidate = {
        "skills": [
            {"name": "nlp", "duration_months": 2, "endorsements": 10}, # Too short
            {"name": "search", "duration_months": 12, "endorsements": 0} # No endorsements
        ]
    }
    assert skill_trust(candidate) == 0.0

def test_feature_score_in_range():
    """Verify total weighted score remains bounded [0, 1]."""
    candidate = {
        "profile": {"current_title": "AI Engineer", "years_of_experience": 6},
        "career_history": [{"title": "AI Engineer", "company": "Google", "description": "built retrieval system"}],
        "skills": [{"name": "nlp", "duration_months": 12, "endorsements": 5}],
        "redrob_signals": {"skill_assessment_scores": {"nlp": 90}}
    }
    
    score, breakdown = feature_score(candidate, 0.8, MOCK_WEIGHTS, constants=MockConstants())
    
    assert 0.0 <= score <= 1.0
    assert len(breakdown) == 8
    assert all(0.0 <= v <= 1.0 for v in breakdown.values())

def test_strong_vs_weak_candidate():
    """Strong ML candidate should score higher than a keyword stuffer."""
    strong = {
        "profile": {"current_title": "Senior AI Engineer", "years_of_experience": 7},
        "career_history": [{"title": "Senior AI Engineer", "company": "Google", "description": "built retrieval system"}],
        "skills": [{"name": "nlp", "duration_months": 24, "endorsements": 20}],
        "redrob_signals": {"skill_assessment_scores": {"nlp": 95}, "github_activity_score": 8}
    }
    
    weak = {
        "profile": {"current_title": "Marketing Manager", "years_of_experience": 1},
        "career_history": [{"title": "Marketing Manager", "company": "TCS", "description": "marketing"}],
        "skills": [],
        "redrob_signals": {}
    }
    
    strong_score, _ = feature_score(strong, 0.9, MOCK_WEIGHTS, constants=MockConstants())
    weak_score, _ = feature_score(weak, 0.1, MOCK_WEIGHTS, constants=MockConstants())
    
    assert strong_score > weak_score