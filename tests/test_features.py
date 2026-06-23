"""
Tests for src/scoring/features.py.

Verify each feature returns a value in [0, 1] and that the weighted total
behaves sensibly (strong ML candidate > weak/keyword candidate).
Fill in once features.py is implemented.
"""

import pytest

# from src.scoring.features import feature_score, experience_fit, skill_trust


@pytest.mark.skip(reason="placeholder — implement features.py first")
def test_experience_fit_peaks_in_band():
    # 7 yrs => high; 1 yr => low; 20 yrs => reduced
    ...


@pytest.mark.skip(reason="placeholder")
def test_assessment_bonus_never_penalizes_absence():
    # candidate with no skill_assessment_scores must NOT be penalized
    ...


@pytest.mark.skip(reason="placeholder")
def test_feature_score_in_range():
    # weighted total stays within [0, 1]
    ...
