"""
Tests that the produced submission CSV passes the organizer's validator.

This is the safety net against auto-rejection: exactly 100 rows, unique ranks
1-100, non-increasing scores, ties broken by candidate_id ascending, etc.
Fill in once rank.py produces an output CSV.
"""

import pytest

# import subprocess, sys
# from pathlib import Path


@pytest.mark.skip(reason="placeholder — run after rank.py produces output/<id>.csv")
def test_submission_passes_validator():
    # result = subprocess.run([sys.executable, "scripts/validate_submission.py", "output/test.csv"])
    # assert result.returncode == 0
    ...


@pytest.mark.skip(reason="placeholder")
def test_exactly_100_rows():
    ...


@pytest.mark.skip(reason="placeholder")
def test_scores_non_increasing():
    ...
