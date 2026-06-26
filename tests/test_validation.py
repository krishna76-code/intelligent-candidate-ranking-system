"""
Tests that the produced submission CSV passes the organizer's validator.

This is the safety net against auto-rejection: exactly 100 rows, unique ranks
1-100, non-increasing scores, ties broken by candidate_id ascending, etc.
"""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

OUTPUT_DIR = Path("output")
VALIDATOR_SCRIPT = Path("scripts/validate_submission.py")

@pytest.fixture
def submission_csv():
    """
    Locates the generated submission CSV. 
    Skips tests gracefully if rank.py hasn't been run yet.
    """
    csv_files = list(OUTPUT_DIR.glob("*.csv"))
    if not csv_files:
        pytest.skip("No submission CSV found in output/ directory. Run rank.py first.")
    # Assuming there's only one CSV in the output folder for the submission
    return csv_files[0]

def test_submission_passes_validator(submission_csv):
    """Executes the official organizer validation script on the output."""
    if not VALIDATOR_SCRIPT.exists():
        pytest.skip(f"Validator script not found at {VALIDATOR_SCRIPT}.")
        
    result = subprocess.run(
        [sys.executable, str(VALIDATOR_SCRIPT), str(submission_csv)],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, f"Organizer validator failed!\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

def test_exactly_100_rows_and_correct_columns(submission_csv):
    """Ensures exactly 100 rows exist with the exact required header schema."""
    with open(submission_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        
    assert len(reader) > 0, "CSV is completely empty."
    
    # 1. Check exact column headers
    expected_headers = {"candidate_id", "score", "reasoning"}
    actual_headers = set(reader[0].keys())
    assert actual_headers == expected_headers, f"Expected headers {expected_headers}, got {actual_headers}"
    
    # 2. Check exact row count (Rule: exactly 100 candidates)
    assert len(reader) == 100, f"Submission must have exactly 100 rows, found {len(reader)}"
    
    # 3. Check for empty values or bad formatting
    seen_ids = set()
    for i, row in enumerate(reader):
        c_id = row.get("candidate_id", "").strip()
        score_str = row.get("score", "").strip()
        reasoning = row.get("reasoning", "").strip()
        
        assert c_id, f"Row {i + 1} is missing a candidate_id."
        assert c_id not in seen_ids, f"Duplicate candidate_id found: {c_id}"
        seen_ids.add(c_id)
        
        assert score_str, f"Row {i + 1} ({c_id}) is missing a score."
        assert reasoning, f"Row {i + 1} ({c_id}) is missing reasoning text."

def test_scores_non_increasing_and_ties_broken(submission_csv):
    """
    Ensures the ranker sorted properly: highest score first.
    If scores tie, candidate_id must be sorted ascending.
    """
    with open(submission_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        
    for i in range(len(reader) - 1):
        try:
            curr_score = float(reader[i]["score"])
            next_score = float(reader[i+1]["score"])
        except ValueError:
            pytest.fail(f"Invalid score format found at rows {i+1} or {i+2}")
            
        curr_id = reader[i]["candidate_id"]
        next_id = reader[i+1]["candidate_id"]
        
        # 1. Scores must be descending
        assert curr_score >= next_score, (
            f"Sorting error: Row {i+1} ({curr_score}) is lower than Row {i+2} ({next_score})"
        )
        
        # 2. Tie-breaker rule (candidate_id ascending)
        if curr_score == next_score:
            assert curr_id < next_id, (
                f"Tie-breaker error: Score tie ({curr_score}) at rows {i+1} and {i+2}, "
                f"but IDs '{curr_id}' and '{next_id}' are not ascending."
            )