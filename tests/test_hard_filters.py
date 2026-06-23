"""
Tests for src/scoring/hard_filters.py.

Each test feeds a minimal candidate dict and asserts the filter fires correctly.
Fill in once hard_filters.py is implemented.
"""

import pytest

# from src.scoring.hard_filters import (
#     is_honeypot_company_founding, is_honeypot_zero_duration_expert,
#     is_consulting_only, cannot_relocate, is_buzzword_stuffer, is_disqualified,
# )


@pytest.mark.skip(reason="placeholder — implement hard_filters.py first")
def test_honeypot_krutrim_before_2023():
    # candidate with Krutrim start_date 2020 => disqualified
    ...


@pytest.mark.skip(reason="placeholder")
def test_zero_duration_expert_threshold():
    # >=3 expert skills with duration_months == 0 => disqualified
    ...


@pytest.mark.skip(reason="placeholder")
def test_consulting_only():
    # all companies in {TCS, Infosys} => disqualified
    ...


@pytest.mark.skip(reason="placeholder")
def test_cannot_relocate():
    # country=USA, willing_to_relocate=False => disqualified
    ...


@pytest.mark.skip(reason="placeholder")
def test_buzzword_stuffer():
    # current_title='Marketing Manager' + PyTorch/LLM skills => disqualified
    ...
