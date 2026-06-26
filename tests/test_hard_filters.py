"""
Tests for src/scoring/hard_filters.py.

Each test feeds a minimal candidate dict and asserts the filter fires correctly.
Run: python -m pytest tests/test_hard_filters.py -v
"""

import sys
from pathlib import Path

# Make the repo root importable when pytest runs from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scoring.hard_filters import (  # noqa: E402
    is_honeypot_company_founding,
    is_honeypot_zero_duration_expert,
    is_honeypot_experience_inflation,
    is_consulting_only,
    cannot_relocate,
    is_buzzword_stuffer,
    is_disqualified,
)
from src.utils.constants import (  # noqa: E402
    CONSULTING_FIRMS,
    COMPANY_FOUNDED,
    NON_ENG_TITLES,
    AI_SKILL_TERMS,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def cand(profile=None, career=None, skills=None, signals=None):
    return {
        "candidate_id": "CAND_0000001",
        "profile": profile or {},
        "career_history": career or [],
        "skills": skills or [],
        "redrob_signals": signals or {},
    }


# --------------------------------------------------------------------------- #
# 1. Honeypot — company founding
# --------------------------------------------------------------------------- #
def test_honeypot_krutrim_before_2023():
    c = cand(career=[{"company": "Krutrim", "start_date": "2020-01-01"}])
    assert is_honeypot_company_founding(c, COMPANY_FOUNDED) is True


def test_honeypot_company_founding_ok_after_founding():
    c = cand(career=[{"company": "Krutrim", "start_date": "2024-01-01"}])
    assert is_honeypot_company_founding(c, COMPANY_FOUNDED) is False


def test_honeypot_company_founding_unknown_company_ignored():
    c = cand(career=[{"company": "Swiggy", "start_date": "2015-01-01"}])
    assert is_honeypot_company_founding(c, COMPANY_FOUNDED) is False


# --------------------------------------------------------------------------- #
# 2. Honeypot — zero-duration expert (threshold = 3)
# --------------------------------------------------------------------------- #
def test_zero_duration_expert_three_triggers():
    skills = [{"name": s, "proficiency": "expert", "duration_months": 0} for s in ("a", "b", "c")]
    assert is_honeypot_zero_duration_expert(cand(skills=skills)) is True


def test_zero_duration_expert_two_does_not_trigger():
    skills = [{"name": s, "proficiency": "expert", "duration_months": 0} for s in ("a", "b")]
    assert is_honeypot_zero_duration_expert(cand(skills=skills)) is False


def test_zero_duration_expert_with_duration_ignored():
    # expert but with real duration -> not a honeypot signal
    skills = [{"name": s, "proficiency": "expert", "duration_months": 24} for s in ("a", "b", "c")]
    assert is_honeypot_zero_duration_expert(cand(skills=skills)) is False


# --------------------------------------------------------------------------- #
# 3. Honeypot — experience inflation
# --------------------------------------------------------------------------- #
def test_experience_inflation_triggers():
    # claims 14 yrs, career history sums to ~1 yr (12 months)
    c = cand(
        profile={"years_of_experience": 14.0},
        career=[{"company": "X", "duration_months": 12}],
    )
    assert is_honeypot_experience_inflation(c) is True


def test_experience_inflation_consistent_ok():
    # claims 7 yrs, career sums to ~7 yrs (84 months) -> fine
    c = cand(
        profile={"years_of_experience": 7.0},
        career=[{"company": "X", "duration_months": 84}],
    )
    assert is_honeypot_experience_inflation(c) is False


# --------------------------------------------------------------------------- #
# 4. Consulting-only
# --------------------------------------------------------------------------- #
def test_consulting_only_all_consulting():
    c = cand(career=[{"company": "TCS"}, {"company": "Infosys"}])
    assert is_consulting_only(c, CONSULTING_FIRMS) is True


def test_consulting_only_mixed_kept():
    # has product-company experience too -> not consulting-only
    c = cand(career=[{"company": "TCS"}, {"company": "Swiggy"}])
    assert is_consulting_only(c, CONSULTING_FIRMS) is False


def test_consulting_only_empty_history_kept():
    assert is_consulting_only(cand(career=[]), CONSULTING_FIRMS) is False


# --------------------------------------------------------------------------- #
# 5. Cannot relocate
# --------------------------------------------------------------------------- #
def test_cannot_relocate_foreign_unwilling():
    c = cand(profile={"country": "USA"}, signals={"willing_to_relocate": False})
    assert cannot_relocate(c) is True


def test_relocate_foreign_but_willing_kept():
    c = cand(profile={"country": "USA"}, signals={"willing_to_relocate": True})
    assert cannot_relocate(c) is False


def test_relocate_india_unwilling_kept():
    # India-based + won't relocate must be KEPT (may already be in-region)
    c = cand(profile={"country": "India"}, signals={"willing_to_relocate": False})
    assert cannot_relocate(c) is False


# --------------------------------------------------------------------------- #
# 6. Buzzword stuffer (substring title + AI skill + no real work)
# --------------------------------------------------------------------------- #
def test_buzzword_stuffer_triggers():
    c = cand(
        profile={"current_title": "Marketing Manager", "summary": "ran ad campaigns"},
        career=[{"description": "managed marketing budgets and brand campaigns"}],
        skills=[{"name": "PyTorch"}, {"name": "LLM"}],
    )
    assert is_buzzword_stuffer(c, NON_ENG_TITLES, AI_SKILL_TERMS) is True


def test_buzzword_stuffer_substring_title():
    # "Senior Marketing Manager" should still match via substring
    c = cand(
        profile={"current_title": "Senior Marketing Manager", "summary": "campaigns"},
        career=[{"description": "brand strategy"}],
        skills=[{"name": "NLP"}],
    )
    assert is_buzzword_stuffer(c, NON_ENG_TITLES, AI_SKILL_TERMS) is True


def test_buzzword_stuffer_spared_if_real_ai_job_title():
    # non-eng CURRENT title + AI skill BUT previously held a real AI job title -> spared
    c = cand(
        profile={"current_title": "Marketing Manager", "summary": "courses on rag"},
        career=[
            {"title": "ML Engineer", "description": "built a production ranking system"},
            {"title": "Marketing Manager", "description": "campaigns"},
        ],
        skills=[{"name": "PyTorch"}],
    )
    assert is_buzzword_stuffer(c, NON_ENG_TITLES, AI_SKILL_TERMS) is False


def test_buzzword_stuffer_not_spared_by_seeded_summary():
    # the dataset's trap: AI buzzwords seeded in summary/description but NO real AI job title
    c = cand(
        profile={
            "current_title": "Project Manager",
            "summary": "took online courses on rag and vector databases, langchain side projects",
        },
        career=[{"title": "Project Manager", "description": "led project delivery; rag experiments"}],
        skills=[{"name": "Vector Database"}, {"name": "LLM"}],
    )
    assert is_buzzword_stuffer(c, NON_ENG_TITLES, AI_SKILL_TERMS) is True


def test_buzzword_stuffer_eng_title_not_flagged():
    c = cand(
        profile={"current_title": "ML Engineer"},
        skills=[{"name": "PyTorch"}],
    )
    assert is_buzzword_stuffer(c, NON_ENG_TITLES, AI_SKILL_TERMS) is False


# --------------------------------------------------------------------------- #
# 7. Master gate + robustness
# --------------------------------------------------------------------------- #
def test_is_disqualified_clean_candidate():
    c = cand(
        profile={"country": "India", "current_title": "ML Engineer", "years_of_experience": 6.0},
        career=[{"company": "Swiggy", "start_date": "2019-01-01", "duration_months": 60}],
        skills=[{"name": "PyTorch", "proficiency": "advanced", "duration_months": 36}],
        signals={"willing_to_relocate": True},
    )
    dq, reason = is_disqualified(c)
    assert dq is False and reason == ""


def test_is_disqualified_returns_reason():
    c = cand(career=[{"company": "Sarvam AI", "start_date": "2019-01-01"}])
    dq, reason = is_disqualified(c)
    assert dq is True and reason == "Impossible company joining date"


def test_robustness_empty_dict():
    # must not crash on a completely empty candidate
    dq, reason = is_disqualified({})
    assert dq is False


def test_robustness_none_fields():
    # fields present but explicitly None -> must not crash
    c = {
        "profile": None,
        "career_history": None,
        "skills": None,
        "redrob_signals": None,
    }
    dq, reason = is_disqualified(c)
    assert dq is False


def test_robustness_none_inner_values():
    c = {
        "profile": {"current_title": None, "country": None, "years_of_experience": None},
        "career_history": [{"company": None, "start_date": None, "duration_months": None}],
        "skills": [{"name": None, "proficiency": None, "duration_months": None}],
        "redrob_signals": {"willing_to_relocate": None},
    }
    dq, reason = is_disqualified(c)
    assert dq is False
