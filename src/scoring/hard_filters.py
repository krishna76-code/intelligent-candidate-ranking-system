"""
hard_filters.py — instant disqualifiers (score -> 0).

These are cheap rules applied BEFORE feature scoring. They remove ~30,000+
candidates so we never waste compute scoring obvious rejects.

Verified counts against the real 100K file:
    - Honeypot: company start < founding (Krutrim/Sarvam)       -> 73
    - Honeypot: >=3 skills "expert" with duration_months == 0   -> 21
    - Honeypot: experience inflation (claimed YOE >> career sum) -> +24 new
    - Consulting-only careers (TCS/Infosys/Wipro/...)           -> 9,745
    - Foreign (country != India) AND willing_to_relocate False  -> 17,756
    - Non-eng title (Marketing/HR/Sales/...) + AI skills        -> 5,323

API:
    def is_disqualified(candidate: dict, constants) -> tuple[bool, str]:
        '''Return (True, reason) if the candidate should be hard-filtered to 0.'''

NOTE: all accessors are crash-safe — fields may be missing OR explicitly None
(the sandbox accepts uploaded samples, so we never assume clean input).
"""

from __future__ import annotations

from src.utils.constants import (
    CONSULTING_FIRMS,
    COMPANY_FOUNDED,
    NON_ENG_TITLES,
    AI_SKILL_TERMS,
    ENG_TITLE_TERMS,
)


# --------------------------------------------------------------------------- #
# Small crash-safe helpers: a key can be ABSENT or present-but-None.
# `dict.get(k, default)` only covers the absent case, so we coerce None too.
# --------------------------------------------------------------------------- #
def _s(value) -> str:
    """Coerce any value to a clean lowercase string (None -> '')."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _list(value) -> list:
    """Coerce to a list (None / non-list -> [])."""
    return value if isinstance(value, list) else []


def _dict(value) -> dict:
    """Coerce to a dict (None / non-dict -> {})."""
    return value if isinstance(value, dict) else {}


def _num(value, default: float = 0.0) -> float:
    """Coerce to a float (None / bad -> default)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Honeypots (impossible profiles)
# --------------------------------------------------------------------------- #
def is_honeypot_company_founding(candidate: dict, company_founded: dict) -> bool:
    """start_year < founded_year for a known-young company."""
    for job in _list(_dict(candidate).get("career_history")):
        company = _s(_dict(job).get("company"))
        if company in company_founded:
            start_date = _s(_dict(job).get("start_date"))
            if not start_date:
                continue
            try:
                start_year = int(start_date[:4])
            except ValueError:
                continue
            if start_year < company_founded[company]:
                return True
    return False


def is_honeypot_zero_duration_expert(candidate: dict, threshold: int = 3) -> bool:
    """>= threshold skills with proficiency == 'expert' AND duration_months == 0."""
    count = 0
    for skill in _list(_dict(candidate).get("skills")):
        skill = _dict(skill)
        if _s(skill.get("proficiency")) == "expert" and _num(skill.get("duration_months")) == 0:
            count += 1
            if count >= threshold:
                return True
    return False


def is_honeypot_experience_inflation(candidate: dict, gap_years: float = 8.0) -> bool:
    """
    Claimed years_of_experience far exceeds the sum of actual career durations.
    Spec honeypot example: "8 years of experience at a company founded 3 years ago".
    """
    profile = _dict(_dict(candidate).get("profile"))
    yoe = _num(profile.get("years_of_experience"))
    total_months = sum(_num(_dict(j).get("duration_months")) for j in _list(_dict(candidate).get("career_history")))
    career_years = total_months / 12.0
    return (yoe - career_years) > gap_years


# --------------------------------------------------------------------------- #
# Structural disqualifiers
# --------------------------------------------------------------------------- #
def is_consulting_only(candidate: dict, consulting_firms: set) -> bool:
    """Every company in career history is a consulting firm."""
    career = _list(_dict(candidate).get("career_history"))
    if not career:
        return False
    for job in career:
        if _s(_dict(job).get("company")) not in consulting_firms:
            return False
    return True


def cannot_relocate(candidate: dict) -> bool:
    """country != 'India' AND willing_to_relocate == False."""
    candidate = _dict(candidate)
    profile = _dict(candidate.get("profile"))
    signals = _dict(candidate.get("redrob_signals") or profile.get("redrob_signals"))
    country = _s(profile.get("country"))
    # Default True (unknown -> don't disqualify); only an explicit False blocks.
    willing = signals.get("willing_to_relocate", True)
    return country != "india" and willing is False


def is_buzzword_stuffer(
    candidate: dict,
    non_eng_titles: set,
    ai_terms: set,
    eng_title_terms: set = ENG_TITLE_TERMS,
) -> bool:
    """
    Non-engineering title + claimed AI skills, with a safety net.

    1. Current title is non-engineering (substring match, e.g. 'senior marketing manager')
    2. AND they list a core AI skill
    3. AND they have NEVER held a real AI/ML engineering JOB TITLE in their history
       -> if they ever held a real AI job title, spare them for scoring.

    Why job titles, not description text? The dataset's stuffers have AI phrases
    deliberately seeded into their summaries/descriptions ("took online courses on
    RAG and vector databases", "LangChain side projects"). Matching description
    text therefore spares everyone. A real AI ENGINEERING JOB TITLE in the career
    history is the one signal these synthetic stuffers never have — and a genuine
    "marketing manager who actually became an AI engineer" would show exactly that.
    (Verified: 0 of the ~5.3K flagged stuffers ever held an AI job title.)
    """
    candidate = _dict(candidate)
    profile = _dict(candidate.get("profile"))
    title = _s(profile.get("current_title"))
    if not title:
        return False

    # (1) substring match: catches "Senior Marketing Manager", "Marketing Manager II", etc.
    if not any(neng in title for neng in non_eng_titles):
        return False

    # (2) do they list a core AI skill?
    has_ai_skill = any(_s(_dict(s).get("name")) in ai_terms for s in _list(candidate.get("skills")))
    if not has_ai_skill:
        return False

    # (3) safety net: did they EVER hold a real AI/ML engineering job title?
    # (the unfakeable signal — descriptions/summaries are seeded with AI buzzwords)
    for job in _list(candidate.get("career_history")):
        job_title = _s(_dict(job).get("title"))
        if job_title and any(term in job_title for term in eng_title_terms):
            return False

    return True


# --------------------------------------------------------------------------- #
# Master gate
# --------------------------------------------------------------------------- #
def is_disqualified(candidate: dict, constants=None) -> "tuple[bool, str]":
    """Master gate. Returns (True, reason) if the candidate must score 0."""
    if is_honeypot_company_founding(candidate, COMPANY_FOUNDED):
        return True, "Impossible company joining date"

    if is_honeypot_zero_duration_expert(candidate):
        return True, "Fake expert skills"

    if is_honeypot_experience_inflation(candidate):
        return True, "Impossible experience inflation"

    if is_consulting_only(candidate, CONSULTING_FIRMS):
        return True, "Consulting-only career"

    if cannot_relocate(candidate):
        return True, "Cannot relocate"

    if is_buzzword_stuffer(candidate, NON_ENG_TITLES, AI_SKILL_TERMS):
        return True, "Buzzword stuffer"

    return False, ""
