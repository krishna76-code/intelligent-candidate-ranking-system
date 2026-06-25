"""
hard_filters.py — instant disqualifiers (score -> 0).

These are cheap rules applied BEFORE feature scoring. They remove ~30,000+
candidates so we never waste compute scoring obvious rejects.

Verified counts against the real 100K file:
    - Honeypot: Krutrim start < 2023 (founded Dec 2023)        -> 38
    - Honeypot: Sarvam start < 2023 (founded 2023)             -> 36
    - Honeypot: >=3 skills "expert" with duration_months == 0  -> 21
    - Consulting-only careers (TCS/Infosys/Wipro/...)          -> 9,745
    - Foreign (country != India) AND willing_to_relocate False -> 17,756
    - Non-eng title (Marketing/HR/Sales/...) + AI skills       -> 4,931

API to implement:
    def is_disqualified(candidate: dict, constants) -> tuple[bool, str]:
        '''Return (True, reason) if the candidate should be hard-filtered to 0.'''

Hard filters implementation.
"""

from __future__ import annotations

from src.utils.constants import (
    CONSULTING_FIRMS,
    COMPANY_FOUNDED,
    NON_ENG_TITLES,
    AI_SKILL_TERMS,
)


def is_honeypot_company_founding(candidate: dict, company_founded: dict) -> bool:
    """start_year < founded_year for a known-young company."""

    career = candidate.get("career_history", [])

    for job in career:

        company = job.get("company", "").strip().lower()

        if company in company_founded:

            start_date = job.get("start_date", "")

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

    skills = candidate.get("skills", [])

    count = 0

    for skill in skills:

        proficiency = skill.get("proficiency", "").strip().lower()
        duration = skill.get("duration_months", 0)

        if proficiency == "expert" and duration == 0:
            count += 1

        if count >= threshold:
            return True

    return False


def is_consulting_only(candidate: dict, consulting_firms: set) -> bool:
    """Every company in career history is a consulting firm."""
    career = candidate.get("career_history", [])
    if not career:
        return False
    
    for job in career:
        company = job.get("company", "").strip().lower()

        if company not in consulting_firms:
            return False
    
    return True


def cannot_relocate(candidate: dict) -> bool:
    """country != 'India' AND willing_to_relocate == False."""
    profile = candidate.get("profile", {})
    signals = (
        candidate.get("redrob_signals")
        or profile.get("redrob_signals", {})
    )
    country = profile.get("country", "").strip().lower()
    willing = signals.get("willing_to_relocate", True)
    return country != "india" and not willing


def is_buzzword_stuffer(candidate: dict, non_eng_titles: set, ai_terms: set) -> bool:
    """Non-engineering current title but claims core AI skills."""

    profile = candidate.get("profile", {})

    title = profile.get("current_title", "").strip().lower()

    if title not in non_eng_titles:
        return False

    skills = candidate.get("skills", [])

    for skill in skills:

        skill_name = skill.get("name", "").strip().lower()

        if skill_name in ai_terms:
            return True

    return False


def is_disqualified(candidate: dict, constants=None) -> tuple[bool, str]:
    """Master gate. Returns (True, reason) if the candidate must score 0."""

    if is_honeypot_company_founding(candidate, COMPANY_FOUNDED):
        return True, "Impossible company joining date"

    if is_honeypot_zero_duration_expert(candidate):
        return True, "Fake expert skills"

    if is_consulting_only(candidate, CONSULTING_FIRMS):
        return True, "Consulting-only career"

    if cannot_relocate(candidate):
        return True, "Cannot relocate"

    if is_buzzword_stuffer(candidate, NON_ENG_TITLES, AI_SKILL_TERMS):
        return True, "Buzzword stuffer"

    return False, ""
