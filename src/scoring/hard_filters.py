from __future__ import annotations

from src.utils.constants import (
    CONSULTING_COMPANIES,
    COMPANY_FOUNDED,
    NON_ENGINEERING_TITLES,
    AI_SKILL_TERMS,
)


def is_honeypot_company_founding(candidate: dict,
                                 company_founded: dict) -> bool:
    """
    start_year < founded_year for known young companies.
    """

    for exp in candidate.get("career_history", []):

        company = exp.get("company", "").strip().lower()
        start_year = exp.get("start_year")

        if company in company_founded:

            if start_year is not None and start_year < company_founded[company]:
                return True

    return False


def is_honeypot_zero_duration_expert(candidate: dict,
                                     threshold: int = 3) -> bool:
    """
    >= threshold skills with proficiency='expert'
    and duration_months == 0.
    """

    count = 0

    for skill in candidate.get("skills", []):

        proficiency = skill.get("proficiency", "").lower()
        duration = skill.get("duration_months", -1)

        if proficiency == "expert" and duration == 0:
            count += 1

    return count >= threshold


def is_consulting_only(candidate: dict,
                       consulting_firms: set) -> bool:
    """
    Every company in career history belongs to consulting.
    """

    companies = []

    for exp in candidate.get("career_history", []):

        company = exp.get("company", "").strip().lower()

        if company:
            companies.append(company)

    if not companies:
        return False

    return all(company in consulting_firms for company in companies)


def cannot_relocate(candidate: dict) -> bool:

    country = candidate.get("country", "").strip().lower()

    willing = candidate.get(
        "willing_to_relocate",
        True
    )

    return country != "india" and not willing


def is_buzzword_stuffer(candidate: dict,
                        non_eng_titles: set,
                        ai_terms: set) -> bool:
    """
    Marketing/HR/Sales/etc title + AI skills.
    """

    title = candidate.get(
        "headline",
        ""
    ).lower()

    title_is_non_eng = any(
        term in title
        for term in non_eng_titles
    )

    if not title_is_non_eng:
        return False

    skill_text = " ".join(
        skill.get("name", "").lower()
        for skill in candidate.get("skills", [])
    )

    claims_ai = any(
        term in skill_text
        for term in ai_terms
    )

    return claims_ai


def is_disqualified(candidate: dict,
                    constants=None) -> tuple[bool, str]:
    """
    Master gate.
    """

    if is_honeypot_company_founding(
            candidate,
            COMPANY_FOUNDED):
        return True, "honeypot_company"

    if is_honeypot_zero_duration_expert(candidate):
        return True, "fake_expert"

    if is_consulting_only(
            candidate,
            CONSULTING_COMPANIES):
        return True, "consulting_only"

    if cannot_relocate(candidate):
        return True, "cannot_relocate"

    if is_buzzword_stuffer(
            candidate,
            NON_ENGINEERING_TITLES,
            AI_SKILL_TERMS):
        return True, "buzzword_stuffer"

    return False, ""