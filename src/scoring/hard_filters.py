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

TODO: implement. Placeholder skeleton.
"""

from __future__ import annotations

# from src.utils.constants import CONSULTING_FIRMS, COMPANY_FOUNDED, NON_ENG_TITLES, AI_SKILL_TERMS


def is_honeypot_company_founding(candidate: dict, company_founded: dict) -> bool:
    """start_year < founded_year for a known-young company."""
    raise NotImplementedError


def is_honeypot_zero_duration_expert(candidate: dict, threshold: int = 3) -> bool:
    """>= threshold skills with proficiency == 'expert' AND duration_months == 0."""
    raise NotImplementedError


def is_consulting_only(candidate: dict, consulting_firms: set) -> bool:
    """Every company in career history is a consulting firm."""
    raise NotImplementedError


def cannot_relocate(candidate: dict) -> bool:
    """country != 'India' AND willing_to_relocate == False."""
    raise NotImplementedError


def is_buzzword_stuffer(candidate: dict, non_eng_titles: set, ai_terms: set) -> bool:
    """Non-engineering current/primary title but claims core AI skills."""
    raise NotImplementedError


def is_disqualified(candidate: dict, constants=None) -> "tuple[bool, str]":
    """Master gate. Returns (True, reason) if the candidate must score 0."""
    raise NotImplementedError
