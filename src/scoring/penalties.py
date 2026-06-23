"""
penalties.py — JD-specific soft penalties (reduce score, do NOT zero out).

Straight from the JD's "Things we explicitly do NOT want" section:
    - title_chaser        : >=4 jobs with >=3 tenures < 20 months   (~5,019 candidates)
    - langchain_only      : only recent LangChain/OpenAI work, no pre-2022 real ML
    - cv_speech_robotics  : computer vision / speech / robotics without NLP/IR exposure
    - research_only       : academic titles, no "production/deployed/shipped/users/scale"

API to implement:
    def penalty_multiplier(candidate, constants) -> tuple[float, list[str]]:
        '''Return (multiplier in ~0.3..1.0, list of penalty reasons applied).'''

TODO: implement. Placeholder skeleton.
"""

from __future__ import annotations


def is_title_chaser(candidate: dict, short_tenure_months: int = 20, min_jobs: int = 4, min_short: int = 3) -> bool:
    raise NotImplementedError


def is_langchain_only(candidate: dict) -> bool:
    raise NotImplementedError


def is_cv_speech_robotics_only(candidate: dict, nlp_ir_terms: set) -> bool:
    raise NotImplementedError


def is_research_only(candidate: dict, production_terms: set) -> bool:
    raise NotImplementedError


def penalty_multiplier(candidate: dict, constants=None) -> "tuple[float, list]":
    """Combine all JD penalties into a single multiplier + reasons list."""
    raise NotImplementedError
