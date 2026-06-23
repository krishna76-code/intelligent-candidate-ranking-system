"""
reasoning.py — generates the 1-2 sentence reasoning string per candidate.

This is the Stage-4 make-or-break. The organizer validator/reviewer PENALIZES:
    - empty reasoning
    - all-identical strings
    - templated reasoning that just inserts the candidate's name
    - hallucinated skills/companies not in the profile
    - reasoning that contradicts the rank

Each reasoning MUST:
    1. Reference SPECIFIC evidence (company + what they actually built, years, signals)
    2. Connect to a JD requirement (retrieval/ranking/eval/product/etc.)
    3. At lower ranks, ACKNOWLEDGE the concern honestly (e.g. "120-day notice period")
    4. Vary in structure across candidates; tone matches rank
    5. NEVER invent anything not in the candidate's JSON

NOTE: the shipped sample_submission.csv is a deliberate LOW-quality anti-example
("HR Manager with 6.1 yrs; 9 AI core skills; response rate 0.76") — do NOT imitate it.

API to implement:
    def build_reasoning(candidate, rank, feature_breakdown, penalties, behavioral) -> str

TODO: implement. Placeholder skeleton.
"""

from __future__ import annotations


def build_reasoning(candidate: dict, rank: int, feature_breakdown: dict,
                    penalties=None, behavioral=None) -> str:
    raise NotImplementedError
