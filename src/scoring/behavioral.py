"""
behavioral.py — reachability multiplier (~0.5..1.1).

"Can we actually hire them?" A perfect-on-paper candidate with a 5% recruiter
response rate is effectively unhireable. This is a MULTIPLIER, not a filter.

Signals used (from redrob_signals):
    - recruiter_response_rate : low -> big down-weight
    - last_active_date        : continuous decay by months-since-active
                                (today is 2026; 2025 is NOT automatically stale)
    - open_to_work_flag       : small bonus
    - notice_period_days      : short -> small bonus (JD loves sub-30-day)

API to implement:
    def behavioral_multiplier(candidate, today=date(2026,6,23)) -> tuple[float, dict]:
        '''Return (multiplier, breakdown) for explainability.'''

TODO: implement. Placeholder skeleton.
"""

from __future__ import annotations

# from datetime import date


def recency_decay(last_active_date: str, today=None) -> float:
    """Continuous decay by months since last active. NOT a 2025 cutoff."""
    raise NotImplementedError


def behavioral_multiplier(candidate: dict, today=None) -> "tuple[float, dict]":
    raise NotImplementedError
