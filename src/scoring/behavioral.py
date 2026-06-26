from __future__ import annotations
from datetime import date, datetime
import math


def recency_decay(last_active_date: str, today=None) -> float:
    """
    Continuous decay by months since last active. NOT a 2025 cutoff.
    Uses an exponential decay model to smoothly reduce the multiplier over time.
    """
    if today is None:
        today = date(2026, 6, 23)
        
    if not last_active_date:
        return 0.5  # Base floor if tracking date is completely missing
        
    try:
        active_dt = datetime.strptime(str(last_active_date).strip(), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 0.5
        
    delta_days = (today - active_dt).days
    
    # Handle future dates or same-day activity safely
    if delta_days <= 0:
        return 1.0
        
    elapsed_months = delta_days / 30.4375
    
    # Exponential decay half-life equation: e^(-ln(2) * t / half_life)
    # A 9-month half-life keeps 2025 candidates viable but noticeably decayed
    half_life_months = 9.0
    decay_factor = math.exp(-0.693147 * elapsed_months / half_life_months)
    
    # Compress the output floor to 0.6 so extremely stale profiles aren't zeroed out
    return 0.6 + (0.4 * decay_factor)


def behavioral_multiplier(candidate: dict, today=None) -> tuple[float, dict]:
    """
    Combine reachability signals into a multiplier bounded between ~0.5 and 1.1.
    Return (multiplier, breakdown) for explainability.
    """
    if today is None:
        today = date(2026, 6, 23)
        
    signals = candidate.get("redrob_signals", {})
    breakdown = {}
    
    # 1. Recruiter Response Rate Filter
    # Default to 1.0 (neutral) to avoid punishing candidates without historical data
    response_rate = float(signals.get("recruiter_response_rate", 1.0))
    if response_rate < 0.10:
        breakdown["response_rate_factor"] = 0.65
    elif response_rate < 0.30:
        breakdown["response_rate_factor"] = 0.85
    else:
        breakdown["response_rate_factor"] = 1.0
        
    # 2. Continuous Recency Decay
    last_active = signals.get("last_active_date", "")
    breakdown["recency_decay_factor"] = float(recency_decay(last_active, today=today))
    
    # 3. Open To Work Bonus
    open_to_work = bool(signals.get("open_to_work_flag", False))
    breakdown["open_to_work_bonus"] = 0.05 if open_to_work else 0.0
    
    # 4. Notice Period Target Bonus
    # JD strongly prefers immediate/sub-30-day availability
    notice_days = signals.get("notice_period_days")
    if notice_days is not None:
        try:
            days = float(notice_days)
            if days <= 30.0:
                breakdown["notice_period_bonus"] = 0.05
            elif days <= 60.0:
                breakdown["notice_period_bonus"] = 0.0
            else:
                breakdown["notice_period_bonus"] = -0.05  # Mild friction for >60 days
        except (ValueError, TypeError):
            breakdown["notice_period_bonus"] = 0.0
    else:
        breakdown["notice_period_bonus"] = 0.0
        
    # Combine signals: multiplicative for core filters, additive for soft bonuses
    base_multiplier = breakdown["response_rate_factor"] * breakdown["recency_decay_factor"]
    raw_multiplier = base_multiplier + breakdown["open_to_work_bonus"] + breakdown["notice_period_bonus"]
    
    # Enforce strict hackathon boundary constraints cleanly
    final_multiplier = max(0.50, min(1.10, raw_multiplier))
    
    return float(final_multiplier), breakdown