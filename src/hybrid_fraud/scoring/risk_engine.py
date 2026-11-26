"""Risk scoring engine utilities."""

from __future__ import annotations

import math
from typing import Literal

Decision = Literal["approve", "review", "decline"]


def risk_to_decision(
    risk_score: float,
    approve_threshold: float = 0.30,
    review_threshold: float = 0.65,
) -> Decision:
    """
    Convert a continuous risk score in [0, 1] to a discrete decision.

    * ``risk < approve_threshold``  → ``"approve"``
    * ``risk < review_threshold``   → ``"review"``
    * otherwise                     → ``"decline"``
    """
    if not 0.0 <= approve_threshold <= review_threshold <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= approve <= review <= 1.")
    if risk_score < approve_threshold:
        return "approve"
    if risk_score < review_threshold:
        return "review"
    return "decline"


def calibrate_score(raw_logit: float) -> float:
    """Numerically stable sigmoid (placeholder for a learned calibrator, e.g. Platt/isotonic)."""
    if raw_logit >= 0:
        return 1.0 / (1.0 + math.exp(-raw_logit))
    z = math.exp(raw_logit)
    return z / (1.0 + z)
