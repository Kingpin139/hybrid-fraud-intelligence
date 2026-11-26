import math

import pytest

from hybrid_fraud.scoring.risk_engine import calibrate_score, risk_to_decision


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, "approve"),
        (0.29, "approve"),
        (0.30, "review"),
        (0.64, "review"),
        (0.65, "decline"),
        (1.0, "decline"),
    ],
)
def test_risk_to_decision(score, expected):
    assert risk_to_decision(score) == expected


def test_risk_to_decision_custom_and_invalid_thresholds():
    assert risk_to_decision(0.5, 0.6, 0.9) == "approve"
    with pytest.raises(ValueError):
        risk_to_decision(0.5, 0.7, 0.3)


def test_calibrate_score_is_stable_and_monotonic():
    assert calibrate_score(0.0) == pytest.approx(0.5)
    assert calibrate_score(-1000.0) == pytest.approx(0.0, abs=1e-12)  # no OverflowError
    assert calibrate_score(1000.0) == pytest.approx(1.0)
    assert calibrate_score(1.0) == pytest.approx(1 / (1 + math.exp(-1.0)))
    assert calibrate_score(-1.0) < calibrate_score(1.0)
