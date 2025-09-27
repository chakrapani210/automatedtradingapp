import math
from utils.strategy_utils import apply_volatility_gate


def test_apply_volatility_gate_no_penalty_when_disabled():
    score = 0.75
    new_score = apply_volatility_gate(score, price=100.0, atr_value=9.0,
                                      enabled=False, gate_atr_pct=0.08, confidence_penalty=0.10)
    assert new_score == score


def test_apply_volatility_gate_penalty_applied():
    score = 0.80
    # atr/price = 10% > 8% gate, expect 0.70
    new_score = apply_volatility_gate(score, price=100.0, atr_value=10.0,
                                      enabled=True, gate_atr_pct=0.08, confidence_penalty=0.10)
    assert math.isclose(new_score, 0.70, rel_tol=1e-9)


def test_apply_volatility_gate_with_nan_atr():
    score = 0.55
    new_score = apply_volatility_gate(score, price=100.0, atr_value=float('nan'),
                                      enabled=True, gate_atr_pct=0.08, confidence_penalty=0.10)
    assert new_score == score


def test_apply_volatility_gate_floor_zero():
    score = 0.05
    new_score = apply_volatility_gate(score, price=100.0, atr_value=20.0,
                                      enabled=True, gate_atr_pct=0.08, confidence_penalty=0.10)
    assert new_score == 0.0
