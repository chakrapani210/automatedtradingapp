import pandas as pd
from utils.strategy_utils import (
    scale_risk_fraction,
    compute_confidence,
    compute_high_priority_override_series,
    evaluate_partial_exit,
    update_trailing_atr,
    compute_gated_buy,
    compute_composite_sell,
    apply_drawdown_dampening
)


def test_scale_risk_fraction_basic():
    base = 0.02
    # below low -> floor
    rf1 = scale_risk_fraction(base, score=0.10, low=0.40, high=0.80, floor_frac=0.25)
    assert abs(rf1 - base * 0.25) < 1e-9
    # above high -> base
    rf2 = scale_risk_fraction(base, score=0.90, low=0.40, high=0.80, floor_frac=0.25)
    assert abs(rf2 - base * 1.0) < 1e-9
    # mid
    mid_score = 0.60
    rf3 = scale_risk_fraction(base, score=mid_score, low=0.40, high=0.80, floor_frac=0.25)
    # linear fraction expected: floor + ((0.60-0.40)/(0.40))*(0.75) = 0.25 + 0.5*0.75 = 0.625
    assert abs(rf3 - base * 0.625) < 1e-9


def test_compute_confidence_blend():
    comps_last = pd.Series({'a': True, 'b': False, 'c': True})
    rel = {'a': 0.8, 'b': 0.4, 'c': 0.9}
    comp_score = 0.70
    conf = compute_confidence(comp_score, comps_last, rel, blend_weight=0.5)
    # reliability-weighted correctness = (1*0.8 + 0*0.4 + 1*0.9)/(0.8+0.4+0.9)= (1.7)/(2.1)=0.809523...
    # blended = 0.5*0.70 + 0.5*0.809523 = 0.75476
    assert abs(conf - 0.75476) < 1e-3


def test_high_priority_override_series():
    df = pd.DataFrame({
        'trend': [True, True, False],
        'ema': [True, False, False],
        'mom': [True, True, True]
    })
    hp = ['trend', 'ema']
    series = compute_high_priority_override_series(df, hp, override_thr=0.75)
    # Row wise share of hp bullish: row0=2/2=1 -> True; row1=1/2=0.5 -> False; row2=0/2=0 -> False
    assert list(series) == [True, False, False]


def test_evaluate_partial_exit():
    # Trigger composite drop partial exit
    new_sig, pc, pconf = evaluate_partial_exit(
        last_signal=1,
        enable_partial=True,
        enable_composite=True,
        peak_composite=0.90,
        peak_confidence=0.85,
        latest_comp=0.70,
        current_confidence=0.82,
        sell_score_threshold=0.40,
        drop_thr=0.15,
        conf_partial=True
    )
    assert new_sig == 2
    # No trigger scenario
    new_sig2, _, _ = evaluate_partial_exit(
        last_signal=1,
        enable_partial=True,
        enable_composite=True,
        peak_composite=0.90,
        peak_confidence=0.85,
        latest_comp=0.80,
        current_confidence=0.84,
        sell_score_threshold=0.40,
        drop_thr=0.15,
        conf_partial=True
    )
    assert new_sig2 == 1


def test_update_trailing_atr():
    # Initialize
    trail, exit_flag = update_trailing_atr(
        trail_stop=None,
        enable_trailing_atr=True,
        last_signal=1,
        entry_price=100.0,
        close_price=105.0,
        atr_value=2.0,
        trail_mult=2.0
    )
    assert not exit_flag
    assert trail is not None
    # Price falls below updated trail
    trail2, exit_flag2 = update_trailing_atr(
        trail_stop=trail,
        enable_trailing_atr=True,
        last_signal=1,
        entry_price=100.0,
        close_price=trail - 0.01,
        atr_value=2.0,
        trail_mult=2.0
    )
    assert exit_flag2 is True


def test_gating_and_composite_sell():
    import pandas as pd
    idx = pd.date_range('2024-01-01', periods=5)
    composite = pd.Series([0.5,0.65,0.7,0.55,0.30], index=idx)
    override = pd.Series([False,False,True,False,False], index=idx)
    bull_counts = pd.Series([2,2,2,1,0], index=idx)
    bull_series = pd.Series([True,True,True,True,False], index=idx)
    gated = compute_gated_buy(composite, override, bull_counts, 2, bull_series, 0.60)
    # Expect True on indices 1 (0.65 pass) & 2 (override True) only
    assert list(gated) == [False, True, True, False, False]
    comp_sell = compute_composite_sell(composite, 0.40, bull_counts, 2)
    # Sell when composite <=0.40 OR bull_counts<2
    assert list(comp_sell) == [False, False, False, True, True]


def test_drawdown_dampening():
    holder = {'peak': None}
    rf = 0.02
    # First price sets peak
    rf1 = apply_drawdown_dampening(rf, 100.0, holder, 0.10, 0.50)
    assert holder['peak'] == 100.0 and rf1 == rf
    # New higher peak
    rf2 = apply_drawdown_dampening(rf1, 110.0, holder, 0.10, 0.50)
    assert holder['peak'] == 110.0 and rf2 == rf1
    # 5% drawdown - below threshold, unchanged
    rf3 = apply_drawdown_dampening(rf2, 104.5, holder, 0.10, 0.50)
    assert abs(rf3 - rf2) < 1e-12
    # 15% drawdown triggers dampening -> 0.02 * 0.50 = 0.01
    price_after_dd = 93.5  # (110-93.5)/110 ~ 15%
    rf4 = apply_drawdown_dampening(rf3, price_after_dd, holder, 0.10, 0.50)
    assert abs(rf4 - 0.01) < 1e-9

