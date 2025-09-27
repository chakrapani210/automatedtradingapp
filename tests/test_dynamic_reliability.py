import pandas as pd
import math
from utils.strategy_utils import dynamic_reliability_weighting

class DummyCfg:
    enable_dynamic_calibration = True
    warmup_bars_for_reliability = 5
    calibration_update_interval = 1
    reliability_smoothing = 0.5
    reliability_regression_to_mean = 0.0
    micro_move_threshold = 0.0
    reliability_floor = 0.1
    instability_lookback = 4
    instability_penalty = 0.5

# Provide attribute names expected by utility
setattr(DummyCfg, 'reliability_smoothing', 0.5)
setattr(DummyCfg, 'reliability_regression_to_mean', 0.0)
setattr(DummyCfg, 'micro_move_threshold', 0.0)
setattr(DummyCfg, 'reliability_floor', 0.1)
setattr(DummyCfg, 'instability_lookback', 4)
setattr(DummyCfg, 'instability_penalty', 0.5)


def test_dynamic_reliability_weighting_basic():
    cfg = DummyCfg()
    # Simulate 12 bars with 2 components that alternate correctness differently
    idx = pd.date_range('2024-01-01', periods=12)
    # Component A mostly bullish, Component B alternating to create instability
    comp_a = [1,1,1,1,1,1,1,0,1,1,1,1]
    comp_b = [0,1,0,1,0,1,0,1,0,1,0,1]
    components_df = pd.DataFrame({'A': comp_a, 'B': comp_b}, index=idx).astype(bool)
    # Price series rising modestly then flat to evaluate correctness
    prices = pd.Series([100,101,102,103,104,105,106,105,106,107,108,109], index=idx)
    raw = components_df.astype(int).mean(axis=1)
    state = {'reliability': {}, 'flips': {}, 'effective_weights': {}, 'last_close': None}
    reliability_ts = []
    # Feed sequentially like real-time to trigger updates after warmup
    for i in range(1, len(idx)+1):
        sub_df = components_df.iloc[:i]
        sub_raw = raw.iloc[:i]
        sub_prices = prices.iloc[:i]
        dynamic_reliability_weighting(state, sub_df, sub_raw, sub_prices, cfg, reliability_ts)
    # After warmup+calibration expect reliability assigned
    assert 'A' in state['reliability'] and 'B' in state['reliability']
    rel_a = state['reliability']['A']
    rel_b = state['reliability']['B']
    # A should have higher reliability than B due to fewer flips / better directional consistency
    assert rel_a > rel_b
    # Instability penalty should reduce effective weight of B relative to its raw reliability
    wA = state['effective_weights']['A']
    wB = state['effective_weights']['B']
    assert wA > wB
    # Reliability time-series should have last date entries for both components
    assert any(r['date'] == idx[-1] and r['component']=='A' for r in reliability_ts)
    assert any(r['date'] == idx[-1] and r['component']=='B' for r in reliability_ts)
