import os
import sys
import copy
import json
import time
import importlib
from dataclasses import asdict
from typing import Dict, Any, List

import pandas as pd

# Ensure project root on path for direct script execution
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# Assumptions: main.py loads config + runs strategies producing diagnostics.
# We simulate A/B parameter variants by:
#   1. Loading base strategy_talib.yaml
#   2. Writing temporary variant YAML overlays (strategy_talib_variant_X.yaml)
#   3. Invoking main entry (import main; main.run_backtest_like()) if such API exists.
# If no programmatic API, instruct user to run manually; here we implement a lightweight
# internal backtest loop using existing strategy class to compute signals & sizing on a single ticker dataset.
# This is a simplified harness focusing on relative comparisons (not full broker simulation).

# Lightweight loader for TALibStrategy and config dataclass
from utils.app_config import TalibStrategyConfig
from strategies.ta_strategy import TALibStrategy

DATA_PATH = os.path.join('data', 'AAPL.csv')  # adjust if needed
OUTPUT_DIR = 'analysis_output'
VARIANTS_FILE = os.path.join('analysis', 'talib_ab_variants.json')

DEFAULT_VARIANTS = [
    {
        "name": "baseline",
        "overrides": {}
    },
    {
        "name": "faster_reliability",
        "overrides": {"reliability_smoothing": 0.30, "warmup_bars_for_reliability": 30}
    },
    {
        "name": "stricter_components",
        "overrides": {"min_components": 3}
    },
    {
        "name": "lower_buy_threshold",
        "overrides": {"buy_score_threshold": 0.55}
    },
    {
        "name": "reduced_drawdown_dampening",
        "overrides": {"drawdown_dampening_threshold": 0.12, "drawdown_dampening_factor": 0.65}
    }
]


def load_price_data() -> pd.DataFrame:
    # Expect CSV with columns including 'Date' or index already set; fallback to 'Close','High','Low'
    # Try popular locations
    candidates = [DATA_PATH, os.path.join('data', 'raw', 'AAPL.csv')]
    for c in candidates:
        if os.path.exists(c):
            df = pd.read_csv(c)
            # Normalize date index
            date_col = None
            for cand in ['Date', 'date', 'timestamp']:
                if cand in df.columns:
                    date_col = cand
                    break
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
            return df
    # Graceful fallback: return empty DF; caller will handle
    return pd.DataFrame()


def merge_config_overrides(base: TalibStrategyConfig, overrides: Dict[str, Any]) -> TalibStrategyConfig:
    new_cfg = copy.deepcopy(base)
    for k, v in overrides.items():
        if hasattr(new_cfg, k):
            setattr(new_cfg, k, v)
    return new_cfg


def evaluate_variant(prices: pd.DataFrame, cfg: TalibStrategyConfig) -> Dict[str, Any]:
    strat = TALibStrategy()
    # Generate signals gradually to simulate per-bar run; accumulate equity using a naive model.
    capital = 100000.0
    position_shares = 0
    equity_curve = []
    # We'll process sequentially to mimic bar-by-bar updates.
    # Construct a rolling window DataFrame for generate_signals
    all_rows = []
    for i in range(len(prices)):
        window = prices.iloc[: i + 1]
        try:
            strat.generate_signals(window, config=type('Tmp', (), {'talib_strategy': cfg}))
        except Exception:
            continue
        # Decide trades only on last signal change event for this bar using strat._changes
        if not strat._diag_rows:
            equity_curve.append(capital)
            continue
        last_diag = strat._diag_rows[-1]
        last_price = last_diag.get('close') or window['Close'].iloc[-1]
        signal = last_diag.get('signal', 0)
        # Entry
        if signal == 1 and position_shares == 0:
            notional = strat.calculate_position_size(window, capital, config=type('Tmp', (), {'talib_strategy': cfg}))
            shares = int(notional / last_price) if last_price else 0
            position_shares = shares
            capital -= shares * last_price  # allocate
        # Partial exit (signal == 2)
        elif signal == 2 and position_shares > 0:
            reduction = int(position_shares * getattr(cfg, 'partial_exit_reduction', 0.5))
            position_shares -= reduction
            capital += reduction * last_price
        # Full exit
        elif signal == -1 and position_shares > 0:
            capital += position_shares * last_price
            position_shares = 0
        # Mark-to-market equity
        equity = capital + position_shares * last_price
        equity_curve.append(equity)
        all_rows.append({
            'date': window.index[-1],
            'equity': equity,
            'signal': signal,
            'confidence': last_diag.get('confidence'),
            'composite_score': last_diag.get('composite_score')
        })
    if not all_rows:
        return {"trades": 0, "final_equity": 100000.0, "cagr": 0.0, "max_drawdown": 0.0, "name": cfg.__dict__.get('name','variant')}    
    eq_df = pd.DataFrame(all_rows).set_index('date')
    # Metrics
    start_val = eq_df['equity'].iloc[0]
    end_val = eq_df['equity'].iloc[-1]
    years = (eq_df.index[-1] - eq_df.index[0]).days / 365.25 if len(eq_df) > 1 else 1
    cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 else 0.0
    roll_max = eq_df['equity'].cummax()
    dd = (eq_df['equity'] / roll_max) - 1
    max_dd = dd.min()
    # Trade count approximation: count BUY signals
    trades = sum(1 for r in eq_df['signal'].values if r == 1)
    return {
        'name': getattr(cfg, 'name', 'variant'),
        'trades': trades,
        'final_equity': end_val,
        'cagr': cagr,
        'max_drawdown': max_dd,
        'equity_start': start_val,
        'equity_end': end_val
    }


def run_variants(variants: List[Dict[str, Any]]):
    prices = load_price_data()
    if prices.empty:
        print("No price data found (expected data/AAPL.csv). Skipping A/B evaluation.")
        return
    base_cfg = TalibStrategyConfig()
    results = []
    for var in variants:
        overrides = var.get('overrides', {})
        cfg = merge_config_overrides(base_cfg, overrides)
        setattr(cfg, 'name', var.get('name', 'variant'))
        res = evaluate_variant(prices, cfg)
        results.append(res)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'talib_ab_results.csv'), index=False)
    print('Wrote talib_ab_results.csv')


def load_variants_from_file():
    if os.path.exists(VARIANTS_FILE):
        with open(VARIANTS_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_VARIANTS


def main():
    variants = load_variants_from_file()
    run_variants(variants)

if __name__ == '__main__':
    main()
