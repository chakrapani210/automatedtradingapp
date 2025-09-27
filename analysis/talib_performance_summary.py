import os
import math
import pandas as pd
from datetime import datetime

"""Generate trade-level and aggregate performance summary for TALib strategy.

Inputs:
  - orders_diagnostics.csv (root) or analysis_output/orders_diagnostics.csv
  - equity_curve.csv (optional) for CAGR / max DD
  - talib_strategy_signals.csv for holding period augmentation (optional)
Outputs:
  - analysis_output/talib_performance_summary.csv (aggregate metrics key,value)
  - analysis_output/talib_trades_detailed.csv (per-trade stats)

Trade Detection Logic:
  - A BUY row with positive quantity establishes/adjusts a position; track cumulative position.
  - A SELL row reducing position to zero closes a trade; partial exits recorded but not closing.
  - SIGNAL_* rows ignored for PnL, used only to mark intent.
Assumptions:
  - quantity column reflects signed share change when action in {BUY, SELL}.
  - 'value' column in orders file expresses signed notional (we recompute for safety).
"""

def load_orders():
    candidates = [
        'orders_diagnostics.csv',
        os.path.join('analysis_output', 'orders_diagnostics.csv')
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError('orders_diagnostics.csv not found in root or analysis_output/.')


def detect_trades(df: pd.DataFrame):
    # Filter actionable rows
    act_df = df[df['action'].isin(['BUY', 'SELL'])].copy()
    if act_df.empty:
        return []
    trades = []
    position = 0.0
    trade = None
    for _, row in act_df.iterrows():
        qty = row.get('quantity')
        if pd.isna(qty):
            # If not provided, attempt to infer from order_ref sequence (skip)
            continue
        qty = float(qty)
        price = float(row.get('price', math.nan))
        timestamp = row.get('timestamp')
        if isinstance(timestamp, str):
            try:
                ts = pd.to_datetime(timestamp)
            except Exception:
                ts = None
        else:
            ts = timestamp

        # BUY increases position
        if row['action'] == 'BUY':
            if position == 0:
                # new trade
                trade = {
                    'entry_time': ts,
                    'entry_price': price,
                    'entry_confidence': row.get('confidence'),
                    'shares': qty,
                    'cost_basis': price,
                    'adds': 0,
                    'scale_in_value': 0.0
                }
            else:
                # scale in
                if trade:
                    total_shares = trade['shares'] + qty
                    if total_shares > 0:
                        trade['cost_basis'] = (trade['cost_basis'] * trade['shares'] + price * qty) / total_shares
                    trade['shares'] = total_shares
                    trade['adds'] += 1
                    trade['scale_in_value'] += price * qty
            position += qty
        elif row['action'] == 'SELL':
            position += qty  # qty negative
            if position <= 0 and trade:
                # trade closed
                exit_price = price
                ret_pct = (exit_price / trade['cost_basis']) - 1.0 if trade['cost_basis'] else math.nan
                holding_days = (ts - trade['entry_time']).days if (ts and trade['entry_time']) else math.nan
                trades.append({
                    **trade,
                    'exit_time': ts,
                    'exit_price': exit_price,
                    'return_pct': ret_pct,
                    'holding_days': holding_days,
                })
                trade = None
                position = 0.0
    return trades


def aggregate_metrics(trades):
    if not trades:
        return {
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_return_pct': 0.0,
            'median_return_pct': 0.0,
            'avg_holding_days': 0.0,
            'expectancy_pct': 0.0,
            'avg_winner_pct': 0.0,
            'avg_loser_pct': 0.0
        }
    df = pd.DataFrame(trades)
    wins = df[df['return_pct'] > 0]
    losses = df[df['return_pct'] <= 0]
    win_rate = len(wins) / len(df) if len(df) else 0.0
    avg_ret = df['return_pct'].mean()
    median_ret = df['return_pct'].median()
    avg_hold = df['holding_days'].mean()
    avg_win = wins['return_pct'].mean() if not wins.empty else 0.0
    avg_loss = losses['return_pct'].mean() if not losses.empty else 0.0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    return {
        'num_trades': len(df),
        'win_rate': win_rate,
        'avg_return_pct': avg_ret,
        'median_return_pct': median_ret,
        'avg_holding_days': avg_hold,
        'avg_winner_pct': avg_win,
        'avg_loser_pct': avg_loss,
        'expectancy_pct': expectancy
    }


def main():
    os.makedirs('analysis_output', exist_ok=True)
    orders = load_orders()
    trades = detect_trades(orders)
    metrics = aggregate_metrics(trades)
    # Export detailed trades
    if trades:
        pd.DataFrame(trades).to_csv('analysis_output/talib_trades_detailed.csv', index=False)
    # Export metrics key-value
    pd.DataFrame([metrics]).to_csv('analysis_output/talib_performance_summary.csv', index=False)
    print('Generated talib_performance_summary.csv and talib_trades_detailed.csv')

if __name__ == '__main__':
    main()
