import pandas as pd
import numpy as np
import os
import talib
from typing import Any, Optional
from .strategy_interface import TradingStrategy


class LongTermValueStrategy(TradingStrategy):
    """Pure technical long-term strategy (fundamentals removed).

    Diagnostics: counts golden cross events and reasons they fail confirmations to help
    understand absence of orders. Access via get_diagnostics().
    """

    def __init__(self):
        self._diag = {
            'golden_cross_total': 0,
            'golden_cross_confirmed': 0,
            'blocked_aroon': 0,
            'blocked_obv': 0,
            'blocked_volume': 0
        }
        self._gc_rows = []  # store per golden cross diagnostics windows
        # Defaults (can be overridden by config each call to generate_signals)
        self.confirm_window = 3  # bars after cross to allow confirmation
        self.aroon_up_min = 60   # relaxed thresholds
        self.aroon_down_max = 60

    # ---------------- Indicator Calculations ----------------
    @staticmethod
    def _calc_sma(df: pd.DataFrame, window: int, col: str):
        df[col] = talib.SMA(df['Close'].values, timeperiod=window)

    @staticmethod
    def _calc_aroon(df: pd.DataFrame, period: int = 25):
        a_down, a_up = talib.AROON(df['High'].values, df['Low'].values, timeperiod=period)
        df['aroon_up'] = a_up
        df['aroon_down'] = a_down
        # AROONOSC gives up - down, but we can compute ourselves
        df['aroon_osc'] = df['aroon_up'] - df['aroon_down']

    @staticmethod
    def _calc_obv(df: pd.DataFrame):
        df['obv'] = talib.OBV(df['Close'].values, df['Volume'].values)
        df['obv_sma_20'] = pd.Series(df['obv']).rolling(20).mean().values

    @staticmethod
    def _calc_fib_levels(df: pd.DataFrame, lookback: int = 120):
        if len(df) < lookback:
            lookback = len(df)
        if lookback < 5:
            return
        recent = df.iloc[-lookback:]
        swing_high = recent['High'].max()
        swing_low = recent['Low'].min()
        if swing_high == swing_low:
            return
        diff = swing_high - swing_low
        levels = {
            'fib_236': swing_high - 0.236 * diff,
            'fib_382': swing_high - 0.382 * diff,
            'fib_500': swing_high - 0.500 * diff,
            'fib_618': swing_high - 0.618 * diff,
            'fib_786': swing_high - 0.786 * diff,
        }
        for k, v in levels.items():
            df[k] = v
        df['swing_high'] = swing_high
        df['swing_low'] = swing_low

    # ---------------- Core Signals ----------------
    def generate_signals(self, data: pd.DataFrame, config: Any) -> pd.Series:
        df_in = next(iter(data.values())) if isinstance(data, dict) else data
        ticker = getattr(df_in, 'attrs', {}).get('ticker', 'UNKNOWN')
        df = df_in.copy()
        if df.empty:
            return pd.Series(0, index=df.index)
        if len(df) < 200:
            return pd.Series(0, index=df.index)
        # Pull externalized thresholds if present on config
        cfg = getattr(config, 'strategy', getattr(config, 'long_term_strategy', config))
        self.confirm_window = int(getattr(cfg, 'confirm_window', self.confirm_window))
        self.aroon_up_min = int(getattr(cfg, 'aroon_up_min', self.aroon_up_min))
        self.aroon_down_max = int(getattr(cfg, 'aroon_down_max', self.aroon_down_max))

        # Indicators (TA-Lib based)
        self._calc_sma(df, 50, 'sma_50')
        self._calc_sma(df, 100, 'sma_100')
        self._calc_sma(df, 200, 'sma_200')
        self._calc_aroon(df, 25)
        self._calc_obv(df)
        df['vol_ma_50'] = df['Volume'].rolling(50).mean()
        self._calc_fib_levels(df)

        df['golden_cross'] = (df['sma_50'].shift(1) < df['sma_200'].shift(1)) & (df['sma_50'] > df['sma_200'])
        df['death_cross'] = (df['sma_50'].shift(1) > df['sma_200'].shift(1)) & (df['sma_50'] < df['sma_200'])
        df['vol_ok'] = df['Volume'] >= (0.9 * df['vol_ma_50'])

        signals = pd.Series(0, index=df.index)

        # Pre-capture golden cross indices
        gc_indices = list(np.where(df['golden_cross'])[0])
        for i in gc_indices:
            if i == 0:
                continue
            self._diag['golden_cross_total'] += 1
            window_end = min(i + self.confirm_window, len(df) - 1)
            confirmed = False
            block_reasons = []
            chosen_j = None
            for j in range(i, window_end + 1):
                a_up = df['aroon_up'].iloc[j]
                a_dn = df['aroon_down'].iloc[j]
                obv_val = df['obv'].iloc[j]
                obv_ma = df['obv_sma_20'].iloc[j]
                vol_cur = df['Volume'].iloc[j]
                vol_ma = df['vol_ma_50'].iloc[j]
                aroon_ok = (a_up > self.aroon_up_min) and (a_dn < self.aroon_down_max)
                obv_ok = obv_val > obv_ma if not pd.isna(obv_ma) else False
                vol_ok = bool(df['vol_ok'].iloc[j])
                if aroon_ok and obv_ok and vol_ok:
                    signals.iloc[j] = 1
                    self._diag['golden_cross_confirmed'] += 1
                    confirmed = True
                    chosen_j = j
                    if os.getenv('LONGTERM_DEBUG', '').lower() in ('1','true','yes'):
                        print(f"[LONGTERM_DEBUG] {ticker} BUY confirm_j={j} base_i={i} date={df.index[j].date()} aroon_up={a_up:.1f} aroon_down={a_dn:.1f} obv_ok={obv_ok} vol_ok={vol_ok} vol_mult={(vol_cur/(vol_ma+1e-9)):.2f}")
                    break
                else:
                    if not aroon_ok:
                        self._diag['blocked_aroon'] += 1
                    if not obv_ok:
                        self._diag['blocked_obv'] += 1
                    if not vol_ok:
                        self._diag['blocked_volume'] += 1
                    block_reasons.append({
                        'j': j,
                        'date': df.index[j],
                        'aroon_up': a_up,
                        'aroon_down': a_dn,
                        'obv': obv_val,
                        'obv_sma_20': obv_ma,
                        'vol': vol_cur,
                        'vol_ma_50': vol_ma,
                        'aroon_ok': aroon_ok,
                        'obv_ok': obv_ok,
                        'vol_ok': vol_ok
                    })
            # Store diagnostic window rows (include a slice around the cross)
            start_slice = max(i - 5, 0)
            end_slice = min(window_end + 2, len(df))
            window_df = df.iloc[start_slice:end_slice].copy()
            window_df['gc_base_index'] = i
            window_df['gc_confirmed'] = confirmed
            window_df['gc_chosen_index'] = chosen_j if confirmed else np.nan
            self._gc_rows.append(window_df.assign(_ticker=ticker))
            if not confirmed and os.getenv('LONGTERM_DEBUG', '').lower() in ('1','true','yes'):
                print(f"[LONGTERM_DEBUG] {ticker} GC UNCONFIRMED base_i={i} date={df.index[i].date()} attempts={len(block_reasons)}")

        # Exit logic loop (independent of entry placement)
        for i in range(1, len(df)):
            death = bool(df['death_cross'].iloc[i])
            aroon_bear = (df['aroon_down'].iloc[i] > 70) and (df['aroon_up'].iloc[i] < 50)
            obv_break = (
                df['obv'].iloc[i] < df['obv_sma_20'].iloc[i] and
                df['obv'].iloc[i-1] >= df['obv_sma_20'].iloc[i-1]
            ) if not pd.isna(df['obv_sma_20'].iloc[i]) and not pd.isna(df['obv_sma_20'].iloc[i-1]) else False
            if death or aroon_bear or obv_break:
                signals.iloc[i] = -1
                if os.getenv('LONGTERM_DEBUG', '').lower() in ('1','true','yes'):
                    reason = 'death_cross' if death else 'aroon_bear' if aroon_bear else 'obv_break'
                    print(f"[LONGTERM_DEBUG] {ticker} SELL i={i} date={df.index[i].date()} reason={reason} aroon_up={df['aroon_up'].iloc[i]:.1f} aroon_down={df['aroon_down'].iloc[i]:.1f}")

        return signals

    def get_diagnostics(self) -> dict:
        return dict(self._diag)

    def export_gc_diagnostics(self, output_dir: str = 'analysis_output'):
        if not self._gc_rows:
            return None
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, 'golden_cross_diagnostics.csv')
        all_df = pd.concat(self._gc_rows)
        all_df.to_csv(out_path, index=True)
        return out_path

    # ---------------- Portfolio Helpers ----------------
    def calculate_position_size(self, data: pd.DataFrame, capital: float, config: Any) -> float:
        """Position sizing honoring new max_position_fraction (alias) or legacy max_position_size.

        If use_fraction_cap is True and max_position_fraction provided, allocate that fraction
        of capital. Otherwise fall back to max_position_size behavior.
        """
        cfg = getattr(config, 'strategy', getattr(config, 'long_term_strategy', config))
        frac_cap = float(getattr(cfg, 'max_position_fraction', getattr(cfg, 'max_position_size', 0.15)))
        use_frac = bool(getattr(cfg, 'use_fraction_cap', True))
        if use_frac and frac_cap > 0:
            return capital * min(frac_cap, 1.0)
        # Fallback legacy path
        max_pos = getattr(cfg, 'max_position_size', 0.15)
        return capital * max_pos

    def get_stop_loss(self, data: pd.DataFrame, entry_price: float, position_type: str, config: Any) -> float:
        stop_loss_pct = 0.10
        if position_type == 'long':
            return entry_price * (1 - stop_loss_pct)
        return entry_price * (1 + stop_loss_pct)

    def should_rebalance(self, current_date: pd.Timestamp, last_rebalance: Optional[pd.Timestamp], config: Any) -> bool:
        if last_rebalance is None:
            return True
        return (current_date.year != last_rebalance.year or current_date.quarter != last_rebalance.quarter)