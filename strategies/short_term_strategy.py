import pandas as pd
import numpy as np
from typing import Any, Optional, Dict
from .strategy_interface import TradingStrategy
import talib

class ShortTermTechnicalStrategy(TradingStrategy):
    def __init__(self):
        self.last_rebalance_date = None
        self._score_export_done = False
        self.last_net_score: float = 0.0
        self.last_bull_score: float = 0.0
        self.last_bear_score: float = 0.0
        # Incremental caches (avoid O(N^2) recomputation each Backtrader bar)
        self._cached_len: int = 0
        self._signals_cache: Optional[pd.Series] = None
        self._net_scores = []  # type: list[float]
        self._bull_scores = []  # type: list[float]
        self._bear_scores = []  # type: list[float]
        self._signals_list = []  # type: list[int]
        self._export_components_enabled = False  # type: bool
        self._comp_names = []  # type: list[str]
        self._comp_arrays = None  # type: Optional[Dict[str, list]]
        # Stateful execution (position / streak handling carried across calls)
        self._in_position = False
        self._position_entry_bar = None  # type: Optional[int]
        self._last_exit_bar = None  # type: Optional[int]
        self._recent_entry_streak = 0
        self._recent_exit_streak = 0
        # Adaptive threshold caches
        self._dyn_buy_thr_cache = None  # type: Optional[float]
        self._dyn_sell_thr_cache = None  # type: Optional[float]

    def generate_signals(self, data: pd.DataFrame, config: Any) -> pd.Series:
        """Incremental structured multi-factor signals.

        Optimized to process ONLY newly appended bars each call (Backtrader invokes every bar).
        Maintains internal state for position logic & adaptive thresholds.
        """
        df = next(iter(data.values())) if isinstance(data, dict) else data
        idx = df.index
        cfg = getattr(config, 'strategy', config)
        if not getattr(cfg, 'enable_weighted_signals', True):
            # Return flat series (initialize cache if needed)
            if self._signals_cache is None or not self._signals_cache.index.equals(idx):
                self._signals_cache = pd.Series(0, index=idx)
            return self._signals_cache

        # --- Parameters (static per call) ---
        rsi_window = getattr(cfg, 'rsi_window', 14)
        sma_w = getattr(cfg, 'sma_window', 20)
        sma_long_w = getattr(cfg, 'sma_long_window', 50)
        buy_thr_cfg = float(getattr(cfg, 'buy_score_threshold', 0.50))
        sell_thr_cfg = float(getattr(cfg, 'sell_score_threshold', 0.50))
        hysteresis = getattr(cfg, 'hysteresis_buffer', 0.15)
        rsi_os = getattr(cfg, 'rsi_oversold', 40)
        rsi_ob = getattr(cfg, 'rsi_overbought', 60)
        vwap_scale = float(getattr(cfg, 'vwap_scale', 50.0))
        stoch_os = float(getattr(cfg, 'stoch_oversold', 20.0))
        stoch_ob = float(getattr(cfg, 'stoch_overbought', 80.0))
        export_components = bool(getattr(cfg, 'export_component_arrays', False))
        components_export_path = getattr(cfg, 'component_export_path', 'short_term_components.csv')
        adaptive_enabled = getattr(cfg, 'adaptive_thresholds', False)
        lookback = getattr(cfg, 'adaptive_lookback', 120)
        buy_pct = float(getattr(cfg, 'adaptive_buy_pct', 70.0))
        sell_pct = float(getattr(cfg, 'adaptive_sell_pct', 70.0))
        min_obs = getattr(cfg, 'adaptive_min_obs', 60)
        update_every = int(getattr(cfg, 'adaptive_update_interval', 5))
        min_hold = getattr(cfg, 'min_hold_bars', 0)
        cooldown = getattr(cfg, 'cooldown_bars_after_exit', 0)
        confirm_needed = getattr(cfg, 'entry_confirm_bars', 1)
        exit_confirm_needed = getattr(cfg, 'exit_confirm_bars', 1)
        neutral_band = getattr(cfg, 'neutral_band', 0.0)
        enable_zf = getattr(cfg, 'enable_zscore_filter', False)
        z_lb = getattr(cfg, 'zscore_lookback', 50)
        z_entry = getattr(cfg, 'zscore_entry', 0.0)
        z_exit = getattr(cfg, 'zscore_exit', -999)

        # Ensure cache series aligns with index
        if self._signals_cache is None or not self._signals_cache.index.equals(idx):
            # Rebuild from scratch if index changed unexpectedly (e.g., new session)
            self._signals_cache = pd.Series(0, index=idx)
            self._cached_len = 0
            self._net_scores = []
            self._bull_scores = []
            self._bear_scores = []
            self._signals_list = []
            self._in_position = False
            self._position_entry_bar = None
            self._last_exit_bar = None
            self._recent_entry_streak = 0
            self._recent_exit_streak = 0
            self._dyn_buy_thr_cache = buy_thr_cfg
            self._dyn_sell_thr_cache = sell_thr_cfg
            # Component arrays reset
            self._comp_names = [
                'rsi_mean_reversion','boll_lower','boll_upper','short_above_long','short_below_long',
                'momentum_slope','vwap_trend','stoch_osc','mfi_flow','price_momentum'
            ]
            self._export_components_enabled = export_components
            self._comp_arrays = {n: [] for n in self._comp_names} if export_components else None

        # Slice values
        close = df['Close'].astype(float).values
        high = df['High'].astype(float).values
        low = df['Low'].astype(float).values
        volume = df['Volume'].astype(float).values

        # Recompute indicator arrays (cost O(N); acceptable for now). Further optimization possible later.
        rsi = talib.RSI(close, timeperiod=rsi_window)
        sma = talib.SMA(close, timeperiod=sma_w)
        sma_long = talib.SMA(close, timeperiod=sma_long_w)
        bb_up, bb_mid, bb_low = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=talib.MA_Type.SMA)
        mfi = talib.MFI(high, low, close, volume, timeperiod=14)
        try:
            stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0,
                                           slowd_period=3, slowd_matype=0)
        except Exception:
            stoch_k = np.full(len(close), np.nan)
            stoch_d = np.full(len(close), np.nan)
        typical_price = (high + low + close) / 3.0
        cum_vol = np.cumsum(volume)
        cum_tp_vol = np.cumsum(typical_price * volume)
        with np.errstate(divide='ignore', invalid='ignore'):
            vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)

        # Helper funcs
        def safe(val):
            return 0.0 if val is None or np.isnan(val) else float(val)
        def clamp(x, lo=-1.0, hi=1.0):
            return max(lo, min(hi, x))

        # Weights (merged with config overrides once per call)
        default_weights_bull = {
            'rsi_mean_reversion': 0.18,'boll_lower': 0.18,'short_above_long': 0.12,'momentum_slope': 0.12,
            'vwap_trend': 0.10,'stoch_osc': 0.05,'mfi_flow': 0.10,'price_momentum': 0.15,
        }
        default_weights_bear = {
            'rsi_mean_reversion': 0.18,'boll_upper': 0.18,'short_below_long': 0.12,'momentum_slope': 0.12,
            'vwap_trend': 0.10,'stoch_osc': 0.05,'mfi_flow': 0.10,'price_momentum': 0.15,
        }
        weights_bull = default_weights_bull.copy()
        weights_bear = default_weights_bear.copy()
        cfg_w_bull = getattr(cfg, 'weights_bull', None)
        if isinstance(cfg_w_bull, dict):
            for k, v in cfg_w_bull.items():
                if k in weights_bull:
                    try: weights_bull[k] = float(v)
                    except Exception: pass
        cfg_w_bear = getattr(cfg, 'weights_bear', None)
        if isinstance(cfg_w_bear, dict):
            for k, v in cfg_w_bear.items():
                if k in weights_bear:
                    try: weights_bear[k] = float(v)
                    except Exception: pass

        warmup = max(sma_long_w, 20, rsi_window)
        start_i = self._cached_len  # first unprocessed bar
        n = len(df)
        if start_i < 0: start_i = 0
        # Extend score arrays if needed
        while len(self._net_scores) < n:
            self._net_scores.append(0.0)
            self._bull_scores.append(0.0)
            self._bear_scores.append(0.0)
            if self._comp_arrays is not None:
                for name in self._comp_arrays:
                    self._comp_arrays[name].append(np.nan)

        # Precompute zscores only if enabling and new bars added (simple full recompute)
        if enable_zf:
            net_series_full = pd.Series(self._net_scores[:n])
            zscores = (net_series_full - net_series_full.rolling(z_lb, min_periods=10).mean()) / net_series_full.rolling(z_lb, min_periods=10).std(ddof=0)
        else:
            zscores = pd.Series([0]*n)

        for i in range(start_i, n):
            if (i + 1) < warmup:
                # Ensure signals_list padded
                if i >= len(self._signals_list):
                    self._signals_list.append(0)
                continue  # skip scoring until warmup
            comps_bull: Dict[str, float] = {}
            comps_bear: Dict[str, float] = {}
            rsi_val = safe(rsi[i])
            if rsi_val < rsi_os:
                comps_bull['rsi_mean_reversion'] = clamp((rsi_os - rsi_val)/max(1, rsi_os),0,1)
            elif rsi_val > rsi_ob:
                comps_bear['rsi_mean_reversion'] = clamp((rsi_val - rsi_ob)/max(1, 100-rsi_ob),0,1)
            if not np.isnan(bb_low[i]) and not np.isnan(bb_up[i]):
                band_h = max(1e-9, bb_up[i]-bb_low[i])
                if close[i] < bb_low[i]:
                    comps_bull['boll_lower'] = clamp((bb_low[i]-close[i])/band_h,0,1)
                elif close[i] > bb_up[i]:
                    comps_bear['boll_upper'] = clamp((close[i]-bb_up[i])/band_h,0,1)
            if not np.isnan(sma[i]) and not np.isnan(sma_long[i]):
                if sma[i] > sma_long[i]:
                    comps_bull['short_above_long'] = clamp((sma[i]-sma_long[i]) / max(1e-9, sma_long[i]),0,1)
                else:
                    comps_bear['short_below_long'] = clamp((sma_long[i]-sma[i]) / max(1e-9, sma_long[i]),0,1)
            if i>0 and not np.isnan(sma[i]) and not np.isnan(sma[i-1]):
                slope = (sma[i]-sma[i-1]) / max(1e-9, sma[i-1])
                if slope>0: comps_bull['momentum_slope'] = clamp(slope*50,0,1)
                elif slope<0: comps_bear['momentum_slope'] = clamp(-slope*50,0,1)
            if not np.isnan(vwap[i]) and vwap[i]>0:
                vdiff = (close[i]-vwap[i]) / vwap[i]
                if vdiff>0: comps_bull['vwap_trend'] = clamp(vdiff*vwap_scale,0,1)
                elif vdiff<0: comps_bear['vwap_trend'] = clamp(-vdiff*vwap_scale,0,1)
            k_val = safe(stoch_k[i])
            if k_val < stoch_os:
                comps_bull['stoch_osc'] = clamp((stoch_os-k_val)/max(1,stoch_os),0,1)
            elif k_val > stoch_ob:
                comps_bear['stoch_osc'] = clamp((k_val-stoch_ob)/max(1,100-stoch_ob),0,1)
            mfi_val = safe(mfi[i])
            if mfi_val < 30:
                comps_bull['mfi_flow'] = clamp((30-mfi_val)/30,0,1)
            elif mfi_val > 70:
                comps_bear['mfi_flow'] = clamp((mfi_val-70)/30,0,1)
            if i>0:
                price_chg = (close[i]-close[i-1]) / max(1e-9, close[i-1])
                if price_chg>0: comps_bull['price_momentum'] = clamp(price_chg*200,0,1)
                elif price_chg<0: comps_bear['price_momentum'] = clamp(-price_chg*200,0,1)

            if self._comp_arrays is not None:
                for name, val in comps_bull.items():
                    if name in self._comp_arrays: self._comp_arrays[name][i] = val
                for name, val in comps_bear.items():
                    if name in self._comp_arrays:
                        existing = self._comp_arrays[name][i]
                        self._comp_arrays[name][i] = val if (np.isnan(existing) or existing is None) else max(existing, val)

            bull_score = sum(comps_bull.get(k,0.0)*weights_bull.get(k,0.0) for k in comps_bull.keys())
            bear_score = sum(comps_bear.get(k,0.0)*weights_bear.get(k,0.0) for k in comps_bear.keys())
            net = bull_score - bear_score
            self._net_scores[i] = net
            self._bull_scores[i] = bull_score
            self._bear_scores[i] = bear_score

            # Adaptive threshold update (periodic)
            if adaptive_enabled and (i % update_every == 0):
                start_idx = max(0, i - lookback + 1)
                window = np.array(self._net_scores[start_idx:i+1])
                valid = window[~np.isnan(window)]
                if len(valid) >= min_obs:
                    self._dyn_buy_thr_cache = float(np.percentile(valid, buy_pct))
                    neg_vals = valid[valid < 0]
                    if len(neg_vals) >= max(5, min_obs//4):
                        self._dyn_sell_thr_cache = float(abs(np.percentile(neg_vals, 100 - sell_pct)))
                    else:
                        self._dyn_sell_thr_cache = sell_thr_cfg
            dyn_buy_thr = self._dyn_buy_thr_cache if (adaptive_enabled and self._dyn_buy_thr_cache is not None) else buy_thr_cfg
            dyn_sell_thr = self._dyn_sell_thr_cache if (adaptive_enabled and self._dyn_sell_thr_cache is not None) else sell_thr_cfg

            # Signal decision (stateful) with local variable then append
            sig = 0
            if not self._in_position:
                blocked = (self._last_exit_bar is not None and (i - self._last_exit_bar) < cooldown)
                if blocked or abs(net) < neutral_band or not (bull_score > bear_score and bull_score > 0) or (enable_zf and zscores.iloc[i] < z_entry):
                    self._recent_entry_streak = 0
                else:
                    if net >= dyn_buy_thr:
                        self._recent_entry_streak += 1
                    else:
                        self._recent_entry_streak = 0
                    if self._recent_entry_streak >= confirm_needed:
                        sig = 1
                        self._in_position = True
                        self._position_entry_bar = i
                        self._recent_entry_streak = 0
            else:
                hold_ok = (self._position_entry_bar is None) or ((i - self._position_entry_bar) >= min_hold)
                exit_condition = (net <= (dyn_buy_thr - hysteresis) or net <= -dyn_sell_thr)
                if enable_zf and zscores.iloc[i] <= z_exit:
                    exit_condition = True
                if hold_ok and exit_condition:
                    self._recent_exit_streak += 1
                else:
                    self._recent_exit_streak = 0
                if hold_ok and self._recent_exit_streak >= exit_confirm_needed:
                    sig = -1
                    self._in_position = False
                    self._last_exit_bar = i
                    self._position_entry_bar = None
                    self._recent_exit_streak = 0
            # store signal
            if i == len(self._signals_list):
                self._signals_list.append(sig)
            else:
                # should be equal length; but guard
                if i < len(self._signals_list):
                    self._signals_list[i] = sig
                else:
                    # fill gaps
                    while len(self._signals_list) < i:
                        self._signals_list.append(0)
                    self._signals_list.append(sig)

        # Update last scores for external logging
        if n:
            self.last_net_score = float(self._net_scores[n-1])
            self.last_bull_score = float(self._bull_scores[n-1])
            self.last_bear_score = float(self._bear_scores[n-1])
            if getattr(cfg, 'debug_last_bar_components', False):
                self.last_components = {
                    'net_score': self.last_net_score,
                    'bull_score': self.last_bull_score,
                    'bear_score': self.last_bear_score,
                }

        # One-time export after warmup
        if (not self._score_export_done) and n >= max(sma_long_w, 50):
            try:
                export_df = pd.DataFrame({
                    'bull_score': self._bull_scores[:n],
                    'bear_score': self._bear_scores[:n],
                    'net_score': self._net_scores[:n],
                }, index=idx)
                export_path = 'short_term_scores.csv'
                export_df.to_csv(export_path)
                desc = export_df['net_score'].describe()
                with open(export_path, 'a') as f:
                    f.write('\n# net_score_distribution\n')
                    for k, v in desc.to_dict().items():
                        f.write(f"# {k},{v}\n")
                if self._export_components_enabled and self._comp_arrays is not None:
                    comp_df = pd.DataFrame(self._comp_arrays, index=idx)
                    comp_df.to_csv(components_export_path)
                self._score_export_done = True
            except Exception:
                pass

        # Rebuild / update signals series
        if len(self._signals_cache) != n:
            self._signals_cache = pd.Series(self._signals_list, index=idx)
        else:
            # update last value only (fast path)
            try:
                if n and self._signals_cache.iloc[-1] != self._signals_list[-1]:
                    self._signals_cache.iloc[-1] = self._signals_list[-1]
            except Exception:
                self._signals_cache = pd.Series(self._signals_list, index=idx)

        self._cached_len = n
        return self._signals_cache
        
    def calculate_position_size(self, data: pd.DataFrame, capital: float, config: Any) -> float:
        """Calculate position size based on volatility and risk parameters"""
        if isinstance(data, dict):
            df = next(iter(data.values()))
        else:
            df = data
        
        # Get risk parameters from config
        cfg = getattr(config, 'strategy', config)
        risk_per_trade = getattr(cfg, 'risk_per_trade', 0.02)
        max_position = getattr(cfg, 'max_position_size', 0.10)
        max_fraction = float(getattr(cfg, 'max_position_fraction', max_position))
        use_frac = bool(getattr(cfg, 'use_fraction_cap', True))
        # adapt to attribute names present in config
        stop_loss_atr = getattr(cfg, 'atr_stop_multiplier', getattr(cfg, 'stop_loss_atr_multiplier', 2.0))
        
        # Calculate ATR
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        atr_period = getattr(cfg, 'atr_period', getattr(cfg, 'atr_window', 14))
        atr = talib.ATR(high, low, close, timeperiod=atr_period)[-1]
        
        if pd.isna(atr):
            atr = (high[-1] - low[-1]) * 0.1  # Fallback if ATR is not available
        
        # Risk-based position sizing
        risk_amount = capital * risk_per_trade
        position_size = risk_amount / (atr * stop_loss_atr)
        
        # Apply maximum position size limit
        cap_limit = capital * (max_fraction if use_frac else max_position)
        return min(position_size, cap_limit)
        
    def get_stop_loss(self, data: pd.DataFrame, entry_price: float, position_type: str, config: Any) -> float:
        """Calculate stop loss level based on ATR"""
        if isinstance(data, dict):
            df = next(iter(data.values()))
        else:
            df = data
        cfg = getattr(config, 'strategy', config)
        stop_loss_atr = getattr(cfg, 'atr_stop_multiplier', getattr(cfg, 'stop_loss_atr_multiplier', 2.0))
        atr = talib.ATR(
            df['High'].values,
            df['Low'].values,
            df['Close'].values,
            timeperiod=14
        )[-1]
        
        if pd.isna(atr):
            atr = (df['High'].iloc[-1] - df['Low'].iloc[-1]) * 0.1  # Fallback if ATR is not available
            
        if position_type == 'long':
            return entry_price - (atr * stop_loss_atr)
        return entry_price + (atr * stop_loss_atr)

    def should_rebalance(self, current_date: pd.Timestamp, last_rebalance: Optional[pd.Timestamp], config: Any) -> bool:
        """Check if portfolio should be rebalanced (daily for short-term)"""
        if last_rebalance is None:
            return True
        return current_date.date() != last_rebalance.date()