import talib
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from utils.strategy_utils import (
    dynamic_reliability_weighting,
    scale_risk_fraction,
    compute_realized_vol,
    compute_high_priority_override_series,
    compute_gated_buy,
    compute_composite_sell,
    apply_drawdown_dampening,
    apply_volatility_gate,
    compute_confidence
)

class DayTradingStrategy(BaseStrategy):
    def __init__(self):
        # Diagnostics storage
        self._diag_rows = []
        self._changes = []
        self._last_signal = None
        self._entry_price = None
        self._perf_trades = []
        self.last_net_score = 0.0
        self.last_bull_score = 0.0
        self.last_bear_score = 0.0
        self._peak_composite = None
        self._last_effective_risk_frac = None
        self._last_realized_vol = None
        # Reliability tracking
        self._component_reliability = {}
        self._last_bar_close = None
        self._component_flips = {}
        self._peak_confidence = None
        self._component_effective_weights = {}
        self._peak_price = None
        self._reliability_ts = []

    def generate_signals(self, data, config=None):
        if config is None:
            raise ValueError("Config must be provided")
        strat_config = getattr(config, 'day_trading_strategy', None) or getattr(config, 'strategy', None)
        if strat_config is None:
            raise ValueError("DayTradingStrategy config not found")

        # Config params
        rsi_window = getattr(strat_config, 'rsi_window', 14)
        rsi_oversold = getattr(strat_config, 'rsi_oversold', 30)
        rsi_overbought = getattr(strat_config, 'rsi_overbought', 70)
        macd_fast = getattr(strat_config, 'macd_fast', 12)
        macd_slow = getattr(strat_config, 'macd_slow', 26)
        macd_signal = getattr(strat_config, 'macd_signal', 9)
        bb_period = getattr(strat_config, 'bb_period', 20)
        bb_dev = getattr(strat_config, 'bb_dev', 2.0)
        ema_period = getattr(strat_config, 'ema_period', 20)
        volume_ma_window = getattr(strat_config, 'volume_ma_window', 20)
        enable_composite = getattr(strat_config, 'enable_composite', True)
        buy_score_threshold = getattr(strat_config, 'buy_score_threshold', 0.60)
        sell_score_threshold = getattr(strat_config, 'sell_score_threshold', 0.40)
        min_components = getattr(strat_config, 'min_components', 2)
        enable_rsi_component = getattr(strat_config, 'enable_rsi_component', True)
        enable_macd_component = getattr(strat_config, 'enable_macd_component', True)
        enable_bb_component = getattr(strat_config, 'enable_bb_component', True)
        enable_ema_component = getattr(strat_config, 'enable_ema_component', True)
        enable_volume_component = getattr(strat_config, 'enable_volume_component', True)

        if isinstance(data.columns, pd.MultiIndex):
            close = data[('Close',)] if ('Close',) in data.columns else data[(data.columns[0][0], 'Close')]
            high = data[('High',)] if ('High',) in data.columns else data[(data.columns[0][0], 'High')]
            low = data[('Low',)] if ('Low',) in data.columns else data[(data.columns[0][0], 'Low')]
            volume = data[('Volume',)] if ('Volume',) in data.columns else data[(data.columns[0][0], 'Volume')]
        else:
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']

        close_np = close.to_numpy()
        high_np = high.to_numpy()
        low_np = low.to_numpy()
        volume_np = volume.to_numpy()

        # Indicators
        rsi = talib.RSI(close_np, timeperiod=rsi_window)
        macd, macdsignal, macdhist = talib.MACD(close_np, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
        upperband, middleband, lowerband = talib.BBANDS(close_np, timeperiod=bb_period, nbdevup=bb_dev, nbdevdn=bb_dev, matype=0)
        ema = talib.EMA(close_np, timeperiod=ema_period)
        volume_ma = talib.SMA(volume_np, timeperiod=volume_ma_window)

        # Series
        close_series = pd.Series(close_np, index=close.index)
        rsi_series = pd.Series(rsi, index=close.index)
        macd_series = pd.Series(macd, index=close.index)
        macdsignal_series = pd.Series(macdsignal, index=close.index)
        upperband_series = pd.Series(upperband, index=close.index)
        lowerband_series = pd.Series(lowerband, index=close.index)
        ema_series = pd.Series(ema, index=close.index)
        volume_series = pd.Series(volume_np, index=close.index)
        volume_ma_series = pd.Series(volume_ma, index=close.index)

        signals = pd.Series(0, index=close.index)

        # Components with strength-based scoring
        comp_dict = {}
        if enable_rsi_component:
            # RSI strength: normalized distance from neutral (50)
            # Values > 0.5 favor buying, < 0.5 favor selling
            rsi_neutral = 50.0
            rsi_strength = 1.0 - abs(rsi_series - rsi_neutral) / rsi_neutral
            comp_dict['rsi'] = rsi_strength
            
        if enable_macd_component:
            # MACD strength: normalized difference from signal line
            macd_diff = (macd_series - macdsignal_series)
            # Normalize by recent volatility (use rolling std of MACD diff)
            macd_volatility = macd_diff.rolling(window=min(20, len(macd_diff))).std()
            macd_volatility = macd_volatility.replace(0, macd_volatility.mean() or 1.0)
            macd_strength = (macd_diff / macd_volatility).clip(-3, 3) / 6.0 + 0.5  # Normalize to 0-1
            comp_dict['macd'] = macd_strength
            
        if enable_bb_component:
            # Bollinger Band strength: position within bands
            # 0 = at lower band, 0.5 = at middle, 1 = at upper band
            band_width = upperband_series - lowerband_series
            band_width = band_width.replace(0, band_width.mean() or 1.0)
            bb_position = (close_series - lowerband_series) / band_width
            comp_dict['bb'] = bb_position
            
        if enable_ema_component:
            # EMA strength: percentage distance from EMA
            ema_diff_pct = (close_series - ema_series) / ema_series
            # Normalize using rolling volatility
            ema_volatility = ema_diff_pct.rolling(window=min(20, len(ema_diff_pct))).std()
            ema_volatility = ema_volatility.replace(0, ema_volatility.mean() or 0.02)
            ema_strength = (ema_diff_pct / ema_volatility).clip(-3, 3) / 6.0 + 0.5  # Normalize to 0-1
            comp_dict['ema'] = ema_strength
            
        if enable_volume_component:
            # Volume strength: ratio above/below MA
            volume_ratio = volume_series / volume_ma_series
            # Normalize using rolling volatility of volume ratio
            vol_ratio_volatility = volume_ratio.rolling(window=min(20, len(volume_ratio))).std()
            vol_ratio_volatility = vol_ratio_volatility.replace(0, vol_ratio_volatility.mean() or 0.5)
            volume_strength = (volume_ratio - 1.0) / vol_ratio_volatility
            volume_strength = volume_strength.clip(-3, 3) / 6.0 + 0.5  # Normalize to 0-1
            comp_dict['volume'] = volume_strength
        components_df = pd.DataFrame(comp_dict) if comp_dict else pd.DataFrame(index=close_series.index)
        
        # Calculate bullish components based on strength thresholds
        # Components with strength > 0.6 are considered bullish, < 0.4 bearish
        bull_threshold = 0.6
        bear_threshold = 0.4
        bullish_components = components_df > bull_threshold
        bearish_components = components_df < bear_threshold
        
        bull_counts = bullish_components.sum(axis=1) if not components_df.empty else pd.Series(0, index=close_series.index)
        bear_counts = bearish_components.sum(axis=1) if not components_df.empty else pd.Series(0, index=close_series.index)
        total_components = components_df.shape[1] if not components_df.empty else 0

        # Raw composite
        comp_weights_cfg = getattr(strat_config, 'component_weights', None) or {}
        if total_components > 0 and comp_weights_cfg:
            w = {k: float(comp_weights_cfg.get(k, 1.0)) for k in components_df.columns}
            s = sum(w.values()) or 1.0
            norm_w = {k: v / s for k, v in w.items()}
            weighted = components_df.astype(float).mul(pd.Series(norm_w))
            composite_raw = weighted.sum(axis=1)
        else:
            composite_raw = (bull_counts / total_components).fillna(0.0) if total_components > 0 else pd.Series(0.0, index=close_series.index)

        # Reliability weighting
        if total_components > 0:
            composite_score = dynamic_reliability_weighting(
                state={
                    'reliability': self._component_reliability,
                    'flips': self._component_flips,
                    'effective_weights': self._component_effective_weights,
                    'last_close': self._last_bar_close
                },
                components_df=components_df,
                composite_raw=composite_raw,
                close_series=close_series,
                strat_config=strat_config,
                reliability_ts=self._reliability_ts
            )
            self._component_effective_weights = getattr(self, '_component_effective_weights', {}) or {}
            self._last_bar_close = float(close_series.iloc[-1]) if len(close_series) else None
        else:
            composite_score = composite_raw

        # Track peak
        if enable_composite and total_components > 0:
            latest_comp = composite_score.iloc[-1]
            if self._peak_composite is None or latest_comp > self._peak_composite:
                self._peak_composite = latest_comp

        # Bear signals based on traditional overbought conditions and bearish component strength
        traditional_bear = (rsi_series > rsi_overbought) | (macd_series < macdsignal_series) | (close_series > upperband_series)
        # Also consider when most components are strongly bearish
        strength_based_bear = bear_counts >= max(2, total_components * 0.6)  # 60%+ components bearish
        bear = traditional_bear | strength_based_bear

        if enable_composite and total_components > 0:
            hp = getattr(strat_config, 'high_priority_components', None) or []
            override_thr = getattr(strat_config, 'override_threshold', buy_score_threshold)
            # Convert strength values to boolean for high priority override calculation
            binary_components_df = components_df > 0.5  # Strength > 0.5 considered bullish
            override_buy = compute_high_priority_override_series(binary_components_df, hp, override_thr) if hp else pd.Series(False, index=close_series.index)
            gated_buy = compute_gated_buy(
                composite_score=composite_score,
                override_buy=override_buy,
                bull_counts=bull_counts,
                min_components=min_components,
                bull_series=pd.Series(True, index=close_series.index),  # No streak for simplicity
                buy_score_threshold=buy_score_threshold
            )
        else:
            gated_buy = pd.Series(False, index=close_series.index)

        if enable_composite and total_components > 0:
            composite_sell = compute_composite_sell(
                composite_score=composite_score,
                sell_score_threshold=sell_score_threshold,
                bull_counts=bull_counts,
                min_components=min_components
            )
        else:
            composite_sell = pd.Series([False]*len(close_series), index=close_series.index)

        signals[gated_buy] = 1
        signals[bear | composite_sell] = -1

        # Confidence
        use_conf = getattr(strat_config, 'use_confidence_for_sizing', True)
        blend_w = getattr(strat_config, 'confidence_blend_weight', 0.50)
        if use_conf and total_components > 0:
            current_confidence = compute_confidence(
                composite_score_last=float(composite_score.iloc[-1]),
                components_df_last_row=components_df.iloc[-1] if not components_df.empty else None,
                component_reliability=self._component_reliability,
                blend_weight=blend_w
            )
        else:
            current_confidence = float(composite_score.iloc[-1]) if total_components > 0 else 0.0
        if self._peak_confidence is None or current_confidence > self._peak_confidence:
            self._peak_confidence = current_confidence

        # Diagnostics
        try:
            last_idx = close.index[-1]
            if not self._diag_rows or self._diag_rows[-1]['date'] != last_idx:
                self.last_net_score = float(composite_score.iloc[-1]) if enable_composite else 0.0
                self.last_bull_score = float(bull_counts.iloc[-1] / total_components) if total_components > 0 else 0.0
                self.last_bear_score = 1.0 - self.last_bull_score
                self._diag_rows.append({
                    'date': last_idx,
                    'close': float(close_series.iloc[-1]),
                    'rsi': float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else None,
                    'macd': float(macd_series.iloc[-1]) if pd.notna(macd_series.iloc[-1]) else None,
                    'macd_signal': float(macdsignal_series.iloc[-1]) if pd.notna(macdsignal_series.iloc[-1]) else None,
                    'bb_lower': float(lowerband_series.iloc[-1]) if pd.notna(lowerband_series.iloc[-1]) else None,
                    'bb_upper': float(upperband_series.iloc[-1]) if pd.notna(upperband_series.iloc[-1]) else None,
                    'ema': float(ema_series.iloc[-1]) if pd.notna(ema_series.iloc[-1]) else None,
                    'volume': float(volume_series.iloc[-1]),
                    'rsi_strength': float(components_df.loc[last_idx, 'rsi']) if 'rsi' in components_df.columns else None,
                    'macd_strength': float(components_df.loc[last_idx, 'macd']) if 'macd' in components_df.columns else None,
                    'bb_strength': float(components_df.loc[last_idx, 'bb']) if 'bb' in components_df.columns else None,
                    'ema_strength': float(components_df.loc[last_idx, 'ema']) if 'ema' in components_df.columns else None,
                    'volume_strength': float(components_df.loc[last_idx, 'volume']) if 'volume' in components_df.columns else None,
                    'composite_score': float(composite_score.iloc[-1]) if enable_composite else None,
                    'bull_components': int(bull_counts.iloc[-1]) if total_components > 0 else None,
                    'bear_components': int(bear_counts.iloc[-1]) if total_components > 0 else None,
                    'total_components': total_components,
                    'signal': int(signals.iloc[-1]) if pd.notna(signals.iloc[-1]) else 0,
                    'confidence': current_confidence,
                    'peak_confidence': float(self._peak_confidence) if self._peak_confidence is not None else None,
                    'effective_risk_fraction': self._last_effective_risk_frac,
                    'realized_vol': self._last_realized_vol
                })
                cur_sig = int(signals.iloc[-1]) if pd.notna(signals.iloc[-1]) else 0
                if self._last_signal is None or cur_sig != self._last_signal:
                    etype = ('BUY' if cur_sig == 1 else ('SELL' if cur_sig == -1 else 'NEUTRAL'))
                    self._changes.append({
                        'date': last_idx,
                        'signal': cur_sig,
                        'event_type': etype,
                        'close': float(close_series.iloc[-1]),
                        'rsi': float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else None,
                        'macd': float(macd_series.iloc[-1]) if pd.notna(macd_series.iloc[-1]) else None,
                        'ema': float(ema_series.iloc[-1]) if pd.notna(ema_series.iloc[-1]) else None,
                        'composite_score': float(composite_score.iloc[-1]) if enable_composite else None,
                        'bull_components': int(bull_counts.iloc[-1]) if total_components > 0 else None,
                        'total_components': total_components,
                        'confidence': current_confidence,
                        'effective_risk_fraction': self._last_effective_risk_frac,
                        'realized_vol': self._last_realized_vol
                    })
                self._last_signal = cur_sig
        except Exception:
            pass

        return signals

    def calculate_position_size(self, data, capital, config=None):
        try:
            price = data['Close'].iloc[-1]
        except Exception:
            price = None
        if price is None or price != price or price <= 0:
            return 0.0

        strat_config = getattr(config, 'day_trading_strategy', None) or getattr(config, 'strategy', None)
        base_risk_fraction = getattr(strat_config, 'risk_per_trade', 0.01)
        atr_mult = getattr(strat_config, 'atr_stop_multiplier', 2.0)
        latest_atr = None
        if self._diag_rows:
            last = self._diag_rows[-1]
            latest_atr = last.get('atr')  # Assuming ATR added in future

        if latest_atr and latest_atr == latest_atr and latest_atr > 0:
            risk_per_share = atr_mult * latest_atr
        else:
            risk_per_share = 0.02 * price

        use_conf = getattr(strat_config, 'use_confidence_for_sizing', True)
        if use_conf and self._diag_rows:
            comp_score = self._diag_rows[-1].get('confidence')
            if comp_score is not None:
                low = getattr(strat_config, 'min_confidence', 0.40)
                high = getattr(strat_config, 'full_confidence', 0.80)
                floor_frac = getattr(strat_config, 'confidence_floor_fraction', 0.25)
                risk_fraction = scale_risk_fraction(base_risk_fraction, comp_score, low, high, floor_frac)
                self._last_effective_risk_frac = risk_fraction
            else:
                risk_fraction = base_risk_fraction
                self._last_effective_risk_frac = risk_fraction
        else:
            risk_fraction = base_risk_fraction
            self._last_effective_risk_frac = risk_fraction

        if getattr(strat_config, 'enable_drawdown_dampening', True):
            dd_thr = getattr(strat_config, 'drawdown_dampening_threshold', 0.10)
            damp = getattr(strat_config, 'drawdown_dampening_factor', 0.50)
            holder = {'peak': self._peak_price}
            risk_fraction = apply_drawdown_dampening(risk_fraction, price, holder, dd_thr, damp)
            self._peak_price = holder['peak']

        dollar_risk = capital * risk_fraction
        size_shares = int(dollar_risk / risk_per_share) if risk_per_share > 0 else int((capital * risk_fraction) / price)

        max_pos_fraction = getattr(strat_config, 'max_position_fraction', 0.20)
        max_pos_size = int((capital * max_pos_fraction) / price)
        size_shares = max(0, min(size_shares, max_pos_size))

        min_shares = getattr(strat_config, 'min_position_shares', 1)
        if size_shares > 0 and size_shares < min_shares:
            size_shares = min_shares if min_shares <= max_pos_size else 0

        return float(size_shares * price)

    def export_diagnostics(self, output_dir='analysis_output'):
        import os
        if not self._diag_rows:
            return
        try:
            os.makedirs(output_dir, exist_ok=True)
            pd.DataFrame(self._diag_rows).to_csv(os.path.join(output_dir, 'day_trading_strategy_signals.csv'), index=False)
            if self._changes:
                pd.DataFrame(self._changes).to_csv(os.path.join(output_dir, 'day_trading_strategy_signal_changes.csv'), index=False)
            if self._component_reliability:
                rows = []
                for comp, rel in self._component_reliability.items():
                    flips_hist = self._component_flips.get(comp, [])
                    instability_rate = sum(flips_hist[-15:]) / min(len(flips_hist), 15) if flips_hist else 0.0
                    total_flips = sum(flips_hist)
                    rows.append({
                        'component': comp,
                        'reliability': rel,
                        'total_flips': total_flips,
                        'recent_instability_rate': instability_rate,
                        'effective_weight': self._component_effective_weights.get(comp)
                    })
                pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'day_trading_component_reliability.csv'), index=False)
            if self._reliability_ts:
                try:
                    pd.DataFrame(self._reliability_ts).to_csv(os.path.join(output_dir, 'day_trading_component_reliability_timeseries.csv'), index=False)
                except Exception:
                    pass
            return output_dir
        except Exception:
            pass
        return None

    def reset(self):
        self._diag_rows.clear()
        self._changes.clear()
        self._last_signal = None
        self._entry_price = None
        self.last_net_score = 0.0
        self.last_bull_score = 0.0
        self.last_bear_score = 0.0
        self._peak_composite = None
        self._last_effective_risk_frac = None
        self._last_realized_vol = None
        self._component_reliability = {}
        self._component_flips = {}
        self._last_bar_close = None
        self._peak_confidence = None
        self._component_effective_weights = {}
        self._peak_price = None