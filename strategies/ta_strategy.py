import talib
import pandas as pd
from .base_strategy import BaseStrategy
from utils.strategy_utils import (
    dynamic_reliability_weighting,
    scale_risk_fraction,
    compute_realized_vol,
    compute_high_priority_override,
    evaluate_partial_exit,
    update_trailing_atr,
    compute_confidence,
    compute_high_priority_override_series,
    compute_gated_buy,
    compute_composite_sell,
    apply_drawdown_dampening,
    apply_volatility_gate
)

class TALibStrategy(BaseStrategy):
    def __init__(self):
        # store per-bar latest indicators for diagnostics
        self._diag_rows = []  # list of dicts
        self._changes = []    # compressed signal changes
        self._last_signal = None
        self._last_atr_stop = None
        self._bull_streak = 0
        self._entry_price = None
        self._trail_stop = None
        self._perf_trades = []  # list of trade dicts for simple performance summarization
        # last composite diagnostics (for external logging / sizing scaling)
        self.last_net_score = 0.0
        self.last_bull_score = 0.0
        self.last_bear_score = 0.0
        # Partial exit / scaling diagnostics
        self._peak_composite = None
        self._last_effective_risk_frac = None
        self._last_realized_vol = None
        # Reliability tracking per component (EMA of correctness)
        self._component_reliability = {}
        self._last_bar_close = None
        self._component_flips = {}  # comp -> list of 0/1 flips history
        self._peak_confidence = None
        self._component_effective_weights = {}  # last effective normalized weight per component
        self._peak_price = None  # for drawdown dampening of sizing
        # Reliability time-series (appended each calibration interval for longitudinal analysis)
        self._reliability_ts = []  # list of {date, component, reliability, effective_weight, instability}

    def generate_signals(self, data, config=None):
        # Clean implementation replaced due to earlier indentation corruption
        import pandas as pd
        if config is None:
            raise ValueError("Config must be provided")
        strat_config = getattr(config, 'talib_strategy', None) or getattr(config, 'short_term_strategy', None) or getattr(config, 'strategy', None)
        sma_window = strat_config.sma_window
        sma_long_window = strat_config.sma_long_window
        rsi_window = strat_config.rsi_window
        rsi_oversold = strat_config.rsi_oversold
        rsi_overbought = strat_config.rsi_overbought
        ema_window = getattr(strat_config, 'ema_window', 21)
        ema_long_window = getattr(strat_config, 'ema_long_window', 55)
        momentum_window = getattr(strat_config, 'momentum_window', 10)
        roc_window = getattr(strat_config, 'roc_window', 10)
        adx_window = getattr(strat_config, 'adx_window', 14)
        enable_composite = getattr(strat_config, 'enable_composite', True)
        buy_score_threshold = getattr(strat_config, 'buy_score_threshold', 0.60)
        sell_score_threshold = getattr(strat_config, 'sell_score_threshold', 0.40)
        min_components = getattr(strat_config, 'min_components', 2)
        enable_trend_component = getattr(strat_config, 'enable_trend_component', True)
        enable_ema_component = getattr(strat_config, 'enable_ema_component', True)
        enable_mom_component = getattr(strat_config, 'enable_mom_component', True)
        enable_roc_component = getattr(strat_config, 'enable_roc_component', True)
        enable_adx_component = getattr(strat_config, 'enable_adx_component', True)

        if isinstance(data.columns, pd.MultiIndex):
            close = data[('Close',)] if ('Close',) in data.columns else data[(data.columns[0][0], 'Close')]
            high = data[('High',)] if ('High',) in data.columns else data[(data.columns[0][0], 'High')]
            low = data[('Low',)] if ('Low',) in data.columns else data[(data.columns[0][0], 'Low')]
        else:
            close = data['Close']
            high = data['High']
            low = data['Low']

        close_np = close.to_numpy()
        sma = talib.SMA(close_np, timeperiod=sma_window)
        sma_long = talib.SMA(close_np, timeperiod=sma_long_window)
        rsi = talib.RSI(close_np, timeperiod=rsi_window)
        ema = talib.EMA(close_np, timeperiod=ema_window)
        ema_long = talib.EMA(close_np, timeperiod=ema_long_window)
        mom = talib.MOM(close_np, timeperiod=momentum_window)
        roc = talib.ROC(close_np, timeperiod=roc_window)
        try:
            adx = talib.ADX(high.to_numpy(), low.to_numpy(), close_np, timeperiod=adx_window)
        except Exception:
            adx = [float('nan')]*len(close_np)
        try:
            atr = talib.ATR(high.to_numpy(), low.to_numpy(), close_np, timeperiod=getattr(strat_config,'atr_period',14))
        except Exception:
            atr = [float('nan')]*len(close_np)

        close_series = pd.Series(close_np, index=close.index)
        sma_series = pd.Series(sma, index=close.index)
        sma_long_series = pd.Series(sma_long, index=close.index)
        rsi_series = pd.Series(rsi, index=close.index)
        ema_series = pd.Series(ema, index=close.index)
        ema_long_series = pd.Series(ema_long, index=close.index)
        mom_series = pd.Series(mom, index=close.index)
        roc_series = pd.Series(roc, index=close.index)
        adx_series = pd.Series(adx, index=close.index)

        signals = pd.Series(0, index=close.index)
        bull_raw = (close_series > sma_series) & (sma_series > sma_long_series) & (rsi_series > rsi_oversold)

        comp_dict = {}
        if enable_trend_component:
            comp_dict['trend'] = (close_series > sma_series) & (sma_series > sma_long_series)
        if enable_ema_component:
            comp_dict['ema'] = (close_series > ema_series) & (ema_series > ema_long_series)
        if enable_mom_component:
            comp_dict['mom'] = mom_series > 0
        if enable_roc_component:
            comp_dict['roc'] = roc_series > 0
        if enable_adx_component:
            comp_dict['adx'] = adx_series >= 20
        components_df = pd.DataFrame(comp_dict) if comp_dict else pd.DataFrame(index=close_series.index)
        bull_counts = components_df.sum(axis=1) if not components_df.empty else pd.Series(0, index=close_series.index)
        total_components = components_df.shape[1] if not components_df.empty else 0
        # --- Calibration / weighting ---
        comp_weights_cfg = getattr(strat_config, 'component_weights', None) or {}
        if total_components > 0 and comp_weights_cfg:
            # Normalize provided weights for existing components
            w = {k: float(comp_weights_cfg.get(k, 1.0)) for k in components_df.columns}
            s = sum(w.values()) or 1.0
            norm_w = {k: v / s for k, v in w.items()}
            weighted = components_df.astype(float).mul(pd.Series(norm_w))
            composite_raw = weighted.sum(axis=1)  # already in 0..1 due to normalized weights
        else:
            composite_raw = (bull_counts / total_components).fillna(0.0) if total_components > 0 else pd.Series(0.0, index=close_series.index)

        # Dynamic reliability calibration via shared utility
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
            # Pull back effective weights updated by utility
            self._component_effective_weights = getattr(self, '_component_effective_weights', {}) or {}
            self._last_bar_close = float(close_series.iloc[-1]) if len(close_series) else None
        else:
            composite_score = composite_raw

        confirm_needed = getattr(strat_config, 'bull_confirm_bars', 1)
        bull_flags = []
        # Optional RSI overbought entry gate: skip counting streak progress when RSI already overbought if enabled.
        rsi_entry_gate = getattr(strat_config, 'enable_rsi_overbought_entry_gate', False)
        for i in range(len(close_series)):
            raw_pass = bool(bull_raw.iloc[i])
            if raw_pass and rsi_entry_gate:
                # Prevent initiating/continuing streak if RSI considered overbought (avoid chasing peak)
                if rsi_series.iloc[i] >= rsi_overbought:
                    raw_pass = False
            if raw_pass:
                self._bull_streak += 1
            else:
                self._bull_streak = 0
            bull_flags.append(self._bull_streak >= confirm_needed)
        bull = pd.Series(bull_flags, index=close_series.index)

        if enable_composite and total_components > 0:
            hp = getattr(strat_config, 'high_priority_components', None) or []
            override_thr = getattr(strat_config, 'override_threshold', buy_score_threshold)
            override_buy = compute_high_priority_override_series(components_df, hp, override_thr) if hp else pd.Series(False, index=close_series.index)
            gated_buy = compute_gated_buy(
                composite_score=composite_score,
                override_buy=override_buy,
                bull_counts=bull_counts,
                min_components=min_components,
                bull_series=bull,
                buy_score_threshold=buy_score_threshold
            )
        else:
            gated_buy = bull

        # Track peak composite for partial exit logic
        if enable_composite and total_components > 0:
            latest_comp = composite_score.iloc[-1]
            if self._peak_composite is None or latest_comp > self._peak_composite:
                self._peak_composite = latest_comp
        # Reset peak composite on fresh entry (previous bar not long, new signal will be 1)
        # We'll detect this after signals computed; handled later once we know current signal

        bear = (close_series < sma_series) | (sma_series < sma_long_series) | (rsi_series > rsi_overbought)
        if enable_composite and total_components > 0:
            composite_sell = compute_composite_sell(
                composite_score=composite_score,
                sell_score_threshold=sell_score_threshold,
                bull_counts=bull_counts,
                min_components=min_components
            )
        else:
            composite_sell = pd.Series([False]*len(close_series), index=close_series.index)

        enable_neutral = getattr(strat_config, 'enable_neutral', True)
        if enable_neutral:
            buy_condition = gated_buy
            sell_condition = (bear | composite_sell) & (~gated_buy)
        else:
            buy_condition = gated_buy
            sell_condition = bear | composite_sell

        signals[buy_condition] = 1
        signals[sell_condition] = -1

        # Confidence calculation (blended) for sizing / partial exits
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
            current_confidence = float(composite_score.iloc[-1]) if total_components > 0 else (1.0 if bull.iloc[-1] else 0.0)
        # Track peak confidence (for confidence-based partial exit)
        if self._peak_confidence is None or current_confidence > self._peak_confidence:
            self._peak_confidence = current_confidence

        # Partial exit logic: allow trim while still in long (signal 1) OR neutral (0) but not on full sell (-1)
        enable_partial = getattr(strat_config, 'enable_partial_exits', False)
        conf_partial = getattr(strat_config, 'enable_confidence_partial_exit', True)
        drop_thr = getattr(strat_config, 'partial_exit_score_drop', 0.15)
        if self._last_signal == 1 and signals.iloc[-1] != -1 and enable_composite:
            latest_comp = float(composite_score.iloc[-1])
            new_sig, new_peak_comp, new_peak_conf = evaluate_partial_exit(
                last_signal=1,
                enable_partial=enable_partial,
                enable_composite=enable_composite,
                peak_composite=self._peak_composite,
                peak_confidence=self._peak_confidence,
                latest_comp=latest_comp,
                current_confidence=current_confidence,
                sell_score_threshold=sell_score_threshold,
                drop_thr=drop_thr,
                conf_partial=conf_partial
            )
            if new_sig == 2:
                signals.iloc[-1] = 2
            self._peak_composite = new_peak_comp
            self._peak_confidence = new_peak_conf

        atr_series = pd.Series(atr, index=close.index)
        trail_mult = getattr(strat_config, 'trail_atr_multiplier', getattr(strat_config,'atr_stop_multiplier',2.0))
        self._trail_stop, exit_trail = update_trailing_atr(
            trail_stop=self._trail_stop,
            enable_trailing_atr=getattr(strat_config, 'enable_trailing_atr', True),
            last_signal=self._last_signal,
            entry_price=self._entry_price,
            close_price=close_series.iloc[-1],
            atr_value=atr_series.iloc[-1],
            trail_mult=trail_mult
        )
        if exit_trail:
            signals.iloc[-1] = -1

        try:
            latest_atr = float(atr_series.iloc[-1])
        except Exception:
            latest_atr = float('nan')

        try:
            last_idx = close.index[-1]
            if not self._diag_rows or self._diag_rows[-1]['date'] != last_idx:
                self.last_net_score = float(composite_score.iloc[-1]) if enable_composite else (1.0 if bull.iloc[-1] else 0.0)
                self.last_bull_score = float(bull_counts.iloc[-1] / total_components) if total_components > 0 else 0.0
                self.last_bear_score = 1.0 - self.last_bull_score
                self._diag_rows.append({
                    'date': last_idx,
                    'close': float(close_series.iloc[-1]),
                    'sma': float(sma_series.iloc[-1]) if pd.notna(sma_series.iloc[-1]) else None,
                    'sma_long': float(sma_long_series.iloc[-1]) if pd.notna(sma_long_series.iloc[-1]) else None,
                    'rsi': float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else None,
                    'ema': float(ema_series.iloc[-1]) if pd.notna(ema_series.iloc[-1]) else None,
                    'ema_long': float(ema_long_series.iloc[-1]) if pd.notna(ema_long_series.iloc[-1]) else None,
                    'mom': float(mom_series.iloc[-1]) if pd.notna(mom_series.iloc[-1]) else None,
                    'roc': float(roc_series.iloc[-1]) if pd.notna(roc_series.iloc[-1]) else None,
                    'adx': float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else None,
                    'composite_score': float(composite_score.iloc[-1]) if enable_composite else None,
                    'bull_components': int(bull_counts.iloc[-1]) if total_components > 0 else None,
                    'total_components': total_components,
                    'signal': int(signals.iloc[-1]) if pd.notna(signals.iloc[-1]) else 0,
                    'atr': latest_atr,
                    'peak_composite': float(self._peak_composite) if self._peak_composite is not None else None,
                    'confidence': current_confidence,
                    'peak_confidence': float(self._peak_confidence) if self._peak_confidence is not None else None,
                    'effective_risk_fraction': self._last_effective_risk_frac,
                    'realized_vol': self._last_realized_vol
                })
                cur_sig = int(signals.iloc[-1]) if pd.notna(signals.iloc[-1]) else 0
                if self._last_signal is None or cur_sig != self._last_signal:
                    etype = 'PARTIAL_EXIT' if cur_sig == 2 else ('BUY' if cur_sig == 1 else ('SELL' if cur_sig == -1 else 'NEUTRAL'))
                    self._changes.append({
                        'date': last_idx,
                        'signal': cur_sig,
                        'event_type': etype,
                        'close': float(close_series.iloc[-1]),
                        'sma': float(sma_series.iloc[-1]) if pd.notna(sma_series.iloc[-1]) else None,
                        'sma_long': float(sma_long_series.iloc[-1]) if pd.notna(sma_long_series.iloc[-1]) else None,
                        'rsi': float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else None,
                        'ema': float(ema_series.iloc[-1]) if pd.notna(ema_series.iloc[-1]) else None,
                        'ema_long': float(ema_long_series.iloc[-1]) if pd.notna(ema_long_series.iloc[-1]) else None,
                        'mom': float(mom_series.iloc[-1]) if pd.notna(mom_series.iloc[-1]) else None,
                        'roc': float(roc_series.iloc[-1]) if pd.notna(roc_series.iloc[-1]) else None,
                        'adx': float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else None,
                        'composite_score': float(composite_score.iloc[-1]) if enable_composite else None,
                        'bull_components': int(bull_counts.iloc[-1]) if total_components > 0 else None,
                        'total_components': total_components,
                        'atr': latest_atr,
                        'trail_stop': float(self._trail_stop) if self._trail_stop is not None else None,
                        'peak_composite': float(self._peak_composite) if self._peak_composite is not None else None,
                        'confidence': current_confidence,
                        'peak_confidence': float(self._peak_confidence) if self._peak_confidence is not None else None,
                        'effective_risk_fraction': self._last_effective_risk_frac,
                        'realized_vol': self._last_realized_vol
                    })
                self._last_signal = cur_sig
                # If fresh entry, set peak to current composite
                if cur_sig == 1 and self._last_signal != 1 and enable_composite and total_components > 0:
                    try:
                        self._peak_composite = float(composite_score.iloc[-1])
                    except Exception:
                        pass
        except Exception:
            pass

        return signals

    def calculate_position_size(self, data, capital, config=None):
        """Return dollar notional (NOT shares) for new position.

        Logic:
          - ATR based risk-per-share
          - Composite score scaling of risk fraction
          - Volatility cap (ATR% of price)
          - Enforce max position fraction
          - Record effective risk fraction & realized volatility diagnostics
        """
        # Price from data (supports being passed a DataFrame consistent with other strategies)
        try:
            price = data['Close'].iloc[-1]
        except Exception:
            price = None
        if price is None or price != price or price <= 0:
            return 0.0

        strat_config = getattr(config, 'talib_strategy', None) or getattr(config, 'short_term_strategy', None) or getattr(config, 'strategy', None)
        base_risk_fraction = getattr(strat_config, 'risk_per_trade', 0.01)
        atr_mult = getattr(strat_config, 'atr_stop_multiplier', 2.0)
        latest_atr = None
        if self._diag_rows:
            last = self._diag_rows[-1]
            latest_atr = last.get('atr')

        # Determine risk-per-share via ATR if possible
        if latest_atr and latest_atr == latest_atr and latest_atr > 0:
            risk_per_share = atr_mult * latest_atr
        else:
            # fallback: assume 2% of price stop distance
            risk_per_share = 0.02 * price

        # Composite score scaling (confidence-aware)
        use_conf = getattr(strat_config, 'use_confidence_for_sizing', True)
        if use_conf and self._diag_rows:
            comp_score = self._diag_rows[-1].get('confidence')
            if comp_score is not None:
                low = getattr(strat_config, 'min_confidence', getattr(strat_config, 'score_min_risk_at', 0.40))
                high = getattr(strat_config, 'full_confidence', getattr(strat_config, 'score_full_risk_at', 0.80))
                floor_frac = getattr(strat_config, 'confidence_floor_fraction', getattr(strat_config, 'position_score_floor_fraction', 0.25))
                latest_atr_local = self._diag_rows[-1].get('atr') if self._diag_rows else None
                comp_score = apply_volatility_gate(
                    score=comp_score,
                    price=price,
                    atr_value=latest_atr_local,
                    enabled=getattr(strat_config, 'enable_volatility_gate', True),
                    gate_atr_pct=getattr(strat_config, 'vol_gate_atr_pct', 0.08),
                    confidence_penalty=getattr(strat_config, 'vol_gate_confidence_penalty', 0.10)
                )
                risk_fraction = scale_risk_fraction(base_risk_fraction, comp_score, low, high, floor_frac)
                self._last_effective_risk_frac = risk_fraction
            else:
                risk_fraction = base_risk_fraction
                self._last_effective_risk_frac = risk_fraction
        elif getattr(strat_config, 'position_size_scale_with_score', True) and self._diag_rows:
            comp_score = self._diag_rows[-1].get('composite_score')
            if comp_score is not None:
                low = getattr(strat_config, 'score_min_risk_at', 0.40)
                high = getattr(strat_config, 'score_full_risk_at', 0.80)
                floor_frac = getattr(strat_config, 'position_score_floor_fraction', 0.25)
                if comp_score <= low:
                    score_scale = floor_frac
                elif comp_score >= high:
                    score_scale = 1.0
                else:
                    score_scale = floor_frac + ( (comp_score - low) / (high - low) ) * (1.0 - floor_frac)
                risk_fraction = base_risk_fraction * max(0.0, min(score_scale, 1.5))
                self._last_effective_risk_frac = risk_fraction
            else:
                risk_fraction = base_risk_fraction
                self._last_effective_risk_frac = risk_fraction
        else:
            risk_fraction = base_risk_fraction
            self._last_effective_risk_frac = risk_fraction
        # High-priority size boost if last diag row flagged override (approximate by comparing last_net_score vs buy threshold & many HP comps passed)
        try:
            strat_cfg = strat_config
            hp_comps = getattr(strat_cfg, 'high_priority_components', None) or []
            if hp_comps and self._diag_rows:
                # Reconstruct HP bullish share from last diag if possible (not stored explicitly); approximate using confidence > high - small epsilon
                if self._diag_rows[-1].get('confidence') and self._diag_rows[-1]['confidence'] >= getattr(strat_cfg, 'override_threshold', 0.60):
                    boost = getattr(strat_cfg, 'high_priority_size_boost', 0.20)
                    risk_fraction *= (1.0 + min(0.50, max(0.0, boost)))
        except Exception:
            pass

        # Drawdown dampening: if in drawdown from peak price reduce risk fraction (applies to fresh sizing for new entry as precaution)
        if getattr(strat_config, 'enable_drawdown_dampening', True):
            dd_thr = getattr(strat_config, 'drawdown_dampening_threshold', 0.10)
            damp = getattr(strat_config, 'drawdown_dampening_factor', 0.50)
            holder = {'peak': self._peak_price}
            risk_fraction = apply_drawdown_dampening(risk_fraction, price, holder, dd_thr, damp)
            self._peak_price = holder['peak']

        dollar_risk = capital * risk_fraction
        size_shares = int(dollar_risk / risk_per_share) if risk_per_share > 0 else int((capital * risk_fraction) / price)

        # Volatility cap: if ATR% of price exceeds threshold, downscale
        if latest_atr and latest_atr == latest_atr and price > 0:
            atr_pct = latest_atr / price
            max_atr_pct = getattr(strat_config, 'max_atr_pct', 0.05)
            if atr_pct > max_atr_pct:
                # scale size inversely proportional to excess volatility
                scale = max(0.2, max_atr_pct / atr_pct)
                size_shares = int(size_shares * scale)
        # realized volatility (close-to-close) for diagnostics using last N closes from diag rows
        try:
            if len(self._diag_rows) >= 25:
                closes = [row['close'] for row in self._diag_rows[-25:] if row.get('close') is not None]
                self._last_realized_vol = compute_realized_vol(closes) if len(closes) >= 20 else None
            else:
                self._last_realized_vol = None
        except Exception:
            self._last_realized_vol = None

        # Absolute notional cap
        max_pos_fraction = getattr(strat_config, 'max_position_fraction', getattr(strat_config, 'max_position_size', 0.20))
        max_pos_size = int((capital * max_pos_fraction) / price)
        size_shares = max(0, min(size_shares, max_pos_size))

        # Enforce minimum shares if any position at all
        min_shares = getattr(strat_config, 'min_position_shares', 1)
        if size_shares > 0 and size_shares < min_shares:
            size_shares = min_shares if min_shares <= max_pos_size else 0
        # Return notional dollars
        return float(size_shares * price)

    def export_diagnostics(self, output_dir='analysis_output'):
        import os
        if not self._diag_rows:
            return
        try:
            os.makedirs(output_dir, exist_ok=True)
            pd.DataFrame(self._diag_rows).to_csv(os.path.join(output_dir, 'talib_strategy_signals.csv'), index=False)
            if self._changes:
                pd.DataFrame(self._changes).to_csv(os.path.join(output_dir, 'talib_strategy_signal_changes.csv'), index=False)
            # reliability snapshot export
            if self._component_reliability:
                rows = []
                for comp, rel in self._component_reliability.items():
                    flips_hist = self._component_flips.get(comp, [])
                    if flips_hist:
                        instability_rate = sum(flips_hist[-15:]) / min(len(flips_hist), 15)
                        total_flips = sum(flips_hist)
                    else:
                        instability_rate = 0.0
                        total_flips = 0
                    rows.append({
                        'component': comp,
                        'reliability': rel,
                        'total_flips': total_flips,
                        'recent_instability_rate': instability_rate,
                        'effective_weight': self._component_effective_weights.get(comp)
                    })
                pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'talib_component_reliability.csv'), index=False)
            # Reliability longitudinal timeline (one row per component per calibration point)
            if self._reliability_ts:
                try:
                    pd.DataFrame(self._reliability_ts).to_csv(os.path.join(output_dir, 'talib_component_reliability_timeseries.csv'), index=False)
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
        self._bull_streak = 0
        self._entry_price = None
        self._trail_stop = None
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
