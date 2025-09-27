import math
from typing import Dict, Any, List
import pandas as pd

# ---------- Reliability / Composite Utilities ----------

def dynamic_reliability_weighting(state: Dict[str, Any],
                                  components_df: pd.DataFrame,
                                  composite_raw: pd.Series,
                                  close_series: pd.Series,
                                  strat_config: Any,
                                  reliability_ts: List[dict]):
    """Apply dynamic reliability weighting to component boolean signals.

    Parameters:
      state: mutable dict tracking reliability state
        keys used: reliability, flips, effective_weights, last_close
      components_df: DataFrame of boolean component states (columns = component names)
      composite_raw: Raw (unweighted) composite score series
      close_series: Price series aligned with rows
      strat_config: strategy configuration object (expects fields used below)
      reliability_ts: list to append longitudinal reliability entries ({date, component, reliability,...})

    Returns:
      composite_score (Series)
      updated state (mutated in place)
    """
    if components_df.empty or not getattr(strat_config, 'enable_dynamic_calibration', True):
        return composite_raw

    total_components = components_df.shape[1]
    if total_components == 0:
        return composite_raw

    warmup = getattr(strat_config, 'warmup_bars_for_reliability', 40)
    if len(components_df) <= max(5, warmup):
        return composite_raw

    calibrate_every = max(1, getattr(strat_config, 'calibration_update_interval', 1))
    if (len(components_df) % calibrate_every) != 0:
        return composite_raw

    # Pull state pieces
    reliability = state.setdefault('reliability', {})
    flips = state.setdefault('flips', {})
    last_close = state.get('last_close')

    smooth = getattr(strat_config, 'reliability_smoothing', 0.2)
    micro_thr = getattr(strat_config, 'micro_move_threshold', 0.002)
    reg_mean = getattr(strat_config, 'reliability_regression_to_mean', 0.02)
    reliability_floor = getattr(strat_config, 'reliability_floor', 0.30)
    lookback_inst = getattr(strat_config, 'instability_lookback', 15)
    inst_penalty = getattr(strat_config, 'instability_penalty', 0.40)

    # Update reliability using last bar outcome
    if last_close is not None and len(close_series) >= 2:
        ret = (close_series.iloc[-1] / last_close) - 1.0
        if abs(ret) < micro_thr:
            ret = 0.0
        try:
            last_row = components_df.iloc[-2]
            for comp_name, val in last_row.items():
                bullish_pred = bool(val)
                if ret == 0.0:
                    correct = 0.5
                else:
                    correct = 1.0 if ((ret > 0 and bullish_pred) or (ret < 0 and not bullish_pred)) else 0.0
                prev = reliability.get(comp_name, 0.5)
                updated = (1 - smooth) * prev + smooth * correct
                updated = (1 - reg_mean) * updated + reg_mean * 0.5
                if updated < reliability_floor:
                    updated = reliability_floor
                reliability[comp_name] = updated
                prev_state = flips.get(comp_name, [bullish_pred])[-1] if flips.get(comp_name) else bullish_pred
                hist = flips.setdefault(comp_name, [])
                hist.append(1 if prev_state != bullish_pred else 0)
                if len(hist) > lookback_inst:
                    del hist[0]
        except Exception:
            pass

    # --- Pruning (optional) ---
    # Pruning parameters now expected to be defined in external YAML / dataclass config.
    # We purposefully avoid embedding numeric fallbacks here to force explicit configuration
    # (or reliance on dataclass defaults) and eliminate silent drift.
    enable_prune = getattr(strat_config, 'enable_reliability_pruning', False)
    prune_thr = strat_config.reliability_prune_threshold if enable_prune else None
    prune_consec = strat_config.reliability_prune_consecutive if enable_prune else 0
    react_thr = strat_config.reactivation_threshold if enable_prune else None
    react_consec = strat_config.reactivation_consecutive if enable_prune else 0
    min_hist = strat_config.min_history_for_prune if enable_prune else 10**9  # effectively disable if off
    max_pruned_frac = strat_config.max_pruned_fraction if enable_prune else 0.0
    grace = strat_config.prune_grace_bars if enable_prune else 0
    inst_prune_thr = getattr(strat_config, 'instability_prune_threshold', None) if enable_prune else None
    protected = (getattr(strat_config, 'protected_components', None) or []) if enable_prune else []
    pruned = state.setdefault('pruned', set())
    prune_meta = state.setdefault('prune_meta', {})  # comp -> dict counters

    # Only attempt pruning at calibration points post warmup+grace and with enough history
    allow_prune_cycle = enable_prune and len(components_df) > max(min_hist, warmup + grace)

    # Compute instability snapshot for reuse
    instability_snapshot = {}
    for k in components_df.columns:
        flips_hist = state.setdefault('flips', {}).get(k, [])
        instability_snapshot[k] = (sum(flips_hist) / len(flips_hist)) if flips_hist else 0.0

    if allow_prune_cycle and total_components > 1:  # need at least 2 to consider pruning
        # Update streak counters & decide prune / reactivate
        active_comps = [c for c in components_df.columns]
        for comp in active_comps:
            meta = prune_meta.setdefault(comp, {'below_streak': 0, 'reactivate_streak': 0})
            r = reliability.get(comp, 0.5)
            instab = instability_snapshot.get(comp, 0.0)
            # Skip protected components
            if comp in protected:
                meta['below_streak'] = 0
                meta['reactivate_streak'] = 0
                if comp in pruned:
                    pruned.discard(comp)
                continue
            if comp not in pruned:
                if r < prune_thr or (inst_prune_thr is not None and instab > inst_prune_thr and r < react_thr):
                    meta['below_streak'] += 1
                else:
                    meta['below_streak'] = 0
                if meta['below_streak'] >= prune_consec:
                    # Guard max pruned fraction
                    prospective = len(pruned) + 1
                    if (prospective / total_components) <= max_pruned_frac and (total_components - prospective) >= 1:
                        pruned.add(comp)
                        meta['reactivate_streak'] = 0
            else:
                # Currently pruned: check reactivation conditions
                if r >= react_thr and (inst_prune_thr is None or instability_snapshot.get(comp, 0.0) <= (inst_prune_thr or 1.0)):
                    meta['reactivate_streak'] += 1
                else:
                    meta['reactivate_streak'] = 0
                if meta['reactivate_streak'] >= react_consec:
                    pruned.discard(comp)
                    meta['below_streak'] = 0
                    meta['reactivate_streak'] = 0

    # Apply pruning by filtering components_df for weighting (do not mutate original for diagnostics)
    effective_components_df = components_df.drop(columns=list(pruned), errors='ignore') if pruned else components_df
    # If all got pruned accidentally, fallback to original
    if effective_components_df.empty:
        effective_components_df = components_df

    comp_weights_cfg = getattr(strat_config, 'component_weights', None) or {}
    base_weights = {k: comp_weights_cfg.get(k, 1.0) if comp_weights_cfg else 1.0 for k in effective_components_df.columns}
    mult = {}
    for k in effective_components_df.columns:
        rel_w = reliability.get(k, 0.5)
        flips_hist = flips.get(k, [])
        instability = (sum(flips_hist) / len(flips_hist)) if flips_hist else 0.0
        penalty = (1 - inst_penalty * instability)
        mult[k] = rel_w * base_weights[k] * max(0.10, penalty)
    s2 = sum(mult.values()) or 1.0
    norm_mult = {k: v / s2 for k, v in mult.items()}
    composite_score = effective_components_df.astype(float).mul(pd.Series(norm_mult)).sum(axis=1)

    state['effective_weights'] = norm_mult
    state['last_close'] = float(close_series.iloc[-1]) if len(close_series) else None

    # Append reliability longitudinal rows
    cur_date = close_series.index[-1]
    for k in components_df.columns:
        flips_hist = flips.get(k, [])
        instability = (sum(flips_hist) / len(flips_hist)) if flips_hist else 0.0
        meta = prune_meta.get(k, {})
        reliability_ts.append({
                'date': cur_date,
                'component': k,
                'reliability': reliability.get(k, 0.5),
                'effective_weight': norm_mult.get(k, 0.0),
                'instability': instability,
                'pruned': k in pruned,
                'below_streak': meta.get('below_streak', 0),
                'reactivate_streak': meta.get('reactivate_streak', 0)
        })
    return composite_score

# ---------- Risk / Sizing Utilities ----------

def scale_risk_fraction(base_risk_fraction: float,
                        score: float,
                        low: float,
                        high: float,
                        floor_frac: float) -> float:
    if score <= low:
        scale = floor_frac
    elif score >= high:
        scale = 1.0
    else:
        scale = floor_frac + ((score - low) / (high - low)) * (1.0 - floor_frac)
    return base_risk_fraction * max(0.0, min(scale, 1.5))

# ---------- Volatility / Diagnostics ----------

def compute_realized_vol(closes: list, annualize: bool = True):
    import numpy as np
    if len(closes) < 2:
        return 0.0
    rets = np.diff(closes) / np.array(closes[:-1])
    if rets.size <= 1:
        return 0.0
    vol = rets.std()
    if annualize:
        import math
        vol *= math.sqrt(252)
    return float(vol)

# ---------- Entry / Exit Utilities ----------

def compute_high_priority_override(components_df_last_row, high_priority: list, override_thr: float) -> bool:
    if not high_priority:
        return False
    existing = [c for c in high_priority if c in components_df_last_row.index]
    if not existing:
        return False
    bullish_share = components_df_last_row[existing].sum() / len(existing)
    return bullish_share >= override_thr

def evaluate_partial_exit(last_signal: int,
                          enable_partial: bool,
                          enable_composite: bool,
                          peak_composite: float | None,
                          peak_confidence: float | None,
                          latest_comp: float,
                          current_confidence: float,
                          sell_score_threshold: float,
                          drop_thr: float,
                          conf_partial: bool) -> tuple[int, float | None, float | None]:
    """Return (new_signal, new_peak_comp, new_peak_conf) after partial exit evaluation."""
    new_signal = last_signal
    pc = peak_composite
    pconf = peak_confidence
    if enable_partial and last_signal == 1 and enable_composite:
        comp_drop_trigger = (pc is not None and (pc - latest_comp) >= drop_thr and latest_comp > sell_score_threshold)
        conf_drop_trigger = False
        if conf_partial and pconf is not None:
            conf_drop_trigger = (pconf - current_confidence) >= drop_thr and current_confidence > sell_score_threshold
        if comp_drop_trigger or conf_drop_trigger:
            new_signal = 2
            pc = latest_comp
            pconf = current_confidence
    return new_signal, pc, pconf

def update_trailing_atr(trail_stop: float | None,
                        enable_trailing_atr: bool,
                        last_signal: int,
                        entry_price: float | None,
                        close_price: float,
                        atr_value: float | None,
                        trail_mult: float) -> tuple[float | None, bool]:
    """Update trailing ATR stop; return (new_trail_stop, exit_triggered)."""
    if not enable_trailing_atr or last_signal != 1 or entry_price is None or atr_value is None or atr_value != atr_value:
        return trail_stop, False
    candidate = close_price - trail_mult * atr_value
    if trail_stop is None:
        trail_stop = candidate
    else:
        trail_stop = max(trail_stop, candidate)
    if close_price < trail_stop:
        return trail_stop, True
    return trail_stop, False

# ---------- Confidence / Override Series Utilities ----------

def compute_confidence(composite_score_last: float,
                       components_df_last_row: pd.Series | None,
                       component_reliability: Dict[str, float],
                       blend_weight: float) -> float:
    """Blend composite score and reliability-weighted component correctness.

    Parameters:
      composite_score_last: float composite (0..1) for latest bar
      components_df_last_row: Series of latest boolean component states
      component_reliability: mapping component->reliability (0..1)
      blend_weight: weight toward composite (0..1); remainder toward reliability-weighted correctness
    Returns:
      blended confidence float
    """
    if components_df_last_row is None or components_df_last_row.empty:
        return float(composite_score_last)
    rel_w_scores = []
    rel_w_vals = []
    try:
        for comp, val in components_df_last_row.items():
            rel = component_reliability.get(comp, 0.5)
            rel_w_scores.append(float(bool(val)) * rel)
            rel_w_vals.append(rel)
        if sum(rel_w_vals) > 0:
            reliability_weighted = sum(rel_w_scores) / sum(rel_w_vals)
        else:
            reliability_weighted = composite_score_last
        bw = max(0.0, min(1.0, blend_weight))
        return bw * float(composite_score_last) + (1 - bw) * float(reliability_weighted)
    except Exception:
        return float(composite_score_last)

def compute_high_priority_override_series(components_df: pd.DataFrame,
                                          high_priority: list,
                                          override_thr: float) -> pd.Series:
    """Return boolean Series where high-priority component share >= override threshold.

    Parameters:
      components_df: DataFrame of boolean components
      high_priority: list of component names considered high priority
      override_thr: threshold (0..1) of bullish share among HP components to trigger override
    """
    if components_df is None or components_df.empty or not high_priority:
        return pd.Series([False]*len(components_df.index), index=components_df.index if components_df is not None else None)
    existing_hp = [c for c in high_priority if c in components_df.columns]
    if not existing_hp:
        return pd.Series(False, index=components_df.index)
    hp_share = components_df[existing_hp].sum(axis=1) / len(existing_hp)
    return hp_share >= override_thr

# ---------- Buy/Sell Gating Utilities ----------

def compute_gated_buy(composite_score: pd.Series,
                      override_buy: pd.Series,
                      bull_counts: pd.Series,
                      min_components: int,
                      bull_series: pd.Series,
                      buy_score_threshold: float) -> pd.Series:
    """Return Series of buy eligibility after composite + override + streak gating."""
    return ((composite_score >= buy_score_threshold) | override_buy) & (bull_counts >= min_components) & bull_series

def compute_composite_sell(composite_score: pd.Series,
                           sell_score_threshold: float,
                           bull_counts: pd.Series,
                           min_components: int) -> pd.Series:
    return (composite_score <= sell_score_threshold) | (bull_counts < min_components)

# ---------- Drawdown Dampening Utility ----------

def apply_drawdown_dampening(risk_fraction: float,
                             latest_price: float,
                             peak_price_ref: dict,
                             dd_threshold: float,
                             damp_factor: float) -> float:
    """Adjust risk fraction based on drawdown from peak price.

    peak_price_ref: dict acting as mutable reference holding {'peak': value}
    Returns new risk fraction; updates peak in place.
    """
    peak = peak_price_ref.get('peak')
    if peak is None or latest_price > peak:
        peak_price_ref['peak'] = latest_price
        return risk_fraction
    if latest_price < peak:
        dd = 1.0 - (latest_price / peak)
        if dd >= dd_threshold:
            return risk_fraction * max(0.10, damp_factor)
    return risk_fraction

# ---------- Volatility Gate Utility ----------

def apply_volatility_gate(score: float,
                          price: float | None,
                          atr_value: float | None,
                          enabled: bool,
                          gate_atr_pct: float,
                          confidence_penalty: float) -> float:
    """Apply volatility gate penalty to a score/confidence if ATR% exceeds threshold.

    Parameters:
      score: current confidence / composite score value (0..1)
      price: latest close price
      atr_value: latest ATR value
      enabled: master enable flag
      gate_atr_pct: threshold (e.g. 0.08 means 8% ATR/price)
      confidence_penalty: amount to subtract from score when gate triggers

    Returns new (possibly penalized) score bounded to 0..1.
    """
    if not enabled or atr_value is None or price is None or price <= 0 or atr_value != atr_value:
        return score
    try:
        atr_pct = atr_value / price
        if atr_pct > gate_atr_pct:
            return max(0.0, min(1.0, score - confidence_penalty))
        return score
    except Exception:
        return score
