from dataclasses import dataclass, field
from typing import List, Optional
import yaml

@dataclass
class VolumeConfig:
    # Short-term Volume Indicators
    obv_ma_window: int = 20         # OBV moving average window
    mfi_period: int = 14            # Money Flow Index period
    mfi_oversold: int = 20          # MFI oversold level
    mfi_overbought: int = 80        # MFI overbought level
    cmf_fast_period: int = 3        # Chaikin Money Flow fast period
    cmf_slow_period: int = 10       # Chaikin Money Flow slow period
    vwap_period: str = "D"          # VWAP calculation period
    volume_profile_bins: int = 12    # Volume profile bins
    
    # Long-term Volume Analysis
    institutional_volume_threshold: int = 1000000  # Min institutional volume
    volume_breakout_mult: float = 2.0  # Volume breakout multiplier
    adl_ma_window: int = 50        # ADL moving average window
    vpt_change_threshold: float = 0.02  # VPT change threshold
    
    @staticmethod
    def from_dict(data: dict) -> 'VolumeConfig':
        volume_data = data.get('volume', {})
        return VolumeConfig(
            obv_ma_window=volume_data.get('obv_ma_window', 20),
            mfi_period=volume_data.get('mfi_period', 14),
            mfi_oversold=volume_data.get('mfi_oversold', 20),
            mfi_overbought=volume_data.get('mfi_overbought', 80),
            cmf_fast_period=volume_data.get('cmf_fast_period', 3),
            cmf_slow_period=volume_data.get('cmf_slow_period', 10),
            vwap_period=volume_data.get('vwap_period', "D"),
            volume_profile_bins=volume_data.get('volume_profile_bins', 12),
            institutional_volume_threshold=volume_data.get('institutional_volume_threshold', 1000000),
            volume_breakout_mult=volume_data.get('volume_breakout_mult', 2.0),
            adl_ma_window=volume_data.get('adl_ma_window', 50),
            vpt_change_threshold=volume_data.get('vpt_change_threshold', 0.02)
        )

@dataclass
class RiskManagementConfig:
    max_portfolio_drawdown: float = 0.15    # 15% max portfolio drawdown
    portfolio_var_limit: float = 0.02       # 2% daily VaR limit
    correlation_threshold: float = 0.75     # Max correlation between positions
    rebalance_frequency: str = "monthly"    # Portfolio rebalancing frequency

    @staticmethod
    def from_dict(data: dict) -> 'RiskManagementConfig':
        risk_data = data.get('risk_management', {})
        return RiskManagementConfig(
            max_portfolio_drawdown=risk_data.get('max_portfolio_drawdown', 0.15),
            portfolio_var_limit=risk_data.get('portfolio_var_limit', 0.02),
            correlation_threshold=risk_data.get('correlation_threshold', 0.75),
            rebalance_frequency=risk_data.get('rebalance_frequency', "monthly")
        )

@dataclass
class LongTermStrategyConfig:
    enabled: bool = True
    # Moving Averages
    sma_window: int = 50
    sma_long_window: int = 200
    
    # Fundamentals
    pe_ratio_max: float = 25.0
    pb_ratio_max: float = 3.0
    dividend_yield_min: float = 0.02
    market_cap_min: int = 10_000_000_000
    
    # Risk Management
    risk_per_trade: float = 0.01
    max_position_size: float = 0.15
    atr_period: int = 14
    atr_stop_multiplier: float = 3.0
    trailing_stop: bool = True
    trailing_stop_atr: float = 4.0
    
    # Position Entry/Exit
    entry_threshold: float = 0.02
    exit_threshold: float = 0.02
    holding_period_min: int = 30
    sell_score_threshold: float = 0.40 # composite exit gating
    # Composite scoring parameters (optional)
    # Component toggles (allow disabling individual pieces)
    enable_trend_component: bool = True
    enable_ema_component: bool = True
    enable_mom_component: bool = True
    enable_roc_component: bool = True
    enable_adx_component: bool = True
    composite_trend_weight: float = 0.5
    composite_adl_weight: float = 0.3
    composite_volume_weight: float = 0.2
    composite_entry_threshold: float = 0.50
    composite_persistent_days: int = 5
    golden_volume_confirm_factor: float = 1.2
    value_min_fraction: float = 0.5  # fraction of value criteria that must pass
    # Newly externalized technical confirmation & sizing parameters
    confirm_window: int = 3              # bars after golden cross to confirm
    aroon_up_min: int = 60               # minimum aroon up for confirmation
    aroon_down_max: int = 60             # maximum aroon down for confirmation
    max_position_fraction: float = 0.15  # explicit capital fraction cap (mirrors max_position_size)
    use_fraction_cap: bool = True        # toggle to use max_position_fraction over max_position_size * capital
    
    @staticmethod
    def from_dict(data: dict) -> 'LongTermStrategyConfig':
        strategy_data = data.get('strategies', {}).get('long_term', {})
        return LongTermStrategyConfig(
            enabled=strategy_data.get('enabled', True),
            sma_window=strategy_data.get('sma_window', 50),
            sma_long_window=strategy_data.get('sma_long_window', 200),
            pe_ratio_max=strategy_data.get('pe_ratio_max', 25.0),
            pb_ratio_max=strategy_data.get('pb_ratio_max', 3.0),
            dividend_yield_min=strategy_data.get('dividend_yield_min', 0.02),
            market_cap_min=strategy_data.get('market_cap_min', 10_000_000_000),
            risk_per_trade=strategy_data.get('risk_per_trade', 0.01),
            max_position_size=strategy_data.get('max_position_size', 0.15),
            atr_period=strategy_data.get('atr_period', 14),
            atr_stop_multiplier=strategy_data.get('atr_stop_multiplier', 3.0),
            trailing_stop=strategy_data.get('trailing_stop', True),
            trailing_stop_atr=strategy_data.get('trailing_stop_atr', 4.0),
            entry_threshold=strategy_data.get('entry_threshold', 0.02),
            exit_threshold=strategy_data.get('exit_threshold', 0.02),
            holding_period_min=strategy_data.get('holding_period_min', 30),
            composite_trend_weight=strategy_data.get('composite_trend_weight', 0.5),
            composite_adl_weight=strategy_data.get('composite_adl_weight', 0.3),
            composite_volume_weight=strategy_data.get('composite_volume_weight', 0.2),
            composite_entry_threshold=strategy_data.get('composite_entry_threshold', 0.50),
            composite_persistent_days=strategy_data.get('composite_persistent_days', 5),
            golden_volume_confirm_factor=strategy_data.get('golden_volume_confirm_factor', 1.2),
            value_min_fraction=strategy_data.get('value_min_fraction', 0.5),
            confirm_window=strategy_data.get('confirm_window', 3),
            aroon_up_min=strategy_data.get('aroon_up_min', 60),
            aroon_down_max=strategy_data.get('aroon_down_max', 60),
            max_position_fraction=strategy_data.get('max_position_fraction', strategy_data.get('max_position_size', 0.15)),
            use_fraction_cap=strategy_data.get('use_fraction_cap', True)
        )

@dataclass
class ShortTermStrategyConfig:
    enabled: bool = True
    # Moving Averages
    sma_window: int = 20
    sma_long_window: int = 50
    
    # RSI Parameters
    rsi_window: int = 14
    rsi_oversold: int = 40
    rsi_overbought: int = 60
    
    # Bollinger Bands
    bb_period: int = 20
    bb_devfactor: float = 2.0
    
    # Volume Analysis
    volume_factor: float = 1.2
    volume_ma_window: int = 20
    
    # Risk Management
    risk_per_trade: float = 0.02
    max_position_size: float = 0.10
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0
    trailing_stop: bool = True
    trailing_stop_atr: float = 2.5
    # Fractional cap (new)
    max_position_fraction: float = 0.10
    use_fraction_cap: bool = True
    # Weighted signal framework parameters
    buy_score_threshold: float = 0.50
    sell_score_threshold: float = 0.50
    hysteresis_buffer: float = 0.15  # buffer to avoid flip-flop
    enable_weighted_signals: bool = True
    # Adaptive thresholding & debug
    adaptive_thresholds: bool = True              # enable percentile-based dynamic thresholds
    adaptive_lookback: int = 120                  # bars for percentile computation
    adaptive_buy_pct: float = 70.0                # percentile of net distribution for long trigger
    adaptive_sell_pct: float = 70.0               # percentile (absolute or negative side) for exit
    adaptive_min_obs: int = 60                    # minimum observations before enabling adaptivity
    debug_last_bar_components: bool = False       # dump component contributions for last processed bar
    enable_fallback: bool = True                  # allow SMA crossover fallback if weighted nets dead
    # Trade lifecycle controls
    min_hold_bars: int = 5                        # minimum bars to hold a position before normal exit
    cooldown_bars_after_exit: int = 3             # bars to wait after exiting before a new entry
    # Advanced confirmation / filtering
    neutral_band: float = 0.15                    # absolute net score dead zone (no entries)
    min_bull_components: int = 2                  # minimum bullish components active for entry
    entry_confirm_bars: int = 2                   # consecutive bars satisfying entry to trigger
    exit_confirm_bars: int = 1                    # consecutive bars satisfying exit to trigger
    enable_zscore_filter: bool = True             # gate entries by net score z-score
    zscore_lookback: int = 50                     # lookback for z-score computation
    zscore_entry: float = 0.75                    # min z-score for long entry
    zscore_exit: float = -0.25                    # optional z threshold for early exit (if below)
    
    @staticmethod
    def from_dict(data: dict) -> 'ShortTermStrategyConfig':
        strategy_data = data.get('strategies', {}).get('short_term', {})
        return ShortTermStrategyConfig(
            enabled=strategy_data.get('enabled', True),
            sma_window=strategy_data.get('sma_window', 20),
            sma_long_window=strategy_data.get('sma_long_window', 50),
            rsi_window=strategy_data.get('rsi_window', 14),
            rsi_oversold=strategy_data.get('rsi_oversold', 40),
            rsi_overbought=strategy_data.get('rsi_overbought', 60),
            bb_period=strategy_data.get('bb_period', 20),
            bb_devfactor=strategy_data.get('bb_devfactor', 2.0),
            volume_factor=strategy_data.get('volume_factor', 1.2),
            volume_ma_window=strategy_data.get('volume_ma_window', 20),
            risk_per_trade=strategy_data.get('risk_per_trade', 0.02),
            max_position_size=strategy_data.get('max_position_size', 0.10),
            atr_period=strategy_data.get('atr_period', 14),
            atr_stop_multiplier=strategy_data.get('atr_stop_multiplier', 2.0),
            trailing_stop=strategy_data.get('trailing_stop', True),
            trailing_stop_atr=strategy_data.get('trailing_stop_atr', 2.5),
            max_position_fraction=strategy_data.get('max_position_fraction', strategy_data.get('max_position_size', 0.10)),
            use_fraction_cap=strategy_data.get('use_fraction_cap', True),
            buy_score_threshold=strategy_data.get('buy_score_threshold', 0.60),
            sell_score_threshold=strategy_data.get('sell_score_threshold', 0.40),
            hysteresis_buffer=strategy_data.get('hysteresis_buffer', 0.25),
            enable_weighted_signals=strategy_data.get('enable_weighted_signals', True)
            ,adaptive_thresholds=strategy_data.get('adaptive_thresholds', True)
            ,adaptive_lookback=strategy_data.get('adaptive_lookback', 120)
            ,adaptive_buy_pct=strategy_data.get('adaptive_buy_pct', 85.0)
            ,adaptive_sell_pct=strategy_data.get('adaptive_sell_pct', 55.0)
            ,adaptive_min_obs=strategy_data.get('adaptive_min_obs', 60)
            ,debug_last_bar_components=strategy_data.get('debug_last_bar_components', False)
            ,enable_fallback=strategy_data.get('enable_fallback', False)
            ,min_hold_bars=strategy_data.get('min_hold_bars', 5)
            ,cooldown_bars_after_exit=strategy_data.get('cooldown_bars_after_exit', 3)
            ,neutral_band=strategy_data.get('neutral_band', 0.15)
            ,min_bull_components=strategy_data.get('min_bull_components', 2)
            ,entry_confirm_bars=strategy_data.get('entry_confirm_bars', 2)
            ,exit_confirm_bars=strategy_data.get('exit_confirm_bars', 1)
            ,enable_zscore_filter=strategy_data.get('enable_zscore_filter', True)
            ,zscore_lookback=strategy_data.get('zscore_lookback', 50)
            ,zscore_entry=strategy_data.get('zscore_entry', 0.75)
            ,zscore_exit=strategy_data.get('zscore_exit', -0.25)
        )

@dataclass
class TalibStrategyConfig:
    enabled: bool = True
    sma_window: int = 20
    sma_long_window: int = 50
    rsi_window: int = 14
    rsi_oversold: int = 40
    rsi_overbought: int = 60
    risk_per_trade: float = 0.01
    max_position_size: float = 0.10
    max_position_fraction: float = 0.20  # optional explicit fraction cap used in sizing if provided
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0
    trailing_stop: bool = True
    enable_neutral: bool = True
    bull_confirm_bars: int = 1
    trail_atr_multiplier: float = 2.0
    enable_trailing_atr: bool = True
    # New composite / momentum options (no MACD / Bollinger / Stoch / OBV / Aroon per user request)
    enable_composite: bool = True
    ema_window: int = 21
    ema_long_window: int = 55
    momentum_window: int = 10       # MOM window
    roc_window: int = 10            # Rate of Change window
    adx_window: int = 14            # ADX trend strength window
    buy_score_threshold: float = 0.58  # tuned composite entry (slightly earlier)
    sell_score_threshold: float = 0.42 # tuned composite exit (slightly more tolerance)
    min_components: int = 2           # minimum bullish component count
    # Component toggles
    enable_trend_component: bool = True
    enable_ema_component: bool = True
    enable_mom_component: bool = True
    enable_roc_component: bool = True
    enable_adx_component: bool = True
    # Advanced position sizing controls
    position_size_scale_with_score: bool = True   # scale size by composite score strength
    score_min_risk_at: float = 0.40               # composite score at/below -> floor fraction
    score_full_risk_at: float = 0.80              # composite score at/above -> full base size
    position_score_floor_fraction: float = 0.25   # fraction of base size at low score
    max_atr_pct: float = 0.05                     # if ATR/price exceeds this, downscale size
    min_position_shares: int = 1                  # never allocate below this if any position
    # Partial exit controls
    # Partial exit controls
    enable_partial_exits: bool = False
    partial_exit_reduction: float = 0.50
    partial_exit_score_drop: float = 0.18  # require larger deterioration before trim
    # Signal prioritization & calibration
    component_weights: dict | None = None  # base weights per component
    high_priority_components: list | None = None
    override_threshold: float = 0.60       # weighted bullish share of high-priority comps to override
    min_confidence: float = 0.32           # tuned confidence floor
    full_confidence: float = 0.78          # reach full size slightly earlier
    enable_dynamic_calibration: bool = True  # adapt weights by component predictive win rate
    calibration_update_interval: int = 1     # evaluate reliability every bar (lightweight)
    reliability_smoothing: float = 0.18       # slightly faster adaptation
    # Extended calibration & confidence sizing
    warmup_bars_for_reliability: int = 40
    reliability_floor: float = 0.30
    reliability_regression_to_mean: float = 0.02
    instability_lookback: int = 15
    instability_penalty: float = 0.35        # slightly softer penalty
    micro_move_threshold: float = 0.002   # ignore returns smaller than this when updating correctness
    use_confidence_for_sizing: bool = True
    confidence_blend_weight: float = 0.55  # tilt toward raw composite for responsiveness
    confidence_floor_fraction: float = 0.25  # floor risk scaling fraction (can override position_score_floor_fraction)
    high_priority_size_boost: float = 0.15   # tempered boost to avoid oversizing
    enable_confidence_partial_exit: bool = True  # use confidence drop instead of composite
    # Drawdown & volatility gating
    enable_drawdown_dampening: bool = True
    drawdown_dampening_threshold: float = 0.08  # start dampening earlier
    drawdown_dampening_factor: float = 0.55     # slightly less severe dampening
    enable_volatility_gate: bool = True
    vol_gate_atr_pct: float = 0.075             # slightly tighter volatility gate
    vol_gate_confidence_penalty: float = 0.08   # gentler penalty
    # Reliability pruning (optional adaptive component exclusion)
    enable_reliability_pruning: bool = True
    reliability_prune_threshold: float = 0.40
    reliability_prune_consecutive: int = 5
    reactivation_threshold: float = 0.52
    reactivation_consecutive: int = 4
    min_history_for_prune: int = 80
    max_pruned_fraction: float = 0.50
    prune_grace_bars: int = 20
    instability_prune_threshold: float | None = None  # set to e.g. 0.55 to enable instability-driven pruning
    protected_components: list | None = None

    def validate(self) -> 'TalibStrategyConfig':
        """Validate and sanitize inter-dependent pruning & calibration parameters.

        Adjusts obviously unsafe values and raises ValueError for irreconcilable ones.
        Returns self for fluent usage.
        """
        # Ensure thresholds make sense
        if self.enable_reliability_pruning:
            if self.reactivation_threshold <= self.reliability_prune_threshold:
                # Nudge reactivation threshold above prune threshold
                self.reactivation_threshold = min(0.99, self.reliability_prune_threshold + 0.05)
            # Clamp fractions
            self.max_pruned_fraction = max(0.0, min(self.max_pruned_fraction, 0.90))
            # Minimum history should exceed warmup
            if self.min_history_for_prune <= self.warmup_bars_for_reliability:
                self.min_history_for_prune = self.warmup_bars_for_reliability + 20
            # Positive consecutive counts
            if self.reliability_prune_consecutive < 1:
                self.reliability_prune_consecutive = 1
            if self.reactivation_consecutive < 1:
                self.reactivation_consecutive = 1
            # Grace bars at least 0
            if self.prune_grace_bars < 0:
                self.prune_grace_bars = 0
            # Instability threshold sanity (if provided)
            if self.instability_prune_threshold is not None:
                self.instability_prune_threshold = max(0.0, min(self.instability_prune_threshold, 1.0))
        return self

    @staticmethod
    def from_dict(data: dict) -> 'TalibStrategyConfig':
        strategy_data = data.get('strategies', {}).get('talib', {})
        cfg = TalibStrategyConfig(
            enabled=strategy_data.get('enabled', True),
            sma_window=strategy_data.get('sma_window', 20),
            sma_long_window=strategy_data.get('sma_long_window', 50),
            rsi_window=strategy_data.get('rsi_window', 14),
            rsi_oversold=strategy_data.get('rsi_oversold', 40),
            rsi_overbought=strategy_data.get('rsi_overbought', 60),
            risk_per_trade=strategy_data.get('risk_per_trade', 0.01),
            max_position_size=strategy_data.get('max_position_size', 0.10),
            max_position_fraction=strategy_data.get('max_position_fraction', 0.20),
            atr_period=strategy_data.get('atr_period', 14),
            atr_stop_multiplier=strategy_data.get('atr_stop_multiplier', 2.0),
            trailing_stop=strategy_data.get('trailing_stop', True),
            enable_neutral=strategy_data.get('enable_neutral', True)
            ,bull_confirm_bars=strategy_data.get('bull_confirm_bars', 1)
            ,trail_atr_multiplier=strategy_data.get('trail_atr_multiplier', 2.0)
            ,enable_trailing_atr=strategy_data.get('enable_trailing_atr', True)
            ,enable_composite=strategy_data.get('enable_composite', True)
            ,ema_window=strategy_data.get('ema_window', 21)
            ,ema_long_window=strategy_data.get('ema_long_window', 55)
            ,momentum_window=strategy_data.get('momentum_window', 10)
            ,roc_window=strategy_data.get('roc_window', 10)
            ,adx_window=strategy_data.get('adx_window', 14)
            ,buy_score_threshold=strategy_data.get('buy_score_threshold', 0.58)
            ,sell_score_threshold=strategy_data.get('sell_score_threshold', 0.42)
            ,min_components=strategy_data.get('min_components', 2)
            ,enable_trend_component=strategy_data.get('enable_trend_component', True)
            ,enable_ema_component=strategy_data.get('enable_ema_component', True)
            ,enable_mom_component=strategy_data.get('enable_mom_component', True)
            ,enable_roc_component=strategy_data.get('enable_roc_component', True)
            ,enable_adx_component=strategy_data.get('enable_adx_component', True)
            ,position_size_scale_with_score=strategy_data.get('position_size_scale_with_score', True)
            ,score_min_risk_at=strategy_data.get('score_min_risk_at', 0.40)
            ,score_full_risk_at=strategy_data.get('score_full_risk_at', 0.80)
            ,position_score_floor_fraction=strategy_data.get('position_score_floor_fraction', 0.25)
            ,max_atr_pct=strategy_data.get('max_atr_pct', 0.05)
            ,min_position_shares=strategy_data.get('min_position_shares', 1)
            ,enable_partial_exits=strategy_data.get('enable_partial_exits', False)
            ,partial_exit_reduction=strategy_data.get('partial_exit_reduction', 0.50)
            ,partial_exit_score_drop=strategy_data.get('partial_exit_score_drop', 0.18)
            ,component_weights=strategy_data.get('component_weights')
            ,high_priority_components=strategy_data.get('high_priority_components')
            ,override_threshold=strategy_data.get('override_threshold', 0.60)
            ,min_confidence=strategy_data.get('min_confidence', 0.32)
            ,full_confidence=strategy_data.get('full_confidence', 0.78)
            ,enable_dynamic_calibration=strategy_data.get('enable_dynamic_calibration', True)
            ,calibration_update_interval=strategy_data.get('calibration_update_interval', 1)
            ,reliability_smoothing=strategy_data.get('reliability_smoothing', 0.18)
            ,warmup_bars_for_reliability=strategy_data.get('warmup_bars_for_reliability', 40)
            ,reliability_floor=strategy_data.get('reliability_floor', 0.30)
            ,reliability_regression_to_mean=strategy_data.get('reliability_regression_to_mean', 0.02)
            ,instability_lookback=strategy_data.get('instability_lookback', 15)
            ,instability_penalty=strategy_data.get('instability_penalty', 0.35)
            ,micro_move_threshold=strategy_data.get('micro_move_threshold', 0.002)
            ,use_confidence_for_sizing=strategy_data.get('use_confidence_for_sizing', True)
            ,confidence_blend_weight=strategy_data.get('confidence_blend_weight', 0.55)
            ,confidence_floor_fraction=strategy_data.get('confidence_floor_fraction', 0.25)
            ,high_priority_size_boost=strategy_data.get('high_priority_size_boost', 0.15)
            ,enable_confidence_partial_exit=strategy_data.get('enable_confidence_partial_exit', True)
            ,enable_drawdown_dampening=strategy_data.get('enable_drawdown_dampening', True)
            ,drawdown_dampening_threshold=strategy_data.get('drawdown_dampening_threshold', 0.08)
            ,drawdown_dampening_factor=strategy_data.get('drawdown_dampening_factor', 0.55)
            ,enable_volatility_gate=strategy_data.get('enable_volatility_gate', True)
            ,vol_gate_atr_pct=strategy_data.get('vol_gate_atr_pct', 0.075)
            ,vol_gate_confidence_penalty=strategy_data.get('vol_gate_confidence_penalty', 0.08)
            ,enable_reliability_pruning=strategy_data.get('enable_reliability_pruning', False)
            ,reliability_prune_threshold=strategy_data.get('reliability_prune_threshold', 0.40)
            ,reliability_prune_consecutive=strategy_data.get('reliability_prune_consecutive', 5)
            ,reactivation_threshold=strategy_data.get('reactivation_threshold', 0.52)
            ,reactivation_consecutive=strategy_data.get('reactivation_consecutive', 4)
            ,min_history_for_prune=strategy_data.get('min_history_for_prune', 80)
            ,max_pruned_fraction=strategy_data.get('max_pruned_fraction', 0.50)
            ,prune_grace_bars=strategy_data.get('prune_grace_bars', 20)
            ,instability_prune_threshold=strategy_data.get('instability_prune_threshold')
            ,protected_components=strategy_data.get('protected_components')
        )
        return cfg.validate()

@dataclass
class DayTradingStrategyConfig:
    enabled: bool = True
    # Indicator Parameters
    rsi_window: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_dev: float = 2.0
    ema_period: int = 20
    volume_ma_window: int = 20
    # Composite Scoring
    enable_composite: bool = True
    buy_score_threshold: float = 0.60
    sell_score_threshold: float = 0.40
    min_components: int = 2
    component_weights: dict = field(default_factory=lambda: {'rsi': 1.0, 'macd': 1.0, 'bb': 1.0, 'ema': 1.0, 'volume': 1.0})
    high_priority_components: list = field(default_factory=list)
    override_threshold: float = 0.60
    # Component Enables
    enable_rsi_component: bool = True
    enable_macd_component: bool = True
    enable_bb_component: bool = True
    enable_ema_component: bool = True
    enable_volume_component: bool = True
    # Reliability and Pruning
    enable_reliability_pruning: bool = True
    reliability_prune_threshold: float = 0.40
    reliability_prune_consecutive: int = 5
    reactivation_threshold: float = 0.52
    reactivation_consecutive: int = 4
    min_history_for_prune: int = 80
    max_pruned_fraction: float = 0.50
    prune_grace_bars: int = 20
    instability_prune_threshold: float | None = None
    protected_components: list | None = None
    reliability_smoothing: float = 0.18
    warmup_bars_for_reliability: int = 40
    reliability_floor: float = 0.30
    reliability_regression_to_mean: float = 0.02
    instability_lookback: int = 15
    instability_penalty: float = 0.35
    micro_move_threshold: float = 0.002
    # Sizing and Risk
    risk_per_trade: float = 0.01
    max_position_fraction: float = 0.20
    min_position_shares: int = 1
    atr_stop_multiplier: float = 2.0
    use_confidence_for_sizing: bool = True
    confidence_blend_weight: float = 0.50
    min_confidence: float = 0.40
    full_confidence: float = 0.80
    confidence_floor_fraction: float = 0.25
    enable_drawdown_dampening: bool = True
    drawdown_dampening_threshold: float = 0.10
    drawdown_dampening_factor: float = 0.50
    enable_volatility_gate: bool = True
    vol_gate_atr_pct: float = 0.075
    vol_gate_confidence_penalty: float = 0.08

    @staticmethod
    def from_dict(data: dict) -> 'DayTradingStrategyConfig':
        strategy_data = data.get('strategies', {}).get('day_trading', {})
        cfg = DayTradingStrategyConfig(
            enabled=strategy_data.get('enabled', True),
            rsi_window=strategy_data.get('rsi_window', 14),
            rsi_oversold=strategy_data.get('rsi_oversold', 30),
            rsi_overbought=strategy_data.get('rsi_overbought', 70),
            macd_fast=strategy_data.get('macd_fast', 12),
            macd_slow=strategy_data.get('macd_slow', 26),
            macd_signal=strategy_data.get('macd_signal', 9),
            bb_period=strategy_data.get('bb_period', 20),
            bb_dev=strategy_data.get('bb_dev', 2.0),
            ema_period=strategy_data.get('ema_period', 20),
            volume_ma_window=strategy_data.get('volume_ma_window', 20),
            enable_composite=strategy_data.get('enable_composite', True),
            buy_score_threshold=strategy_data.get('buy_score_threshold', 0.60),
            sell_score_threshold=strategy_data.get('sell_score_threshold', 0.40),
            min_components=strategy_data.get('min_components', 2),
            component_weights=strategy_data.get('component_weights', {'rsi': 1.0, 'macd': 1.0, 'bb': 1.0, 'ema': 1.0, 'volume': 1.0}),
            high_priority_components=strategy_data.get('high_priority_components', []),
            override_threshold=strategy_data.get('override_threshold', 0.60),
            enable_rsi_component=strategy_data.get('enable_rsi_component', True),
            enable_macd_component=strategy_data.get('enable_macd_component', True),
            enable_bb_component=strategy_data.get('enable_bb_component', True),
            enable_ema_component=strategy_data.get('enable_ema_component', True),
            enable_volume_component=strategy_data.get('enable_volume_component', True),
            enable_reliability_pruning=strategy_data.get('enable_reliability_pruning', True),
            reliability_prune_threshold=strategy_data.get('reliability_prune_threshold', 0.40),
            reliability_prune_consecutive=strategy_data.get('reliability_prune_consecutive', 5),
            reactivation_threshold=strategy_data.get('reactivation_threshold', 0.52),
            reactivation_consecutive=strategy_data.get('reactivation_consecutive', 4),
            min_history_for_prune=strategy_data.get('min_history_for_prune', 80),
            max_pruned_fraction=strategy_data.get('max_pruned_fraction', 0.50),
            prune_grace_bars=strategy_data.get('prune_grace_bars', 20),
            instability_prune_threshold=strategy_data.get('instability_prune_threshold'),
            protected_components=strategy_data.get('protected_components'),
            reliability_smoothing=strategy_data.get('reliability_smoothing', 0.18),
            warmup_bars_for_reliability=strategy_data.get('warmup_bars_for_reliability', 40),
            reliability_floor=strategy_data.get('reliability_floor', 0.30),
            reliability_regression_to_mean=strategy_data.get('reliability_regression_to_mean', 0.02),
            instability_lookback=strategy_data.get('instability_lookback', 15),
            instability_penalty=strategy_data.get('instability_penalty', 0.35),
            micro_move_threshold=strategy_data.get('micro_move_threshold', 0.002),
            risk_per_trade=strategy_data.get('risk_per_trade', 0.01),
            max_position_fraction=strategy_data.get('max_position_fraction', 0.20),
            min_position_shares=strategy_data.get('min_position_shares', 1),
            atr_stop_multiplier=strategy_data.get('atr_stop_multiplier', 2.0),
            use_confidence_for_sizing=strategy_data.get('use_confidence_for_sizing', True),
            confidence_blend_weight=strategy_data.get('confidence_blend_weight', 0.50),
            min_confidence=strategy_data.get('min_confidence', 0.40),
            full_confidence=strategy_data.get('full_confidence', 0.80),
            confidence_floor_fraction=strategy_data.get('confidence_floor_fraction', 0.25),
            enable_drawdown_dampening=strategy_data.get('enable_drawdown_dampening', True),
            drawdown_dampening_threshold=strategy_data.get('drawdown_dampening_threshold', 0.10),
            drawdown_dampening_factor=strategy_data.get('drawdown_dampening_factor', 0.50),
            enable_volatility_gate=strategy_data.get('enable_volatility_gate', True),
            vol_gate_atr_pct=strategy_data.get('vol_gate_atr_pct', 0.075),
            vol_gate_confidence_penalty=strategy_data.get('vol_gate_confidence_penalty', 0.08)
        )
        return cfg.validate()

    def validate(self) -> 'DayTradingStrategyConfig':
        if self.enable_reliability_pruning:
            if self.reactivation_threshold <= self.reliability_prune_threshold:
                self.reactivation_threshold = min(0.99, self.reliability_prune_threshold + 0.05)
            self.max_pruned_fraction = max(0.0, min(self.max_pruned_fraction, 0.90))
            if self.min_history_for_prune <= self.warmup_bars_for_reliability:
                self.min_history_for_prune = self.warmup_bars_for_reliability + 20
            if self.reliability_prune_consecutive < 1:
                self.reliability_prune_consecutive = 1
            if self.reactivation_consecutive < 1:
                self.reactivation_consecutive = 1
            if self.prune_grace_bars < 0:
                self.prune_grace_bars = 0
        return self

@dataclass
class OptionsStrategyConfig:
    enabled: bool = True
    # Strategy Parameters
    strategy_type: str = "iron_condor"
    days_to_expiry_min: int = 30
    days_to_expiry_max: int = 45
    
    # Greeks Thresholds
    delta_short: float = 0.16
    delta_long: float = 0.08
    theta_min: float = 0.1
    vega_max: float = 0.5
    
    # Profit/Loss Parameters
    profit_target_pct: float = 0.50
    stop_loss_pct: float = 0.75
    max_loss_per_trade: float = 0.01
    
    # Position Management
    max_position_size: float = 0.05
    portfolio_margin_limit: float = 0.15
    max_contracts_per_trade: int = 10
    
    # IV Parameters
    iv_rank_min: float = 0.3
    iv_rank_max: float = 0.8
    iv_percentile_threshold: int = 50
    # Advanced sizing / volatility controls
    risk_per_trade: float = 0.01                 # fraction of capital at risk per trade
    scale_with_vol: bool = True                  # enable dynamic size scaling by volatility regime
    vol_low_threshold: float = 0.15              # annualized volatility low boundary
    vol_high_threshold: float = 0.30             # annualized volatility high boundary
    vol_floor_fraction: float = 0.30             # size fraction floor at/above vol_high
    vol_ema_period: int = 10                     # smoothing period for realized vol
    max_position_fraction: float = 0.05          # hard cap fraction of capital in one options position
    min_position_shares: int = 1                 # minimum share equivalent (for underlying proxy)
    
    @staticmethod
    def from_dict(data: dict) -> 'OptionsStrategyConfig':
        strategy_data = data.get('strategies', {}).get('options', {})
        return OptionsStrategyConfig(
            enabled=strategy_data.get('enabled', True),
            strategy_type=strategy_data.get('strategy_type', "iron_condor"),
            days_to_expiry_min=strategy_data.get('days_to_expiry_min', 30),
            days_to_expiry_max=strategy_data.get('days_to_expiry_max', 45),
            delta_short=strategy_data.get('delta_short', 0.16),
            delta_long=strategy_data.get('delta_long', 0.08),
            theta_min=strategy_data.get('theta_min', 0.1),
            vega_max=strategy_data.get('vega_max', 0.5),
            profit_target_pct=strategy_data.get('profit_target_pct', 0.50),
            stop_loss_pct=strategy_data.get('stop_loss_pct', 0.75),
            max_loss_per_trade=strategy_data.get('max_loss_per_trade', 0.01),
            max_position_size=strategy_data.get('max_position_size', 0.05),
            portfolio_margin_limit=strategy_data.get('portfolio_margin_limit', 0.15),
            max_contracts_per_trade=strategy_data.get('max_contracts_per_trade', 10),
            iv_rank_min=strategy_data.get('iv_rank_min', 0.3),
            iv_rank_max=strategy_data.get('iv_rank_max', 0.8),
            iv_percentile_threshold=strategy_data.get('iv_percentile_threshold', 50),
            risk_per_trade=strategy_data.get('risk_per_trade', 0.01),
            scale_with_vol=strategy_data.get('scale_with_vol', True),
            vol_low_threshold=strategy_data.get('vol_low_threshold', 0.15),
            vol_high_threshold=strategy_data.get('vol_high_threshold', 0.30),
            vol_floor_fraction=strategy_data.get('vol_floor_fraction', 0.30),
            vol_ema_period=strategy_data.get('vol_ema_period', 10),
            max_position_fraction=strategy_data.get('max_position_fraction', 0.05),
            min_position_shares=strategy_data.get('min_position_shares', 1)
        )

@dataclass
class AppConfig:
    # Basic Settings
    tickers: List[str] = field(default_factory=lambda: ['AAPL'])
    start_date: str = '2023-01-01'
    end_date: str = '2023-12-31'
    initial_cash: float = 100000
    risk_free_rate: float = 0.0
    commission: float = 0.001  # 0.1% commission
    
    # Portfolio Allocation
    portfolio_allocation: dict = field(default_factory=lambda: {
        'long_term': 0.50,
        'short_term': 0.25,
        'day_trading': 0.0,
        'options': 0.25
    })
    
    # Strategy Configurations
    long_term_strategy: LongTermStrategyConfig = field(default_factory=lambda: LongTermStrategyConfig())
    short_term_strategy: ShortTermStrategyConfig = field(default_factory=lambda: ShortTermStrategyConfig())
    talib_strategy: TalibStrategyConfig = field(default_factory=lambda: TalibStrategyConfig())
    day_trading_strategy: DayTradingStrategyConfig = field(default_factory=lambda: DayTradingStrategyConfig())
    options_strategy: OptionsStrategyConfig = field(default_factory=lambda: OptionsStrategyConfig())
    
    # Global Configurations
    volume: VolumeConfig = field(default_factory=lambda: VolumeConfig())
    risk_management: RiskManagementConfig = field(default_factory=lambda: RiskManagementConfig())
    # New sections
    data_cache: 'DataCacheConfig' = field(default_factory=lambda: None)  # type: ignore
    visualization: 'VisualizationConfig' = field(default_factory=lambda: None)  # type: ignore

    @staticmethod
    def load_from_yaml(path: str = 'config.yaml'):
        """Load main config plus optional per-strategy YAML overlays.

        Strategy overlay files (if present in same directory):
          - strategy_long_term.yaml (expects top-level key 'long_term')
          - strategy_short_term.yaml (top-level 'short_term')
          - strategy_talib.yaml (top-level 'talib')
          - strategy_day_trading.yaml (top-level 'day_trading')
          - strategy_options.yaml (top-level 'options')

        Merging precedence (later overrides earlier): base config.yaml -> per-strategy file.
        """
        import os
        base_dir = os.path.dirname(path) or '.'
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        # Collect base strategies dict (may be absent)
        combined_strats = dict(data.get('strategies', {}))
        overlay_map = {
            'long_term': 'strategy_long_term.yaml',
            'short_term': 'strategy_short_term.yaml',
            'talib': 'strategy_talib.yaml',
            'day_trading': 'strategy_day_trading.yaml',
            'options': 'strategy_options.yaml'
        }
        for key, fname in overlay_map.items():
            fpath = os.path.join(base_dir, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, 'r') as sf:
                        overlay_raw = yaml.safe_load(sf) or {}
                    # overlay file can either have top-level key or be direct mapping
                    if key in overlay_raw:
                        overlay = overlay_raw.get(key, {})
                    else:
                        overlay = overlay_raw
                    base_entry = combined_strats.get(key, {})
                    merged = {**base_entry, **overlay}
                    combined_strats[key] = merged
                except Exception:
                    pass
        # Rebuild synthetic data dict for from_dict helpers
        data_merged = dict(data)
        data_merged['strategies'] = combined_strats

        # Optional sections
        data_cache_cfg = DataCacheConfig.from_dict(data_merged) if 'data_cache' in data_merged else DataCacheConfig()
        visualization_cfg = VisualizationConfig.from_dict(data_merged) if 'visualization' in data_merged else VisualizationConfig()

        return AppConfig(
            tickers=data_merged.get('tickers', ['AAPL']),
            start_date=data_merged.get('start_date', '2023-01-01'),
            end_date=data_merged.get('end_date', '2023-12-31'),
            initial_cash=data_merged.get('initial_cash', 100000),
            risk_free_rate=data_merged.get('risk_free_rate', 0.0),
            commission=data_merged.get('commission', 0.001),
            portfolio_allocation=data_merged.get('portfolio_allocation', {
                'long_term': 0.50,
                'short_term': 0.25,
                'options': 0.25
            }),
            long_term_strategy=LongTermStrategyConfig.from_dict(data_merged),
            short_term_strategy=ShortTermStrategyConfig.from_dict(data_merged),
            talib_strategy=TalibStrategyConfig.from_dict(data_merged),
            day_trading_strategy=DayTradingStrategyConfig.from_dict(data_merged),
            options_strategy=OptionsStrategyConfig.from_dict(data_merged),
            volume=VolumeConfig.from_dict(data_merged),
            risk_management=RiskManagementConfig.from_dict(data_merged),
            data_cache=data_cache_cfg,
            visualization=visualization_cfg
        )

@dataclass
class DataCacheConfig:
    enable: bool = True
    force_refresh: bool = False
    format: str = 'parquet'  # 'parquet' or 'csv'
    dir: str = 'data/cache'
    auto_adjust: bool = True
    ttl_days: int | None = None  # optional TTL for cache files

    @staticmethod
    def from_dict(data: dict) -> 'DataCacheConfig':
        cfg = data.get('data_cache', {}) or {}
        return DataCacheConfig(
            enable=cfg.get('enable', True),
            force_refresh=cfg.get('force_refresh', False),
            format=cfg.get('format', 'parquet').lower(),
            dir=cfg.get('dir', 'data/cache'),
            auto_adjust=cfg.get('auto_adjust', True),
            ttl_days=cfg.get('ttl_days')
        )

@dataclass
class VisualizationConfig:
    generate_charts: bool = True
    output_dir: str = 'charts'

    @staticmethod
    def from_dict(data: dict) -> 'VisualizationConfig':
        cfg = data.get('visualization', {}) or {}
        return VisualizationConfig(
            generate_charts=cfg.get('generate_charts', True),
            output_dir=cfg.get('output_dir', 'charts')
        )
