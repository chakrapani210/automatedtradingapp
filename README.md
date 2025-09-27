# Automated Trading App

A fully managed, modular automated trading application supporting stocks and options, with backtesting, risk management, and live trading.

## Features
- Data acquisition (yfinance)
- Strategy development (TA-Lib, pandas, custom logic)
- Backtesting (Backtrader, PyPortfolioOpt, options pricing with Optopsy/QuantLib)
- Risk management (PyPortfolioOpt, custom rules)
- Live trading (Backtrader live broker API integration)

## Setup
1. Install dependencies:
   ```
pip install -r requirements.txt
   ```
2. Run the app:
   ```
python main.py
   ```

## Structure
- `data/`: Data acquisition modules
- `strategies/`: Trading strategies
- `backtest/`: Backtesting engine
- `portfolio/`: Risk/account management
- `live/`: Live trading integration
- `utils/`: Utilities
- `analysis/`: Post-run analysis scripts (e.g., TALib performance summary)
- `analysis_output/`: Generated diagnostics (signals, component reliability, performance summaries)

## TALib Strategy Diagnostics & Analysis
After running a backtest including the TALib strategy, the following CSVs are produced in `analysis_output/`:
- `talib_strategy_signals.csv`: Per-bar indicator & composite diagnostics
- `talib_strategy_signal_changes.csv`: Compressed event stream (entries, exits, partial exits)
- `talib_component_reliability.csv`: Latest snapshot of component reliabilities and effective weights
- `talib_component_reliability_timeseries.csv`: Longitudinal reliability timeline (one row per component per calibration interval)
- `orders_diagnostics.csv`: (root) Order intent and executions with confidence & sizing context
- `talib_trades_detailed.csv`: Parsed trade list with holding period & return
- `talib_performance_summary.csv`: Aggregate trade metrics (win rate, expectancy, avg holding days, etc.)

### Generate / Refresh Performance Summary
If you re-run backtests or modify orders, regenerate summaries:

```
python analysis/talib_performance_summary.py
```

### Key Metrics
- Expectancy: WinRate * AvgWin + (1 - WinRate) * AvgLoss
- Reliability Timeseries: Use to assess stability / predictive drift of each component and inform pruning or gating.

## DayTrading Strategy
A short-term trading strategy optimized for intraday signals using RSI, MACD, Bollinger Bands, and Volume indicators.

### Components
- **RSI Component**: Triggers buy when RSI < 30 (oversold), sell when RSI > 70 (overbought).
- **MACD Component**: Buy when MACD > signal line, sell when MACD < signal line.
- **Bollinger Bands Component**: Buy when price < lower band, sell when price > upper band.
- **Volume Component**: Buy when volume > volume MA (high volume confirmation).

### Configuration
Located in `config.yaml` under `strategies.day_trading` or via overlay file `strategy_day_trading.yaml`.

Key parameters:
- `rsi_window`: RSI period (default 14)
- `macd_fast/slow/signal`: MACD periods (default 12/26/9)
- `bb_period`: Bollinger Bands period (default 20)
- `volume_ma_window`: Volume MA period (default 20)
- `buy_score_threshold`: Minimum composite score for buy (default 0.60)
- `sell_score_threshold`: Maximum composite score for sell (default 0.40)
- `enable_reliability_pruning`: Adaptive component exclusion (default true)
- `risk_per_trade`: Base risk fraction (default 0.01)
- `max_position_fraction`: Max capital per trade (default 0.20)

### Diagnostics Output
After backtesting, generates CSVs in `analysis_output/`:
- `day_trading_strategy_signals.csv`: Per-bar diagnostics (RSI, MACD, BB, composite score, signal)
- `day_trading_strategy_signal_changes.csv`: Entry/exit events
- `day_trading_component_reliability.csv`: Component reliability metrics
- `day_trading_component_reliability_timeseries.csv`: Longitudinal reliability data

### Usage
Enable in portfolio allocation:
```yaml
portfolio_allocation:
  long_term: 0.50
  short_term: 0.25
  day_trading: 0.15  # Allocate 15% to day trading
  options: 0.10
```

## Strategy Utility Layer (Shared Mechanics)
The `utils/strategy_utils.py` module centralizes reusable mechanics consumed by `TALibStrategy` and intended for future strategies.

### Reliability & Dynamic Weighting
Function: `dynamic_reliability_weighting`
Purpose: Adjust component contribution based on an EMA of recent correctness, while penalizing instability (frequent flips) and gently regressing toward neutrality.
Key Config Fields:
- `warmup_bars_for_reliability`: Bars to accumulate before first calibration.
- `calibration_update_interval`: Calibration cadence (e.g. every bar, every 5 bars).
- `reliability_smoothing`: EMA alpha for correctness updates.
- `micro_move_threshold`: Ignore tiny price moves when scoring correctness.
- `instability_lookback`: Rolling window length for flip rate.
- `instability_penalty`: Multiplier intensity reducing weight for unstable components.
- `reliability_floor`: Minimum reliability to avoid total starvation.
Outputs: Updated composite score series and longitudinal entries (component, reliability, effective_weight, instability) appended to `reliability_ts`.

### Confidence Blending
Function: `compute_confidence`
Blends: Raw composite score with reliability-weighted instantaneous component correctness. Controlled by `confidence_blend_weight`.
Used For: Position sizing scaling, partial exit logic.

### High-Priority Overrides
Functions: `compute_high_priority_override`, `compute_high_priority_override_series`
Allows a subset of components to force (or accelerate) entries when their collective bullish share crosses `override_threshold`.

### Buy/Sell Gating
Functions: `compute_gated_buy`, `compute_composite_sell`
Combine composite score thresholds, minimum bullish component counts, and confirmation streak logic to form robust entry/exit eligibility.

### Partial Exit Logic
Function: `evaluate_partial_exit`
Triggers a trim signal (encoded as 2) when composite or confidence drops by a configured delta (`partial_exit_score_drop`) from its peak but remains above full-exit thresholds.

### Trailing ATR Stops
Function: `update_trailing_atr`
Maintains a ratcheting stop using ATR multiples after entry; returns exit trigger when price pierces stop.

### Volatility Gate
Function: `apply_volatility_gate` (NEW abstraction)
Purpose: Penalize confidence/score when normalized volatility (ATR / price) exceeds `vol_gate_atr_pct`. Deducts `vol_gate_confidence_penalty` while preserving bounds [0,1]. Centralizes logic previously inline in sizing.

### Reliability Pruning
Enabled by default. Configuration keys (under `strategies.talib`):
```
enable_reliability_pruning: true
reliability_prune_threshold: 0.40
reliability_prune_consecutive: 5
reactivation_threshold: 0.52
reactivation_consecutive: 4
min_history_for_prune: 80
prune_grace_bars: 20
max_pruned_fraction: 0.50
instability_prune_threshold: null   # set e.g. 0.55 to incorporate instability penalty as trigger
protected_components: []            # components never pruned
```
Tracking fields appended to `talib_component_reliability_timeseries.csv`: `pruned`, `below_streak`, `reactivate_streak`.

### Risk Fraction Scaling
Function: `scale_risk_fraction`
Maps confidence/score into a scaled risk fraction between a floor and full allocation thresholds (`min_confidence`/`full_confidence`).

### Drawdown Dampening
Function: `apply_drawdown_dampening`
Reduces new position risk fraction when current price is under peak by more than `drawdown_dampening_threshold`, multiplying by `drawdown_dampening_factor`.

### Realized Volatility Diagnostics
Function: `compute_realized_vol`
Calculates rolling close-to-close annualized volatility for recent bars (used only for logging & analysis).

## Testing Additions
Recent tests cover:
- Reliability calibration & instability penalty (`test_dynamic_reliability.py`).
- Gating and composite sell logic.
- Drawdown dampening scaling behavior.
- Volatility gate indirectly via sizing adjustments (confidence penalty path).

## Extension Guidelines
When adding a new component (e.g., volume divergence):
1. Add its boolean signal column to the components DataFrame.
2. (Optional) Include a static weight via `component_weights` config.
3. Run with reliability enabled; inspect `talib_component_reliability_timeseries.csv` for drift.
4. Consider adding it to `high_priority_components` only after observing stable positive reliability.

## Planned Enhancements
- Separate volatility gate unit test to explicitly assert penalty behavior.
- Adaptive instability penalty (higher during regime shifts).
- Optional pruning of chronically low-reliability components.

