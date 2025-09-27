import argparse
import traceback
import os
import yfinance as yf
import pandas as pd
import backtrader as bt

from data.data_acquisition import DataAcquisition
from utils.app_config import AppConfig

from visualization.plot_signals import build_signals_dataframe
from strategies.long_term_strategy import LongTermValueStrategy
from strategies.short_term_strategy import ShortTermTechnicalStrategy
from strategies.options_strategy import OptionsStrategy
from strategies.ta_strategy import TALibStrategy
from strategies.day_trading_strategy import DayTradingStrategy
from strategies.multi_asset_strategy import MultiAssetStrategy
from utils.performance import analyze_performance


def parse_cli_args():
    p = argparse.ArgumentParser(description="Run automated trading backtest")
    p.add_argument('--config', default='config.yaml', help='Path to config YAML')
    p.add_argument('--force-refresh', action='store_true', help='Force data cache refresh')
    p.add_argument('--no-charts', action='store_true', help='Disable chart generation regardless of config')
    p.add_argument('--ttl-days', type=int, default=None, help='Override cache TTL days (default from config)')
    p.add_argument('--start-date', type=str, default=None, help='Override start date (YYYY-MM-DD)')
    p.add_argument('--end-date', type=str, default=None, help='Override end date (YYYY-MM-DD)')
    p.add_argument('--only', type=str, default=None, help='Comma-separated list of strategies to run (overrides enabled flags). Options: long_term,short_term,day_trading,options,talib')
    p.add_argument('--interval', type=str, default=None, help='Explicit data interval override (e.g. 1m,5m,15m,1h,1d). Useful to run short_term on intraday data when day_trading is disabled.')
    return p.parse_args()


def main():
    args = parse_cli_args()
    import sys
    log_file = open('backtest_log.txt', 'w', buffering=1)

    # Load configuration (YAML first)
    try:
        config = AppConfig.load_from_yaml(args.config)
    except Exception as e:
        print(f"Could not load config file '{args.config}'; using defaults. Error: {e}")
        config = AppConfig()

    print("\n=== Automated Trading System Starting ===")
    print(f"Config file: {args.config}")
    print("\nConfiguration:")
    print(f"Tickers: {config.tickers}")
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Initial Cash: ${config.initial_cash:,.2f}")
    if getattr(config, 'data_cache', None):
        dc = config.data_cache
        print(f"Data Cache: enable={dc.enable} format={dc.format} dir={dc.dir} force_refresh={dc.force_refresh}")
    if getattr(config, 'visualization', None):
        vc = config.visualization
        print(f"Visualization: generate_charts={vc.generate_charts} output_dir={vc.output_dir}")

    st = config.short_term_strategy
    lt = config.long_term_strategy

    print("\nShort-term Strategy:")
    print(f"  SMA {st.sma_window}/{st.sma_long_window} | RSI {st.rsi_window} (oversold {st.rsi_oversold} / overbought {st.rsi_overbought})")
    print(f"  Risk/MaxPos: {st.risk_per_trade*100:.1f}% / {st.max_position_size*100:.1f}%  WeightedSignals={st.enable_weighted_signals}")
    print(f"  Adaptive Thresholds: {st.adaptive_thresholds} lookback={st.adaptive_lookback}")

    print("\nLong-term Strategy:")
    print(f"  SMA {lt.sma_window}/{lt.sma_long_window}  Risk/MaxPos: {lt.risk_per_trade*100:.1f}% / {lt.max_position_size*100:.1f}%")

    print("\n=== Fetching Market Data ===")
    # Optional date overrides
    if args.start_date:
        config.start_date = args.start_date
    if args.end_date:
        config.end_date = args.end_date

        import os as _os
        if _os.getenv('LONGTERM_DEBUG', '').lower() in ('1','true','yes'):
            print("[LONGTERM_DEBUG] Environment flag detected in main")
    # Use tickers exactly as provided in config (no hard-coded additions)
    tickers = list(dict.fromkeys(config.tickers))
    print(f"Universe: {tickers}")

    dc = config.data_cache or type('X', (), {'enable': True, 'force_refresh': False, 'dir': 'data/cache', 'format': 'parquet', 'auto_adjust': True})()
    eff_force_refresh = dc.force_refresh or args.force_refresh
    ttl_days = args.ttl_days if args.ttl_days is not None else getattr(dc, 'ttl_days', None) if hasattr(dc, 'ttl_days') else None

    allocations = config.portfolio_allocation
    # Determine strategy selection (CLI overrides enabled flags if provided)
    requested = None
    if args.only:
        requested = {s.strip() for s in args.only.split(',') if s.strip()}
        print(f"CLI override: only running strategies = {sorted(requested)}")

    strategies_cfg = {}
    allocation_filtered = {}
    skipped = []

    def maybe_add(name: str, enabled_flag: bool, cls, alloc_key: str):
        if requested is not None and name not in requested:
            skipped.append((name, 'cli_filter'))
            return
        if not enabled_flag:
            skipped.append((name, 'disabled_flag'))
            return
        strategies_cfg[name] = {'class': cls, 'params': {'config': config}}
        allocation_filtered[name] = allocations.get(alloc_key, 0) * 100

    maybe_add('long_term', getattr(config.long_term_strategy, 'enabled', True), LongTermValueStrategy, 'long_term')
    maybe_add('short_term', getattr(config.short_term_strategy, 'enabled', True), ShortTermTechnicalStrategy, 'short_term')
    # TALibStrategy: lightweight TA-Lib based technical strategy with its own allocation
    maybe_add('talib', getattr(config.talib_strategy, 'enabled', True), TALibStrategy, 'talib')
    maybe_add('day_trading', getattr(config.day_trading_strategy, 'enabled', True), DayTradingStrategy, 'day_trading')
    maybe_add('options', getattr(config.options_strategy, 'enabled', True), OptionsStrategy, 'options')

    # Determine data interval
    # Priority: explicit CLI --interval override > day_trading presence heuristic
    has_day_trading = 'day_trading' in strategies_cfg
    if args.interval:
        data_interval = args.interval
        print(f"Data interval override via CLI: {data_interval}")
    else:
        data_interval = '1m' if has_day_trading else '1d'
        print(f"Data interval: {data_interval} ({'intraday for day trading' if has_day_trading else 'daily for other strategies'})")

    data_feeds = DataAcquisition.get_stock_data(
        tickers,
        config.start_date,
        config.end_date,
        interval=data_interval,
        auto_adjust=getattr(dc, 'auto_adjust', True),
        enable_cache=getattr(dc, 'enable', True),
        cache_dir=getattr(dc, 'dir', 'data/cache'),
        force_refresh=eff_force_refresh,
        cache_format=getattr(dc, 'format', 'parquet'),
        ttl_days=ttl_days
    )

    # If day trading is enabled but no intraday data available, fall back to lower frequency data
    if has_day_trading and not data_feeds and data_interval == '1m':
        print("Warning: 1-minute intraday data not available. Falling back to 5-minute data for day trading.")
        data_interval = '5m'
        data_feeds = DataAcquisition.get_stock_data(
            tickers,
            config.start_date,
            config.end_date,
            interval=data_interval,
            auto_adjust=getattr(dc, 'auto_adjust', True),
            enable_cache=getattr(dc, 'enable', True),
            cache_dir=getattr(dc, 'dir', 'data/cache'),
            force_refresh=eff_force_refresh,
            cache_format=getattr(dc, 'format', 'parquet'),
            ttl_days=ttl_days
        )
        
    if has_day_trading and not data_feeds and data_interval == '5m':
        print("Warning: 5-minute intraday data not available. Falling back to daily data for day trading.")
        data_interval = '1d'
        data_feeds = DataAcquisition.get_stock_data(
            tickers,
            config.start_date,
            config.end_date,
            interval=data_interval,
            auto_adjust=getattr(dc, 'auto_adjust', True),
            enable_cache=getattr(dc, 'enable', True),
            cache_dir=getattr(dc, 'dir', 'data/cache'),
            force_refresh=eff_force_refresh,
            cache_format=getattr(dc, 'format', 'parquet'),
            ttl_days=ttl_days
        )

    if not data_feeds:
        raise RuntimeError("No data feeds available. Check ticker symbols and date range.")

    print("\n=== Initializing Trading Strategy ===")
    cerebro = bt.Cerebro()
    for ticker, feed in data_feeds.items():
        cerebro.adddata(feed, name=ticker)
        print(f"Added {ticker} data feed")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if not strategies_cfg:
        raise RuntimeError("No strategies enabled. Please enable at least one strategy in config.")

    # Normalize allocation if some disabled: scale remaining to sum 100
    total_pct = sum(allocation_filtered.values())
    if total_pct <= 0:
        raise RuntimeError("Enabled strategies have zero allocation in portfolio_allocation section.")
    if abs(total_pct - 100.0) > 1e-6:
        scale = 100.0 / total_pct
        allocation_filtered = {k: round(v * scale, 4) for k, v in allocation_filtered.items()}
        print(f"Rescaled allocations to 100%: {allocation_filtered}")
    else:
        print(f"Final allocations (already 100%): {allocation_filtered}")

    print(f"Active strategies: {list(strategies_cfg.keys())}")

    cerebro.addstrategy(
        MultiAssetStrategy,
        strategies=strategies_cfg,
        allocation=allocation_filtered
    )
    cerebro.broker.setcash(config.initial_cash)
    cerebro.broker.setcommission(commission=config.commission)
    print(f"Broker initialized with ${config.initial_cash:,.2f} and {config.commission*100:.2f}% commission")

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    print("Analyzers added")

    print("\n=== Starting Backtest ===")
    print(f"Initial portfolio value: ${cerebro.broker.getvalue():,.2f}")
    try:
        results = cerebro.run()
        strat = results[0]
        print("\n=== Backtest Completed ===")
        print(f"Final portfolio value: ${cerebro.broker.getvalue():,.2f}")
        # Export equity curve
        try:
            ec = strat.get_equity_curve()
            if not ec.empty:
                ec.to_csv('equity_curve.csv')
                print("Saved equity_curve.csv")
            # Export long-term diagnostics if method exists
            try:
                # Access underlying logic instance for long_term
                if hasattr(strat, 'logic') and 'long_term' in strat.logic:
                    lt_logic = strat.logic['long_term']
                    if hasattr(lt_logic, 'export_diagnostics'):
                        lt_logic.export_diagnostics()
                        print("Exported long_term diagnostics CSV(s) to analysis_output/")
                # Export talib diagnostics if present
                if hasattr(strat, 'logic') and 'talib' in strat.logic:
                    t_logic = strat.logic['talib']
                    if hasattr(t_logic, 'export_diagnostics'):
                        outp = t_logic.export_diagnostics()
                        if outp:
                            print(f"Exported talib diagnostics to {outp}")
                # Export day_trading diagnostics if present
                if hasattr(strat, 'logic') and 'day_trading' in strat.logic:
                    dt_logic = strat.logic['day_trading']
                    if hasattr(dt_logic, 'export_diagnostics'):
                        outp = dt_logic.export_diagnostics()
                        if outp:
                            print(f"Exported day_trading diagnostics to {outp}")
            except Exception as de:
                print(f"Could not export long-term diagnostics: {de}")
        except Exception as eqe:
            print(f"Could not export equity curve: {eqe}")
    except Exception as e:
        print("\nERROR during backtest run:", e)
        traceback.print_exc()
        strat = None
    finally:
        log_file.close()

    metrics, drawdown = analyze_performance(cerebro, data_feeds)
    print("\nPerformance Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nTotal Return: {(cerebro.broker.getvalue()/config.initial_cash - 1)*100:.2f}%")

    if strat is not None:
        try:
            strat_perf = strat.get_strategy_performance()
            print("\nPer-Strategy Performance:")
            for name, det in strat_perf.items():
                m = det['metrics']
                print(f"  {name}: return={m['total_return']:.2%} trades={m['total_trades']} win_rate={m['win_rate']:.2%} pf={m['profit_factor']:.2f} max_dd={m['max_drawdown']:.2%}")
        except Exception as e:
            print(f"Could not retrieve strategy performance: {e}")

    # Long-term golden cross diagnostics export / print
    try:
        if strat is not None and hasattr(strat, 'logic') and 'long_term' in strat.logic:
            lt_logic = strat.logic['long_term']
            if hasattr(lt_logic, 'get_diagnostics'):
                d = lt_logic.get_diagnostics()
                print("\nLong-term Diagnostics:")
                for k, v in d.items():
                    print(f"  {k}: {v}")
            if hasattr(lt_logic, 'export_gc_diagnostics'):
                outp = lt_logic.export_gc_diagnostics()
                if outp:
                    print(f"Exported golden cross diagnostics to {outp}")
    except Exception as de:
        print(f"Diagnostics export error: {de}")

    viz_cfg = config.visualization or type('V', (), {'generate_charts': True, 'output_dir': 'charts'})()
    
    # Create strategy-specific output directory
    active_strategies = list(strategies_cfg.keys())
    if len(active_strategies) == 1:
        strategy_name = active_strategies[0]
        viz_cfg.output_dir = f"charts/{strategy_name}"
    else:
        viz_cfg.output_dir = "charts/combined"
    
    # Ensure output directory exists
    os.makedirs(viz_cfg.output_dir, exist_ok=True)
    
    charts_enabled = viz_cfg.generate_charts and not args.no_charts
    if charts_enabled and strat is not None:
        print("Launching interactive chart viewer...")
        try:
            # Import and run the chart viewer
            import subprocess
            import sys

            # Prepare data for chart viewer
            sig_hist = getattr(strat, 'get_signal_history', lambda: {})()
            orders_log = getattr(strat, 'get_orders_log', lambda: [])()
            orders_df = pd.DataFrame(orders_log) if orders_log else None

            # Get the first ticker's data for chart viewer
            ticker = list(data_feeds.keys())[0]
            feed = data_feeds[ticker]
            price_df = feed._dataname.copy()

            # Capitalize column names
            rename = {c: c.capitalize() for c in price_df.columns if c.lower() in ('open','high','low','close','volume')}
            price_df.rename(columns=rename, inplace=True)

            # Get strategy name
            active_strategies = list(strategies_cfg.keys())
            strategy_name = active_strategies[0] if len(active_strategies) == 1 else 'multi_strategy'

            # Save data to temporary CSV for chart viewer
            chart_data_path = f"temp_{ticker}_{strategy_name}_data.csv"
            price_df.to_csv(chart_data_path)

            # Launch chart viewer with the data
            cmd = [sys.executable, 'chart_viewer.py',
                   '--ticker', ticker,
                   '--strategy', strategy_name,
                   '--data', chart_data_path]

            if orders_df is not None and not orders_df.empty:
                orders_path = "temp_orders.csv"
                orders_df.to_csv(orders_path)
                cmd.extend(['--orders', orders_path])

            print(f"Launching chart viewer for {ticker} ({strategy_name})...")
            try:
                subprocess.Popen(cmd)
                print("Chart viewer launched successfully")
            except Exception as e:
                print(f"Failed to launch chart viewer: {e}")

            if orders_df is not None:
                orders_df.to_csv('orders_diagnostics.csv', index=False)
                print("Exported orders_diagnostics.csv")

        except Exception as e:
            print(f"Chart viewer launch error: {e}")
    else:
        print("Charts disabled (config or CLI flag)")

    print("Done.")


if __name__ == '__main__':
    main()