import backtrader as bt
from data.data_acquisition import DataAcquisition
from portfolio.portfolio_manager import PortfolioManager
from utils.app_config import AppConfig
from utils.performance import analyze_performance, plot_equity_curve
from strategies.standard_strategy import StandardStrategy
import pandas as pd
import quantstats as qs
import yfinance as yf

def main():
    # Load configuration
    config = AppConfig()
    
    # Initialize data acquisition
    tickers = config.tickers
    start_date = config.start_date
    end_date = config.end_date
    
    # Get data feeds
    data_feeds = DataAcquisition.get_stock_data(tickers, start_date, end_date)
    
    # Initialize Backtrader cerebro
    cerebro = bt.Cerebro()
    
    # Add data feeds to cerebro
    for ticker, data in data_feeds.items():
        cerebro.adddata(data, name=ticker)
    
    # Add our standard strategy
    cerebro.addstrategy(StandardStrategy,
                       sma_window=config.strategy.sma_window,
                       sma_long_window=config.strategy.sma_long_window,
                       rsi_window=config.strategy.rsi_window,
                       rsi_oversold=config.strategy.rsi_oversold,
                       rsi_overbought=config.strategy.rsi_overbought,
                       risk_per_trade=config.strategy.risk_per_trade,
                       max_position_size=config.strategy.max_position_size)
    
    # Set broker parameters
    cerebro.broker.setcash(config.initial_cash)
    cerebro.broker.setcommission(commission=config.commission)  # 0.1% commission
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Print initial portfolio value
    print(f'Initial portfolio value: ${cerebro.broker.getvalue():,.2f}')
    
    # Run backtest
    results = cerebro.run()
    strat = results[0]
    
    # Print final portfolio value
    print(f'Final portfolio value: ${cerebro.broker.getvalue():,.2f}')
    
    # Analyze performance
    metrics, drawdown = analyze_performance(cerebro, data_feeds)
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2%}" if isinstance(value, float) else f"{metric}: {value}")
    
    # Plot results
    plot_equity_curve(cerebro)
    
    # Generate detailed HTML report with QuantStats
    portfolio_returns = pd.Series(strat._value_history).pct_change().dropna()
    benchmark_data = yf.download('^GSPC', start=start_date, end=end_date)['Close'].pct_change().dropna()
    qs.reports.html(portfolio_returns, benchmark_data, output='trading_report.html')
    # Load configuration
    config = AppConfig.load_from_yaml()
    
    # Download data
    data = DataAcquisition.get_stock_data(
        config.tickers,
        config.start_date,
        config.end_date
    )
    
    # Initialize portfolio manager
    portfolio_manager = PortfolioManager(config)
    
    # Prepare data dictionary for each ticker
    ticker_data = {}
    for ticker in config.tickers:
        if isinstance(data.columns, pd.MultiIndex):
            ticker_cols = [col for col in data.columns if col[0] == ticker]
            if ticker_cols:
                df = data[ticker_cols].copy()
                df.columns = [col[1] for col in ticker_cols]
                ticker_data[ticker] = df
        else:
            ticker_data[ticker] = data
    
    # Run portfolio manager
    portfolio_values = portfolio_manager.run(ticker_data)
    
    # Analyze and plot results
    performance_metrics = analyze_performance(portfolio_values, config.risk_free_rate)
    plot_equity_curve(portfolio_values)
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
    # Generate signals for each asset
    signals = {}
    explanations = {}
    for ticker in tickers:
        # Robustly extract per-ticker data from MultiIndex or single-index DataFrame
        if isinstance(data.columns, pd.MultiIndex):
            ticker_cols = [col for col in data.columns if (len(col) > 1 and col[0] == ticker)]
            if ticker_cols:
                asset_data = data[ticker_cols].copy()
                asset_data.columns = [col[1] for col in ticker_cols]  # Flatten to ['Open', 'High', ...]
            else:
                logger.warning(f'No columns found for ticker {ticker}')
                continue
        else:
            asset_data = data
        strategy = TALibStrategy()
        sig = strategy.generate_signals(asset_data, config=config)
        # Align signal index to asset_data index for Backtrader compatibility
        sig = sig.reindex(asset_data.index)
        signals[ticker] = sig
        # Debug: Print first few signal values
        print(f"DEBUG: First 5 signals for {ticker}:")
        print(sig.head())
        # Generate explanations for signals
        if hasattr(strategy, 'explain_signals'):
            asset_data['SMA'] = asset_data['Close'].rolling(sma_window).mean()
            explanations[ticker] = strategy.explain_signals(asset_data, sig)
            #for exp in explanations[ticker]:
            #    logger.info(f"{ticker} {exp['date']}: Signal={exp['signal']}, Close={exp['close']:.2f}, SMA={exp['sma']:.2f}, Reason={exp['reason']}")
    logger.info('Generated trading signals for all assets')

    # Prepare price data for risk manager (Close prices for all assets)
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = pd.concat([
            data[(ticker, 'Close')] for ticker in tickers
        ], axis=1)
    else:
        close_prices = pd.concat([
            data['Close'] for _ in tickers
        ], axis=1)
    close_prices.columns = tickers

    # Risk management
    risk_manager = RiskManager(close_prices, risk_free_rate=risk_free_rate)
    weights = risk_manager.optimize_portfolio()
    logger.info(f'Optimized portfolio weights: {weights}')
    
    # Debug: Check if weights sum to ~1.0
    weight_sum = sum(weights.values())
    print(f"DEBUG: Sum of portfolio weights: {weight_sum}")



    # Prepare Backtrader data feeds
    data_dict = {}
    for ticker in tickers:
        if isinstance(data.columns, pd.MultiIndex):
            ticker_cols = [col for col in data.columns if (len(col) > 1 and col[0] == ticker)]
            if ticker_cols:
                df = data[ticker_cols].copy()
                df.columns = [col[1] for col in ticker_cols]  # Flatten to ['Open', 'High', ...]
            else:
                logger.warning(f'No columns found for ticker {ticker} for Backtrader feed')
                continue
        else:
            df = data
        bt_df = pd.DataFrame({
            'open': df['Open'],
            'high': df['High'],
            'low': df['Low'],
            'close': df['Close'],
            'volume': df['Volume'],
        }, index=df.index)
        datafeed = bt.feeds.PandasData(dataname=bt_df)
        data_dict[ticker] = datafeed

    # Run backtest
    # Use MultiAssetStrategy directly
    backtest_engine = BacktestEngine(MultiAssetStrategy, data_dict, signals, weights, cash=initial_cash)
    results = backtest_engine.run()
    logger.info('Backtest complete')

    # Performance analysis (simple version)
    strat = results[0]
    # Try to extract portfolio value history from Backtrader
    if hasattr(strat.broker, '_value_history'):
        portfolio_values = pd.Series(strat.broker._value_history, index=bt_df.index[:len(strat.broker._value_history)])
        metrics, drawdown = analyze_performance(portfolio_values)
        logger.info(f'Performance metrics: {metrics}')
        # Plot portfolio, ticker performances, signals and executed trades
        plot_equity_curve(portfolio_values, drawdown, data, signals, strat.executed_trades)

    else:
        logger.warning('Portfolio value history not available for analysis.')

    # Log account status info at the end using Backtrader's built-in features
    final_value = strat.broker.getvalue() if hasattr(strat, 'broker') else None
    cash = strat.broker.get_cash() if hasattr(strat, 'broker') else None
    positions = {}
    unrealized = {}
    realized = 0.0
    for data in strat.datas:
        pos = strat.getposition(data)
        if pos.size != 0:
            positions[data._name] = pos.size
            # Calculate unrealized P&L for each open position
            unrealized[data._name] = (data.close[0] - pos.price) * pos.size
            # Calculate realized P&L from closed trades (if needed, accumulate from trade notifications)
    logger.info(f'Account Status: Portfolio Value = {final_value}, Cash = {cash}, Open Positions = {positions}')
    logger.info(f'Unrealized P&L: {unrealized}')

if __name__ == '__main__':
    main()
