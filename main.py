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
    # Set up logging to file
    import sys
    log_file = open('backtest_log.txt', 'w')
    sys.stdout = log_file
    
    # Load configuration
    config = AppConfig()
    
    print("\n=== Automated Trading System Starting ===")
    print("\nConfiguration:")
    print(f"Tickers: {config.tickers}")
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Initial Cash: ${config.initial_cash:,.2f}")
    print("\nStrategy Parameters:")
    print(f"  - SMA Window: {config.strategy.sma_window}")
    print(f"  - SMA Long Window: {config.strategy.sma_long_window}")
    print(f"  - RSI Window: {config.strategy.rsi_window}")
    print(f"  - RSI Oversold: {config.strategy.rsi_oversold}")
    print(f"  - RSI Overbought: {config.strategy.rsi_overbought}")
    print(f"  - Risk Per Trade: {config.strategy.risk_per_trade * 100}%")
    print(f"  - Max Position Size: {config.strategy.max_position_size * 100}%")
    
    print("\n=== Fetching Market Data ===")
    # Initialize data acquisition
    tickers = config.tickers
    start_date = config.start_date
    end_date = config.end_date
    
    # Get data feeds
    data_feeds = DataAcquisition.get_stock_data(tickers, start_date, end_date)
    
    print("\n=== Initializing Trading Strategy ===")
    # Initialize Backtrader cerebro
    cerebro = bt.Cerebro()
    
    # Add data feeds to cerebro
    for ticker, data in data_feeds.items():
        cerebro.adddata(data, name=ticker)
        print(f"Added {ticker} data feed")
    
    # Add our standard strategy
    cerebro.addstrategy(StandardStrategy,
                       sma_window=config.strategy.sma_window,
                       sma_long_window=config.strategy.sma_long_window,
                       rsi_window=config.strategy.rsi_window,
                       rsi_oversold=config.strategy.rsi_oversold,
                       rsi_overbought=config.strategy.rsi_overbought,
                       risk_per_trade=config.strategy.risk_per_trade,
                       max_position_size=config.strategy.max_position_size)
    print("Strategy initialized with parameters")
    
    # Set broker parameters
    cerebro.broker.setcash(config.initial_cash)
    cerebro.broker.setcommission(commission=config.commission)
    print(f"Broker initialized with ${config.initial_cash:,.2f} and {config.commission*100}% commission")
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    print("Performance analyzers added")
    
    print("\n=== Starting Backtest ===")
    print(f'Initial portfolio value: ${cerebro.broker.getvalue():,.2f}')
    
    # Run backtest
    results = cerebro.run()
    strat = results[0]
    
    print("\n=== Backtest Completed ===")
    print(f'Final portfolio value: ${cerebro.broker.getvalue():,.2f}')
    
    # Close log file
    log_file.close()
    sys.stdout = sys.__stdout__
    
    # Analyze performance
    metrics, drawdown = analyze_performance(cerebro, data_feeds)
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2%}" if isinstance(value, float) else f"{metric}: {value}")
    
    # Print final performance summary
    print("\nFinal Performance Summary:")
    print(f"Starting Portfolio Value: ${config.initial_cash:,.2f}")
    print(f"Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
    print(f"Total Return: {(cerebro.broker.getvalue() / config.initial_cash - 1) * 100:.2f}%")

    # Plot the equity curve and returns analysis
    cerebro.plot()

if __name__ == '__main__':
    main()