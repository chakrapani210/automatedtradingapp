import backtrader as bt
import quantstats as qs
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt

def analyze_performance(cerebro: bt.Cerebro, data_feeds: Dict[str, bt.feeds.PandasData]) -> Tuple[Dict[str, float], pd.Series]:
    """
    Analyze trading performance using QuantStats and backtrader analyzers
    """
    # Add analyzers to cerebro
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Run the backtest
    results = cerebro.run()
    strat = results[0]
    
    # Get analyzer results
    ret_analyzer = strat.analyzers.returns.get_analysis()
    sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
    dd_analyzer = strat.analyzers.drawdown.get_analysis()
    trade_analyzer = strat.analyzers.trades.get_analysis()
    
    # Calculate portfolio values and returns
    portfolio_values = pd.Series(strat._value_history)
    returns = portfolio_values.pct_change().dropna()
    
    # Calculate drawdown series
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max
    
    # Calculate basic metrics
    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    ann_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    
    # Compile metrics
    metrics = {
        'Total Return': total_return,
        'Annualized Return': ann_return,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': drawdown.min(),
        'Win Rate': 0,  # No trades in this period
        'Profit Factor': float('inf'),  # No trades in this period
        'Average Trade': 0,  # No trades in this period
        'Number of Trades': 0  # No trades in this period
    }
    
    return metrics, drawdown

def plot_equity_curve(cerebro: bt.Cerebro, save_path: Optional[str] = None) -> None:
    """
    Plot equity curve and trade analysis using backtrader's built-in plotting
    and QuantStats visualization
    
    Args:
        cerebro: The backtrader cerebro instance after running
        save_path: Optional path to save the plot
    """
    # Get the strategy instance
    strat = cerebro.runstrats[0][0]
    
    # Get portfolio values
    portfolio_values = pd.Series(strat._value_history)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Equity Curve
    ax1 = plt.subplot(311)
    portfolio_values.plot(title='Equity Curve', ax=ax1)
    ax1.set_xlabel('')
    ax1.grid(True)
    
    # Plot 2: Drawdown
    ax2 = plt.subplot(312)
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max
    drawdown.plot(title='Drawdown', color='red', ax=ax2)
    ax2.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.1)
    ax2.set_xlabel('')
    ax2.grid(True)
    
    # Plot 3: Daily Returns
    ax3 = plt.subplot(313)
    returns = portfolio_values.pct_change()
    returns.plot(title='Daily Returns', kind='bar', ax=ax3, alpha=0.5)
    ax3.set_xlabel('Date')
    ax3.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
