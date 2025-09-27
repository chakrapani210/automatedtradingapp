import backtrader as bt
import quantstats as qs
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt

def analyze_performance(cerebro: bt.Cerebro, data_feeds: Dict[str, bt.feeds.PandasData]) -> Tuple[Dict[str, float], pd.Series]:
    """Analyze trading performance after a run.

    Assumes `cerebro.run()` already executed externally (to avoid double execution) and that
    analyzers were added prior to run (main.py does this). Falls back gracefully if private
    attributes (e.g. _value_history) are not present.
    """
    # Retrieve strategy instance from last run
    try:
        strat = cerebro.runstrats[0][0]
    except Exception:
        return {}, pd.Series(dtype=float)

    # Analyzer results (guarded)
    def safe_analyzer(name):
        return getattr(strat.analyzers, name).get_analysis() if hasattr(strat.analyzers, name) else {}
    ret_analyzer = safe_analyzer('returns')
    sharpe_analyzer = safe_analyzer('sharpe')
    dd_analyzer = safe_analyzer('drawdown')
    trade_analyzer = safe_analyzer('trades')

    # Portfolio value history: prefer strategy _value_history else reconstruct simplistic series
    if hasattr(strat, '_value_history') and len(getattr(strat, '_value_history')) > 1:
        portfolio_values = pd.Series(strat._value_history)
    else:
        # Fallback: single-point series using current broker value
        portfolio_values = pd.Series([cerebro.broker.getvalue()])
    returns = portfolio_values.pct_change().dropna()
    
    # Calculate drawdown series
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max
    
    # Calculate basic metrics
    if len(portfolio_values) > 1:
        total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
        ann_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
    else:
        total_return = 0.0
        ann_return = 0.0
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    
    # Compile metrics
    metrics = {
        'Total Return': total_return,
        'Annualized Return': ann_return,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': float(drawdown.min()) if len(drawdown) else 0.0,
        'Win Rate': 0,
        'Profit Factor': float('inf'),
        'Average Trade': 0,
        'Number of Trades': trade_analyzer.get('total', {}).get('total', 0) if isinstance(trade_analyzer, dict) else 0
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
    try:
        cerebro.plot(style='candlestick', volume=True)
    except Exception as e:
        print(f"Warning: Could not create detailed plot: {e}")
        print("Falling back to basic plotting...")
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
