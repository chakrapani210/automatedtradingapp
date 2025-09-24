import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_performance(portfolio_values, freq='D'):
    returns = portfolio_values.pct_change().dropna()
    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    ann_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1 if freq == 'D' else total_return
    ann_vol = returns.std() * np.sqrt(252 if freq == 'D' else 1)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = drawdown.min()
    metrics = {
        'Total Return': total_return,
        'Annualized Return': ann_return,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown
    }
    return metrics, drawdown

def plot_equity_curve(portfolio_values, drawdown=None):
    plt.figure(figsize=(12,6))
    plt.plot(portfolio_values, label='Equity Curve')
    if drawdown is not None:
        plt.fill_between(drawdown.index, portfolio_values.min(), portfolio_values.max(), where=drawdown<0, color='red', alpha=0.1, label='Drawdown')
    plt.title('Portfolio Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.tight_layout()
    plt.show()
