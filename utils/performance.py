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

def plot_equity_curve(portfolio_values, drawdown=None, ticker_data=None, signals=None, executed_trades=None):
    # Portfolio performance window
    plt.figure(figsize=(15, 8))
    plt.plot(portfolio_values, 'b-', label='Portfolio', linewidth=2)
    if drawdown is not None:
        plt.fill_between(drawdown.index, portfolio_values.min(), portfolio_values.max(), 
                        where=drawdown<0, color='red', alpha=0.1, label='Drawdown')
    plt.title('Overall Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Individual ticker windows with signals and executed trades
    if ticker_data is not None and signals is not None:
        for ticker in ticker_data.columns.levels[0]:
            # Create new figure for each ticker
            plt.figure(figsize=(15, 8))
            
            # Plot price
            close_prices = ticker_data[(ticker, 'Close')]
            plt.plot(close_prices.index, close_prices.values, 'b-', label='Price', linewidth=1)
            
            # Plot signals
            if signals and ticker in signals:
                signal_series = signals[ticker]
                
                # Buy signals (1)
                buy_signals = signal_series[signal_series > 0]
                if not buy_signals.empty:
                    plt.scatter(buy_signals.index, 
                              close_prices[buy_signals.index],
                              marker='^', color='g', s=100, alpha=0.3, label='Buy Signal')
                
                # Sell signals (-1)
                sell_signals = signal_series[signal_series < 0]
                if not sell_signals.empty:
                    plt.scatter(sell_signals.index,
                              close_prices[sell_signals.index],
                              marker='v', color='r', s=100, alpha=0.3, label='Sell Signal')
            
            # Plot executed trades
            if executed_trades and ticker in executed_trades:
                trades = executed_trades[ticker]
                for trade in trades:
                    if trade['type'] == 'buy':
                        plt.scatter(trade['date'], trade['price'], 
                                  marker='>', color='darkgreen', s=150, 
                                  label='Executed Buy' if trade == trades[0] else "")
                    else:  # sell
                        plt.scatter(trade['date'], trade['price'], 
                                  marker='<', color='darkred', s=150, 
                                  label='Executed Sell' if trade == trades[0] else "")
            
            plt.title(f'{ticker} Price and Signals')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
    
    # Show all windows
    plt.show()
