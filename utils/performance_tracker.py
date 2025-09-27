from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np

@dataclass
class StrategyMetrics:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades_count: int

class PerformanceTracker:
    def __init__(self):
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.daily_returns: List[Dict] = []
        self.initial_capital: float = 0
        self.current_capital: float = 0
        
    def initialize(self, capital: float):
        """Initialize tracker with starting capital"""
        self.initial_capital = capital
        self.current_capital = capital
        
    def update_position(self, ticker: str, quantity: int, price: float):
        """Update position information"""
        if ticker not in self.positions:
            self.positions[ticker] = {
                'quantity': 0,
                'avg_price': 0,
                'cost_basis': 0
            }
            
        position = self.positions[ticker]
        old_quantity = position['quantity']
        old_cost = position['cost_basis']
        
        # Update position
        if old_quantity == 0:
            position['avg_price'] = price
            position['quantity'] = quantity
            position['cost_basis'] = price * quantity
        else:
            # Calculate new average price and cost basis
            new_quantity = old_quantity + quantity
            if new_quantity != 0:
                new_cost = old_cost + (price * quantity)
                position['avg_price'] = new_cost / new_quantity
                position['quantity'] = new_quantity
                position['cost_basis'] = new_cost
            else:
                # Position closed
                position['avg_price'] = 0
                position['quantity'] = 0
                position['cost_basis'] = 0
                
    def record_trade(self, ticker: str, action: str, quantity: int, 
                    price: float, timestamp: datetime):
        """Record a trade"""
        trade = {
            'timestamp': timestamp,
            'ticker': ticker,
            'action': action,
            'quantity': quantity,
            'price': price,
            'value': quantity * price
        }
        self.trades.append(trade)
        
    def update_daily_return(self, total_value: float, timestamp: datetime):
        """Record daily portfolio value and calculate return"""
        daily_return = {
            'timestamp': timestamp,
            'total_value': total_value,
            'daily_return': (total_value - self.current_capital) / self.current_capital
        }
        self.daily_returns.append(daily_return)
        self.current_capital = total_value
        
    def calculate_metrics(self) -> StrategyMetrics:
        """Calculate strategy performance metrics"""
        if not self.daily_returns:
            return StrategyMetrics(0, 0, 0, 0, 0, 0)
            
        # Convert daily returns to DataFrame
        df_returns = pd.DataFrame(self.daily_returns)
        returns = df_returns['daily_return'].values
        
        # Calculate metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # Annualized Sharpe Ratio (assuming 252 trading days)
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) if len(returns) > 1 else 0
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Win Rate and Profit Factor
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            profits = df_trades[df_trades['value'] > 0]['value'].sum()
            losses = abs(df_trades[df_trades['value'] < 0]['value'].sum())
            win_count = len(df_trades[df_trades['value'] > 0])
            
            win_rate = win_count / len(df_trades)
            profit_factor = profits / losses if losses != 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0
            
        return StrategyMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades_count=len(self.trades)
        )
        
    def get_position_summary(self) -> pd.DataFrame:
        """Get current positions summary"""
        return pd.DataFrame.from_dict(self.positions, orient='index')
        
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        return pd.DataFrame(self.trades)
        
    def get_returns_history(self) -> pd.DataFrame:
        """Get returns history as DataFrame"""
        return pd.DataFrame(self.daily_returns)