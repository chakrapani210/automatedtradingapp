from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

@dataclass
class TradeInfo:
    timestamp: datetime
    ticker: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    value: float
    pnl: Optional[float] = None

@dataclass
class PositionInfo:
    ticker: str
    quantity: int
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float

class StrategyPerformance:
    def __init__(self, strategy_name: str, initial_capital: float):
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Performance tracking
        self.trades: List[TradeInfo] = []
        self.positions: Dict[str, PositionInfo] = {}
        self.daily_stats: List[Dict] = []
        self.high_water_mark = initial_capital
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
    def update_position(self, ticker: str, quantity: int, current_price: float):
        """Update position information for a ticker"""
        if ticker not in self.positions:
            if quantity == 0:
                return
            
            self.positions[ticker] = PositionInfo(
                ticker=ticker,
                quantity=quantity,
                avg_entry_price=current_price,
                current_price=current_price,
                market_value=quantity * current_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0
            )
        else:
            position = self.positions[ticker]
            old_value = position.quantity * position.current_price
            new_value = quantity * current_price
            
            if quantity == 0:
                # Position closed
                realized_pnl = position.unrealized_pnl
                self.total_pnl += realized_pnl
                del self.positions[ticker]
            else:
                # Position updated
                position.quantity = quantity
                position.current_price = current_price
                position.market_value = new_value
                position.unrealized_pnl = new_value - (quantity * position.avg_entry_price)
                
    def record_trade(self, trade: TradeInfo):
        """Record a new trade"""
        self.trades.append(trade)
        
        # Update win/loss counters
        if trade.pnl is not None:
            if trade.pnl > 0:
                self.winning_trades += 1
            elif trade.pnl < 0:
                self.losing_trades += 1
                
    def update_daily_stats(self, timestamp: datetime, portfolio_value: float):
        """Update daily performance statistics"""
        if self.current_capital == 0:
            daily_return = 0.0
        else:
            daily_return = (portfolio_value - self.current_capital) / self.current_capital
        
        # Update high water mark and drawdown
        if portfolio_value > self.high_water_mark:
            self.high_water_mark = portfolio_value
            self.current_drawdown = 0.0
        else:
            if self.high_water_mark > 0:
                self.current_drawdown = (self.high_water_mark - portfolio_value) / self.high_water_mark
                self.max_drawdown = min(self.max_drawdown, -self.current_drawdown)
            else:
                self.current_drawdown = 0.0
            
        self.daily_stats.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'daily_return': daily_return,
            'drawdown': self.current_drawdown,
            'realized_pnl': self.total_pnl,
            'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions.values())
        })
        
        self.current_capital = portfolio_value
        
    def get_metrics(self) -> Dict:
        """Calculate and return performance metrics"""
        if not self.daily_stats:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
            
        df = pd.DataFrame(self.daily_stats)
        returns = df['daily_return'].values
        
        total_trades = self.winning_trades + self.losing_trades
        win_rate = self.winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate metrics
        if self.initial_capital > 0:
            total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        else:
            total_return = 0.0
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) if len(returns) > 1 else 0.0
        
        # Calculate profit factor
        winning_trades_value = sum(t.pnl for t in self.trades if t.pnl and t.pnl > 0)
        losing_trades_value = sum(-t.pnl for t in self.trades if t.pnl and t.pnl < 0)
        eps = 1e-9
        profit_factor = (winning_trades_value / (losing_trades_value + eps)) if (winning_trades_value > 0) else 0.0
        avg_win = (winning_trades_value / self.winning_trades) if self.winning_trades else 0.0
        avg_loss = (-losing_trades_value / self.losing_trades) if self.losing_trades else 0.0
        expectancy = ((win_rate * avg_win) - ((1 - win_rate) * avg_loss)) if (avg_win or avg_loss) else 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy
        }
        
    def get_position_summary(self) -> pd.DataFrame:
        """Get current positions summary"""
        if not self.positions:
            return pd.DataFrame()
        return pd.DataFrame([vars(p) for p in self.positions.values()])
        
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([vars(t) for t in self.trades])
        
    def get_daily_stats(self) -> pd.DataFrame:
        """Get daily performance statistics"""
        if not self.daily_stats:
            return pd.DataFrame()
        return pd.DataFrame(self.daily_stats)