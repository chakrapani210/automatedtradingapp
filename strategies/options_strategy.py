from typing import Any, Optional
import pandas as pd
from .strategy_interface import TradingStrategy

class OptionsStrategy(TradingStrategy):
    def generate_signals(self, data: pd.DataFrame, config: Any) -> pd.Series:
        """Generate trading signals for options strategy"""
        if isinstance(data, dict):
            # Get the first ticker's data
            df = next(iter(data.values()))
        else:
            df = data
            
        signals = pd.Series(0, index=df.index)
        
        # Simple volatility-based options strategy
        # Calculate historical volatility
        returns = df['Close'].pct_change()
        volatility = returns.rolling(window=20).std() * (252 ** 0.5)  # Annualized volatility
        
        # Generate signals based on volatility regime
        signals[volatility > 0.3] = 1  # High volatility - potential for writing options
        signals[volatility < 0.15] = -1  # Low volatility - potential for buying options
        
        return signals

    def calculate_position_size(self, data: pd.DataFrame, capital: float, config: Any) -> float:
        """Calculate position size for options"""
        # Use smaller position sizes for options due to leverage
        base_position = capital * 0.05  # Start with 5% of capital
        
        # Adjust based on implied volatility if available
        # For now, using a conservative fixed size
        return min(base_position, capital * 0.1)  # Cap at 10% of capital

    def get_stop_loss(self, data: pd.DataFrame, entry_price: float, position_type: str, config: Any) -> float:
        """Calculate stop loss level for options positions"""
        if position_type == 'long':
            # For long options, risk 50% of premium
            return entry_price * 0.5
        else:
            # For short options, risk 200% of premium
            return entry_price * 2.0
            
    def rebalance(self, data: pd.DataFrame, positions: dict, config: Any) -> dict:
        """Rebalance options positions"""
        # Start with a clean slate for options positions
        new_positions = {}
        
        # Get current volatility
        returns = data['Close'].pct_change()
        current_volatility = returns.rolling(window=20).std().iloc[-1] * (252 ** 0.5)
        
        # Adjust positions based on volatility regime
        if current_volatility > 0.3:
            # High volatility - focus on writing options
            new_positions['options_short'] = 0.6  # 60% short options
            new_positions['options_long'] = 0.4   # 40% long options for hedge
        elif current_volatility < 0.15:
            # Low volatility - focus on buying options
            new_positions['options_long'] = 0.7   # 70% long options
            new_positions['options_short'] = 0.3  # 30% short options for income
        else:
            # Medium volatility - balanced approach
            new_positions['options_long'] = 0.5
            new_positions['options_short'] = 0.5
            
        return new_positions
        """Calculate stop loss for options position"""
        # For options, stop loss is typically the premium paid
        return entry_price * 0.5  # 50% of premium as stop loss

    def should_rebalance(self, current_date: pd.Timestamp, last_rebalance: Optional[pd.Timestamp], config: Any) -> bool:
        """Check if options positions should be rebalanced (weekly)"""
        if last_rebalance is None:
            return True
        # Rebalance weekly for options
        return current_date.isocalendar()[1] != last_rebalance.isocalendar()[1]
