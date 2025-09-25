from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional

class TradingStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, config: Any) -> pd.Series:
        """Generate trading signals for the given data"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, data: pd.DataFrame, capital: float, config: Any) -> float:
        """Calculate the position size based on available capital and risk parameters"""
        pass
    
    @abstractmethod
    def get_stop_loss(self, data: pd.DataFrame, entry_price: float, position_type: str, config: Any) -> float:
        """Calculate stop loss level for a position"""
        pass
    
    @abstractmethod
    def should_rebalance(self, current_date: pd.Timestamp, last_rebalance: Optional[pd.Timestamp], config: Any) -> bool:
        """Determine if the portfolio should be rebalanced"""
        pass