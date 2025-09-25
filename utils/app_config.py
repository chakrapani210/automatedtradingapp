from dataclasses import dataclass, field
from typing import List
import yaml

@dataclass
class StrategyConfig:
    # Moving Average Parameters
    sma_window: int = 50          # Medium-term SMA window
    sma_long_window: int = 200    # Long-term SMA window
    
    # Risk Management Parameters
    risk_per_trade: float = 0.02   # Maximum risk per trade (2% of portfolio)
    max_position_size: float = 0.2  # Maximum position size (20% of portfolio)
    max_drawdown: float = 0.10     # Maximum drawdown allowed per trade
    stop_loss_atr_multiplier: float = 2.0  # ATR multiplier for stop loss
    
    # Technical Indicators
    rsi_window: int = 14          # RSI lookback period
    rsi_oversold: int = 30        # RSI oversold level
    rsi_overbought: int = 70      # RSI overbought level
    
    # Position Sizing and Volatility
    atr_window: int = 14          # ATR period for volatility-based sizing
    atr_risk_factor: float = 2.0  # ATR multiplier for risk calculation
    trailing_stop: bool = False    # Enable/disable trailing stop losses
    trailing_stop_atr: float = 3.0 # ATR multiplier for trailing stop

    @staticmethod
    def from_dict(data: dict):
        strategy_data = data.get('strategy', {})
        return StrategyConfig(
            sma_window=strategy_data.get('sma_window', 50),
            sma_long_window=strategy_data.get('sma_long_window', 200),
            risk_per_trade=strategy_data.get('risk_per_trade', 0.02),
            max_position_size=strategy_data.get('max_position_size', 0.2),
            max_drawdown=strategy_data.get('max_drawdown', 0.10),
            stop_loss_atr_multiplier=strategy_data.get('stop_loss_atr_multiplier', 2.0),
            rsi_window=strategy_data.get('rsi_window', 14),
            rsi_oversold=strategy_data.get('rsi_oversold', 30),
            rsi_overbought=strategy_data.get('rsi_overbought', 70),
            atr_window=strategy_data.get('atr_window', 14),
            atr_risk_factor=strategy_data.get('atr_risk_factor', 2.0),
            trailing_stop=strategy_data.get('trailing_stop', False),
            trailing_stop_atr=strategy_data.get('trailing_stop_atr', 3.0)
        )

@dataclass
class AppConfig:
    # Basic Settings
    tickers: List[str] = field(default_factory=lambda: ['AAPL'])
    start_date: str = '2022-01-01'
    end_date: str = '2023-01-01'
    initial_cash: float = 100000
    risk_free_rate: float = 0.0
    
    # Strategy Configuration
    strategy: StrategyConfig = field(default_factory=lambda: StrategyConfig())

    @staticmethod
    def load_from_yaml(path: str = 'config.yaml'):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            
        strategy_config = StrategyConfig.from_dict(data)
        
        return AppConfig(
            tickers=data.get('tickers', ['AAPL']),
            start_date=data.get('start_date', '2022-01-01'),
            end_date=data.get('end_date', '2023-01-01'),
            initial_cash=data.get('initial_cash', 100000),
            risk_free_rate=data.get('risk_free_rate', 0.0),
            strategy=strategy_config
        )
