from dataclasses import dataclass, field
from typing import List
import yaml

@dataclass
class VolumeConfig:
    # Common Volume Parameters
    volume_ma_window: int = 20      # Volume moving average window
    min_tick_volume: int = 100000   # Minimum volume per tick
    
    # Short-term Volume Indicators
    obv_ma_window: int = 20         # OBV moving average window
    mfi_period: int = 14            # Money Flow Index period
    mfi_oversold: int = 20          # MFI oversold level
    mfi_overbought: int = 80        # MFI overbought level
    cmf_fast_period: int = 3        # Chaikin Money Flow fast period
    cmf_slow_period: int = 10       # Chaikin Money Flow slow period
    volume_ratio_threshold: float = 1.5  # Volume ratio threshold
    vwap_period: str = "D"          # VWAP calculation period
    volume_profile_bins: int = 12    # Volume profile bins
    volume_price_correlation_window: int = 20  # Volume-price correlation window
    
    # Long-term Volume Analysis
    institutional_volume_threshold: int = 1000000  # Min institutional volume
    volume_breakout_mult: float = 2.0  # Volume breakout multiplier
    adl_ma_window: int = 50        # ADL moving average window
    vpt_change_threshold: float = 0.02  # VPT change threshold
    
    @staticmethod
    def from_dict(data: dict) -> 'VolumeConfig':
        return VolumeConfig(
            volume_ma_window=data.get('volume_ma_window', 20),
            min_tick_volume=data.get('min_tick_volume', 100000),
            obv_ma_window=data.get('obv_ma_window', 20),
            mfi_period=data.get('mfi_period', 14),
            mfi_oversold=data.get('mfi_oversold', 20),
            mfi_overbought=data.get('mfi_overbought', 80),
            cmf_fast_period=data.get('cmf_fast_period', 3),
            cmf_slow_period=data.get('cmf_slow_period', 10),
            volume_ratio_threshold=data.get('volume_ratio_threshold', 1.5),
            vwap_period=data.get('vwap_period', "D"),
            volume_profile_bins=data.get('volume_profile_bins', 12),
            volume_price_correlation_window=data.get('volume_price_correlation_window', 20),
            institutional_volume_threshold=data.get('institutional_volume_threshold', 1000000),
            volume_breakout_mult=data.get('volume_breakout_mult', 2.0),
            adl_ma_window=data.get('adl_ma_window', 50),
            vpt_change_threshold=data.get('vpt_change_threshold', 0.02)
        )

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
    
    # Long-term Strategy Parameters
    pe_ratio_max: float = 25.0     # Maximum P/E ratio
    pb_ratio_max: float = 3.0      # Maximum P/B ratio
    dividend_yield_min: float = 0.02 # Minimum dividend yield
    market_cap_min: int = 10000000000 # Minimum market cap
    
    # Volume Configuration
    volume: VolumeConfig = field(default_factory=VolumeConfig)

    @staticmethod
    def from_dict(data: dict):
        strategy_data = data.get('strategies', {}).get('long_term', {})
        return StrategyConfig(
            sma_window=strategy_data.get('sma_window', 50),
            pe_ratio_max=strategy_data.get('pe_ratio_max', 25.0),
            pb_ratio_max=strategy_data.get('pb_ratio_max', 3.0),
            dividend_yield_min=strategy_data.get('dividend_yield_min', 0.02),
            market_cap_min=strategy_data.get('market_cap_min', 10000000000),
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
    commission: float = 0.001  # 0.1% commission
    
    # Portfolio Allocation
    portfolio_allocation: dict = field(default_factory=lambda: {
        'long_term': 0.50,
        'short_term': 0.25,
        'options': 0.25
    })
    
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
            commission=data.get('commission', 0.001),
            portfolio_allocation=data.get('portfolio_allocation', {
                'long_term': 0.50,
                'short_term': 0.25,
                'options': 0.25
            }),
            strategy=strategy_config
        )
