import pytest
import pandas as pd
import numpy as np
from strategies.day_trading_strategy import DayTradingStrategy
from utils.app_config import DayTradingStrategyConfig

def test_day_trading_strategy_basic():
    """Test basic functionality of DayTradingStrategy."""
    strategy = DayTradingStrategy()
    config = DayTradingStrategyConfig()

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Close': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(105, 205, 100),
        'Low': np.random.uniform(95, 195, 100),
        'Volume': np.random.uniform(1000000, 5000000, 100)
    }, index=dates)

    # Wrap config in an object with day_trading_strategy attribute
    class MockConfig:
        def __init__(self, strat_config):
            self.day_trading_strategy = strat_config

    mock_config = MockConfig(config)

    # Generate signals
    signals = strategy.generate_signals(data, mock_config)

    # Check that signals are generated
    assert len(signals) == len(data)
    assert signals.index.equals(data.index)
    assert all(sig in [-1, 0, 1] for sig in signals)

def test_day_trading_strategy_config():
    """Test DayTradingStrategyConfig defaults and validation."""
    config = DayTradingStrategyConfig()

    # Check defaults
    assert config.enabled == True
    assert config.rsi_window == 14
    assert config.buy_score_threshold == 0.60

    # Test validation
    validated = config.validate()
    assert validated is config

def test_day_trading_strategy_sizing():
    """Test position sizing in DayTradingStrategy."""
    strategy = DayTradingStrategy()
    config = DayTradingStrategyConfig()

    # Sample data
    data = pd.DataFrame({
        'Close': [150.0],
        'High': [155.0],
        'Low': [145.0],
        'Volume': [2000000]
    })

    capital = 100000.0

    # Wrap config
    class MockConfig:
        def __init__(self, strat_config):
            self.day_trading_strategy = strat_config

    mock_config = MockConfig(config)

    size = strategy.calculate_position_size(data, capital, mock_config)

    # Check that size is reasonable
    assert size >= 0
    assert size <= capital * config.max_position_fraction