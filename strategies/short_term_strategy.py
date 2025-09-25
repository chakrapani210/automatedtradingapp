import pandas as pd
import numpy as np
from typing import Any, Optional
from .strategy_interface import TradingStrategy
import talib

class ShortTermTechnicalStrategy(TradingStrategy):
    def __init__(self):
        self.last_rebalance_date = None

    def generate_signals(self, data: pd.DataFrame, config: Any) -> pd.Series:
        """Generate trading signals based on technical indicators"""
        if isinstance(data, dict):
            df = next(iter(data.values()))
        else:
            df = data
            
        signals = pd.Series(0, index=df.index)
        
        # Calculate technical indicators
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        # RSI
        rsi = talib.RSI(close, timeperiod=config.strategy.rsi_window)
        
        # Moving averages
        sma = talib.SMA(close, timeperiod=config.strategy.sma_window)
        sma_long = talib.SMA(close, timeperiod=config.strategy.sma_long_window)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=talib.MA_Type.SMA
        )
        
        # Volume indicators
        obv = talib.OBV(close, volume)
        mfi = talib.MFI(high, low, close, volume, timeperiod=14)
        
        # Generate signals
        for i in range(len(signals)):
            if i < config.strategy.sma_long_window:
                continue
                
            # Buy conditions
            if (rsi[i] < config.strategy.rsi_oversold and 
                sma[i] > sma[i-1] and  # Short-term uptrend
                close[i] < lower[i] and  # Price below lower BB
                volume[i] > volume[i-1] * 1.2 and  # Volume confirmation
                mfi[i] < 30):  # Money flow oversold
                signals[i] = 1
                
            # Sell conditions
            elif (rsi[i] > config.strategy.rsi_overbought and 
                  sma[i] < sma[i-1] and  # Short-term downtrend
                  close[i] > upper[i] and  # Price above upper BB
                  volume[i] > volume[i-1] * 1.2 and  # Volume confirmation
                  mfi[i] > 70):  # Money flow overbought
                signals[i] = -1
                
        return signals
        
    def calculate_position_size(self, data: pd.DataFrame, capital: float, config: Any) -> float:
        """Calculate position size based on volatility and risk parameters"""
        if isinstance(data, dict):
            df = next(iter(data.values()))
        else:
            df = data
            
        # Get risk parameters from config
        risk_per_trade = config.strategy.risk_per_trade
        max_position = config.strategy.max_position_size
        stop_loss_atr = config.strategy.stop_loss_atr_multiplier
        
        # Calculate ATR
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        atr = talib.ATR(high, low, close, timeperiod=config.strategy.atr_window)[-1]
        
        if pd.isna(atr):
            atr = (high[-1] - low[-1]) * 0.1  # Fallback if ATR is not available
        
        # Risk-based position sizing
        risk_amount = capital * risk_per_trade
        position_size = risk_amount / (atr * stop_loss_atr)
        
        # Apply maximum position size limit
        max_position_size = capital * max_position
        return min(position_size, max_position_size)
        
    def get_stop_loss(self, data: pd.DataFrame, entry_price: float, position_type: str, config: Any) -> float:
        """Calculate stop loss level based on ATR"""
        if isinstance(data, dict):
            df = next(iter(data.values()))
        else:
            df = data
            
        stop_loss_atr = config.strategy.stop_loss_atr_multiplier
        atr = talib.ATR(
            df['High'].values,
            df['Low'].values,
            df['Close'].values,
            timeperiod=14
        )[-1]
        
        if pd.isna(atr):
            atr = (df['High'].iloc[-1] - df['Low'].iloc[-1]) * 0.1  # Fallback if ATR is not available
            
        if position_type == 'long':
            return entry_price - (atr * stop_loss_atr)
        return entry_price + (atr * stop_loss_atr)

    def should_rebalance(self, current_date: pd.Timestamp, last_rebalance: Optional[pd.Timestamp], config: Any) -> bool:
        """Check if portfolio should be rebalanced (daily for short-term)"""
        if last_rebalance is None:
            return True
        return current_date.date() != last_rebalance.date()