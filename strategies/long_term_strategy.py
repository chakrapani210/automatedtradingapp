import pandas as pd
import numpy as np
from typing import Any, Optional
from .strategy_interface import TradingStrategy
import yfinance as yf

class LongTermValueStrategy(TradingStrategy):
    def __init__(self):
        self.last_rebalance_date = None

    def get_fundamental_data(self, ticker: str) -> dict:
        """Fetch fundamental data for value analysis"""
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'pe_ratio': info.get('forwardPE', float('inf')),
            'pb_ratio': info.get('priceToBook', float('inf')),
            'dividend_yield': info.get('dividendYield', 0),
            'market_cap': info.get('marketCap', 0)
        }

    def generate_signals(self, data: pd.DataFrame, config: Any) -> pd.Series:
        """Generate trading signals based on value metrics and trend"""
        signals = pd.Series(0, index=data.index)
        
        # Get configuration parameters from strategy config
        pe_max = config.strategy.pe_ratio_max
        pb_max = config.strategy.pb_ratio_max
        div_min = config.strategy.dividend_yield_min
        mcap_min = config.strategy.market_cap_min
        
        # Calculate technical indicators
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['SMA200'] = data['Close'].rolling(window=200).mean()
        
        # Volume analysis for long-term trends
        # Calculate 50-day volume moving average
        data['Volume_MA50'] = data['Volume'].rolling(window=50).mean()
        
        # Calculate Accumulation/Distribution Line
        data['ADL'] = (
            ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) /
            (data['High'] - data['Low']) * data['Volume']
        ).cumsum()
        
        # Get the ticker from the data
        ticker = list(data.keys())[0] if isinstance(data, dict) else data.columns[0][0] if isinstance(data.columns, pd.MultiIndex) else None
        if ticker is None:
            # If no ticker found, return empty signals
            return pd.Series(0, index=next(iter(data.values())).index if isinstance(data, dict) else data.index)
            
        # Get the actual price DataFrame
        df = data[ticker] if isinstance(data, dict) else data
        
        # Calculate moving averages
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate ADL and VPT
        df['ADL_MA50'] = df['ADL'].rolling(window=50).mean() if 'ADL' in df.columns else 0
        df['VPT'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * df['Volume']).cumsum()
        
        # Get fundamental data
        fundamentals = self.get_fundamental_data(ticker)
        
        # Value criteria
        value_criteria = (
            fundamentals['pe_ratio'] < pe_max and
            fundamentals['pb_ratio'] < pb_max and
            fundamentals['dividend_yield'] >= div_min and
            fundamentals['market_cap'] >= mcap_min
        )
        
        # Technical criteria (Golden Cross / Death Cross)
        df['golden_cross'] = (df['SMA50'] > df['SMA200'])
        
        # Calculate Volume Moving Average
        df['Volume_MA50'] = df['Volume'].rolling(window=50).mean()
        
        # Initialize signals series
        signals = pd.Series(0, index=df.index)
        
        # Generate signals
        for i in range(1, len(df)):
            # Volume trend analysis
            volume_trend_strong = (
                df['Volume'].iloc[i] > df['Volume_MA50'].iloc[i] * 1.2  # 20% above average volume
            )
            if 'ADL' in df.columns:
                volume_trend_strong &= df['ADL'].iloc[i] > df['ADL_MA50'].iloc[i]  # Rising accumulation
            volume_trend_strong &= df['VPT'].iloc[i] > df['VPT'].iloc[i-1]  # Rising volume price trend
            
            if value_criteria:
                # Buy signals with volume confirmation
                if (df['golden_cross'].iloc[i] and not df['golden_cross'].iloc[i-1] and
                    volume_trend_strong):
                    signals.iloc[i] = 1  # Buy on golden cross with strong volume
                
                # Sell signals
                elif (not df['golden_cross'].iloc[i] and df['golden_cross'].iloc[i-1] or
                      df['Volume'].iloc[i] > df['Volume_MA50'].iloc[i] * 2.0):  # Volume spike might indicate trend reversal
                    signals.iloc[i] = -1  # Sell on death cross or unusual volume spike
        
        return signals

    def calculate_position_size(self, data: pd.DataFrame, capital: float, config: Any) -> float:
        """Calculate position size based on portfolio allocation"""
        max_position = config.strategy.max_position_size
        return capital * max_position

    def get_stop_loss(self, data: pd.DataFrame, entry_price: float, position_type: str, config: Any) -> float:
        """Calculate stop loss level"""
        stop_loss_pct = 0.05  # Default 5% stop loss
        if position_type == 'long':
            return entry_price * (1 - stop_loss_pct)
        return entry_price * (1 + stop_loss_pct)

    def should_rebalance(self, current_date: pd.Timestamp, last_rebalance: Optional[pd.Timestamp], config: Any) -> bool:
        """Check if portfolio should be rebalanced (monthly)"""
        if last_rebalance is None:
            return True
            
        # Rebalance monthly
        return (current_date.year != last_rebalance.year or
                current_date.month != last_rebalance.month)