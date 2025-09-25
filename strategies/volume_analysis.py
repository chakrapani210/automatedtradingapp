import pandas as pd
import numpy as np
from typing import Tuple

class VolumeAnalysis:
    @staticmethod
    def calculate_vwap(data: pd.DataFrame, period: str = 'D') -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        return (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    @staticmethod
    def volume_trend_strength(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate volume trend strength"""
        vol_ma = data['Volume'].rolling(window=window).mean()
        trend = pd.Series(0.0, index=data.index)
        for i in range(window, len(data)):
            recent_vol = data['Volume'].iloc[i-window:i]
            vol_slope = np.polyfit(range(window), recent_vol, 1)[0]
            trend.iloc[i] = vol_slope / vol_ma.iloc[i]
        return trend.fillna(0)

    @staticmethod
    def volume_price_correlation(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate rolling correlation between price and volume"""
        return data['Close'].rolling(window).corr(data['Volume'])

    @staticmethod
    def calculate_volume_profile(data: pd.DataFrame, bins: int = 12) -> pd.Series:
        """Calculate volume profile (price levels with highest volume)"""
        price_bins = pd.qcut(data['Close'], bins)
        return data.groupby(price_bins)['Volume'].sum()

    @staticmethod
    def money_flow_strength(data: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate positive and negative money flow"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
    @staticmethod
    def on_balance_volume(data: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume (OBV)"""
        obv = pd.Series(0, index=data.index)
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv
        
        return (positive_flow.rolling(period).sum() / negative_flow.rolling(period).sum(),
                money_flow.rolling(period).sum())

    @staticmethod
    def volume_trend_strength(data: pd.DataFrame, ma_window: int = 20) -> pd.Series:
        """Calculate volume trend strength indicator"""
        volume_ma = data['Volume'].rolling(window=ma_window).mean()
        return (data['Volume'] - volume_ma) / volume_ma.rolling(window=ma_window).std()

    @staticmethod
    def institutional_activity(data: pd.DataFrame, threshold: int = 1000000) -> pd.Series:
        """Detect potential institutional activity through volume analysis"""
        volume_ma = data['Volume'].rolling(window=50).mean()
        large_trades = data['Volume'] > threshold
        price_impact = (data['Close'] - data['Open']).abs() / data['Open']
        
        return pd.Series(
            large_trades & (data['Volume'] > 2 * volume_ma) & (price_impact < 0.02),
            index=data.index
        )

    @staticmethod
    def volume_breakout_confirmation(data: pd.DataFrame, mult: float = 2.0, window: int = 20) -> bool:
        """Confirm price breakouts with volume"""
        recent_volume = data['Volume'].iloc[-window:]
        avg_volume = recent_volume.mean()
        current_volume = data['Volume'].iloc[-1]
        
        price_breakout = abs(data['Close'].pct_change().iloc[-1]) > 0.02
        volume_breakout = current_volume > avg_volume * mult
        
        return price_breakout and volume_breakout

    @staticmethod
    def calculate_volume_zones(data: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """Calculate support and resistance zones based on volume"""
        volume_profile = VolumeAnalysis.calculate_volume_profile(data.tail(window))
        high_volume_prices = volume_profile.nlargest(2).index
        
        support = float(high_volume_prices[0].left)
        resistance = float(high_volume_prices[-1].right)
        
        return support, resistance

    @staticmethod
    def smart_money_index(data: pd.DataFrame) -> pd.Series:
        """Calculate Smart Money Index (institutional activity indicator)"""
        first_hour_change = data['Open'] - data['Low']
        last_hour_change = data['Close'] - data['High']
        
        smi = pd.Series(0.0, index=data.index)  # Initialize as float
        smi.iloc[0] = float(data['Close'].iloc[0])  # Explicitly convert to float
        
        for i in range(1, len(data)):
            smi.iloc[i] = float(smi.iloc[i-1] - first_hour_change.iloc[i] + last_hour_change.iloc[i])
        
        return smi