import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import talib

class EnhancedVolumeAnalysis:
    """Additional volume-based indicators and analysis tools"""
    
    @staticmethod
    def calculate_klinger_oscillator(data: pd.DataFrame, fast: int = 34, slow: int = 55) -> pd.Series:
        """Calculate Klinger Volume Oscillator"""
        trend = (data['Close'] - data['Close'].shift(1)).fillna(0)
        dm = pd.Series(0, index=data.index)
        dm[trend > 0] = data['Volume'][trend > 0]
        dm[trend < 0] = -data['Volume'][trend < 0]
        
        # Convert to float64 for talib
        dm_values = dm.astype(float).values
        
        fast_ema = talib.EMA(dm_values, timeperiod=fast)
        slow_ema = talib.EMA(dm_values, timeperiod=slow)
        
        return pd.Series(fast_ema - slow_ema, index=data.index)

    @staticmethod
    def ease_of_movement(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Ease of Movement indicator"""
        distance_moved = ((data['High'] + data['Low']) / 2) - \
                        ((data['High'].shift(1) + data['Low'].shift(1)) / 2)
        box_ratio = (data['Volume'] / 100000000) / (data['High'] - data['Low'])
        emv = distance_moved / box_ratio
        return emv.rolling(window=period).mean()
        
    @staticmethod
    def calculate_volume_volatility(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate volume volatility"""
        return data['Volume'].rolling(window=window).std() / data['Volume'].rolling(window=window).mean()
    
    @staticmethod
    def volume_zone_oscillator(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Volume Zone Oscillator"""
        vol_ma = data['Volume'].rolling(window=period).mean()
        vol_diff = data['Volume'] - vol_ma
        pos_vol = vol_diff.where(data['Close'] > data['Close'].shift(1), 0)
        neg_vol = vol_diff.where(data['Close'] < data['Close'].shift(1), 0)
        vzo = 100 * (pos_vol.rolling(window=period).sum() - neg_vol.rolling(window=period).sum()) / \
              vol_ma.rolling(window=period).sum()
        return vzo.fillna(0)

    @staticmethod
    def volume_force_index(data: pd.DataFrame, period: int = 13) -> pd.Series:
        """Calculate Force Index"""
        fi = (data['Close'] - data['Close'].shift(1)) * data['Volume']
        fi = fi.fillna(0)  # Fill NaN values with 0 for the first period
        fi_ema = talib.EMA(fi.values.astype(float), timeperiod=period)
        # Use bfill() instead of deprecated fillna(method='bfill')
        return pd.Series(fi_ema, index=data.index).bfill()

    @staticmethod
    def calculate_volume_intensity(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Volume Intensity Indicator"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        intensity = (typical_price - typical_price.shift(1)) * data['Volume']
        return intensity.rolling(window=window).mean()

    @staticmethod
    def volume_weighted_macd(data: pd.DataFrame, 
                           fast: int = 12, 
                           slow: int = 26, 
                           signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate Volume-Weighted MACD"""
        volume_price = data['Close'] * data['Volume']
        fast_ema = talib.EMA(volume_price.values, timeperiod=fast)
        slow_ema = talib.EMA(volume_price.values, timeperiod=slow)
        macd = fast_ema - slow_ema
        signal_line = talib.EMA(macd, timeperiod=signal)
        return pd.Series(macd, index=data.index), pd.Series(signal_line, index=data.index)

    @staticmethod
    def market_facilitation_index(data: pd.DataFrame) -> pd.Series:
        """Calculate Market Facilitation Index (BW MFI)"""
        return (data['High'] - data['Low']) / data['Volume']

    @staticmethod
    def volume_price_confirmation(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Volume-Price Confirmation Indicator"""
        price_change = data['Close'].pct_change()
        volume_change = data['Volume'].pct_change()
        confirmation = price_change * volume_change
        return confirmation.rolling(window=window).sum()

    @staticmethod
    def calculate_volume_volatility(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Volume Volatility"""
        log_volume = np.log(data['Volume'])
        return log_volume.rolling(window=window).std()

    @staticmethod
    def intraday_intensity(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Intraday Intensity Index"""
        intensity = (2 * data['Close'] - data['High'] - data['Low']) / \
                   (data['High'] - data['Low']) * data['Volume']
        return intensity.rolling(window=window).mean()

    @staticmethod
    def relative_volume_factor(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Relative Volume Factor"""
        volume_ma = data['Volume'].rolling(window=window).mean()
        return data['Volume'] / volume_ma

    @staticmethod
    def volume_zone_oscillator(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Volume Zone Oscillator"""
        bullish_volume = pd.Series(0, index=data.index)
        bearish_volume = pd.Series(0, index=data.index)
        
        bullish_volume[data['Close'] > data['Open']] = data['Volume'][data['Close'] > data['Open']]
        bearish_volume[data['Close'] < data['Open']] = data['Volume'][data['Close'] < data['Open']]
        
        bullish_ma = bullish_volume.rolling(window=window).mean()
        bearish_ma = bearish_volume.rolling(window=window).mean()
        
        return (bullish_ma - bearish_ma) / (bullish_ma + bearish_ma) * 100

    @staticmethod
    def volume_based_support_resistance(data: pd.DataFrame, 
                                      window: int = 20, 
                                      volume_threshold: float = 1.5) -> Dict[str, float]:
        """Identify support and resistance levels based on volume"""
        # Get recent data
        recent_data = data.tail(window)
        
        # Create price bins (10 equal-sized bins)
        min_price = recent_data['Low'].min()
        max_price = recent_data['High'].max()
        price_range = max_price - min_price
        bin_size = price_range / 10
        
        # Create bins manually
        bins = [min_price + i * bin_size for i in range(11)]
        labels = [(bins[i], bins[i+1]) for i in range(10)]
        
        # Assign each price to a bin
        price_bins = pd.cut(recent_data['Close'], bins=bins, labels=labels)
        # Explicit observed=False to silence future pandas default change warning
        volume_profile = recent_data.groupby(price_bins, observed=False)['Volume'].sum()
        
        # Find high volume areas
        mean_volume = volume_profile.mean()
        high_volume_levels = volume_profile[volume_profile > mean_volume * volume_threshold]
        
        if len(high_volume_levels) == 0:
            # If no high volume levels found, use more recent price action
            return {
                'support': recent_data['Low'].tail(5).min(),
                'resistance': recent_data['High'].tail(5).max()
            }
        
        # Sort levels by price
        sorted_levels = sorted(high_volume_levels.index, key=lambda x: x[0])
        
        # Find closest levels to current price
        current_price = data['Close'].iloc[-1]
        levels_below = [level for level in sorted_levels if level[1] <= current_price]
        levels_above = [level for level in sorted_levels if level[0] >= current_price]
        
        support_level = levels_below[-1][1] if levels_below else recent_data['Low'].min()
        resistance_level = levels_above[0][0] if levels_above else recent_data['High'].max()
        
        return {
            'support': float(support_level),
            'resistance': float(resistance_level),
            'volume_profile': volume_profile.to_dict()  # Include full volume profile
        }