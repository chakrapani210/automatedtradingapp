import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import talib
from strategies.volume_analysis import VolumeAnalysis
from strategies.enhanced_volume import EnhancedVolumeAnalysis

class VolumeBasedPositionSizer:
    def __init__(self, config: any):
        self.config = config
        self.va = VolumeAnalysis()
        self.eva = EnhancedVolumeAnalysis()

    def calculate_liquidity_score(self, data: pd.DataFrame) -> float:
        """Calculate liquidity score based on volume metrics"""
        # Get recent volume data
        recent_volume = data['Volume'].tail(20)
        avg_volume = recent_volume.mean()
        volume_std = recent_volume.std()
        
        # Calculate volume-based metrics
        relative_volume = data['Volume'].iloc[-1] / avg_volume
        volume_consistency = 1 - (volume_std / avg_volume)  # Higher is better
        
        # Volume trend strength
        volume_trend = self.va.volume_trend_strength(data)
        trend_score = volume_trend.iloc[-1]
        
        # Combine metrics into liquidity score
        liquidity_score = (
            0.4 * relative_volume +
            0.3 * volume_consistency +
            0.3 * abs(trend_score)
        )
        
        return min(max(liquidity_score, 0), 1)  # Normalize between 0 and 1

    def calculate_volume_risk_factor(self, data: pd.DataFrame) -> float:
        """Calculate risk factor based on volume patterns"""
        # Get volume volatility
        vol_volatility = self.eva.calculate_volume_volatility(data)
        
        # Calculate volume-based indicators
        vfi = self.eva.volume_force_index(data)
        vzo = self.eva.volume_zone_oscillator(data)
        
        # Normalize indicators
        vol_vol_norm = vol_volatility.iloc[-1] / vol_volatility.max()
        vfi_norm = (vfi.iloc[-1] - vfi.min()) / (vfi.max() - vfi.min())
        vzo_norm = (vzo.iloc[-1] + 100) / 200  # VZO ranges from -100 to +100
        
        # Combine into risk factor
        risk_factor = (
            0.4 * (1 - vol_vol_norm) +  # Lower volatility = lower risk
            0.3 * vfi_norm +            # Higher force index = stronger trend
            0.3 * vzo_norm              # Higher VZO = more bullish volume
        )
        
        return risk_factor

    def adjust_position_for_institutional_activity(self, 
                                                data: pd.DataFrame,
                                                base_size: float,
                                                direction: int) -> float:
        """Adjust position size based on institutional activity"""
        inst_activity = self.va.institutional_activity(data)
        recent_inst = inst_activity.tail(5).sum()  # Check last 5 periods
        
        # Get volume metrics
        volume_profile = self.va.calculate_volume_profile(data)
        current_price = data['Close'].iloc[-1]
        
        # Check if current price is near high-volume level
        volume_levels = volume_profile.nlargest(3).index
        near_volume_level = any(
            abs(current_price - float(level.mid)) / current_price < 0.02 
            for level in volume_levels
        )
        
        # Calculate adjustment factor
        adj_factor = 1.0
        if recent_inst >= 3:  # Strong institutional activity
            adj_factor *= 1.2
        if near_volume_level:  # Price near significant volume level
            adj_factor *= 1.1
        
        return base_size * adj_factor

    def calculate_position_size(self,
                              data: pd.DataFrame,
                              capital: float,
                              signal: int,
                              risk_per_trade: float) -> Tuple[float, Dict[str, float]]:
        """Calculate position size based on volume analysis"""
        if len(data) < 20:
            raise ValueError("Insufficient data for position sizing calculation")
        
        # Get basic metrics
        liquidity_score = self.calculate_liquidity_score(data)
        risk_factor = self.calculate_volume_risk_factor(data)
        
        # Handle NaN values in metrics
        if pd.isna(liquidity_score) or pd.isna(risk_factor):
            liquidity_score = 0.5  # Default to medium liquidity
            risk_factor = 0.5      # Default to medium risk
        
        # Calculate base position size
        max_position = capital * self.config.strategy.max_position_size
        base_size = max_position * liquidity_score * risk_factor
        
        # Ensure base size is finite
        base_size = min(base_size, max_position)
        
        # Apply risk limits
        risk_amount = capital * risk_per_trade
        price = data['Close'].iloc[-1]
        
        # Calculate ATR with error handling
        atr = talib.ATR(
            data['High'].values,
            data['Low'].values,
            data['Close'].values,
            timeperiod=14
        )[-1]
        
        if pd.isna(atr) or atr == 0:
            atr = data['Close'].std()  # Use standard deviation as fallback
        
        # Calculate max position size based on ATR risk
        max_shares = risk_amount / (atr * 2)  # Using 2x ATR for stop loss
        final_size = float(min(base_size, max_shares))  # Ensure float type
        
        # Ensure final size is valid
        if pd.isna(final_size) or final_size <= 0:
            final_size = 0
        
        # Round to nearest lot size
        lot_size = 100
        final_size = round(final_size / lot_size) * lot_size
        
        metrics = {
            'liquidity_score': float(liquidity_score),
            'risk_factor': float(risk_factor),
            'base_size': float(base_size),
            'atr': float(atr)
        }
        
        return final_size, metrics
    
    def adjust_position_for_institutional_activity(self,
                                               data: pd.DataFrame,
                                               base_size: float,
                                               signal: int) -> float:
        """Adjust position size based on institutional activity"""
        # Get recent volume data
        recent_volume = data['Volume'].tail(5)
        avg_volume = recent_volume.mean()
        
        # Check for institutional activity
        volume_threshold = avg_volume * 1.5  # 50% above average
        if recent_volume.iloc[-1] > volume_threshold:
            # Reduce position size on high volume to avoid getting caught in institutional moves
            return base_size * 0.8
        return base_size
        
        metrics = {
            'liquidity_score': liquidity_score,
            'risk_factor': risk_factor,
            'base_size': base_size,
            'adjusted_size': adjusted_size,
            'final_size': final_size
        }
        
        return final_size, metrics

    def get_size_adjustment_factors(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get all factors that influence position sizing"""
        # Volume trend factors
        volume_trend = self.va.volume_trend_strength(data)
        klinger = self.eva.calculate_klinger_oscillator(data)
        ease_of_movement = self.eva.ease_of_movement(data)
        
        # Price-volume relationship
        vol_price_conf = self.eva.volume_price_confirmation(data)
        mfi = talib.MFI(
            data['High'].values,
            data['Low'].values,
            data['Close'].values,
            data['Volume'].values,
            timeperiod=14
        )
        
        return {
            'volume_trend': volume_trend.iloc[-1],
            'klinger_osc': klinger.iloc[-1],
            'eom': ease_of_movement.iloc[-1],
            'vol_price_conf': vol_price_conf.iloc[-1],
            'mfi': mfi[-1]
        }