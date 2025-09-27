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
        """Advanced position sizing for options strategy.

        Logic:
          1. Base risk fraction (risk_per_trade) of capital.
          2. Scale with realized volatility (annualized) between vol_low_threshold and vol_high_threshold.
             - Below low: full base size.
             - Above high: floor fraction (vol_floor_fraction) of base size.
             - Linear interpolation between.
          3. Apply hard caps: max_position_fraction and max_position_size (legacy field) if present.
        """
        strat_cfg = getattr(config, 'strategy', config)
        risk_per_trade = getattr(strat_cfg, 'risk_per_trade', 0.01)
        scale_with_vol = getattr(strat_cfg, 'scale_with_vol', True)
        vol_low = getattr(strat_cfg, 'vol_low_threshold', 0.15)
        vol_high = getattr(strat_cfg, 'vol_high_threshold', 0.30)
        vol_floor = getattr(strat_cfg, 'vol_floor_fraction', 0.30)
        max_pos_frac = getattr(strat_cfg, 'max_position_fraction', getattr(strat_cfg, 'max_position_size', 0.05))

        # Realized volatility
        returns = data['Close'].pct_change()
        realized_vol = returns.rolling(window=20).std().iloc[-1] * (252 ** 0.5) if len(returns) >= 20 else 0.0

        base_notional = capital * risk_per_trade
        scale = 1.0
        if scale_with_vol and realized_vol > 0:
            if realized_vol <= vol_low:
                scale = 1.0
            elif realized_vol >= vol_high:
                scale = vol_floor
            else:
                # linear interpolation high -> low size
                span = vol_high - vol_low if vol_high > vol_low else 1.0
                rel = (realized_vol - vol_low) / span
                scale = 1.0 - rel * (1.0 - vol_floor)
        sized_notional = base_notional * scale
        cap_notional = capital * max_pos_frac
        final_notional = min(sized_notional, cap_notional)
        # Return dollar notional (framework above converts to shares/contracts externally)
        return max(0.0, final_notional)

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
