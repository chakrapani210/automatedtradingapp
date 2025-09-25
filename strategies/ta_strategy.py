import talib
import pandas as pd
from .base_strategy import BaseStrategy

class TALibStrategy(BaseStrategy):
    def generate_signals(self, data, config=None):
        import pandas as pd
        
        # Get parameters from config
        sma_window = config.strategy.sma_window
        sma_long_window = config.strategy.sma_long_window
        rsi_window = config.strategy.rsi_window
        rsi_oversold = config.strategy.rsi_oversold
        rsi_overbought = config.strategy.rsi_overbought
        
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            close = data[('Close',)] if ('Close',) in data.columns else data[(data.columns[0][0], 'Close')]
            high = data[('High',)] if ('High',) in data.columns else data[(data.columns[0][0], 'High')]
            low = data[('Low',)] if ('Low',) in data.columns else data[(data.columns[0][0], 'Low')]
        else:
            close = data['Close']
            high = data['High']
            low = data['Low']
            
        # Calculate indicators
        close_np = close.to_numpy()
        sma = talib.SMA(close_np, timeperiod=sma_window)
        sma_long = talib.SMA(close_np, timeperiod=sma_long_window)
        rsi = talib.RSI(close_np, timeperiod=rsi_window)
        
        # Convert to series for easier comparison
        close_series = pd.Series(close_np, index=close.index)
        sma_series = pd.Series(sma, index=close.index)
        sma_long_series = pd.Series(sma_long, index=close.index)
        rsi_series = pd.Series(rsi, index=close.index)
        
        # Initialize signals
        signals = pd.Series(0, index=close.index)
        
        # Buy conditions: Price > SMAs and RSI > oversold
        buy_condition = (
            (close_series > sma_series) & 
            (sma_series > sma_long_series) & 
            (rsi_series > rsi_oversold)
        )
        
        # Sell conditions: Price < SMAs or RSI > overbought
        sell_condition = (
            (close_series < sma_series) | 
            (sma_series < sma_long_series) | 
            (rsi_series > rsi_overbought)
        )
        
        # Set signals
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals

    def explain_signals(self, data, signals):
        # Example explanation: why each signal was generated
        explanations = []
        for idx in signals.index:
            close = data['Close'].loc[idx]
            sma = data['SMA'].loc[idx] if 'SMA' in data else float('nan')
            if signals.loc[idx] == 1:
                reason = f"BUY: Close ({close:.2f}) > SMA ({sma:.2f})"
            elif signals.loc[idx] == -1:
                reason = f"SELL: Close ({close:.2f}) < SMA ({sma:.2f})"
            else:
                reason = f"HOLD: Close ({close:.2f}) ~= SMA ({sma:.2f})"
            explanations.append({
                'date': idx,
                'signal': signals.loc[idx],
                'close': close,
                'sma': sma,
                'reason': reason
            })
        return explanations
