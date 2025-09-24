import talib
import pandas as pd
from .base_strategy import BaseStrategy

class TALibStrategy(BaseStrategy):
    def generate_signals(self, data, sma_window=5):
        import pandas as pd
        print('DEBUG: data.columns:', data.columns)
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            close_col = [col for col in data.columns if col[0] == 'Close'][0]
            close = data[close_col]
        else:
            close = data['Close']
        print('DEBUG: type(close):', type(close))
        close_flat = close.to_numpy().flatten()
        print('DEBUG after flatten: close_flat type:', type(close_flat), 'shape:', close_flat.shape)
        sma = talib.SMA(close_flat, timeperiod=sma_window)
        close_series = pd.Series(close_flat, index=close.index)
        sma_series = pd.Series(sma, index=close.index)
        # Generate signals: 1 for buy, -1 for sell, 0 for hold
        signals = (close_series > sma_series).astype(int) - (close_series < sma_series).astype(int)
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
