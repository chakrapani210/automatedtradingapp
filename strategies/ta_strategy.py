import talib
import pandas as pd
from .base_strategy import BaseStrategy

class TALibStrategy(BaseStrategy):
    def generate_signals(self, data):
        data['SMA'] = talib.SMA(data['Close'], timeperiod=20)
        data['Signal'] = 0
        data.loc[data['Close'] > data['SMA'], 'Signal'] = 1
        data.loc[data['Close'] < data['SMA'], 'Signal'] = -1
        return data['Signal']
