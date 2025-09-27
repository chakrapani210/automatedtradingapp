import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class StandardStrategy(bt.Strategy):
    """A strategy that uses standard libraries and best practices for signal generation"""
    
    params = (
        ('config', None),  # Will be set from AppConfig
    )

    def __init__(self):
        """Initialize indicators using backtrader's built-in functionality"""
        # Initialize value history list
        self._value_history = []
        
        # Get strategy configuration based on allocation type
        if self.p.config is None:
            raise ValueError("Strategy configuration not provided")
            
        strat_config = self.p.config.short_term_strategy  # Using short-term strategy config by default
        
        # Price indicators
        self.sma = bt.indicators.SMA(period=strat_config.sma_window)
        self.sma_long = bt.indicators.SMA(period=strat_config.sma_long_window)
        
        # Momentum indicators
        self.rsi = bt.indicators.RSI(period=strat_config.rsi_window)
        
        # Volatility indicators
        self.bb = bt.indicators.BollingerBands(
            period=strat_config.bb_period,
            devfactor=strat_config.bb_devfactor
        )
        self.atr = bt.indicators.ATR(period=strat_config.atr_period)
        
        # Volume indicators
        self.volume_ma = bt.indicators.SMA(
            self.data.volume,
            period=strat_config.volume_ma_window
        )
        self.volume_sma_ratio = self.data.volume / self.volume_ma
        
        # Track order and position management
        self.order = None
        self.stop_loss = None
        self.last_price = None
        
        # Initialize trade metrics
        self.trades = []
        self.current_position = None

    def next(self):
        """
        Core strategy logic using built-in indicators
        Returns position signals: 1 (long), -1 (short), 0 (neutral)
        """
        # Update value history
        self._value_history.append(self.broker.getvalue())
        if self.order or not len(self.data):
            return

        # Get current indicator values
        price = self.data.close[0]
        volume = self.data.volume[0]
        
        if not self.position:  # No position - look for entry signals
            # Buy conditions
            # Log signal data for debugging
            self.log_signal_data()
            
            buy_signal = (
                self.data.close > self.sma and  # Price above short MA
                self.data.close > self.sma_long and  # Price above long MA
                self.volume_sma_ratio > 1.0  # Above average volume
            )
            
            if buy_signal:
                self.log(f"BUY SIGNAL Generated - Close: {self.data.close[0]:.2f}, RSI: {self.rsi[0]:.2f}")
            
            # Sell conditions
            sell_signal = (
                self.data.close < self.sma and  # Price below short MA
                self.data.close < self.sma_long and  # Price below long MA
                self.volume_sma_ratio > 1.0  # Above average volume
            )
            
            if sell_signal:
                self.log(f"SELL SIGNAL Generated - Close: {self.data.close[0]:.2f}, RSI: {self.rsi[0]:.2f}")
            
            if buy_signal:
                self.buy_signal()
            elif sell_signal:
                self.sell_signal()
                
        else:  # Have position - look for exit signals
            if self.position.size > 0:  # Long position
                strat_config = self.p.config.short_term_strategy
                if (self.data.close < self.sma or 
                    self.rsi > strat_config.rsi_overbought or 
                    self.data.close < self.stop_loss):
                    self.close()
                    
            else:  # Short position
                if (self.data.close > self.sma or 
                    self.rsi < strat_config.rsi_oversold or 
                    self.data.close > self.stop_loss):
                    self.close()

    def buy_signal(self):
        """Execute buy signal with position sizing and stop loss"""
        size = self.position_size()
        price = self.data.close[0]
        stop_price = price - self.atr[0] * 2  # 2 ATR stop loss
        
        self.order = self.buy(size=size)
        self.stop_loss = stop_price
        self.current_position = {
            'entry_price': price,
            'size': size,
            'stop_loss': stop_price,
            'entry_time': len(self.data)
        }

    def sell_signal(self):
        """Execute sell signal with position sizing and stop loss"""
        size = self.position_size()
        price = self.data.close[0]
        stop_price = price + self.atr[0] * 2  # 2 ATR stop loss
        
        self.order = self.sell(size=size)
        self.stop_loss = stop_price
        self.current_position = {
            'entry_price': price,
            'size': -size,
            'stop_loss': stop_price,
            'entry_time': len(self.data)
        }

    def position_size(self):
        """Calculate position size based on ATR and account risk"""
        strat_config = self.p.config.short_term_strategy
        price = self.data.close[0]
        atr = self.atr[0]
        
        # Risk amount based on account equity
        risk_amount = self.broker.getvalue() * strat_config.risk_per_trade
        
        # Position size based on ATR stop loss
        size = risk_amount / (atr * strat_config.atr_stop_multiplier)
        
        # Limit position size
        max_size = self.broker.getvalue() * strat_config.max_position_size / price
        size = min(size, max_size)
        
        return int(size)

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            # Order has been submitted/accepted - show basic info
            self.log(f'ORDER SUBMITTED/ACCEPTED - Type: {"BUY" if order.isbuy() else "SELL"}, Size: {order.size}')
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED - Price: {order.executed.price:.2f}, Size: {order.executed.size}, '
                        f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.log(f'Portfolio Value: ${self.broker.getvalue():.2f}, Cash: ${self.broker.get_cash():.2f}')
            else:
                self.log(f'SELL EXECUTED - Price: {order.executed.price:.2f}, Size: {order.executed.size}, '
                        f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.log(f'Portfolio Value: ${self.broker.getvalue():.2f}, Cash: ${self.broker.get_cash():.2f}')
                
        elif order.status in [order.Canceled]:
            self.log('Order Canceled')
        elif order.status in [order.Margin]:
            self.log('Order Margin - Not enough funds')
        elif order.status in [order.Rejected]:
            self.log('Order Rejected - Check size and margin requirements')
            
        # Reset the order reference
        self.order = None
        
        # Print current position info
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size != 0:
                self.log(f'Current Position [{data._name}]: {pos.size} shares at {pos.price:.2f}')

    def notify_trade(self, trade):
        """Handle trade notifications"""
        if not trade.isclosed:
            return
            
        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
        
        if self.current_position:
            self.trades.append({
                'entry_time': self.current_position['entry_time'],
                'exit_time': len(self.data),
                'entry_price': self.current_position['entry_price'],
                'exit_price': trade.price,
                'size': self.current_position['size'],
                'pnl': trade.pnlcomm
            })
            self.current_position = None

    def log(self, txt):
        """Logging function"""
        dt = self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')
        
    def log_signal_data(self):
        """Log current indicator values for debugging"""
        for i, d in enumerate(self.datas):
            self.log(f'===== {d._name} Signal Analysis =====')
            self.log(f'Close: {d.close[0]:.2f}')
            self.log(f'SMA: {self.sma[0]:.2f}')
            self.log(f'SMA Long: {self.sma_long[0]:.2f}')
            self.log(f'RSI: {self.rsi[0]:.2f}')
            self.log(f'Volume Ratio: {self.volume_sma_ratio[0]:.2f}')
            self.log(f'ATR: {self.atr[0]:.2f}')

    def get_analysis(self):
        """Return strategy analysis"""
        return {
            'trades': self.trades,
            'sharpe_ratio': bt.analyzers.SharpeRatio(),
            'drawdown': bt.analyzers.DrawDown(),
            'returns': bt.analyzers.Returns(),
            'trade_analyzer': bt.analyzers.TradeAnalyzer()
        }