
import backtrader as bt
import pandas as pd


class MultiAssetStrategy(bt.Strategy):
    params = (
        ('signals', None),  # dict of {ticker: pd.Series}
        ('weights', None),  # dict of {ticker: float}
    )

    def __init__(self):
        self.tickers = list(self.params.signals.keys())
        self.datas_by_ticker = {d._name: d for d in self.datas}
        self.order_refs = {ticker: None for ticker in self.tickers}
        self.current_date = None
        # Track executed trades
        self.executed_trades = {ticker: [] for ticker in self.tickers}
        # Track stop losses and entry prices
        self.stop_losses = {ticker: None for ticker in self.tickers}
        self.entry_prices = {ticker: None for ticker in self.tickers}
        # Get config from params (passed from main.py)
        self.config = self.params.config if hasattr(self.params, 'config') else None
        # Initialize portfolio value history
        if not hasattr(self.broker, '_value_history'):
            self.broker._value_history = []

    def next(self):
        dt = pd.Timestamp(self.datas[0].datetime.date(0))  # Convert to pandas Timestamp
        current_value = self.broker.getvalue()
        print(f"\nDEBUG: ===== next() called for date {dt} =====")
        print(f"Current portfolio value: ${current_value:.2f}")
        # Record portfolio value at each step
        self.broker._value_history.append(current_value)
        if self.current_date == dt:
            return
        self.current_date = dt
        for ticker in self.tickers:
            data = self.datas_by_ticker[ticker]
            signal = self.params.signals[ticker]
            weight = self.params.weights.get(ticker, 0)
            print(f"DEBUG: Ticker={ticker}, Weight={weight}, Signal index sample={signal.index[:5]}")
            # Get signal for current date, safely handle missing dates
            try:
                sig = signal.loc[dt]
                if pd.isna(sig):
                    print(f"DEBUG: {ticker} - Skipping date {dt} (signal is NaN)")
                    continue
                print(f"DEBUG: {ticker} - Signal value for {dt}: {sig}")
                pos = self.getposition(data).size
                target_value = self.broker.getvalue() * weight
                price = data.close[0]
                cash = self.broker.get_cash()
            except KeyError as e:
                print(f"DEBUG: {ticker} - Error getting signal for date {dt}: {e}")
                continue
            # Detailed trade analysis
            print(f"\nDEBUG: Analysis for {ticker} on {dt}:")
            print(f"  - Current position: {pos} shares")
            print(f"  - Portfolio weight: {weight:.4f}")
            print(f"  - Target value: ${target_value:.2f}")
            print(f"  - Available cash: ${cash:.2f}")
            print(f"  - Current price: ${price:.2f}")
            print(f"  - Signal value: {sig}")
            # Debug: print all relevant variables
            # Check stop loss
            if pos != 0 and self.stop_losses[ticker] is not None:
                if pos > 0 and price <= self.stop_losses[ticker]:  # Long position stop loss
                    print(f"[{dt}] {ticker}: Stop loss triggered at {price:.2f} (stop: {self.stop_losses[ticker]:.2f})")
                    self.order_refs[ticker] = self.close(data=data)
                    continue
                elif pos < 0 and price >= self.stop_losses[ticker]:  # Short position stop loss
                    print(f"[{dt}] {ticker}: Stop loss triggered at {price:.2f} (stop: {self.stop_losses[ticker]:.2f})")
                    self.order_refs[ticker] = self.close(data=data)
                    continue

            # Check max drawdown
            if pos != 0 and self.entry_prices[ticker] is not None:
                entry = self.entry_prices[ticker]
                drawdown = (price - entry) / entry if pos > 0 else (entry - price) / entry
                max_drawdown = self.config.get('max_drawdown', 0.10) if self.config else 0.10
                if drawdown < -max_drawdown:
                    print(f"[{dt}] {ticker}: Max drawdown exceeded ({drawdown:.2%})")
                    self.order_refs[ticker] = self.close(data=data)
                    continue

            if sig > 0 and pos == 0:
                size = self.calculate_position_size(ticker, price)
                print(f"DEBUG: Calculated BUY size for {ticker}: {size} (price={price})")
                if size > 0:
                    print(f"[{dt}] {ticker}: Placing BUY order for {size} @ {price:.2f}")
                    self.order_refs[ticker] = self.buy(data=data, size=size)
                else:
                    print(f"[{dt}] {ticker}: BUY signal but size=0 (price={price})")
            elif sig < 0 and pos != 0:
                print(f"[{dt}] {ticker}: Placing SELL order for {pos} @ {price:.2f}")
                self.order_refs[ticker] = self.close(data=data)
            elif sig < 0 and pos == 0:
                print(f"[{dt}] {ticker}: SELL signal but no position to close.")
            elif sig > 0 and pos != 0:
                print(f"[{dt}] {ticker}: BUY signal but already in position (pos={pos}).")

    def calculate_position_size(self, ticker, price):
        """Calculate position size based on volatility (ATR) and risk per trade"""
        if not self.config:
            return 100  # Default size if no config

        equity = self.broker.getvalue()
        risk_per_trade = self.config.get('risk_per_trade', 0.02)  # 2% default
        max_position_size = self.config.get('max_position_size', 0.20)  # 20% default

        # Get ATR value for risk calculation
        data = self.datas_by_ticker[ticker]
        atr = data.atr[0] if hasattr(data, 'atr') else price * 0.02  # Use 2% if ATR not available

        # Calculate position size based on risk and ATR
        risk_amount = equity * risk_per_trade
        shares = risk_amount / (atr * 2)  # Use 2x ATR for stop loss

        # Limit position size
        max_shares = (equity * max_position_size) / price
        shares = min(shares, max_shares)

        return int(shares)

    def set_stop_loss(self, ticker, entry_price, direction):
        """Set stop loss for a position based on ATR"""
        if not self.config:
            return None

        data = self.datas_by_ticker[ticker]
        atr = data.atr[0] if hasattr(data, 'atr') else entry_price * 0.02

        # ATR-based stop loss
        atr_multiplier = self.config.get('stop_loss_atr_multiplier', 2.0)
        if direction == "buy":
            stop_price = entry_price - (atr * atr_multiplier)
        else:
            stop_price = entry_price + (atr * atr_multiplier)

        return stop_price

    def notify_order(self, order):
        symbol = order.data._name
        status_map = {
            order.Submitted: 'Submitted',
            order.Accepted: 'Accepted',
            order.Partial: 'Partial',
            order.Completed: 'Completed',
            order.Canceled: 'Canceled',
            order.Margin: 'Margin',
            order.Rejected: 'Rejected',
        }
        status_str = status_map.get(order.status, str(order.status))
        print(f"notify_order: {symbol} status={status_str}")
        if order.status in [order.Completed]:
            qty = order.executed.size
            price = order.executed.price
            dt = self.datas[0].datetime.date(0)  # Current date
            signed_qty = qty if order.isbuy() else -qty
            order_type = 'buy' if order.isbuy() else 'sell'
            
            # Set or clear stop loss based on order type
            if order_type == 'buy':
                stop_price = self.set_stop_loss(symbol, price, order_type)
                self.stop_losses[symbol] = stop_price
                self.entry_prices[symbol] = price
            else:
                self.stop_losses[symbol] = None
                self.entry_prices[symbol] = None

            # Store executed trade info
            self.executed_trades[symbol].append({
                'date': dt,
                'price': price,
                'qty': signed_qty,
                'type': order_type,
                'stop_loss': self.stop_losses[symbol]
            })
            print(f"Order completed: {symbol} {signed_qty} @ {price:.2f}")
