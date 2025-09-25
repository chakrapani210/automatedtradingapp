
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
        self.executed_trades = {ticker: [] for ticker in self.tickers}  # Store executed orders for each ticker
    # No custom account manager needed
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
            if sig > 0 and pos == 0:
                size = int(target_value // price)
                print(f"DEBUG: Calculated BUY size for {ticker}: {size} (target_value={target_value}, price={price})")
                if size > 0:
                    print(f"[{dt}] {ticker}: Placing BUY order for {size} @ {price:.2f}")
                    self.order_refs[ticker] = self.buy(data=data, size=size)
                else:
                    print(f"[{dt}] {ticker}: BUY signal but size=0 (target_value={target_value}, price={price})")
            elif sig < 0 and pos != 0:
                print(f"[{dt}] {ticker}: Placing SELL order for {pos} @ {price:.2f}")
                self.order_refs[ticker] = self.close(data=data)
            elif sig < 0 and pos == 0:
                print(f"[{dt}] {ticker}: SELL signal but no position to close.")
            elif sig > 0 and pos != 0:
                print(f"[{dt}] {ticker}: BUY signal but already in position (pos={pos}).")

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
            # Store executed trade info
            self.executed_trades[symbol].append({
                'date': dt,
                'price': price,
                'qty': signed_qty,
                'type': 'buy' if order.isbuy() else 'sell'
            })
            print(f"Order completed: {symbol} {signed_qty} @ {price:.2f}")
