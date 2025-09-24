
import backtrader as bt



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
    # No custom account manager needed
        # Initialize portfolio value history
        if not hasattr(self.broker, '_value_history'):
            self.broker._value_history = []

    def next(self):
        print(f"DEBUG: next() called for date {self.datas[0].datetime.date(0)}")
        dt = self.datas[0].datetime.date(0)
        # Record portfolio value at each step
        self.broker._value_history.append(self.broker.getvalue())
        if self.current_date == dt:
            return
        self.current_date = dt
        for ticker in self.tickers:
            data = self.datas_by_ticker[ticker]
            signal = self.params.signals[ticker]
            weight = self.params.weights.get(ticker, 0)
            print(f"DEBUG: Ticker={ticker}, Weight={weight}, Signal index sample={signal.index[:5]}")
            # Get signal for current date
            if dt not in signal.index:
                print(f"DEBUG: {ticker} - date {dt} not in signal index")
                continue
            sig = signal.loc[dt]
            print(f"DEBUG: {ticker} - Signal value for {dt}: {sig}")
            pos = self.getposition(data).size
            target_value = self.broker.getvalue() * weight
            price = data.close[0]
            # Detailed trade logging
            print(f"[{dt}] {ticker}: Signal={sig}, Position={pos}, TargetValue={target_value:.2f}, Price={price:.2f}")
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
            signed_qty = qty if order.isbuy() else -qty
            print(f"Order completed: {symbol} {signed_qty} @ {price:.2f}")
