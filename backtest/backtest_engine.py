import backtrader as bt

class BacktestEngine:
    def __init__(self, strategy_cls, data_dict, signals, weights, cash=100000):
        self.strategy_cls = strategy_cls
        self.data_dict = data_dict  # dict of {ticker: bt.feeds.PandasData}
        self.signals = signals      # dict of {ticker: pd.Series}
        self.weights = weights      # dict of {ticker: float}
        self.cash = cash

    def run(self):
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(self.cash)
        # Add data feeds
        for ticker, datafeed in self.data_dict.items():
            cerebro.adddata(datafeed, name=ticker)
        # Pass signals and weights to strategy
        cerebro.addstrategy(self.strategy_cls, signals=self.signals, weights=self.weights)
        result = cerebro.run()
        return result
