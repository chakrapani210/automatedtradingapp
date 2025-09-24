import backtrader as bt

class BacktestEngine:
    def __init__(self, strategy, data, cash=100000):
        self.strategy = strategy
        self.data = data
        self.cash = cash

    def run(self):
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(self.cash)
        cerebro.addstrategy(self.strategy)
        cerebro.adddata(self.data)
        result = cerebro.run()
        return result
